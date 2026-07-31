from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from research_wp_v19_recall import (
    artifact,
    assert_unique,
    json_safe,
    load_recall_frontier,
    yearly_metrics,
)
from research_wp_v21_margin import (
    add_yearly_economic_metrics,
    concat_or_empty,
)
from wp.v3.contracts import load_v3_config
from wp.v3.io import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
)
from wp.v3.v16_policy import policy_metrics
from wp.v3.v19_recall import (
    DEFAULT_EXPLORATION_PER_SLOT,
    DEFAULT_TOP_PER_SOURCE,
)
from wp.v3.v21_margin import MARGIN_TARGET_PCT, TAIL_LOSS_TARGET_PCT
from wp.v3.v22_market_license import (
    FIXED_MAX_CANDIDATES_PER_DAY,
    FIXED_TARGET_CANDIDATE_DAY_RATE,
    GATE_CALIBRATION_DAYS,
    GATE_PURGE_DAYS,
    GATE_TRAIN_DAYS,
    MARKET_AGGREGATE_FEATURES,
    SCHEMA_VERSION,
    FrozenMarketLicensePolicy,
    MarketLicenseBundle,
    MarketLicensePolicySpec,
    add_market_economic_metrics,
    apply_market_license_policy,
    build_market_slot_leaders,
    calibrate_market_license_policy,
    fit_market_license,
    rolling_market_license_segments,
    v22_research_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
V21_RUN_ID = 30_624_949_710
V21_ARTIFACT_ID = 8_791_116_655
V21_SHA256 = (
    "028d4d67947f1e83e252092e1d88bfb9be89dbcd9589f3faf2119ebca8f17144"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run preregistered V22 market-license research over immutable "
            "V9 out-of-sample predictions."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--top-per-source",
        type=int,
        default=DEFAULT_TOP_PER_SOURCE,
    )
    parser.add_argument(
        "--exploration-per-slot",
        type=int,
        default=DEFAULT_EXPLORATION_PER_SLOT,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_v3_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frontier, source = load_recall_frontier(
        args.shard_dir,
        evaluation_start="00000000",
        evaluation_end=config.history.evaluation_end_date,
        top_per_source=args.top_per_source,
        exploration_per_slot=args.exploration_per_slot,
    )
    leaders = build_market_slot_leaders(frontier)
    assert_unique(leaders, "V22 V9 OOS market-slot leaders")
    evaluation_mask = leaders["trade_date"].astype(str).between(
        config.history.evaluation_start_date,
        config.history.evaluation_end_date,
    )
    evaluation_dates = sorted(
        leaders.loc[evaluation_mask, "trade_date"].astype(str).unique()
    )
    if not evaluation_dates:
        raise RuntimeError("V22 evaluation window has no leader rows")

    folds = sorted(
        pd.to_numeric(leaders.loc[evaluation_mask, "fold"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    numeric_fold = pd.to_numeric(leaders["fold"], errors="coerce")
    fold_rows: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    model_covered_dates: set[str] = set()
    temporal_integrity = True
    feature_integrity = True
    policy_spec = MarketLicensePolicySpec(
        target_candidate_day_rate=FIXED_TARGET_CANDIDATE_DAY_RATE,
        max_candidates_per_day=FIXED_MAX_CANDIDATES_PER_DAY,
    )

    for fold in folds:
        fold_test = leaders.loc[numeric_fold.eq(fold)].copy()
        if fold_test.empty:
            continue
        fold_dates = sorted(fold_test["trade_date"].astype(str).unique())
        test_start = fold_dates[0]
        test = fold_test.loc[
            fold_test["trade_date"].astype(str).between(
                config.history.evaluation_start_date,
                config.history.evaluation_end_date,
            )
        ].copy()
        test_dates = sorted(test["trade_date"].astype(str).unique())
        if not test_dates:
            continue
        overlap = model_covered_dates.intersection(test_dates)
        if overlap:
            raise RuntimeError(
                f"V22 outer test dates overlap: {sorted(overlap)[:5]}"
            )

        history = leaders.loc[
            leaders["trade_date"].astype(str).lt(test_start)
        ].copy()
        segments = rolling_market_license_segments(
            history["trade_date"].astype(str).unique()
        )
        base = {
            "fold": int(fold),
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "test_rows": int(len(test)),
            "test_days": int(len(test_dates)),
        }
        if segments is None:
            empty_metrics = economic_policy_metrics(
                test.head(0),
                total_days=len(test_dates),
                seed=config.model.random_seed + int(fold),
                bootstrap_samples=args.bootstrap_samples,
            )
            fold_rows.append(
                {
                    **base,
                    "scored": False,
                    "reason": "insufficient_prior_oos_market_history",
                    "selected": empty_metrics,
                }
            )
            continue

        train_dates, calibration_dates = segments
        train = labeled_rows(
            history.loc[
                history["trade_date"].astype(str).isin(train_dates)
            ]
        )
        calibration = labeled_rows(
            history.loc[
                history["trade_date"].astype(str).isin(calibration_dates)
            ]
        )
        fold_temporal_integrity = bool(
            train_dates[-1] < calibration_dates[0]
            and calibration_dates[-1] < test_start
        )
        temporal_integrity &= fold_temporal_integrity
        if not fold_temporal_integrity:
            raise RuntimeError(
                f"V22 fold {fold} historical evidence crosses test start"
            )

        print(
            f"[wp-v22] fold={fold} "
            f"train={train_dates[0]}..{train_dates[-1]} rows={len(train):,} "
            f"calibration={calibration_dates[0]}.."
            f"{calibration_dates[-1]} rows={len(calibration):,} "
            f"test={test_dates[0]}..{test_dates[-1]} rows={len(test):,}",
            flush=True,
        )
        bundle = fit_market_license(
            train,
            calibration,
            random_seed=config.model.random_seed + int(fold) * 22_013,
        )
        fold_feature_integrity = validate_feature_contract(
            bundle.feature_columns
        )
        feature_integrity &= fold_feature_integrity
        scored_calibration = bundle.predict(calibration)
        policy = calibrate_market_license_policy(
            scored_calibration,
            calibration_dates=calibration_dates,
            spec=policy_spec,
        )
        scored_test = bundle.predict(test)
        scored_test["v22_source_fold"] = int(fold)
        selected = apply_market_license_policy(scored_test, policy)
        selected["v22_source_fold"] = int(fold)
        validate_selected_contract(selected, policy)

        scored_frames.append(scored_test)
        if not selected.empty:
            selected_frames.append(selected)
        model_covered_dates.update(test_dates)
        selected_metrics = economic_policy_metrics(
            selected,
            total_days=len(test_dates),
            seed=config.model.random_seed + int(fold) * 67,
            bootstrap_samples=args.bootstrap_samples,
        )
        fold_rows.append(
            {
                **base,
                "scored": True,
                "reason": "fixed_prior_oos_market_license_applied",
                "temporal_integrity": fold_temporal_integrity,
                "feature_integrity": fold_feature_integrity,
                "model": {
                    "train_start": train_dates[0],
                    "train_end": train_dates[-1],
                    "train_days": len(train_dates),
                    "train_rows": len(train),
                    "calibration_start": calibration_dates[0],
                    "calibration_end": calibration_dates[-1],
                    "calibration_days": len(calibration_dates),
                    "calibration_rows": len(calibration),
                    "feature_count": len(bundle.feature_columns),
                    "features": list(bundle.feature_columns),
                },
                "policy": policy.as_dict(),
                "selected": selected_metrics,
            }
        )
        print(
            f"[wp-v22] fold={fold} "
            f"threshold="
            f"{policy.license_probability_lower_threshold:.6f} "
            f"events={selected_metrics['events']} "
            f"days={selected_metrics['candidate_days']} "
            f"win={selected_metrics['win_rate']:.4f} "
            f"mean={selected_metrics['mean_net_return_pct']}",
            flush=True,
        )

    scored_all = concat_or_empty(scored_frames, leaders)
    selected_all = concat_or_empty(selected_frames, scored_all)
    assert_unique(scored_all, "V22 nested OOS scored market leaders")
    assert_unique(selected_all, "V22 nested OOS candidates")
    validate_selected_contract(selected_all, None)
    nested_metrics = economic_policy_metrics(
        selected_all,
        total_days=len(evaluation_dates),
        seed=config.model.random_seed + 22_000,
        bootstrap_samples=max(4_000, args.bootstrap_samples),
    )
    yearly = yearly_metrics(
        selected_all,
        total_dates=evaluation_dates,
        seed=config.model.random_seed + 22,
        bootstrap_samples=args.bootstrap_samples,
    )
    add_yearly_economic_metrics(yearly, selected_all)
    readiness = v22_research_readiness(
        nested_metrics,
        yearly=yearly,
        temporal_integrity=temporal_integrity,
        source_integrity=bool(
            source["source_integrity"] and feature_integrity
        ),
    )
    final_bundle, final_policy, final_model = fit_final_market_license(
        leaders,
        random_seed=config.model.random_seed,
        spec=policy_spec,
    )
    final_bundle_path = output / "wp_v22_frozen_research_bundle.joblib"
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "research_only": True,
            "production_authorized": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "market_license": final_bundle,
            "policy": final_policy,
            "source": source,
            "protocol": {
                "license_target": "leader_net_return_pct_gt_0",
                "margin_target_pct": MARGIN_TARGET_PCT,
                "tail_loss_target_pct": TAIL_LOSS_TARGET_PCT,
                **policy_spec.as_dict(),
            },
        },
        final_bundle_path,
        compress=3,
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_authorized": False,
        "objective": (
            "Maximize positive net return probability for executable "
            "14:20-14:50 candidates under the fixed T+1 close exit after "
            "all established costs, by licensing only supportive market "
            "and opportunity-set states."
        ),
        "evaluation_start": config.history.evaluation_start_date,
        "evaluation_end": config.history.evaluation_end_date,
        "evaluation_days": len(evaluation_dates),
        "model_covered_days": len(model_covered_dates),
        "model_coverage_rate": (
            len(model_covered_dates) / max(len(evaluation_dates), 1)
        ),
        "same_historical_window_already_explored": True,
        "historical_result_role": (
            "research_screen_only; cannot replace future 150-day shadow"
        ),
        "mechanism_rationale": {
            "v21_run_id": V21_RUN_ID,
            "v21_artifact_id": V21_ARTIFACT_ID,
            "v21_sha256": V21_SHA256,
            "finding": (
                "The stock-level V21 economic-margin model failed 13 "
                "preregistered gates. V22 leaves the V9 stock selector "
                "unchanged and tests a distinct market-state permission "
                "mechanism."
            ),
            "v21_post_result_subgroups_used_as_policy_rules": False,
        },
        "execution_contract": {
            "entry": config.execution.entry_price_contract,
            "exit": config.execution.exit_order_contract,
            "baseline_all_in_cost_bps": (
                config.execution.baseline_all_in_cost_bps
            ),
            "stress_cost_bps": list(config.execution.stress_cost_bps),
            "failed_exit_penalty_pct": (
                config.execution.non_fill_penalty_pct
            ),
        },
        "protocol": {
            "mechanism": (
                "V9 chooses one leader per slot; a separate market and "
                "opportunity-set classifier decides whether the slot may "
                "release that leader"
            ),
            "license_target": "leader_net_return_pct_gt_0",
            "gate_train_days": GATE_TRAIN_DAYS,
            "gate_calibration_days": GATE_CALIBRATION_DAYS,
            "purge_days": GATE_PURGE_DAYS,
            "policy_family_size": 1,
            "fixed_policy": policy_spec.as_dict(),
            "market_aggregate_features": list(MARKET_AGGREGATE_FEATURES),
            "selected_stock_identity_used_by_license_model": False,
            "selected_stock_level_features_used_by_license_model": False,
            "first_qualifying_signal_is_immutable": True,
            "no_trade_allowed": True,
            "future_information_allowed": False,
            "post_result_threshold_search_allowed": False,
        },
        "source": source,
        "frontier_rows": int(len(frontier)),
        "market_slot_leader_rows": int(len(leaders)),
        "folds": fold_rows,
        "nested_oos_metrics": nested_metrics,
        "yearly": yearly,
        "temporal_integrity": temporal_integrity,
        "feature_integrity": feature_integrity,
        "research_readiness": readiness,
        "final_model": final_model,
        "final_policy": final_policy.as_dict(),
        "frozen_bundle": artifact(final_bundle_path.resolve()),
    }
    atomic_write_json(output / "wp_v22_research_summary.json", summary)
    atomic_write_csv(
        pd.json_normalize(fold_rows, sep="."),
        output / "wp_v22_folds.csv",
    )
    atomic_write_parquet(
        scored_all,
        output / "wp_v22_nested_oos_scored_market_leaders.parquet",
    )
    atomic_write_csv(
        selected_all,
        output / "wp_v22_nested_oos_candidates.csv",
    )
    atomic_write_parquet(
        selected_all,
        output / "wp_v22_nested_oos_candidates.parquet",
    )
    atomic_write_csv(
        pd.json_normalize(yearly, sep="."),
        output / "wp_v22_yearly.csv",
    )
    atomic_write_csv(
        pd.DataFrame(source["shards"]),
        output / "wp_v22_source_shards.csv",
    )
    print(
        "WP_V22_RESULT="
        + json.dumps(
            json_safe(
                {
                    "evaluation_days": len(evaluation_dates),
                    "model_covered_days": len(model_covered_dates),
                    "market_slot_leader_rows": int(len(leaders)),
                    "nested_oos_metrics": nested_metrics,
                    "yearly": yearly,
                    "research_readiness": readiness,
                    "final_policy": final_policy.as_dict(),
                }
            ),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def economic_policy_metrics(
    selected: pd.DataFrame,
    *,
    total_days: int,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    metrics = policy_metrics(
        selected,
        total_days=total_days,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )
    return add_market_economic_metrics(metrics, selected)


def labeled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    net = pd.to_numeric(frame.get("net_return_pct"), errors="coerce")
    target = pd.to_numeric(
        frame.get("target_net_positive"),
        errors="coerce",
    )
    return frame.loc[net.notna() & target.notna()].copy()


def fit_final_market_license(
    leaders: pd.DataFrame,
    *,
    random_seed: int,
    spec: MarketLicensePolicySpec,
) -> tuple[
    MarketLicenseBundle,
    FrozenMarketLicensePolicy,
    dict[str, Any],
]:
    segments = rolling_market_license_segments(
        leaders["trade_date"].astype(str).unique(),
        reserve_final_purge=False,
    )
    if segments is None:
        raise RuntimeError("V22 final license has insufficient OOS history")
    train_dates, calibration_dates = segments
    train = labeled_rows(
        leaders.loc[
            leaders["trade_date"].astype(str).isin(train_dates)
        ]
    )
    calibration = labeled_rows(
        leaders.loc[
            leaders["trade_date"].astype(str).isin(calibration_dates)
        ]
    )
    bundle = fit_market_license(
        train,
        calibration,
        random_seed=random_seed + 220_022,
    )
    validate_feature_contract(bundle.feature_columns)
    scored_calibration = bundle.predict(calibration)
    policy = calibrate_market_license_policy(
        scored_calibration,
        calibration_dates=calibration_dates,
        spec=spec,
    )
    return bundle, policy, {
        "train_start": train_dates[0],
        "train_end": train_dates[-1],
        "train_days": len(train_dates),
        "train_rows": len(train),
        "calibration_start": calibration_dates[0],
        "calibration_end": calibration_dates[-1],
        "calibration_days": len(calibration_dates),
        "calibration_rows": len(calibration),
        "feature_count": len(bundle.feature_columns),
        "features": list(bundle.feature_columns),
        "research_only": True,
        "production_authorized": False,
    }


def validate_feature_contract(features: tuple[str, ...]) -> bool:
    if len(features) < 12:
        raise RuntimeError("V22 market license has fewer than 12 features")
    invalid = [
        feature
        for feature in features
        if not (
            feature.startswith("v22_market_")
            or feature in MARKET_AGGREGATE_FEATURES
        )
    ]
    contaminated = [
        feature
        for feature in features
        if any(
            token in feature.lower()
            for token in (
                "target",
                "net_return",
                "future",
                "truth",
                "ts_code",
                "stock_score",
                "stock_rank",
            )
        )
    ]
    if invalid or contaminated:
        raise RuntimeError(
            "V22 market license feature contract violated: "
            f"invalid={invalid}, contaminated={contaminated}"
        )
    return True


def validate_selected_contract(
    selected: pd.DataFrame,
    policy: FrozenMarketLicensePolicy | None,
) -> None:
    if selected.empty:
        return
    stock_day_duplicates = selected.duplicated(
        ["trade_date", "ts_code"],
        keep=False,
    )
    if stock_day_duplicates.any():
        raise RuntimeError(
            "V22 selected output rewrote a first qualifying signal"
        )
    maximum = (
        policy.spec.max_candidates_per_day
        if policy is not None
        else FIXED_MAX_CANDIDATES_PER_DAY
    )
    per_day = selected.groupby("trade_date", sort=False).size()
    if int(per_day.max()) > maximum:
        raise RuntimeError("V22 selected output exceeds fixed daily maximum")
    slot_minute = selected["signal_slot"].astype(str).str.replace(":", "")
    if not slot_minute.between("1420", "1450").all():
        raise RuntimeError("V22 selected output contains an illegal slot")


if __name__ == "__main__":
    raise SystemExit(main())
