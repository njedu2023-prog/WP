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
from wp.v3.v21_margin import (
    DEFAULT_LEADERS_PER_SLOT,
    FIXED_MAX_CANDIDATES_PER_DAY,
    FIXED_TARGET_CANDIDATE_DAY_RATE,
    GATE_CALIBRATION_DAYS,
    GATE_PURGE_DAYS,
    GATE_TRAIN_DAYS,
    MARGIN_TARGET_PCT,
    SCHEMA_VERSION,
    TAIL_LOSS_TARGET_PCT,
    FrozenMarginPolicy,
    MarginGateBundle,
    MarginPolicySpec,
    add_economic_metrics,
    apply_margin_policy,
    build_margin_leaders,
    calibrate_margin_policy,
    fit_margin_gate,
    rolling_margin_model_segments,
    v21_research_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
V20_DIAGNOSIS_RUN_ID = 30_624_371_230
V20_DIAGNOSIS_ARTIFACT_ID = 8_790_726_669
V20_DIAGNOSIS_SHA256 = (
    "5cb3cf1c23b8cac8b8053dd64f7d01f2a3659a27650c313d34a2b0d1e40ccbc6"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run preregistered V21 economic-margin and tail-risk research "
            "over immutable V9 out-of-sample predictions."
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
    parser.add_argument(
        "--leaders-per-slot",
        type=int,
        default=DEFAULT_LEADERS_PER_SLOT,
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
    leaders = build_margin_leaders(
        frontier,
        leaders_per_slot=args.leaders_per_slot,
    )
    assert_unique(leaders, "V21 V9 OOS leaders")
    evaluation_mask = leaders["trade_date"].astype(str).between(
        config.history.evaluation_start_date,
        config.history.evaluation_end_date,
    )
    evaluation_dates = sorted(
        leaders.loc[evaluation_mask, "trade_date"].astype(str).unique()
    )
    if not evaluation_dates:
        raise RuntimeError("V21 evaluation window has no leader rows")

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
    policy_spec = MarginPolicySpec(
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
                f"V21 outer test dates overlap: {sorted(overlap)[:5]}"
            )

        history = leaders.loc[
            leaders["trade_date"].astype(str).lt(test_start)
        ].copy()
        segments = rolling_margin_model_segments(
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
                    "reason": "insufficient_prior_oos_margin_history",
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
                f"V21 fold {fold} historical evidence crosses test start"
            )

        print(
            f"[wp-v21] fold={fold} "
            f"train={train_dates[0]}..{train_dates[-1]} rows={len(train):,} "
            f"calibration={calibration_dates[0]}.."
            f"{calibration_dates[-1]} rows={len(calibration):,} "
            f"test={test_dates[0]}..{test_dates[-1]} rows={len(test):,}",
            flush=True,
        )
        bundle = fit_margin_gate(
            train,
            calibration,
            random_seed=config.model.random_seed + int(fold) * 21_011,
        )
        scored_calibration = bundle.predict(calibration)
        policy = calibrate_margin_policy(
            scored_calibration,
            calibration_dates=calibration_dates,
            spec=policy_spec,
        )
        scored_test = bundle.predict(test)
        scored_test["v21_source_fold"] = int(fold)
        selected = apply_margin_policy(scored_test, policy)
        selected["v21_source_fold"] = int(fold)
        validate_selected_contract(selected, policy)

        scored_frames.append(scored_test)
        if not selected.empty:
            selected_frames.append(selected)
        model_covered_dates.update(test_dates)
        selected_metrics = economic_policy_metrics(
            selected,
            total_days=len(test_dates),
            seed=config.model.random_seed + int(fold) * 61,
            bootstrap_samples=args.bootstrap_samples,
        )
        fold_rows.append(
            {
                **base,
                "scored": True,
                "reason": "fixed_prior_oos_margin_gate_applied",
                "temporal_integrity": fold_temporal_integrity,
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
            f"[wp-v21] fold={fold} "
            f"threshold={policy.margin_probability_lower_threshold:.6f} "
            f"events={selected_metrics['events']} "
            f"days={selected_metrics['candidate_days']} "
            f"win={selected_metrics['win_rate']:.4f} "
            f"margin_hit={selected_metrics['margin_hit_rate']:.4f} "
            f"mean={selected_metrics['mean_net_return_pct']}",
            flush=True,
        )

    scored_all = concat_or_empty(scored_frames, leaders)
    selected_all = concat_or_empty(selected_frames, scored_all)
    assert_unique(scored_all, "V21 nested OOS scored leaders")
    assert_unique(selected_all, "V21 nested OOS candidates")
    validate_selected_contract(selected_all, None)
    nested_metrics = economic_policy_metrics(
        selected_all,
        total_days=len(evaluation_dates),
        seed=config.model.random_seed + 21_000,
        bootstrap_samples=max(4_000, args.bootstrap_samples),
    )
    yearly = yearly_metrics(
        selected_all,
        total_dates=evaluation_dates,
        seed=config.model.random_seed + 21,
        bootstrap_samples=args.bootstrap_samples,
    )
    add_yearly_economic_metrics(yearly, selected_all)
    readiness = v21_research_readiness(
        nested_metrics,
        yearly=yearly,
        temporal_integrity=temporal_integrity,
        source_integrity=bool(source["source_integrity"]),
    )
    final_bundle, final_policy, final_model = fit_final_margin_gate(
        leaders,
        random_seed=config.model.random_seed,
        spec=policy_spec,
    )
    final_bundle_path = output / "wp_v21_frozen_research_bundle.joblib"
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "research_only": True,
            "production_authorized": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "gate": final_bundle,
            "policy": final_policy,
            "source": source,
            "protocol": {
                "leaders_per_slot": args.leaders_per_slot,
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
            "all established costs, using a 50bp economic safety target."
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
            "v20_diagnosis_run_id": V20_DIAGNOSIS_RUN_ID,
            "v20_diagnosis_artifact_id": V20_DIAGNOSIS_ARTIFACT_ID,
            "v20_diagnosis_sha256": V20_DIAGNOSIS_SHA256,
            "finding": (
                "V20 composite score and expected-return estimates had near-"
                "zero relationship with realized return; V21 changes the "
                "training target to an economic margin and models tail loss "
                "separately."
            ),
            "diagnostic_subgroups_used_as_policy_rules": False,
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
                "V9 stock ranking followed by a prior-OOS economic-margin "
                "classifier and an independently calibrated tail-loss gate"
            ),
            "leaders_per_slot": args.leaders_per_slot,
            "margin_target_pct": MARGIN_TARGET_PCT,
            "tail_loss_target_pct": TAIL_LOSS_TARGET_PCT,
            "gate_train_days": GATE_TRAIN_DAYS,
            "gate_calibration_days": GATE_CALIBRATION_DAYS,
            "purge_days": GATE_PURGE_DAYS,
            "policy_family_size": 1,
            "fixed_policy": policy_spec.as_dict(),
            "first_qualifying_signal_is_immutable": True,
            "no_trade_allowed": True,
            "future_information_allowed": False,
            "post_result_threshold_search_allowed": False,
        },
        "source": source,
        "frontier_rows": int(len(frontier)),
        "leader_rows": int(len(leaders)),
        "folds": fold_rows,
        "nested_oos_metrics": nested_metrics,
        "yearly": yearly,
        "temporal_integrity": temporal_integrity,
        "research_readiness": readiness,
        "final_model": final_model,
        "final_policy": final_policy.as_dict(),
        "frozen_bundle": artifact(final_bundle_path.resolve()),
    }
    atomic_write_json(output / "wp_v21_research_summary.json", summary)
    atomic_write_csv(
        pd.json_normalize(fold_rows, sep="."),
        output / "wp_v21_folds.csv",
    )
    atomic_write_parquet(
        scored_all,
        output / "wp_v21_nested_oos_scored_leaders.parquet",
    )
    atomic_write_csv(
        selected_all,
        output / "wp_v21_nested_oos_candidates.csv",
    )
    atomic_write_parquet(
        selected_all,
        output / "wp_v21_nested_oos_candidates.parquet",
    )
    atomic_write_csv(
        pd.json_normalize(yearly, sep="."),
        output / "wp_v21_yearly.csv",
    )
    atomic_write_csv(
        pd.DataFrame(source["shards"]),
        output / "wp_v21_source_shards.csv",
    )
    print(
        "WP_V21_RESULT="
        + json.dumps(
            json_safe(
                {
                    "evaluation_days": len(evaluation_dates),
                    "model_covered_days": len(model_covered_dates),
                    "leader_rows": int(len(leaders)),
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
    return add_economic_metrics(metrics, selected)


def add_yearly_economic_metrics(
    yearly: list[dict[str, Any]],
    selected: pd.DataFrame,
) -> None:
    years = selected["trade_date"].astype(str).str[:4]
    for row in yearly:
        year_frame = selected.loc[years.eq(str(row["year"]))]
        economic = add_economic_metrics({}, year_frame)
        row["margin_hit_rate"] = economic["margin_hit_rate"]
        row["tail_loss_rate"] = economic["tail_loss_rate"]


def labeled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    net = pd.to_numeric(frame.get("net_return_pct"), errors="coerce")
    return frame.loc[net.notna()].copy()


def fit_final_margin_gate(
    leaders: pd.DataFrame,
    *,
    random_seed: int,
    spec: MarginPolicySpec,
) -> tuple[MarginGateBundle, FrozenMarginPolicy, dict[str, Any]]:
    segments = rolling_margin_model_segments(
        leaders["trade_date"].astype(str).unique(),
        reserve_final_purge=False,
    )
    if segments is None:
        raise RuntimeError("V21 final gate has insufficient OOS history")
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
    bundle = fit_margin_gate(
        train,
        calibration,
        random_seed=random_seed + 210_021,
    )
    scored_calibration = bundle.predict(calibration)
    policy = calibrate_margin_policy(
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


def validate_selected_contract(
    selected: pd.DataFrame,
    policy: FrozenMarginPolicy | None,
) -> None:
    if selected.empty:
        return
    stock_day_duplicates = selected.duplicated(
        ["trade_date", "ts_code"],
        keep=False,
    )
    if stock_day_duplicates.any():
        raise RuntimeError(
            "V21 selected output rewrote a first qualifying signal"
        )
    maximum = (
        policy.spec.max_candidates_per_day
        if policy is not None
        else FIXED_MAX_CANDIDATES_PER_DAY
    )
    per_day = selected.groupby("trade_date", sort=False).size()
    if int(per_day.max()) > maximum:
        raise RuntimeError("V21 selected output exceeds fixed daily maximum")
    slot_minute = selected["signal_slot"].astype(str).str.replace(":", "")
    if not slot_minute.between("1420", "1450").all():
        raise RuntimeError("V21 selected output contains an illegal signal slot")


def concat_or_empty(
    frames: list[pd.DataFrame],
    template: pd.DataFrame,
) -> pd.DataFrame:
    if not frames:
        return template.head(0).copy()
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    raise SystemExit(main())
