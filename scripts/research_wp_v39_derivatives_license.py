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
from research_wp_v22_market_license import economic_policy_metrics
from wp.v3.contracts import load_v3_config
from wp.v3.io import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    file_sha256,
)
from wp.v3.v19_recall import (
    DEFAULT_EXPLORATION_PER_SLOT,
    DEFAULT_TOP_PER_SOURCE,
)
from wp.v3.v22_market_license import build_market_slot_leaders
from wp.v3.v39_derivatives_license import (
    FIXED_MAX_CANDIDATES_PER_DAY,
    FIXED_TARGET_CANDIDATE_DAY_RATE,
    MODEL_CALIBRATION_DAYS,
    MODEL_PURGE_DAYS,
    MODEL_TRAIN_DAYS,
    SCHEMA_VERSION,
    DerivativesLicenseBundle,
    DerivativesPolicySpec,
    FrozenDerivativesPolicy,
    apply_policy,
    calibrate_policy,
    fit_derivatives_license,
    join_derivative_features,
    research_readiness,
    rolling_segments,
    validate_feature_contract,
    validate_selected_contract,
)


V9_SOURCE_RUN_ID = 30_600_193_544
V39_DATA_RUN_ID = 30_687_183_695


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered V39 T-1 derivatives market-license "
            "nested out-of-sample study."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--v39-data-dir", required=True)
    parser.add_argument("--v39-data-run-id", required=True, type=int)
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
    if args.v39_data_run_id != V39_DATA_RUN_ID:
        raise ValueError(
            f"V39 requires immutable data run {V39_DATA_RUN_ID}; "
            f"received {args.v39_data_run_id}"
        )
    config = load_v3_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    derivatives, data_manifest, data_integrity = load_v39_data(
        args.v39_data_dir
    )
    frontier, source = load_recall_frontier(
        args.shard_dir,
        evaluation_start="00000000",
        evaluation_end=config.history.evaluation_end_date,
        top_per_source=args.top_per_source,
        exploration_per_slot=args.exploration_per_slot,
    )
    leaders = build_market_slot_leaders(frontier)
    assert_unique(leaders, "V39 immutable V9 slot leaders")
    joined = join_derivative_features(leaders, derivatives)
    assert_unique(joined, "V39 joined slot leaders")

    evaluation_dates = sorted(
        derivatives.loc[
            derivatives["trade_date"].astype(str).between(
                config.history.evaluation_start_date,
                config.history.evaluation_end_date,
            ),
            "trade_date",
        ]
        .astype(str)
        .unique()
    )
    if not evaluation_dates:
        raise RuntimeError("V39 evaluation window has no trade dates")
    evaluation_mask = joined["trade_date"].astype(str).isin(
        evaluation_dates
    )
    folds = sorted(
        pd.to_numeric(
            joined.loc[evaluation_mask, "fold"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )
    numeric_fold = pd.to_numeric(joined["fold"], errors="coerce")
    policy_spec = DerivativesPolicySpec(
        target_candidate_day_rate=FIXED_TARGET_CANDIDATE_DAY_RATE,
        max_candidates_per_day=FIXED_MAX_CANDIDATES_PER_DAY,
    )

    fold_rows: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    covered_dates: set[str] = set()
    temporal_integrity = True
    feature_integrity = True

    for fold in folds:
        fold_test = joined.loc[numeric_fold.eq(fold)].copy()
        if fold_test.empty:
            continue
        fold_dates = sorted(fold_test["trade_date"].astype(str).unique())
        test_start = fold_dates[0]
        test_end = fold_dates[-1]
        test_dates = [
            date
            for date in evaluation_dates
            if test_start <= date <= test_end
        ]
        if not test_dates:
            continue
        overlap = covered_dates.intersection(test_dates)
        if overlap:
            raise RuntimeError(
                f"V39 outer test dates overlap: {sorted(overlap)[:5]}"
            )
        test = fold_test.loc[
            fold_test["trade_date"].astype(str).isin(test_dates)
        ].copy()
        prior_dates = [date for date in evaluation_dates if date < test_start]
        segments = rolling_segments(prior_dates)
        base = {
            "fold": int(fold),
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "test_days": len(test_dates),
            "test_rows": len(test),
        }
        if segments is None:
            fold_rows.append(
                skipped_fold(
                    base,
                    test,
                    test_dates,
                    reason="insufficient_prior_oos_derivatives_history",
                    config=config,
                    bootstrap_samples=args.bootstrap_samples,
                )
            )
            continue
        train_dates, calibration_dates = segments
        history = joined.loc[
            joined["trade_date"].astype(str).lt(test_start)
        ].copy()
        train = history.loc[
            history["trade_date"].astype(str).isin(train_dates)
        ].copy()
        calibration = history.loc[
            history["trade_date"].astype(str).isin(calibration_dates)
        ].copy()
        fold_temporal = bool(
            train_dates[-1] < calibration_dates[0]
            and calibration_dates[-1] < test_start
        )
        temporal_integrity &= fold_temporal
        if not fold_temporal:
            raise RuntimeError(
                f"V39 fold {fold} history crosses the outer test"
            )
        try:
            bundle = fit_derivatives_license(
                train,
                calibration,
                random_seed=config.model.random_seed + int(fold) * 39_007,
            )
        except ValueError as error:
            fold_rows.append(
                {
                    **skipped_fold(
                        base,
                        test,
                        test_dates,
                        reason="insufficient_complete_prior_v39_rows",
                        config=config,
                        bootstrap_samples=args.bootstrap_samples,
                    ),
                    "temporal_integrity": fold_temporal,
                    "model_error": str(error),
                    "train_rows": len(train),
                    "calibration_rows": len(calibration),
                }
            )
            print(f"[wp-v39] fold={fold} skipped: {error}", flush=True)
            continue

        fold_feature_integrity = validate_feature_contract(
            bundle.feature_columns
        )
        feature_integrity &= fold_feature_integrity
        scored_calibration = bundle.predict(calibration)
        policy = calibrate_policy(
            scored_calibration,
            calibration_dates=calibration_dates,
            spec=policy_spec,
        )
        scored_test = bundle.predict(test)
        scored_test["v39_source_fold"] = int(fold)
        selected = apply_policy(scored_test, policy)
        selected["v39_source_fold"] = int(fold)
        validate_selected_contract(selected, policy)

        scored_frames.append(scored_test)
        if not selected.empty:
            selected_frames.append(selected)
        covered_dates.update(test_dates)
        metrics = economic_policy_metrics(
            selected,
            total_days=len(test_dates),
            seed=config.model.random_seed + int(fold) * 97,
            bootstrap_samples=args.bootstrap_samples,
        )
        fold_rows.append(
            {
                **base,
                "scored": True,
                "reason": "fixed_prior_oos_v39_policy_applied",
                "temporal_integrity": fold_temporal,
                "feature_integrity": fold_feature_integrity,
                "model": bundle_metadata(
                    bundle,
                    train_dates=train_dates,
                    calibration_dates=calibration_dates,
                ),
                "policy": policy.as_dict(),
                "selected": metrics,
            }
        )
        print(
            f"[wp-v39] fold={fold} "
            f"threshold={policy.score_threshold:.6f} "
            f"events={metrics['events']} "
            f"days={metrics['candidate_days']} "
            f"win={metrics['win_rate']:.4f} "
            f"mean={metrics['mean_net_return_pct']}",
            flush=True,
        )

    scored_all = concat_or_empty(scored_frames, joined)
    selected_all = concat_or_empty(selected_frames, joined)
    assert_unique(scored_all, "V39 nested OOS scored leaders")
    assert_unique(selected_all, "V39 nested OOS candidates")
    validate_selected_contract(selected_all, None)

    nested_metrics = economic_policy_metrics(
        selected_all,
        total_days=len(evaluation_dates),
        seed=config.model.random_seed + 39_000,
        bootstrap_samples=max(4_000, args.bootstrap_samples),
    )
    outcome_audit = selected_outcome_audit(selected_all)
    data_integrity &= bool(
        outcome_audit["all_selected_outcomes_verified"]
        and data_manifest["v39_model_research_authorized"]
    )
    yearly = yearly_metrics(
        selected_all,
        total_dates=evaluation_dates,
        seed=config.model.random_seed + 39,
        bootstrap_samples=args.bootstrap_samples,
    )
    add_yearly_economic_metrics(yearly, selected_all)
    readiness = research_readiness(
        nested_metrics,
        yearly=yearly,
        temporal_integrity=temporal_integrity,
        source_integrity=bool(
            source["source_integrity"] and feature_integrity
        ),
        data_integrity=data_integrity,
    )

    final_model: dict[str, Any] | None = None
    final_policy: FrozenDerivativesPolicy | None = None
    bundle_path: Path | None = None
    final_model_error: str | None = None
    if readiness["all_historical_gates_passed"]:
        try:
            final_bundle, final_policy, final_model = fit_final_model(
                joined,
                calendar_dates=evaluation_dates,
                random_seed=config.model.random_seed,
                policy_spec=policy_spec,
            )
        except ValueError as error:
            final_model_error = str(error)
            data_integrity = False
            readiness = research_readiness(
                nested_metrics,
                yearly=yearly,
                temporal_integrity=temporal_integrity,
                source_integrity=bool(
                    source["source_integrity"] and feature_integrity
                ),
                data_integrity=False,
            )
        else:
            bundle_path = output / "wp_v39_frozen_research_bundle.joblib"
            joblib.dump(
                {
                    "schema_version": SCHEMA_VERSION,
                    "research_only": True,
                    "production_authorized": False,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "derivatives_license": final_bundle,
                    "policy": final_policy,
                    "source": source,
                    "v39_data_manifest": data_manifest,
                },
                bundle_path,
                compress=3,
            )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_authorized": False,
        "objective": (
            "Maximize positive net-return probability for executable "
            "14:20-14:50 candidates under the fixed T+1 close exit."
        ),
        "historical_result_role": (
            "research_screen_only; cannot replace future 150-day shadow"
        ),
        "new_information_family": (
            "strict T-1 index-futures basis and positioning plus ETF-option "
            "risk-premium and skew state"
        ),
        "evaluation_start": evaluation_dates[0],
        "evaluation_end": evaluation_dates[-1],
        "evaluation_days": len(evaluation_dates),
        "model_covered_days": len(covered_dates),
        "model_coverage_rate": len(covered_dates) / len(evaluation_dates),
        "execution_contract": {
            "entry": config.execution.entry_price_contract,
            "exit": config.execution.exit_order_contract,
            "baseline_all_in_cost_bps": (
                config.execution.baseline_all_in_cost_bps
            ),
            "stress_cost_bps": list(config.execution.stress_cost_bps),
            "failed_exit_penalty_pct": config.execution.non_fill_penalty_pct,
        },
        "protocol": {
            "source_run_id": V9_SOURCE_RUN_ID,
            "v39_data_run_id": args.v39_data_run_id,
            "model_train_days": MODEL_TRAIN_DAYS,
            "model_calibration_days": MODEL_CALIBRATION_DAYS,
            "purge_days": MODEL_PURGE_DAYS,
            "policy_family_size": 1,
            "fixed_policy": policy_spec.as_dict(),
            "selected_stock_identity_used_by_license_model": False,
            "selected_stock_raw_features_used_by_license_model": False,
            "first_qualifying_signal_is_immutable": True,
            "maximum_candidates_per_day": FIXED_MAX_CANDIDATES_PER_DAY,
            "no_signal_allowed": True,
            "future_information_allowed": False,
            "post_result_threshold_search_allowed": False,
        },
        "source": source,
        "v39_data_manifest": data_manifest,
        "source_frontier_rows": len(frontier),
        "leader_rows": len(leaders),
        "joined_rows": len(joined),
        "folds": fold_rows,
        "nested_oos_metrics": nested_metrics,
        "selected_outcome_audit": outcome_audit,
        "yearly": yearly,
        "temporal_integrity": temporal_integrity,
        "feature_integrity": feature_integrity,
        "data_integrity": data_integrity,
        "research_readiness": readiness,
        "final_model": final_model,
        "final_model_error": final_model_error,
        "final_policy": (
            final_policy.as_dict() if final_policy is not None else None
        ),
        "frozen_bundle": (
            artifact(bundle_path.resolve())
            if bundle_path is not None
            else None
        ),
    }
    atomic_write_json(output / "wp_v39_research_summary.json", summary)
    atomic_write_csv(
        pd.json_normalize(fold_rows, sep="."),
        output / "wp_v39_folds.csv",
    )
    atomic_write_parquet(
        scored_all,
        output / "wp_v39_nested_oos_scored_leaders.parquet",
    )
    atomic_write_csv(
        selected_all,
        output / "wp_v39_nested_oos_candidates.csv",
    )
    atomic_write_parquet(
        selected_all,
        output / "wp_v39_nested_oos_candidates.parquet",
    )
    atomic_write_csv(
        pd.json_normalize(yearly, sep="."),
        output / "wp_v39_yearly.csv",
    )
    print(
        "WP_V39_RESULT="
        + json.dumps(
            json_safe(
                {
                    "evaluation_days": len(evaluation_dates),
                    "model_covered_days": len(covered_dates),
                    "leader_rows": len(leaders),
                    "nested_oos_metrics": nested_metrics,
                    "selected_outcome_audit": outcome_audit,
                    "yearly": yearly,
                    "research_readiness": readiness,
                    "final_model_error": final_model_error,
                }
            ),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def load_v39_data(
    root: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    base = Path(root)
    feature_paths = sorted(base.rglob("wp_v39_derivative_features.parquet"))
    manifest_paths = sorted(
        base.rglob("wp_v39_derivatives_data_manifest.json")
    )
    if len(feature_paths) != 1 or len(manifest_paths) != 1:
        raise RuntimeError(
            "V39 data artifact must contain exactly one feature file and "
            "one manifest"
        )
    features = pd.read_parquet(feature_paths[0])
    manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    expected = manifest["artifacts"]["features"]["sha256"]
    integrity = bool(
        manifest.get("v39_model_research_authorized")
        and file_sha256(feature_paths[0]) == expected
        and len(features)
        == int(manifest["target_window"]["trade_days"])
        and not features["trade_date"].astype(str).duplicated().any()
    )
    if not integrity:
        raise RuntimeError("V39 immutable data artifact failed integrity")
    return features, manifest, integrity


def fit_final_model(
    joined: pd.DataFrame,
    *,
    calendar_dates: list[str],
    random_seed: int,
    policy_spec: DerivativesPolicySpec,
) -> tuple[
    DerivativesLicenseBundle,
    FrozenDerivativesPolicy,
    dict[str, Any],
]:
    segments = rolling_segments(
        calendar_dates,
        reserve_final_purge=False,
    )
    if segments is None:
        raise RuntimeError("V39 final model has insufficient history")
    train_dates, calibration_dates = segments
    train = joined.loc[
        joined["trade_date"].astype(str).isin(train_dates)
    ].copy()
    calibration = joined.loc[
        joined["trade_date"].astype(str).isin(calibration_dates)
    ].copy()
    bundle = fit_derivatives_license(
        train,
        calibration,
        random_seed=random_seed + 39_039,
    )
    validate_feature_contract(bundle.feature_columns)
    policy = calibrate_policy(
        bundle.predict(calibration),
        calibration_dates=calibration_dates,
        spec=policy_spec,
    )
    return bundle, policy, bundle_metadata(
        bundle,
        train_dates=train_dates,
        calibration_dates=calibration_dates,
    )


def bundle_metadata(
    bundle: DerivativesLicenseBundle,
    *,
    train_dates: list[str],
    calibration_dates: list[str],
) -> dict[str, Any]:
    return {
        "train_start": train_dates[0],
        "train_end": train_dates[-1],
        "train_days": len(train_dates),
        "train_rows": bundle.train_rows,
        "calibration_start": calibration_dates[0],
        "calibration_end": calibration_dates[-1],
        "calibration_days": len(calibration_dates),
        "calibration_rows": bundle.calibration_rows,
        "feature_count": len(bundle.feature_columns),
        "features": list(bundle.feature_columns),
        "return_downside_residual_pct": (
            bundle.return_downside_residual_pct
        ),
    }


def skipped_fold(
    base: dict[str, Any],
    test: pd.DataFrame,
    test_dates: list[str],
    *,
    reason: str,
    config: Any,
    bootstrap_samples: int,
) -> dict[str, Any]:
    metrics = economic_policy_metrics(
        test.head(0),
        total_days=len(test_dates),
        seed=config.model.random_seed + int(base["fold"]),
        bootstrap_samples=bootstrap_samples,
    )
    return {
        **base,
        "scored": False,
        "reason": reason,
        "selected": metrics,
    }


def selected_outcome_audit(selected: pd.DataFrame) -> dict[str, Any]:
    if selected.empty:
        return {
            "selected_rows": 0,
            "verified_rows": 0,
            "all_selected_outcomes_verified": True,
        }
    verified = (
        selected["label_available"].fillna(False).astype(bool)
        & pd.to_numeric(
            selected["net_return_pct"],
            errors="coerce",
        ).notna()
    )
    return {
        "selected_rows": len(selected),
        "verified_rows": int(verified.sum()),
        "all_selected_outcomes_verified": bool(verified.all()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
