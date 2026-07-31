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
from wp.v3.v23_data import (
    MINIMUM_DATASET_COVERAGE,
    SCHEMA_VERSION as DATA_SCHEMA_VERSION,
    V23_FEATURE_COLUMNS,
)
from wp.v3.v23_microstructure import (
    FIXED_MAX_CANDIDATES_PER_DAY,
    FIXED_TARGET_CANDIDATE_DAY_RATE,
    MODEL_CALIBRATION_DAYS,
    MODEL_FEATURES,
    MODEL_PURGE_DAYS,
    MODEL_TRAIN_DAYS,
    SCHEMA_VERSION,
    FrozenMicrostructurePolicy,
    MicrostructureGateBundle,
    MicrostructurePolicySpec,
    apply_microstructure_policy,
    calibrate_microstructure_policy,
    fit_microstructure_gate,
    fold_test_window,
    labeled_complete_rows,
    load_evaluation_calendar,
    load_full_trade_calendar,
    load_v23_research_source,
    rolling_microstructure_segments,
    selected_outcome_audit,
    v23_research_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
V9_SOURCE_RUN_ID = 30_600_193_544


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered V23 microstructure gate over immutable "
            "V9 out-of-sample leaders."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_v3_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    features, data_manifest, data_integrity = load_v23_features(
        args.data_dir,
    )
    leaders, source = load_v23_research_source(
        args.shard_dir,
        evaluation_end=config.history.evaluation_end_date,
        features=features,
        data_manifest=data_manifest,
    )
    assert_unique(leaders, "V23 immutable outcome-blind V9 leaders")
    joined = join_features(leaders, features)
    assert_unique(joined, "V23 joined source leaders")
    evaluation_dates = load_evaluation_calendar(
        data_manifest,
        start_date=config.history.evaluation_start_date,
        end_date=config.history.evaluation_end_date,
    )
    evaluation_mask = joined["trade_date"].astype(str).isin(
        evaluation_dates
    )
    if not set(
        joined.loc[evaluation_mask, "trade_date"].astype(str).unique()
    ).issubset(set(evaluation_dates)):
        raise RuntimeError("V23 source dates are outside the trade calendar")
    if not evaluation_dates:
        raise RuntimeError("V23 evaluation window has no A-share trade dates")
    data_integrity &= bool(
        source["source_integrity"]
        and int(
            (data_manifest.get("requirements") or {}).get(
                "leader_rows",
                -1,
            )
        )
        == len(features)
    )

    folds = sorted(
        pd.to_numeric(joined.loc[evaluation_mask, "fold"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    numeric_fold = pd.to_numeric(joined["fold"], errors="coerce")
    fold_rows: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    covered_dates: set[str] = set()
    temporal_integrity = True
    feature_integrity = True
    policy_spec = MicrostructurePolicySpec(
        target_candidate_day_rate=FIXED_TARGET_CANDIDATE_DAY_RATE,
        max_candidates_per_day=FIXED_MAX_CANDIDATES_PER_DAY,
    )

    for fold in folds:
        fold_source = joined.loc[numeric_fold.eq(fold)].copy()
        if fold_source.empty:
            continue
        test_start, test_end = fold_test_window(fold_source)
        test = fold_source.loc[
            fold_source["trade_date"].astype(str).isin(evaluation_dates)
        ].copy()
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
                f"V23 outer test dates overlap: {sorted(overlap)[:5]}"
            )
        prior_calendar_dates = [
            date for date in load_full_trade_calendar(data_manifest)
            if date < test_start
        ]
        history = joined.loc[
            joined["trade_date"].astype(str).lt(test_start)
        ].copy()
        segments = rolling_microstructure_segments(prior_calendar_dates)
        base = {
            "fold": int(fold),
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "test_rows": int(len(test)),
            "test_days": int(len(test_dates)),
        }
        if segments is None:
            metrics = economic_policy_metrics(
                test.head(0),
                total_days=len(test_dates),
                seed=config.model.random_seed + int(fold),
                bootstrap_samples=args.bootstrap_samples,
            )
            fold_rows.append(
                {
                    **base,
                    "scored": False,
                    "reason": "insufficient_prior_oos_microstructure_history",
                    "selected": metrics,
                }
            )
            continue

        train_dates, calibration_dates = segments
        train = labeled_complete_rows(
            history.loc[
                history["trade_date"].astype(str).isin(train_dates)
            ]
        )
        calibration = labeled_complete_rows(
            history.loc[
                history["trade_date"].astype(str).isin(calibration_dates)
            ]
        )
        fold_temporal = bool(
            train_dates[-1] < calibration_dates[0]
            and calibration_dates[-1] < test_start
        )
        temporal_integrity &= fold_temporal
        if not fold_temporal:
            raise RuntimeError(
                f"V23 fold {fold} historical evidence crosses test start"
            )
        print(
            f"[wp-v23] fold={fold} "
            f"train={train_dates[0]}..{train_dates[-1]} rows={len(train):,} "
            f"calibration={calibration_dates[0]}.."
            f"{calibration_dates[-1]} rows={len(calibration):,} "
            f"test={test_dates[0]}..{test_dates[-1]} rows={len(test):,}",
            flush=True,
        )
        bundle = fit_microstructure_gate(
            train,
            calibration,
            random_seed=config.model.random_seed + int(fold) * 23_003,
        )
        feature_integrity &= validate_feature_contract(
            bundle.feature_columns
        )
        scored_calibration = bundle.predict(calibration)
        policy = calibrate_microstructure_policy(
            scored_calibration,
            calibration_dates=calibration_dates,
            spec=policy_spec,
        )
        scored_test = bundle.predict(test)
        scored_test["v23_source_fold"] = int(fold)
        selected = apply_microstructure_policy(scored_test, policy)
        selected["v23_source_fold"] = int(fold)
        validate_selected_contract(selected, policy)

        scored_frames.append(scored_test)
        if not selected.empty:
            selected_frames.append(selected)
        covered_dates.update(test_dates)
        metrics = economic_policy_metrics(
            selected,
            total_days=len(test_dates),
            seed=config.model.random_seed + int(fold) * 71,
            bootstrap_samples=args.bootstrap_samples,
        )
        fold_rows.append(
            {
                **base,
                "scored": True,
                "reason": "fixed_prior_oos_v23_policy_applied",
                "temporal_integrity": fold_temporal,
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
                    "return_downside_residual_pct": (
                        bundle.return_downside_residual_pct
                    ),
                },
                "policy": policy.as_dict(),
                "selected": metrics,
            }
        )
        print(
            f"[wp-v23] fold={fold} "
            f"threshold={policy.economic_score_threshold:.6f} "
            f"events={metrics['events']} days={metrics['candidate_days']} "
            f"win={metrics['win_rate']:.4f} "
            f"mean={metrics['mean_net_return_pct']}",
            flush=True,
        )

    scored_all = concat_or_empty(scored_frames, joined)
    selected_all = concat_or_empty(selected_frames, scored_all)
    assert_unique(scored_all, "V23 nested OOS scored leaders")
    assert_unique(selected_all, "V23 nested OOS candidates")
    validate_selected_contract(selected_all, None)
    nested_metrics = economic_policy_metrics(
        selected_all,
        total_days=len(evaluation_dates),
        seed=config.model.random_seed + 23_000,
        bootstrap_samples=max(4_000, args.bootstrap_samples),
    )
    outcome_audit = selected_outcome_audit(
        selected_all,
        total_days=len(evaluation_dates),
    )
    data_integrity &= bool(
        outcome_audit["all_selected_outcomes_verified"]
    )
    yearly = yearly_metrics(
        selected_all,
        total_dates=evaluation_dates,
        seed=config.model.random_seed + 23,
        bootstrap_samples=args.bootstrap_samples,
    )
    add_yearly_economic_metrics(yearly, selected_all)
    readiness = v23_research_readiness(
        nested_metrics,
        yearly=yearly,
        temporal_integrity=temporal_integrity,
        source_integrity=bool(
            source["source_integrity"] and feature_integrity
        ),
        data_integrity=data_integrity,
    )
    final_bundle, final_policy, final_model = fit_final_model(
        joined,
        calendar_dates=load_full_trade_calendar(data_manifest),
        random_seed=config.model.random_seed,
        policy_spec=policy_spec,
    )
    final_bundle_path = output / "wp_v23_frozen_research_bundle.joblib"
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "research_only": True,
            "production_authorized": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "microstructure_gate": final_bundle,
            "policy": final_policy,
            "source": source,
            "data_manifest": data_manifest,
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
            "Maximize the probability that executable 14:20-14:50 "
            "candidates earn positive net return under the fixed T+1 close "
            "exit after all established costs."
        ),
        "evaluation_start": config.history.evaluation_start_date,
        "evaluation_end": config.history.evaluation_end_date,
        "evaluation_days": len(evaluation_dates),
        "model_covered_days": len(covered_dates),
        "model_coverage_rate": (
            len(covered_dates) / max(len(evaluation_dates), 1)
        ),
        "historical_result_role": (
            "research_screen_only; cannot replace future 150-day shadow"
        ),
        "same_historical_window_already_explored": True,
        "new_information_family": (
            "causal 1-minute microstructure, completed opening auction, "
            "and previous-trading-day L2 money flow"
        ),
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
            "source_run_id": V9_SOURCE_RUN_ID,
            "model_train_days": MODEL_TRAIN_DAYS,
            "model_calibration_days": MODEL_CALIBRATION_DAYS,
            "purge_days": MODEL_PURGE_DAYS,
            "policy_family_size": 1,
            "fixed_policy": policy_spec.as_dict(),
            "candidate_day_rate_target": (
                FIXED_TARGET_CANDIDATE_DAY_RATE
            ),
            "maximum_candidates_per_day": (
                FIXED_MAX_CANDIDATES_PER_DAY
            ),
            "model_features": list(MODEL_FEATURES),
            "first_qualifying_signal_is_immutable": True,
            "no_trade_allowed": True,
            "future_information_allowed": False,
            "post_result_threshold_search_allowed": False,
        },
        "source": source,
        "data_manifest": data_manifest,
        "source_leader_rows": int(len(leaders)),
        "joined_rows": int(len(joined)),
        "folds": fold_rows,
        "nested_oos_metrics": nested_metrics,
        "selected_outcome_audit": outcome_audit,
        "yearly": yearly,
        "temporal_integrity": temporal_integrity,
        "feature_integrity": feature_integrity,
        "data_integrity": data_integrity,
        "research_readiness": readiness,
        "final_model": final_model,
        "final_policy": final_policy.as_dict(),
        "frozen_bundle": artifact(final_bundle_path.resolve()),
    }
    atomic_write_json(output / "wp_v23_research_summary.json", summary)
    atomic_write_csv(
        pd.json_normalize(fold_rows, sep="."),
        output / "wp_v23_folds.csv",
    )
    atomic_write_parquet(
        scored_all,
        output / "wp_v23_nested_oos_scored_leaders.parquet",
    )
    atomic_write_csv(
        selected_all,
        output / "wp_v23_nested_oos_candidates.csv",
    )
    atomic_write_parquet(
        selected_all,
        output / "wp_v23_nested_oos_candidates.parquet",
    )
    atomic_write_csv(
        pd.json_normalize(yearly, sep="."),
        output / "wp_v23_yearly.csv",
    )
    print(
        "WP_V23_RESULT="
        + json.dumps(
            json_safe(
                {
                    "evaluation_days": len(evaluation_dates),
                    "model_covered_days": len(covered_dates),
                    "source_leader_rows": int(len(leaders)),
                    "joined_rows": int(len(joined)),
                    "nested_oos_metrics": nested_metrics,
                    "selected_outcome_audit": outcome_audit,
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


def load_v23_features(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    root = Path(data_dir)
    manifests = list(root.rglob("wp_v23_data_manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(
            f"expected one V23 data manifest under {root}; found {len(manifests)}"
        )
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATA_SCHEMA_VERSION:
        raise RuntimeError("V23 data manifest schema mismatch")
    if manifest.get("profit_outcomes_read") is not False:
        raise RuntimeError("V23 data build was not outcome blind")
    if not manifest.get("v23_model_research_authorized"):
        raise RuntimeError("V23 data manifest did not authorize model research")
    coverage = manifest.get("coverage_audit") or {}
    if (
        not coverage.get("coverage_passed")
        or float(coverage.get("complete_coverage_rate", 0.0))
        < MINIMUM_DATASET_COVERAGE
    ):
        raise RuntimeError("V23 data coverage contract failed")
    feature_paths = list(root.rglob("wp_v23_point_in_time_features.parquet"))
    if len(feature_paths) != 1:
        raise RuntimeError(
            f"expected one V23 feature parquet; found {len(feature_paths)}"
        )
    expected_sha = str(
        (manifest.get("artifacts") or {})
        .get("features", {})
        .get("sha256", "")
    )
    actual_sha = file_sha256(feature_paths[0])
    if not expected_sha or actual_sha != expected_sha:
        raise RuntimeError("V23 feature digest mismatch")
    features = pd.read_parquet(feature_paths[0])
    required = {
        "trade_date",
        "signal_slot",
        "ts_code",
        "fold",
        "v23_point_in_time_complete",
        *V23_FEATURE_COLUMNS,
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise RuntimeError(f"V23 feature parquet missing columns: {missing}")
    if features.duplicated(
        ["trade_date", "signal_slot", "ts_code"],
        keep=False,
    ).any():
        raise RuntimeError("V23 feature parquet contains duplicate identities")
    expected_rows = int(
        (manifest.get("requirements") or {}).get("leader_rows", -1)
    )
    if expected_rows != len(features):
        raise RuntimeError(
            "V23 feature row count does not match the data manifest"
        )
    return features, manifest, True


def join_features(
    leaders: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    source = leaders.copy()
    source["trade_date"] = source["trade_date"].astype(str)
    source["signal_slot"] = source["signal_slot"].astype(str)
    source["ts_code"] = source["ts_code"].astype(str)
    feature_frame = features.copy()
    feature_frame["trade_date"] = feature_frame["trade_date"].astype(str)
    feature_frame["signal_slot"] = feature_frame["signal_slot"].astype(str)
    feature_frame["ts_code"] = feature_frame["ts_code"].astype(str)
    source_fold = pd.to_numeric(source["fold"], errors="coerce")
    feature_frame.rename(columns={"fold": "_v23_data_fold"}, inplace=True)
    joined = source.merge(
        feature_frame,
        on=["trade_date", "signal_slot", "ts_code"],
        how="left",
        validate="one_to_one",
    )
    joined_fold = pd.to_numeric(joined["_v23_data_fold"], errors="coerce")
    if not source_fold.reset_index(drop=True).equals(
        joined_fold.reset_index(drop=True)
    ):
        raise RuntimeError("V23 feature folds do not match V9 source folds")
    if joined["v23_point_in_time_complete"].isna().any():
        raise RuntimeError("V23 feature join is missing source identities")
    return joined.drop(columns="_v23_data_fold")


def fit_final_model(
    joined: pd.DataFrame,
    *,
    calendar_dates: list[str],
    random_seed: int,
    policy_spec: MicrostructurePolicySpec,
) -> tuple[
    MicrostructureGateBundle,
    FrozenMicrostructurePolicy,
    dict[str, Any],
]:
    segments = rolling_microstructure_segments(
        calendar_dates,
        reserve_final_purge=False,
    )
    if segments is None:
        raise RuntimeError("V23 final model has insufficient OOS history")
    train_dates, calibration_dates = segments
    train = labeled_complete_rows(
        joined.loc[joined["trade_date"].astype(str).isin(train_dates)]
    )
    calibration = labeled_complete_rows(
        joined.loc[
            joined["trade_date"].astype(str).isin(calibration_dates)
        ]
    )
    bundle = fit_microstructure_gate(
        train,
        calibration,
        random_seed=random_seed + 230_023,
    )
    validate_feature_contract(bundle.feature_columns)
    policy = calibrate_microstructure_policy(
        bundle.predict(calibration),
        calibration_dates=calibration_dates,
        spec=policy_spec,
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
        "return_downside_residual_pct": (
            bundle.return_downside_residual_pct
        ),
        "research_only": True,
        "production_authorized": False,
    }


def validate_feature_contract(features: tuple[str, ...]) -> bool:
    invalid = sorted(set(features) - set(MODEL_FEATURES))
    contaminated = [
        feature
        for feature in features
        if any(
            token in feature.lower()
            for token in (
                "target",
                "truth",
                "future",
                "gross_return",
                "net_return",
                "t1_",
                "exit_price",
            )
        )
    ]
    if len(features) < 20 or invalid or contaminated:
        raise RuntimeError(
            "V23 feature contract violated: "
            f"count={len(features)} invalid={invalid} "
            f"contaminated={contaminated}"
        )
    return True


def validate_selected_contract(
    selected: pd.DataFrame,
    policy: FrozenMicrostructurePolicy | None,
) -> None:
    if selected.empty:
        return
    if selected.duplicated(["trade_date", "ts_code"], keep=False).any():
        raise RuntimeError("V23 selected output rewrote a first signal")
    maximum = (
        policy.spec.max_candidates_per_day
        if policy is not None
        else FIXED_MAX_CANDIDATES_PER_DAY
    )
    if int(selected.groupby("trade_date").size().max()) > maximum:
        raise RuntimeError("V23 selected output exceeds fixed daily maximum")
    slot = selected["signal_slot"].astype(str).str.replace(":", "")
    if not slot.between("1420", "1450").all():
        raise RuntimeError("V23 selected output contains an illegal slot")
    if not selected["v23_point_in_time_complete"].fillna(False).all():
        raise RuntimeError("V23 selected output contains incomplete data")


if __name__ == "__main__":
    raise SystemExit(main())
