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
from research_wp_v24_cross_section import (
    join_features,
    load_v24_features,
)
from wp.v3.contracts import load_v3_config
from wp.v3.io import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    file_sha256,
)
from wp.v3.v23_microstructure import (
    fold_test_window,
    load_evaluation_calendar,
    load_full_trade_calendar,
    load_v23_research_source,
    selected_outcome_audit,
)
from wp.v3.v24_cross_section import (
    v24_research_readiness,
    rolling_cross_section_segments,
)
from wp.v3.v25_positioning import (
    SCHEMA_VERSION as DATA_SCHEMA_VERSION,
    V25_FEATURE_COLUMNS,
)
from wp.v3.v25_ranker import (
    FIXED_MAX_CANDIDATES_PER_DAY,
    FIXED_TARGET_CANDIDATE_DAY_RATE,
    MINIMUM_CALIBRATION_PAIR_ROWS,
    MINIMUM_CALIBRATION_ROWS,
    MINIMUM_TRAIN_PAIR_ROWS,
    MINIMUM_TRAIN_ROWS,
    MODEL_CALIBRATION_DAYS,
    MODEL_FEATURES,
    MODEL_PURGE_DAYS,
    MODEL_TRAIN_DAYS,
    SCHEMA_VERSION,
    PositioningPolicySpec,
    PositioningRankBundle,
    FrozenPositioningPolicy,
    apply_positioning_policy,
    calibrate_positioning_policy,
    fit_positioning_ranker,
    labeled_positioning_rows,
    validate_feature_contract,
    validate_selected_contract,
    within_slot_rank_diagnostics,
)


V9_SOURCE_RUN_ID = 30_600_193_544


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run preregistered V25 prior-positioning stock-ranking research."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--v24-data-dir", required=True)
    parser.add_argument("--v25-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_v3_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    v24_features, v24_manifest, v24_integrity = load_v24_features(
        args.v24_data_dir
    )
    v25_features, v25_manifest, v25_integrity = (
        load_v25_positioning_features(
            args.v25_data_dir,
            v24_features=v24_features,
            v24_manifest=v24_manifest,
            v24_data_dir=args.v24_data_dir,
        )
    )
    source_candidates, source = load_v23_research_source(
        args.shard_dir,
        evaluation_end=config.history.evaluation_end_date,
        features=v24_features,
        data_manifest=v24_manifest,
    )
    source = {
        **source,
        "schema_version": "wp_v25_research_source_1",
        "source_candidate_rows": int(len(source_candidates)),
    }
    assert_unique(
        source_candidates,
        "V25 immutable outcome-blind V9 top-five candidates",
    )
    joined = join_features(source_candidates, v24_features)
    joined = join_positioning_features(joined, v25_features)
    assert_unique(joined, "V25 joined source candidates")

    evaluation_dates = load_evaluation_calendar(
        v24_manifest,
        start_date=config.history.evaluation_start_date,
        end_date=config.history.evaluation_end_date,
    )
    if not evaluation_dates:
        raise RuntimeError("V25 evaluation window has no A-share trade dates")
    evaluation_mask = joined["trade_date"].astype(str).isin(
        evaluation_dates
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
    data_integrity = bool(v24_integrity and v25_integrity)
    policy_spec = PositioningPolicySpec(
        target_candidate_day_rate=FIXED_TARGET_CANDIDATE_DAY_RATE,
        max_candidates_per_day=FIXED_MAX_CANDIDATES_PER_DAY,
    )

    for fold in folds:
        fold_source = joined.loc[numeric_fold.eq(fold)].copy()
        if fold_source.empty:
            continue
        test_start, test_end = fold_test_window(fold_source)
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
                f"V25 outer test dates overlap: {sorted(overlap)[:5]}"
            )
        test = fold_source.loc[
            fold_source["trade_date"].astype(str).isin(test_dates)
        ].copy()
        prior_calendar_dates = [
            date
            for date in load_full_trade_calendar(v24_manifest)
            if date < test_start
        ]
        segments = rolling_cross_section_segments(prior_calendar_dates)
        base = {
            "fold": int(fold),
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "test_rows": int(len(test)),
            "test_days": int(len(test_dates)),
        }
        if segments is None:
            fold_rows.append(
                skipped_fold(
                    base,
                    test,
                    test_dates,
                    reason="insufficient_prior_oos_positioning_history",
                    config=config,
                    bootstrap_samples=args.bootstrap_samples,
                )
            )
            continue

        train_dates, calibration_dates = segments
        history = joined.loc[
            joined["trade_date"].astype(str).lt(test_start)
        ].copy()
        train = labeled_positioning_rows(
            history.loc[
                history["trade_date"].astype(str).isin(train_dates)
            ]
        )
        calibration = labeled_positioning_rows(
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
                f"V25 fold {fold} historical evidence crosses test start"
            )
        try:
            bundle = fit_positioning_ranker(
                train,
                calibration,
                random_seed=(
                    config.model.random_seed + int(fold) * 25_003
                ),
            )
        except ValueError as error:
            fold_rows.append(
                {
                    **skipped_fold(
                        base,
                        test,
                        test_dates,
                        reason="insufficient_complete_prior_positioning_rows",
                        config=config,
                        bootstrap_samples=args.bootstrap_samples,
                    ),
                    "temporal_integrity": fold_temporal,
                    "model_error": str(error),
                    "train_rows": int(len(train)),
                    "calibration_rows": int(len(calibration)),
                }
            )
            print(
                f"[wp-v25] fold={fold} skipped: {error}",
                flush=True,
            )
            continue

        feature_integrity &= validate_feature_contract(
            bundle.feature_columns
        )
        scored_calibration = bundle.predict(calibration)
        policy = calibrate_positioning_policy(
            scored_calibration,
            calibration_dates=calibration_dates,
            spec=policy_spec,
        )
        scored_test = bundle.predict(test)
        scored_test["v25_source_fold"] = int(fold)
        selected = apply_positioning_policy(scored_test, policy)
        selected["v25_source_fold"] = int(fold)
        validate_selected_contract(selected, policy)

        scored_frames.append(scored_test)
        if not selected.empty:
            selected_frames.append(selected)
        covered_dates.update(test_dates)
        metrics = economic_policy_metrics(
            selected,
            total_days=len(test_dates),
            seed=config.model.random_seed + int(fold) * 79,
            bootstrap_samples=args.bootstrap_samples,
        )
        rank_diagnostics = within_slot_rank_diagnostics(
            scored_test,
            seed=config.model.random_seed + int(fold) * 83,
            bootstrap_samples=args.bootstrap_samples,
        )
        fold_rows.append(
            {
                **base,
                "scored": True,
                "reason": "fixed_prior_oos_v25_policy_applied",
                "temporal_integrity": fold_temporal,
                "model": bundle_metadata(
                    bundle,
                    train_dates=train_dates,
                    calibration_dates=calibration_dates,
                ),
                "policy": policy.as_dict(),
                "selected": metrics,
                "within_slot_rank": rank_diagnostics,
            }
        )
        print(
            f"[wp-v25] fold={fold} "
            f"events={metrics['events']} days={metrics['candidate_days']} "
            f"win={metrics['win_rate']:.4f} "
            f"mean={metrics['mean_net_return_pct']} "
            f"rank_ic={rank_diagnostics['mean_within_slot_ic']:.4f} "
            f"rank_spread="
            f"{rank_diagnostics['mean_top_minus_bottom_return_pct']:.4f}",
            flush=True,
        )

    scored_all = concat_or_empty(scored_frames, joined)
    selected_all = concat_or_empty(selected_frames, scored_all)
    assert_unique(scored_all, "V25 nested OOS scored candidates")
    assert_unique(selected_all, "V25 nested OOS selected candidates")
    validate_selected_contract(selected_all, None)
    nested_metrics = economic_policy_metrics(
        selected_all,
        total_days=len(evaluation_dates),
        seed=config.model.random_seed + 25_000,
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
        seed=config.model.random_seed + 25,
        bootstrap_samples=args.bootstrap_samples,
    )
    add_yearly_economic_metrics(yearly, selected_all)
    rank_diagnostics = within_slot_rank_diagnostics(
        scored_all,
        seed=config.model.random_seed + 25_025,
        bootstrap_samples=max(4_000, args.bootstrap_samples),
    )
    readiness = v25_research_readiness(
        nested_metrics,
        yearly=yearly,
        rank_diagnostics=rank_diagnostics,
        temporal_integrity=temporal_integrity,
        source_integrity=bool(
            source["source_integrity"] and feature_integrity
        ),
        data_integrity=data_integrity,
    )
    final_bundle, final_policy, final_model = fit_final_model(
        joined,
        calendar_dates=load_full_trade_calendar(v24_manifest),
        random_seed=config.model.random_seed,
        policy_spec=policy_spec,
    )
    bundle_path = output / "wp_v25_frozen_research_bundle.joblib"
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "research_only": True,
            "production_authorized": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "positioning_ranker": final_bundle,
            "policy": final_policy,
            "source": source,
            "v24_data_manifest": v24_manifest,
            "v25_data_manifest": v25_manifest,
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
            "previous-day holder-cost distribution, prior margin "
            "positioning, and prior abnormal-trading disclosure"
        ),
        "evaluation_start": config.history.evaluation_start_date,
        "evaluation_end": config.history.evaluation_end_date,
        "evaluation_days": len(evaluation_dates),
        "model_covered_days": len(covered_dates),
        "model_coverage_rate": (
            len(covered_dates) / max(len(evaluation_dates), 1)
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
            "minimum_train_rows": MINIMUM_TRAIN_ROWS,
            "minimum_calibration_rows": MINIMUM_CALIBRATION_ROWS,
            "minimum_train_pair_rows": MINIMUM_TRAIN_PAIR_ROWS,
            "minimum_calibration_pair_rows": (
                MINIMUM_CALIBRATION_PAIR_ROWS
            ),
            "policy_family_size": 1,
            "fixed_policy": policy_spec.as_dict(),
            "model_features": list(MODEL_FEATURES),
            "same_day_same_slot_pairwise_ranker": True,
            "old_model_outputs_used_as_alpha": False,
            "first_qualifying_signal_is_immutable": True,
            "no_signal_allowed": True,
            "future_information_allowed": False,
            "post_result_threshold_search_allowed": False,
        },
        "source": source,
        "v24_data_manifest": v24_manifest,
        "v25_data_manifest": v25_manifest,
        "source_candidate_rows": int(len(source_candidates)),
        "joined_rows": int(len(joined)),
        "folds": fold_rows,
        "nested_oos_metrics": nested_metrics,
        "within_slot_rank_diagnostics": rank_diagnostics,
        "selected_outcome_audit": outcome_audit,
        "yearly": yearly,
        "temporal_integrity": temporal_integrity,
        "feature_integrity": feature_integrity,
        "data_integrity": data_integrity,
        "research_readiness": readiness,
        "final_model": final_model,
        "final_policy": final_policy.as_dict(),
        "frozen_bundle": artifact(bundle_path.resolve()),
    }
    atomic_write_json(output / "wp_v25_research_summary.json", summary)
    atomic_write_csv(
        pd.json_normalize(fold_rows, sep="."),
        output / "wp_v25_folds.csv",
    )
    atomic_write_parquet(
        scored_all,
        output / "wp_v25_nested_oos_scored_candidates.parquet",
    )
    atomic_write_csv(
        selected_all,
        output / "wp_v25_nested_oos_candidates.csv",
    )
    atomic_write_parquet(
        selected_all,
        output / "wp_v25_nested_oos_candidates.parquet",
    )
    atomic_write_csv(
        pd.json_normalize(yearly, sep="."),
        output / "wp_v25_yearly.csv",
    )
    print(
        "WP_V25_RESULT="
        + json.dumps(
            json_safe(
                {
                    "evaluation_days": len(evaluation_dates),
                    "model_covered_days": len(covered_dates),
                    "source_candidate_rows": int(len(source_candidates)),
                    "joined_rows": int(len(joined)),
                    "nested_oos_metrics": nested_metrics,
                    "within_slot_rank_diagnostics": rank_diagnostics,
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


def load_v25_positioning_features(
    data_dir: str | Path,
    *,
    v24_features: pd.DataFrame,
    v24_manifest: dict[str, Any],
    v24_data_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    root = Path(data_dir)
    manifests = list(root.rglob("wp_v25_data_manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(
            f"expected one V25 data manifest under {root}; "
            f"found {len(manifests)}"
        )
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATA_SCHEMA_VERSION:
        raise RuntimeError("V25 data manifest schema mismatch")
    if manifest.get("profit_outcomes_read") is not False:
        raise RuntimeError("V25 data build was not outcome blind")
    if manifest.get("future_information_allowed") is not False:
        raise RuntimeError("V25 data build permitted future information")
    if not manifest.get("v25_model_research_authorized"):
        raise RuntimeError("V25 data manifest did not authorize research")
    coverage = manifest.get("coverage_audit") or {}
    if not coverage.get("coverage_passed"):
        raise RuntimeError("V25 positioning coverage contract failed")

    feature_paths = list(
        root.rglob("wp_v25_positioning_features.parquet")
    )
    if len(feature_paths) != 1:
        raise RuntimeError(
            f"expected one V25 feature parquet; found {len(feature_paths)}"
        )
    expected_sha = str(
        (manifest.get("artifacts") or {})
        .get("features", {})
        .get("sha256", "")
    )
    if not expected_sha or file_sha256(feature_paths[0]) != expected_sha:
        raise RuntimeError("V25 positioning feature digest mismatch")

    v24_root = Path(v24_data_dir)
    v24_manifest_paths = list(v24_root.rglob("wp_v24_data_manifest.json"))
    v24_feature_paths = list(
        v24_root.rglob("wp_v24_point_in_time_features.parquet")
    )
    if len(v24_manifest_paths) != 1 or len(v24_feature_paths) != 1:
        raise RuntimeError("V25 could not resolve immutable V24 artifacts")
    source = manifest.get("source") or {}
    if (
        str(source.get("v24_manifest_sha256") or "")
        != file_sha256(v24_manifest_paths[0])
        or str(source.get("v24_features_sha256") or "")
        != file_sha256(v24_feature_paths[0])
    ):
        raise RuntimeError("V25 source digests do not match immutable V24")

    features = pd.read_parquet(feature_paths[0])
    required = {
        "trade_date",
        "signal_slot",
        "ts_code",
        "fold",
        "v25_positioning_core_complete",
        *V25_FEATURE_COLUMNS,
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise RuntimeError(f"V25 feature parquet missing columns: {missing}")
    identities = ["trade_date", "signal_slot", "ts_code"]
    if features.duplicated(identities, keep=False).any():
        raise RuntimeError("V25 feature parquet has duplicate identities")
    expected_rows = int(
        (manifest.get("source") or {}).get("candidate_rows", -1)
    )
    if expected_rows != len(features) or len(features) != len(v24_features):
        raise RuntimeError("V25 feature row count changed V24 candidates")
    left = v24_features.loc[:, identities].astype(str).sort_values(
        identities,
        kind="stable",
    )
    right = features.loc[:, identities].astype(str).sort_values(
        identities,
        kind="stable",
    )
    if not left.reset_index(drop=True).equals(
        right.reset_index(drop=True)
    ):
        raise RuntimeError("V25 feature identities differ from immutable V24")
    if str(v24_manifest.get("schema_version") or "") != str(
        source.get("v24_schema_version") or ""
    ):
        raise RuntimeError("V25 source V24 schema mismatch")
    return features, manifest, True


def join_positioning_features(
    joined: pd.DataFrame,
    v25_features: pd.DataFrame,
) -> pd.DataFrame:
    identities = ["trade_date", "signal_slot", "ts_code"]
    columns = [
        *identities,
        "v25_positioning_core_complete",
        *V25_FEATURE_COLUMNS,
    ]
    positioning = v25_features.loc[
        :,
        list(dict.fromkeys(columns)),
    ].copy()
    for column in identities:
        positioning[column] = positioning[column].astype(str)
    result = joined.copy()
    for column in identities:
        result[column] = result[column].astype(str)
    result = result.merge(
        positioning,
        on=identities,
        how="left",
        validate="one_to_one",
    )
    if result["v25_positioning_core_complete"].isna().any():
        raise RuntimeError("V25 positioning join missed source identities")
    return result


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


def bundle_metadata(
    bundle: PositioningRankBundle,
    *,
    train_dates: list[str],
    calibration_dates: list[str],
) -> dict[str, Any]:
    return {
        "train_start": train_dates[0],
        "train_end": train_dates[-1],
        "train_days": len(train_dates),
        "train_rows": bundle.train_rows,
        "train_pair_rows": bundle.train_pair_rows,
        "calibration_start": calibration_dates[0],
        "calibration_end": calibration_dates[-1],
        "calibration_days": len(calibration_dates),
        "calibration_rows": bundle.calibration_rows,
        "calibration_pair_rows": bundle.calibration_pair_rows,
        "feature_count": len(bundle.feature_columns),
        "features": list(bundle.feature_columns),
    }


def v25_research_readiness(
    metrics: dict[str, Any],
    *,
    yearly: list[dict[str, Any]],
    rank_diagnostics: dict[str, Any],
    temporal_integrity: bool,
    source_integrity: bool,
    data_integrity: bool,
) -> dict[str, Any]:
    base = v24_research_readiness(
        metrics,
        yearly=yearly,
        temporal_integrity=temporal_integrity,
        source_integrity=source_integrity,
        data_integrity=data_integrity,
    )
    positive_rank_years = sum(
        float(
            row.get("mean_top_minus_bottom_return_pct") or -999.0
        )
        > 0.0
        for row in rank_diagnostics.get("yearly", [])
    )
    rank_gates = {
        "minimum_1000_evaluable_same_slot_groups": (
            int(rank_diagnostics.get("groups", 0)) >= 1_000
        ),
        "minimum_same_slot_rank_ic": (
            float(
                rank_diagnostics.get("mean_within_slot_ic") or -999.0
            )
            >= 0.05
        ),
        "minimum_same_slot_top_bottom_spread_pct": (
            float(
                rank_diagnostics.get(
                    "mean_top_minus_bottom_return_pct"
                )
                or -999.0
            )
            >= 0.20
        ),
        "clustered_same_slot_spread_lower_positive": (
            float(
                rank_diagnostics.get("clustered_spread_lower_pct")
                or -999.0
            )
            > 0.0
        ),
        "minimum_three_positive_rank_spread_years": (
            positive_rank_years >= 3
        ),
    }
    gates = {**base["gates"], **rank_gates}
    passed = all(gates.values())
    return {
        **base,
        "all_historical_gates_passed": passed,
        "gates": gates,
        "failed_gates": [
            name for name, gate_passed in gates.items() if not gate_passed
        ],
        "production_authorized": False,
        "future_shadow_days_required": 150,
        "future_shadow_min_candidates": 60,
        "future_shadow_min_candidate_days": 40,
        "reason": (
            "historical_screen_passed_future_shadow_still_required"
            if passed
            else "historical_evidence_insufficient"
        ),
    }


def fit_final_model(
    joined: pd.DataFrame,
    *,
    calendar_dates: list[str],
    random_seed: int,
    policy_spec: PositioningPolicySpec,
) -> tuple[
    PositioningRankBundle,
    FrozenPositioningPolicy,
    dict[str, Any],
]:
    segments = rolling_cross_section_segments(
        calendar_dates,
        reserve_final_purge=False,
    )
    if segments is None:
        raise RuntimeError("V25 final model has insufficient OOS history")
    train_dates, calibration_dates = segments
    train = labeled_positioning_rows(
        joined.loc[
            joined["trade_date"].astype(str).isin(train_dates)
        ]
    )
    calibration = labeled_positioning_rows(
        joined.loc[
            joined["trade_date"].astype(str).isin(calibration_dates)
        ]
    )
    bundle = fit_positioning_ranker(
        train,
        calibration,
        random_seed=random_seed + 250_025,
    )
    validate_feature_contract(bundle.feature_columns)
    policy = calibrate_positioning_policy(
        bundle.predict(calibration),
        calibration_dates=calibration_dates,
        spec=policy_spec,
    )
    return bundle, policy, {
        **bundle_metadata(
            bundle,
            train_dates=train_dates,
            calibration_dates=calibration_dates,
        ),
        "research_only": True,
        "production_authorized": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
