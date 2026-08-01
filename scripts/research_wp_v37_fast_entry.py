from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pyarrow.parquet as pq

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
from research_wp_v34_intraday_path import (
    join_path_features,
    load_v34_path_features,
    path_rank_diagnostics,
    validate_v34_v24_source,
)
from wp.v3.contracts import load_v3_config
from wp.v3.io import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    file_sha256,
)
from wp.v3.meta_alpha import IDENTITY_COLUMNS
from wp.v3.v23_microstructure import (
    fold_test_window,
    load_evaluation_calendar,
    load_full_trade_calendar,
    load_v23_research_source,
    selected_outcome_audit,
)
from wp.v3.v24_cross_section import rolling_cross_section_segments
from wp.v3.v34_path_ranker import (
    MARGIN_TARGET_PCT,
    MINIMUM_CALIBRATION_ROWS,
    MINIMUM_TRAIN_ROWS,
    MODEL_CALIBRATION_DAYS,
    MODEL_FEATURES,
    MODEL_PURGE_DAYS,
    MODEL_TRAIN_DAYS,
    SEVERE_TARGET_PCT,
    PathRankBundle,
    add_path_context_features,
    fit_path_ranker,
    labeled_path_rows,
    validate_feature_contract,
)
from wp.v3.v37_fast_entry import (
    BASE_ALERT_SLOTS,
    ENTRY_DELAY_MINUTES,
    FIXED_TARGET_CANDIDATE_DAY_RATE,
    OUTCOME_COLUMNS,
    PUBLICATION_DELAY_MINUTES,
    QUALITY_COLUMNS,
    SCHEMA_VERSION,
    FastEntryPolicySpec,
    FrozenFastEntryPolicy,
    apply_fast_entry_policy,
    audit_fast_entry_outcomes,
    build_fast_entry_outcomes,
    calibrate_fast_entry_policy,
    join_fast_entry_outcomes,
    selected_execution_audit,
    v37_research_readiness,
    validate_selected_contract,
)


V9_SOURCE_RUN_ID = 30_600_193_544
V24_DATA_RUN_ID = 30_635_569_735
V34_DATA_RUN_ID = 30_677_075_531

SOURCE_OUTCOME_RENAME = {
    "entry_price": "source_entry_price",
    "gross_return_pct": "source_gross_return_pct",
    "net_return_pct": "source_net_return_pct",
    "entry_fillable": "source_entry_fillable",
    "exit_fillable": "source_exit_fillable",
    "execution_success": "source_execution_success",
    "label_available": "source_label_available",
    "target_net_positive": "source_target_net_positive",
    "target_severe_loss": "source_target_severe_loss",
}

PANEL_TRUTH_COLUMNS = (
    *IDENTITY_COLUMNS,
    "target_trade_date",
    "t1_close",
    "t1_total_return_close",
    "exit_fillable",
    "label_available",
    "up_limit",
    "down_limit",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered V37 fast-entry-contract nested OOS study."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--v24-data-dir", required=True)
    parser.add_argument("--v34-data-dir", required=True)
    parser.add_argument("--v34-data-run-id", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.v34_data_run_id != V34_DATA_RUN_ID:
        raise ValueError(
            "V37 requires immutable V34 data run "
            f"{V34_DATA_RUN_ID}; received {args.v34_data_run_id}"
        )
    config = load_v3_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    v24_features, v24_manifest, v24_integrity = load_v24_features(
        args.v24_data_dir
    )
    path_features, v34_manifest, v34_integrity = load_v34_path_features(
        args.v34_data_dir
    )
    validate_v34_v24_source(args.v24_data_dir, v34_manifest)
    source_candidates, source = load_v23_research_source(
        args.shard_dir,
        evaluation_end=config.history.evaluation_end_date,
        features=v24_features,
        data_manifest=v24_manifest,
    )
    source = {
        **source,
        "schema_version": "wp_v37_research_source_1",
        "source_candidate_rows": int(len(source_candidates)),
    }
    assert_unique(source_candidates, "V37 immutable V9 top-five candidates")

    joined = join_features(source_candidates, v24_features)
    joined = join_path_features(joined, path_features)
    joined = add_path_context_features(joined)
    joined = joined.loc[
        joined["signal_slot"].astype(str).isin(BASE_ALERT_SLOTS)
    ].copy()
    joined.rename(columns=SOURCE_OUTCOME_RENAME, inplace=True)
    panel_truth, panel_audit = load_candidate_panel_truth(
        args.panel_dir,
        joined,
    )
    joined = attach_panel_truth(joined, panel_truth)
    outcomes, minute_source = build_outcomes_from_v34_partitions(
        joined,
        args.v34_data_dir,
        v34_manifest,
        config=config,
    )
    outcome_audit_all = audit_fast_entry_outcomes(outcomes, joined)
    candidates = join_fast_entry_outcomes(joined, outcomes)
    assert_unique(candidates, "V37 fast-entry candidate frame")

    evaluation_dates = load_evaluation_calendar(
        v24_manifest,
        start_date=config.history.evaluation_start_date,
        end_date=config.history.evaluation_end_date,
    )
    if not evaluation_dates:
        raise RuntimeError("V37 evaluation window has no A-share trade dates")
    evaluation_mask = candidates["trade_date"].astype(str).isin(
        evaluation_dates
    )
    folds = sorted(
        pd.to_numeric(
            candidates.loc[evaluation_mask, "fold"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )
    numeric_fold = pd.to_numeric(candidates["fold"], errors="coerce")
    full_calendar = load_full_trade_calendar(v24_manifest)
    fold_rows: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    covered_dates: set[str] = set()
    temporal_integrity = True
    feature_integrity = True
    data_integrity = bool(
        v24_integrity
        and v34_integrity
        and panel_audit["source_integrity"]
        and minute_source["source_integrity"]
        and outcome_audit_all["coverage_passed"]
    )
    policy_spec = FastEntryPolicySpec(
        target_candidate_day_rate=FIXED_TARGET_CANDIDATE_DAY_RATE,
        max_candidates_per_day=3,
    )

    for fold in folds:
        fold_source = candidates.loc[numeric_fold.eq(fold)].copy()
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
                f"V37 outer test dates overlap: {sorted(overlap)[:5]}"
            )
        test = fold_source.loc[
            fold_source["trade_date"].astype(str).isin(test_dates)
        ].copy()
        prior_dates = [date for date in full_calendar if date < test_start]
        segments = rolling_cross_section_segments(prior_dates)
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
                    test_dates,
                    reason="insufficient_prior_oos_fast_entry_history",
                    config=config,
                    bootstrap_samples=args.bootstrap_samples,
                )
            )
            continue
        train_dates, calibration_dates = segments
        history = candidates.loc[
            candidates["trade_date"].astype(str).lt(test_start)
        ].copy()
        train = labeled_path_rows(
            history.loc[
                history["trade_date"].astype(str).isin(train_dates)
            ]
        )
        calibration = labeled_path_rows(
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
                f"V37 fold {fold} history crosses the outer test"
            )
        try:
            bundle = fit_path_ranker(
                train,
                calibration,
                random_seed=config.model.random_seed + int(fold) * 37_003,
            )
        except ValueError as error:
            fold_rows.append(
                {
                    **skipped_fold(
                        base,
                        test_dates,
                        reason="insufficient_complete_prior_fast_entry_rows",
                        config=config,
                        bootstrap_samples=args.bootstrap_samples,
                    ),
                    "temporal_integrity": fold_temporal,
                    "model_error": str(error),
                    "train_rows": int(len(train)),
                    "calibration_rows": int(len(calibration)),
                }
            )
            print(f"[wp-v37] fold={fold} skipped: {error}", flush=True)
            continue

        feature_integrity &= validate_feature_contract(
            bundle.feature_columns
        )
        scored_calibration = bundle.predict(calibration)
        policy = calibrate_fast_entry_policy(
            scored_calibration,
            calibration_dates=calibration_dates,
            spec=policy_spec,
        )
        scored_test = bundle.predict(test)
        scored_test["v37_source_fold"] = int(fold)
        selected = apply_fast_entry_policy(scored_test, policy)
        selected["v37_source_fold"] = int(fold)
        validate_selected_contract(selected, policy)

        scored_frames.append(scored_test)
        if not selected.empty:
            selected_frames.append(selected)
        covered_dates.update(test_dates)
        metrics = economic_policy_metrics(
            selected,
            total_days=len(test_dates),
            seed=config.model.random_seed + int(fold) * 89,
            bootstrap_samples=args.bootstrap_samples,
        )
        execution = selected_execution_audit(selected)
        rank = path_rank_diagnostics(
            scored_test,
            seed=config.model.random_seed + int(fold) * 97,
            bootstrap_samples=args.bootstrap_samples,
        )
        fold_rows.append(
            {
                **base,
                "scored": True,
                "reason": "fixed_prior_oos_v37_fast_entry_policy_applied",
                "temporal_integrity": fold_temporal,
                "model": bundle_metadata(
                    bundle,
                    train_dates=train_dates,
                    calibration_dates=calibration_dates,
                ),
                "policy": policy.as_dict(),
                "selected": metrics,
                "execution": execution,
                "within_slot_rank": rank,
            }
        )
        print(
            f"[wp-v37] fold={fold} "
            f"threshold={policy.score_threshold:.6f} "
            f"events={metrics['events']} "
            f"days={metrics['candidate_days']} "
            f"entry_fill={execution['entry_fill_rate']:.4f} "
            f"win={metrics['win_rate']:.4f} "
            f"mean={metrics['mean_net_return_pct']} "
            f"rank_ic={rank['mean_within_slot_ic']:.4f}",
            flush=True,
        )

    scored_all = concat_or_empty(scored_frames, candidates)
    selected_all = concat_or_empty(selected_frames, scored_all)
    assert_unique(scored_all, "V37 nested OOS scored candidates")
    assert_unique(selected_all, "V37 nested OOS selected candidates")
    validate_selected_contract(selected_all, None)

    nested_metrics = economic_policy_metrics(
        selected_all,
        total_days=len(evaluation_dates),
        seed=config.model.random_seed + 37_000,
        bootstrap_samples=max(4_000, args.bootstrap_samples),
    )
    rank_diagnostics = path_rank_diagnostics(
        scored_all,
        seed=config.model.random_seed + 37_037,
        bootstrap_samples=max(4_000, args.bootstrap_samples),
    )
    outcome_audit = selected_outcome_audit(
        selected_all,
        total_days=len(evaluation_dates),
    )
    execution_audit = selected_execution_audit(selected_all)
    data_integrity &= bool(
        outcome_audit["all_selected_outcomes_verified"]
    )
    yearly = yearly_metrics(
        selected_all,
        total_dates=evaluation_dates,
        seed=config.model.random_seed + 37,
        bootstrap_samples=args.bootstrap_samples,
    )
    add_yearly_economic_metrics(yearly, selected_all)
    readiness = v37_research_readiness(
        nested_metrics,
        yearly=yearly,
        execution_audit=execution_audit,
        temporal_integrity=temporal_integrity,
        source_integrity=bool(
            source["source_integrity"] and feature_integrity
        ),
        data_integrity=data_integrity,
    )

    final_model: dict[str, Any] | None = None
    final_policy: FrozenFastEntryPolicy | None = None
    bundle_path: Path | None = None
    final_model_error: str | None = None
    if readiness["all_historical_gates_passed"]:
        try:
            final_bundle, final_policy, final_model = fit_final_model(
                candidates,
                calendar_dates=full_calendar,
                random_seed=config.model.random_seed,
                policy_spec=policy_spec,
            )
        except ValueError as error:
            final_model_error = str(error)
            data_integrity = False
            readiness = v37_research_readiness(
                nested_metrics,
                yearly=yearly,
                execution_audit=execution_audit,
                temporal_integrity=temporal_integrity,
                source_integrity=bool(
                    source["source_integrity"] and feature_integrity
                ),
                data_integrity=False,
            )
        else:
            bundle_path = output / "wp_v37_frozen_research_bundle.joblib"
            joblib.dump(
                {
                    "schema_version": SCHEMA_VERSION,
                    "research_only": True,
                    "production_authorized": False,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "path_ranker_retrained_on_fast_entry_labels": (
                        final_bundle
                    ),
                    "policy": final_policy,
                    "source": source,
                    "v24_data_manifest": v24_manifest,
                    "v34_data_manifest": v34_manifest,
                    "panel_source": panel_audit,
                    "minute_source": minute_source,
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
        "same_historical_window_already_explored": True,
        "distinct_hypothesis": (
            "replace the old t+5 close entry benchmark with a fixed "
            "publication-at-t+2 and entry-at-t+3 contract"
        ),
        "evaluation_start": config.history.evaluation_start_date,
        "evaluation_end": config.history.evaluation_end_date,
        "evaluation_days": len(evaluation_dates),
        "model_covered_days": len(covered_dates),
        "model_coverage_rate": (
            len(covered_dates) / max(len(evaluation_dates), 1)
        ),
        "execution_contract": {
            "base_alert_slots": list(BASE_ALERT_SLOTS),
            "publication_delay_minutes": PUBLICATION_DELAY_MINUTES,
            "entry_delay_minutes": ENTRY_DELAY_MINUTES,
            "entry_benchmark": "exact_t_plus_3_one_minute_close_plus_10bps",
            "last_publication": "14:47",
            "last_entry_benchmark": "14:48",
            "exit": config.execution.exit_order_contract,
            "round_trip_cost_bps": (
                config.execution.round_trip_cost_bps
            ),
            "stress_cost_bps": list(config.execution.stress_cost_bps),
            "failed_entry_result": "cash_zero_return",
            "failed_exit_penalty_pct": (
                config.execution.non_fill_penalty_pct
            ),
        },
        "protocol": {
            "source_run_id": V9_SOURCE_RUN_ID,
            "v24_data_run_id": V24_DATA_RUN_ID,
            "v34_data_run_id": args.v34_data_run_id,
            "model_train_days": MODEL_TRAIN_DAYS,
            "model_calibration_days": MODEL_CALIBRATION_DAYS,
            "purge_days": MODEL_PURGE_DAYS,
            "minimum_train_rows": MINIMUM_TRAIN_ROWS,
            "minimum_calibration_rows": MINIMUM_CALIBRATION_ROWS,
            "policy_family_size": 1,
            "fixed_policy": policy_spec.as_dict(),
            "margin_target_pct": MARGIN_TARGET_PCT,
            "severe_loss_target_pct": SEVERE_TARGET_PCT,
            "model_features": list(MODEL_FEATURES),
            "outcome_columns": list(OUTCOME_COLUMNS),
            "quality_columns": list(QUALITY_COLUMNS),
            "first_qualifying_signal_is_immutable": True,
            "maximum_candidates_per_day": 3,
            "no_signal_allowed": True,
            "future_information_allowed": False,
            "entry_bar_feature_use_allowed": False,
            "post_result_threshold_search_allowed": False,
        },
        "source": source,
        "v24_data_manifest": v24_manifest,
        "v34_data_manifest": v34_manifest,
        "panel_source": panel_audit,
        "minute_source": minute_source,
        "fast_entry_outcome_audit": outcome_audit_all,
        "source_candidate_rows": int(len(source_candidates)),
        "legal_base_candidate_rows": int(len(candidates)),
        "folds": fold_rows,
        "nested_oos_metrics": nested_metrics,
        "within_slot_rank_diagnostics": rank_diagnostics,
        "selected_outcome_audit": outcome_audit,
        "selected_execution_audit": execution_audit,
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
    atomic_write_json(output / "wp_v37_research_summary.json", summary)
    atomic_write_parquet(
        outcomes,
        output / "wp_v37_fast_entry_outcomes.parquet",
    )
    atomic_write_csv(
        pd.json_normalize(fold_rows, sep="."),
        output / "wp_v37_folds.csv",
    )
    atomic_write_parquet(
        scored_all,
        output / "wp_v37_nested_oos_scored_candidates.parquet",
    )
    atomic_write_csv(
        selected_all,
        output / "wp_v37_nested_oos_candidates.csv",
    )
    atomic_write_parquet(
        selected_all,
        output / "wp_v37_nested_oos_candidates.parquet",
    )
    atomic_write_csv(
        pd.json_normalize(yearly, sep="."),
        output / "wp_v37_yearly.csv",
    )
    print(
        "WP_V37_RESULT="
        + json.dumps(
            json_safe(
                {
                    "evaluation_days": len(evaluation_dates),
                    "model_covered_days": len(covered_dates),
                    "source_candidate_rows": int(len(source_candidates)),
                    "legal_base_candidate_rows": int(len(candidates)),
                    "fast_entry_outcome_audit": outcome_audit_all,
                    "nested_oos_metrics": nested_metrics,
                    "within_slot_rank_diagnostics": rank_diagnostics,
                    "selected_outcome_audit": outcome_audit,
                    "selected_execution_audit": execution_audit,
                    "yearly": yearly,
                    "research_readiness": readiness,
                    "final_model_error": final_model_error,
                    "final_policy": (
                        final_policy.as_dict()
                        if final_policy is not None
                        else None
                    ),
                }
            ),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def load_candidate_panel_truth(
    panel_dir: str | Path,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(panel_dir)
    required = candidates.loc[
        :,
        [*IDENTITY_COLUMNS],
    ].copy()
    for column in IDENTITY_COLUMNS:
        required[column] = required[column].astype(str)
    required["_v37_month"] = required["trade_date"].str[:6]
    frames: list[pd.DataFrame] = []
    partitions: list[dict[str, Any]] = []
    for month, month_required in required.groupby(
        "_v37_month",
        sort=True,
    ):
        path = root / f"wp_v3_panel_{month}.parquet"
        if not path.is_file():
            raise FileNotFoundError(
                f"V37 panel partition missing: {path.name}"
            )
        available = set(pq.read_schema(path).names)
        missing = sorted(set(PANEL_TRUTH_COLUMNS) - available)
        if missing:
            raise RuntimeError(
                f"V37 panel partition {path.name} missing {missing}"
            )
        dates = sorted(month_required["trade_date"].unique())
        panel = pd.read_parquet(
            path,
            columns=list(PANEL_TRUTH_COLUMNS),
            filters=[("trade_date", "in", dates)],
        )
        for column in IDENTITY_COLUMNS:
            panel[column] = panel[column].astype(str)
        wanted = month_required.drop(columns="_v37_month")
        selected = wanted.merge(
            panel,
            on=list(IDENTITY_COLUMNS),
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        if not selected["_merge"].eq("both").all():
            raise RuntimeError(
                f"V37 panel truth missed identities in {path.name}"
            )
        frames.append(selected.drop(columns="_merge"))
        partitions.append(
            {
                "name": path.name,
                "candidate_rows": int(len(wanted)),
                "selected_rows": int(len(selected)),
                "sha256": file_sha256(path),
            }
        )
    truth = pd.concat(frames, ignore_index=True)
    truth.sort_values(
        list(IDENTITY_COLUMNS),
        kind="stable",
        inplace=True,
    )
    truth.reset_index(drop=True, inplace=True)
    if truth.duplicated(list(IDENTITY_COLUMNS), keep=False).any():
        raise RuntimeError("V37 panel truth identities are duplicated")
    return truth, {
        "schema_version": "wp_v37_panel_truth_source_1",
        "partition_count": len(partitions),
        "candidate_rows": int(len(candidates)),
        "truth_rows": int(len(truth)),
        "partitions": partitions,
        "source_integrity": bool(len(truth) == len(candidates)),
    }


def attach_panel_truth(
    candidates: pd.DataFrame,
    panel_truth: pd.DataFrame,
) -> pd.DataFrame:
    result = candidates.merge(
        panel_truth.rename(
            columns={
                "target_trade_date": "_panel_target_trade_date",
                "t1_close": "_panel_t1_close",
                "exit_fillable": "_panel_exit_fillable",
                "label_available": "_panel_label_available",
            }
        ),
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not result["_merge"].eq("both").all():
        raise RuntimeError("V37 failed to attach panel truth")
    if not result["target_trade_date"].astype(str).eq(
        result["_panel_target_trade_date"].astype(str)
    ).all():
        raise RuntimeError("V37 panel target dates differ from source")
    if not normalized_bool(result["source_exit_fillable"]).eq(
        normalized_bool(result["_panel_exit_fillable"])
    ).all():
        raise RuntimeError("V37 panel exit truth differs from source")
    if not normalized_bool(result["source_label_available"]).eq(
        normalized_bool(result["_panel_label_available"])
    ).all():
        raise RuntimeError("V37 panel label availability differs from source")
    source_t1 = pd.to_numeric(result["t1_close"], errors="coerce")
    panel_t1 = pd.to_numeric(result["_panel_t1_close"], errors="coerce")
    if not source_t1.fillna(-1.0).eq(panel_t1.fillna(-1.0)).all():
        raise RuntimeError("V37 panel T+1 close differs from source")
    return result.drop(
        columns=[
            "_panel_target_trade_date",
            "_panel_t1_close",
            "_panel_exit_fillable",
            "_panel_label_available",
            "_merge",
        ]
    )


def normalized_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y"}
    )


def build_outcomes_from_v34_partitions(
    candidates: pd.DataFrame,
    data_dir: str | Path,
    manifest: dict[str, Any],
    *,
    config: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(data_dir)
    contracts = (
        (manifest.get("artifacts") or {}).get("one_minute_partitions") or []
    )
    if not contracts:
        raise RuntimeError("V37 V34 manifest has no minute partitions")
    expected: dict[str, dict[str, Any]] = {}
    for contract in contracts:
        name = Path(str(contract.get("path") or "")).name
        if not name or name in expected:
            raise RuntimeError("V37 V34 minute partition names are invalid")
        expected[name] = contract
    actual_paths = list(root.rglob("wp_v34_full_session_minutes_*.parquet"))
    actual_by_name: dict[str, Path] = {}
    for path in actual_paths:
        if path.name in actual_by_name:
            raise RuntimeError(
                f"V37 found duplicate V34 minute partition {path.name}"
            )
        actual_by_name[path.name] = path
    if set(actual_by_name) != set(expected):
        raise RuntimeError(
            "V37 V34 minute partitions differ from immutable manifest"
        )

    legal = candidates.loc[
        candidates["signal_slot"].astype(str).isin(BASE_ALERT_SLOTS)
    ].copy()
    legal["_v37_month"] = legal["trade_date"].astype(str).str[:6]
    outcome_frames: list[pd.DataFrame] = []
    partition_audit: list[dict[str, Any]] = []
    total_rows = 0
    for name in sorted(expected):
        contract = expected[name]
        path = actual_by_name[name]
        actual_sha = file_sha256(path)
        actual_rows = int(pq.ParquetFile(path).metadata.num_rows)
        if (
            actual_sha != str(contract.get("sha256") or "")
            or actual_rows != int(contract.get("rows", -1))
        ):
            raise RuntimeError(f"V37 V34 partition digest changed: {name}")
        month = path.stem.rsplit("_", 1)[-1]
        month_candidates = legal.loc[legal["_v37_month"].eq(month)].drop(
            columns="_v37_month"
        )
        minute_frame = pd.read_parquet(path)
        total_rows += len(minute_frame)
        if not month_candidates.empty:
            outcome_frames.append(
                build_fast_entry_outcomes(
                    month_candidates,
                    minute_frame,
                    entry_slippage_bps=(
                        config.execution.entry_slippage_bps
                    ),
                    round_trip_cost_bps=(
                        config.execution.round_trip_cost_bps
                    ),
                    min_slot_amount=config.execution.min_slot_amount,
                    reference_order_notional=(
                        config.execution.reference_order_notional
                    ),
                    max_entry_pct_of_slot_amount=(
                        config.execution.max_entry_pct_of_slot_amount
                    ),
                    min_distance_to_up_limit_pct=(
                        config.execution.min_distance_to_up_limit_pct
                    ),
                    min_distance_to_down_limit_pct=(
                        config.execution.min_distance_to_down_limit_pct
                    ),
                    non_fill_penalty_pct=(
                        config.execution.non_fill_penalty_pct
                    ),
                )
            )
        partition_audit.append(
            {
                "name": name,
                "sha256": actual_sha,
                "rows": actual_rows,
                "candidate_rows": int(len(month_candidates)),
            }
        )
        print(
            f"[wp-v37] minute_partition={name} "
            f"rows={actual_rows:,} candidates={len(month_candidates):,}",
            flush=True,
        )
    outcomes = (
        pd.concat(outcome_frames, ignore_index=True)
        if outcome_frames
        else pd.DataFrame(
            columns=[
                *IDENTITY_COLUMNS,
                "fold",
                "target_trade_date",
                *QUALITY_COLUMNS,
                *OUTCOME_COLUMNS,
                "target_severe_loss",
            ]
        )
    )
    outcomes.sort_values(
        ["fold", *IDENTITY_COLUMNS],
        kind="stable",
        inplace=True,
    )
    outcomes.reset_index(drop=True, inplace=True)
    return outcomes, {
        "schema_version": "wp_v37_v34_minute_source_1",
        "source_run_id": V34_DATA_RUN_ID,
        "partition_count": len(partition_audit),
        "minute_rows": int(total_rows),
        "legal_candidate_rows": int(len(legal)),
        "partitions": partition_audit,
        "source_integrity": True,
    }


def fit_final_model(
    candidates: pd.DataFrame,
    *,
    calendar_dates: list[str],
    random_seed: int,
    policy_spec: FastEntryPolicySpec,
) -> tuple[PathRankBundle, FrozenFastEntryPolicy, dict[str, Any]]:
    segments = rolling_cross_section_segments(
        calendar_dates,
        reserve_final_purge=False,
    )
    if segments is None:
        raise ValueError("V37 final model has insufficient prior history")
    train_dates, calibration_dates = segments
    train = labeled_path_rows(
        candidates.loc[
            candidates["trade_date"].astype(str).isin(train_dates)
        ]
    )
    calibration = labeled_path_rows(
        candidates.loc[
            candidates["trade_date"].astype(str).isin(calibration_dates)
        ]
    )
    bundle = fit_path_ranker(
        train,
        calibration,
        random_seed=random_seed + 37_337,
    )
    validate_feature_contract(bundle.feature_columns)
    policy = calibrate_fast_entry_policy(
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
    bundle: PathRankBundle,
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
    test_dates: list[str],
    *,
    reason: str,
    config: Any,
    bootstrap_samples: int,
) -> dict[str, Any]:
    metrics = economic_policy_metrics(
        pd.DataFrame(columns=["trade_date", "net_return_pct"]),
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


if __name__ == "__main__":
    raise SystemExit(main())
