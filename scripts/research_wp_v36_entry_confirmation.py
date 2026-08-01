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
from research_wp_v24_cross_section import load_v24_features
from research_wp_v34_intraday_path import (
    load_v34_path_features,
    validate_v34_v24_source,
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
from wp.v3.v25_ranker import within_slot_rank_diagnostics
from wp.v3.v36_entry_confirmation import (
    BASE_ALERT_SLOTS,
    EXPECTED_NET_RETURN_LOWER_MIN_PCT,
    FIXED_MAX_CANDIDATES_PER_DAY,
    FIXED_TARGET_CANDIDATE_DAY_RATE,
    MARGIN_PROBABILITY_LOWER_MIN,
    MARGIN_TARGET_PCT,
    MINIMUM_CALIBRATION_ROWS,
    MINIMUM_TRAIN_ROWS,
    MODEL_CALIBRATION_DAYS,
    MODEL_FEATURES,
    MODEL_PURGE_DAYS,
    MODEL_TRAIN_DAYS,
    POSITIVE_PROBABILITY_LOWER_MIN,
    POST_ALERT_FEATURES,
    PROBABILITY_SPREAD_MAX,
    QUALITY_COLUMNS,
    RETURN_SPREAD_MAX_PCT,
    SCHEMA_VERSION,
    SEVERE_PROBABILITY_UPPER_MAX,
    SEVERE_TARGET_PCT,
    EntryConfirmationBundle,
    EntryConfirmationPolicySpec,
    FrozenEntryConfirmationPolicy,
    apply_entry_confirmation_policy,
    audit_confirmation_feature_coverage,
    build_post_alert_confirmation_features,
    calibrate_entry_confirmation_policy,
    fit_entry_confirmation_gate,
    join_confirmation_features,
    labeled_confirmation_rows,
    rolling_confirmation_segments,
    v36_research_readiness,
    validate_feature_contract,
    validate_selected_contract,
)


V9_SOURCE_RUN_ID = 30_600_193_544
V24_DATA_RUN_ID = 30_635_569_735
V34_DATA_RUN_ID = 30_677_075_531


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered V36 post-alert entry-confirmation "
            "nested out-of-sample study."
        )
    )
    parser.add_argument("--config", required=True)
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
            "V36 requires immutable V34 data run "
            f"{V34_DATA_RUN_ID}; received {args.v34_data_run_id}"
        )
    config = load_v3_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    v24_features, v24_manifest, v24_integrity = load_v24_features(
        args.v24_data_dir
    )
    _, v34_manifest, v34_integrity = load_v34_path_features(
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
        "schema_version": "wp_v36_research_source_1",
        "source_candidate_rows": int(len(source_candidates)),
    }
    assert_unique(
        source_candidates,
        "V36 immutable outcome-blind V9 top-five candidates",
    )

    confirmation_features, minute_source = (
        build_confirmation_features_from_v34_partitions(
            source_candidates,
            args.v34_data_dir,
            v34_manifest,
            entry_slippage_bps=config.execution.entry_slippage_bps,
        )
    )
    coverage = audit_confirmation_feature_coverage(
        confirmation_features,
        source_candidates,
    )
    candidates = join_confirmation_features(
        source_candidates,
        confirmation_features,
    )
    assert_unique(candidates, "V36 post-alert candidate frame")

    evaluation_dates = load_evaluation_calendar(
        v24_manifest,
        start_date=config.history.evaluation_start_date,
        end_date=config.history.evaluation_end_date,
    )
    if not evaluation_dates:
        raise RuntimeError("V36 evaluation window has no A-share trade dates")
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
    policy_spec = EntryConfirmationPolicySpec(
        target_candidate_day_rate=FIXED_TARGET_CANDIDATE_DAY_RATE,
        max_candidates_per_day=FIXED_MAX_CANDIDATES_PER_DAY,
    )

    fold_rows: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    covered_dates: set[str] = set()
    temporal_integrity = True
    feature_integrity = True
    data_integrity = bool(
        v24_integrity
        and v34_integrity
        and minute_source["source_integrity"]
        and coverage["coverage_passed"]
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
                f"V36 outer test dates overlap: {sorted(overlap)[:5]}"
            )
        test = fold_source.loc[
            fold_source["trade_date"].astype(str).isin(test_dates)
        ].copy()
        prior_dates = [date for date in full_calendar if date < test_start]
        segments = rolling_confirmation_segments(prior_dates)
        base = {
            "fold": int(fold),
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "test_days": int(len(test_dates)),
            "test_rows": int(len(test)),
        }
        if segments is None:
            fold_rows.append(
                skipped_fold(
                    base,
                    test,
                    test_dates,
                    reason="insufficient_prior_oos_confirmation_history",
                    config=config,
                    bootstrap_samples=args.bootstrap_samples,
                )
            )
            continue
        train_dates, calibration_dates = segments
        history = candidates.loc[
            candidates["trade_date"].astype(str).lt(test_start)
        ].copy()
        train = labeled_confirmation_rows(
            history.loc[
                history["trade_date"].astype(str).isin(train_dates)
            ]
        )
        calibration = labeled_confirmation_rows(
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
                f"V36 fold {fold} history crosses the outer test"
            )
        try:
            bundle = fit_entry_confirmation_gate(
                train,
                calibration,
                random_seed=config.model.random_seed + int(fold) * 36_003,
            )
        except ValueError as error:
            fold_rows.append(
                {
                    **skipped_fold(
                        base,
                        test,
                        test_dates,
                        reason="insufficient_complete_prior_confirmation_rows",
                        config=config,
                        bootstrap_samples=args.bootstrap_samples,
                    ),
                    "temporal_integrity": fold_temporal,
                    "model_error": str(error),
                    "train_rows": int(len(train)),
                    "calibration_rows": int(len(calibration)),
                }
            )
            print(f"[wp-v36] fold={fold} skipped: {error}", flush=True)
            continue

        fold_feature_integrity = validate_feature_contract(
            bundle.feature_columns
        )
        feature_integrity &= fold_feature_integrity
        scored_calibration = bundle.predict(calibration)
        policy = calibrate_entry_confirmation_policy(
            scored_calibration,
            calibration_dates=calibration_dates,
            spec=policy_spec,
        )
        scored_test = bundle.predict(test)
        scored_test["v36_source_fold"] = int(fold)
        selected = apply_entry_confirmation_policy(scored_test, policy)
        selected["v36_source_fold"] = int(fold)
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
        rank = confirmation_rank_diagnostics(
            scored_test,
            seed=config.model.random_seed + int(fold) * 97,
            bootstrap_samples=args.bootstrap_samples,
        )
        fold_rows.append(
            {
                **base,
                "scored": True,
                "reason": "fixed_prior_oos_v36_confirmation_policy_applied",
                "temporal_integrity": fold_temporal,
                "feature_integrity": fold_feature_integrity,
                "model": bundle_metadata(
                    bundle,
                    train_dates=train_dates,
                    calibration_dates=calibration_dates,
                ),
                "policy": policy.as_dict(),
                "selected": metrics,
                "within_slot_rank": rank,
            }
        )
        print(
            f"[wp-v36] fold={fold} "
            f"threshold={policy.score_threshold:.6f} "
            f"events={metrics['events']} "
            f"days={metrics['candidate_days']} "
            f"win={metrics['win_rate']:.4f} "
            f"mean={metrics['mean_net_return_pct']} "
            f"rank_ic={rank['mean_within_slot_ic']:.4f}",
            flush=True,
        )

    scored_all = concat_or_empty(scored_frames, candidates)
    selected_all = concat_or_empty(selected_frames, scored_all)
    assert_unique(scored_all, "V36 nested OOS scored candidates")
    assert_unique(selected_all, "V36 nested OOS selected candidates")
    validate_selected_contract(selected_all, None)

    nested_metrics = economic_policy_metrics(
        selected_all,
        total_days=len(evaluation_dates),
        seed=config.model.random_seed + 36_000,
        bootstrap_samples=max(4_000, args.bootstrap_samples),
    )
    rank_diagnostics = confirmation_rank_diagnostics(
        scored_all,
        seed=config.model.random_seed + 36_036,
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
        seed=config.model.random_seed + 36,
        bootstrap_samples=args.bootstrap_samples,
    )
    add_yearly_economic_metrics(yearly, selected_all)
    readiness = v36_research_readiness(
        nested_metrics,
        yearly=yearly,
        temporal_integrity=temporal_integrity,
        source_integrity=bool(
            source["source_integrity"] and feature_integrity
        ),
        data_integrity=data_integrity,
    )

    final_model: dict[str, Any] | None = None
    final_policy: FrozenEntryConfirmationPolicy | None = None
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
            readiness = v36_research_readiness(
                nested_metrics,
                yearly=yearly,
                temporal_integrity=temporal_integrity,
                source_integrity=bool(
                    source["source_integrity"] and feature_integrity
                ),
                data_integrity=False,
            )
        else:
            bundle_path = output / "wp_v36_frozen_research_bundle.joblib"
            joblib.dump(
                {
                    "schema_version": SCHEMA_VERSION,
                    "research_only": True,
                    "production_authorized": False,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "entry_confirmation": final_bundle,
                    "policy": final_policy,
                    "source": source,
                    "v24_data_manifest": v24_manifest,
                    "v34_data_manifest": v34_manifest,
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
            "tail candidates under the fixed T+1 close exit."
        ),
        "historical_result_role": (
            "research_screen_only; cannot replace future 150-day shadow"
        ),
        "same_historical_window_already_explored": True,
        "new_information_family": (
            "four completed one-minute bars observed after a source alert "
            "and before its immutable next-five-minute entry benchmark"
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
            "confirmation_delay_minutes": 4,
            "entry": config.execution.entry_price_contract,
            "entry_delay_minutes": config.execution.entry_delay_minutes,
            "last_public_signal": "14:49",
            "last_entry_benchmark": "14:50",
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
            "v24_data_run_id": V24_DATA_RUN_ID,
            "v34_data_run_id": args.v34_data_run_id,
            "model_train_days": MODEL_TRAIN_DAYS,
            "model_calibration_days": MODEL_CALIBRATION_DAYS,
            "purge_days": MODEL_PURGE_DAYS,
            "minimum_train_rows": MINIMUM_TRAIN_ROWS,
            "minimum_calibration_rows": MINIMUM_CALIBRATION_ROWS,
            "policy_family_size": 1,
            "fixed_policy": policy_spec.as_dict(),
            "positive_probability_lower_min": (
                POSITIVE_PROBABILITY_LOWER_MIN
            ),
            "margin_probability_lower_min": (
                MARGIN_PROBABILITY_LOWER_MIN
            ),
            "severe_probability_upper_max": (
                SEVERE_PROBABILITY_UPPER_MAX
            ),
            "expected_net_return_lower_min_pct": (
                EXPECTED_NET_RETURN_LOWER_MIN_PCT
            ),
            "probability_spread_max": PROBABILITY_SPREAD_MAX,
            "return_spread_max_pct": RETURN_SPREAD_MAX_PCT,
            "margin_target_pct": MARGIN_TARGET_PCT,
            "severe_loss_target_pct": SEVERE_TARGET_PCT,
            "model_features": list(MODEL_FEATURES),
            "post_alert_features": list(POST_ALERT_FEATURES),
            "first_passing_confirmation_is_immutable": True,
            "source_signal_price_is_immutable": True,
            "entry_price_is_immutable": True,
            "maximum_candidates_per_day": (
                FIXED_MAX_CANDIDATES_PER_DAY
            ),
            "no_signal_allowed": True,
            "future_information_allowed": False,
            "entry_bar_feature_use_allowed": False,
            "post_result_threshold_search_allowed": False,
        },
        "source": source,
        "v24_data_manifest": v24_manifest,
        "v34_data_manifest": v34_manifest,
        "minute_source": minute_source,
        "confirmation_feature_coverage": coverage,
        "source_candidate_rows": int(len(source_candidates)),
        "legal_base_candidate_rows": int(len(candidates)),
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
    atomic_write_json(output / "wp_v36_research_summary.json", summary)
    atomic_write_parquet(
        confirmation_features,
        output / "wp_v36_post_alert_confirmation_features.parquet",
    )
    atomic_write_csv(
        pd.json_normalize(fold_rows, sep="."),
        output / "wp_v36_folds.csv",
    )
    atomic_write_parquet(
        scored_all,
        output / "wp_v36_nested_oos_scored_candidates.parquet",
    )
    atomic_write_csv(
        selected_all,
        output / "wp_v36_nested_oos_candidates.csv",
    )
    atomic_write_parquet(
        selected_all,
        output / "wp_v36_nested_oos_candidates.parquet",
    )
    atomic_write_csv(
        pd.json_normalize(yearly, sep="."),
        output / "wp_v36_yearly.csv",
    )
    print(
        "WP_V36_RESULT="
        + json.dumps(
            json_safe(
                {
                    "evaluation_days": len(evaluation_dates),
                    "model_covered_days": len(covered_dates),
                    "source_candidate_rows": int(len(source_candidates)),
                    "legal_base_candidate_rows": int(len(candidates)),
                    "confirmation_feature_coverage": coverage,
                    "nested_oos_metrics": nested_metrics,
                    "within_slot_rank_diagnostics": rank_diagnostics,
                    "selected_outcome_audit": outcome_audit,
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


def build_confirmation_features_from_v34_partitions(
    candidates: pd.DataFrame,
    data_dir: str | Path,
    manifest: dict[str, Any],
    *,
    entry_slippage_bps: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(data_dir)
    contracts = (
        (manifest.get("artifacts") or {}).get("one_minute_partitions") or []
    )
    if not contracts:
        raise RuntimeError("V36 V34 manifest has no minute partitions")
    expected: dict[str, dict[str, Any]] = {}
    for contract in contracts:
        name = Path(str(contract.get("path") or "")).name
        if not name or name in expected:
            raise RuntimeError("V36 V34 minute partition names are invalid")
        expected[name] = contract
    actual_paths = list(root.rglob("wp_v34_full_session_minutes_*.parquet"))
    actual_by_name: dict[str, Path] = {}
    for path in actual_paths:
        if path.name in actual_by_name:
            raise RuntimeError(
                f"V36 found duplicate V34 minute partition {path.name}"
            )
        actual_by_name[path.name] = path
    if set(actual_by_name) != set(expected):
        raise RuntimeError(
            "V36 V34 minute partitions differ from immutable manifest"
        )

    legal = candidates.loc[
        candidates["signal_slot"].astype(str).isin(BASE_ALERT_SLOTS)
    ].copy()
    legal["_v36_month"] = legal["trade_date"].astype(str).str[:6]
    feature_frames: list[pd.DataFrame] = []
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
            raise RuntimeError(f"V36 V34 partition digest changed: {name}")
        month = path.stem.rsplit("_", 1)[-1]
        month_candidates = legal.loc[legal["_v36_month"].eq(month)].drop(
            columns="_v36_month"
        )
        minute_frame = pd.read_parquet(path)
        total_rows += len(minute_frame)
        if not month_candidates.empty:
            feature_frames.append(
                build_post_alert_confirmation_features(
                    month_candidates,
                    minute_frame,
                    entry_slippage_bps=entry_slippage_bps,
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
            f"[wp-v36] minute_partition={name} "
            f"rows={actual_rows:,} candidates={len(month_candidates):,}",
            flush=True,
        )
    features = (
        pd.concat(feature_frames, ignore_index=True)
        if feature_frames
        else pd.DataFrame(
            columns=[
                "trade_date",
                "signal_slot",
                "ts_code",
                "fold",
                "signal_price",
                "entry_price",
                *QUALITY_COLUMNS,
                *POST_ALERT_FEATURES,
            ]
        )
    )
    features.sort_values(
        ["fold", "trade_date", "signal_slot", "ts_code"],
        kind="stable",
        inplace=True,
    )
    features.reset_index(drop=True, inplace=True)
    return features, {
        "schema_version": "wp_v36_v34_minute_source_1",
        "source_run_id": V34_DATA_RUN_ID,
        "partition_count": len(partition_audit),
        "minute_rows": int(total_rows),
        "legal_candidate_rows": int(len(legal)),
        "partitions": partition_audit,
        "source_integrity": True,
    }


def confirmation_rank_diagnostics(
    scored: pd.DataFrame,
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    diagnostic = scored.loc[
        scored["v36_path_complete"].fillna(False).astype(bool)
    ].copy()
    diagnostic["v25_within_slot_rank_score"] = pd.to_numeric(
        diagnostic["v36_confirmation_score"],
        errors="coerce",
    )
    return within_slot_rank_diagnostics(
        diagnostic,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )


def fit_final_model(
    candidates: pd.DataFrame,
    *,
    calendar_dates: list[str],
    random_seed: int,
    policy_spec: EntryConfirmationPolicySpec,
) -> tuple[
    EntryConfirmationBundle,
    FrozenEntryConfirmationPolicy,
    dict[str, Any],
]:
    segments = rolling_confirmation_segments(
        calendar_dates,
        reserve_final_purge=False,
    )
    if segments is None:
        raise ValueError("V36 final model has insufficient prior history")
    train_dates, calibration_dates = segments
    train = labeled_confirmation_rows(
        candidates.loc[
            candidates["trade_date"].astype(str).isin(train_dates)
        ]
    )
    calibration = labeled_confirmation_rows(
        candidates.loc[
            candidates["trade_date"].astype(str).isin(calibration_dates)
        ]
    )
    bundle = fit_entry_confirmation_gate(
        train,
        calibration,
        random_seed=random_seed + 36_336,
    )
    validate_feature_contract(bundle.feature_columns)
    policy = calibrate_entry_confirmation_policy(
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
    bundle: EntryConfirmationBundle,
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


if __name__ == "__main__":
    raise SystemExit(main())
