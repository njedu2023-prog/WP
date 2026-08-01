from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
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
from research_wp_v34_intraday_path import (
    join_path_features,
    load_v34_path_features,
    validate_v34_v24_source,
)
from wp.v3.contracts import load_v3_config
from wp.v3.io import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
)
from wp.v3.v23_microstructure import (
    fold_test_window,
    load_evaluation_calendar,
    load_full_trade_calendar,
    load_v23_research_source,
    selected_outcome_audit,
)
from wp.v3.v35_regime_gate import (
    FIXED_BASKET_SIZE,
    FIXED_MAX_CANDIDATES_PER_DAY,
    FIXED_TARGET_CANDIDATE_DAY_RATE,
    MARGIN_TARGET_PCT,
    MINIMUM_CALIBRATION_ROWS,
    MINIMUM_TRAIN_ROWS,
    MODEL_CALIBRATION_DAYS,
    MODEL_PURGE_DAYS,
    MODEL_TRAIN_DAYS,
    SCHEMA_VERSION,
    SEVERE_TARGET_PCT,
    V35_REGIME_FEATURES,
    FrozenRegimePolicy,
    RegimeLicenseBundle,
    RegimePolicySpec,
    apply_regime_policy_to_slots,
    build_regime_slot_frame,
    calibrate_regime_policy,
    fit_regime_license,
    labeled_regime_slots,
    rolling_regime_segments,
    select_regime_candidates,
    v35_research_readiness,
    validate_feature_contract,
    validate_selected_contract,
)


V9_SOURCE_RUN_ID = 30_600_193_544
V24_DATA_RUN_ID = 30_635_569_735
V34_DATA_RUN_ID = 30_677_075_531


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered V35 full-session regime-license "
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
            "V35 requires immutable V34 data run "
            f"{V34_DATA_RUN_ID}; received {args.v34_data_run_id}"
        )
    config = load_v3_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    v24_features, v24_manifest, v24_integrity = load_v24_features(
        args.v24_data_dir
    )
    path_features, path_manifest, path_integrity = load_v34_path_features(
        args.v34_data_dir
    )
    validate_v34_v24_source(args.v24_data_dir, path_manifest)
    source_candidates, source = load_v23_research_source(
        args.shard_dir,
        evaluation_end=config.history.evaluation_end_date,
        features=v24_features,
        data_manifest=v24_manifest,
    )
    source = {
        **source,
        "schema_version": "wp_v35_research_source_1",
        "source_candidate_rows": int(len(source_candidates)),
    }
    assert_unique(
        source_candidates,
        "V35 immutable outcome-blind V9 top-five candidates",
    )
    candidates = join_features(source_candidates, v24_features)
    candidates = join_path_features(candidates, path_features)
    assert_unique(candidates, "V35 joined source candidates")
    slots = build_regime_slot_frame(candidates)
    assert_slot_unique(slots, "V35 fixed basket slot frame")

    evaluation_dates = load_evaluation_calendar(
        v24_manifest,
        start_date=config.history.evaluation_start_date,
        end_date=config.history.evaluation_end_date,
    )
    if not evaluation_dates:
        raise RuntimeError("V35 evaluation window has no A-share trade dates")
    evaluation_mask = slots["trade_date"].astype(str).isin(evaluation_dates)
    folds = sorted(
        pd.to_numeric(slots.loc[evaluation_mask, "fold"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    numeric_slot_fold = pd.to_numeric(slots["fold"], errors="coerce")
    numeric_candidate_fold = pd.to_numeric(candidates["fold"], errors="coerce")
    full_calendar = load_full_trade_calendar(v24_manifest)
    policy_spec = RegimePolicySpec(
        target_candidate_day_rate=FIXED_TARGET_CANDIDATE_DAY_RATE,
        basket_size=FIXED_BASKET_SIZE,
        max_candidates_per_day=FIXED_MAX_CANDIDATES_PER_DAY,
    )

    fold_rows: list[dict[str, Any]] = []
    scored_slot_frames: list[pd.DataFrame] = []
    licensed_slot_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    covered_dates: set[str] = set()
    temporal_integrity = True
    feature_integrity = True
    data_integrity = bool(v24_integrity and path_integrity)

    for fold in folds:
        fold_slots = slots.loc[numeric_slot_fold.eq(fold)].copy()
        fold_candidates = candidates.loc[
            numeric_candidate_fold.eq(fold)
        ].copy()
        if fold_slots.empty or fold_candidates.empty:
            continue
        test_start, test_end = fold_test_window(fold_candidates)
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
                f"V35 outer test dates overlap: {sorted(overlap)[:5]}"
            )
        test_slots = fold_slots.loc[
            fold_slots["trade_date"].astype(str).isin(test_dates)
        ].copy()
        test_candidates = fold_candidates.loc[
            fold_candidates["trade_date"].astype(str).isin(test_dates)
        ].copy()
        prior_dates = [date for date in full_calendar if date < test_start]
        segments = rolling_regime_segments(prior_dates)
        base = {
            "fold": int(fold),
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "test_days": int(len(test_dates)),
            "test_slots": int(len(test_slots)),
            "test_candidate_rows": int(len(test_candidates)),
        }
        if segments is None:
            fold_rows.append(
                skipped_fold(
                    base,
                    test_candidates,
                    test_dates,
                    reason="insufficient_prior_oos_regime_history",
                    config=config,
                    bootstrap_samples=args.bootstrap_samples,
                )
            )
            continue
        train_dates, calibration_dates = segments
        history = slots.loc[
            slots["trade_date"].astype(str).lt(test_start)
        ].copy()
        train = labeled_regime_slots(
            history.loc[
                history["trade_date"].astype(str).isin(train_dates)
            ]
        )
        calibration = labeled_regime_slots(
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
                f"V35 fold {fold} history crosses the outer test"
            )
        try:
            bundle = fit_regime_license(
                train,
                calibration,
                random_seed=config.model.random_seed + int(fold) * 35_003,
            )
        except ValueError as error:
            fold_rows.append(
                {
                    **skipped_fold(
                        base,
                        test_candidates,
                        test_dates,
                        reason="insufficient_complete_prior_regime_slots",
                        config=config,
                        bootstrap_samples=args.bootstrap_samples,
                    ),
                    "temporal_integrity": fold_temporal,
                    "model_error": str(error),
                    "train_rows": int(len(train)),
                    "calibration_rows": int(len(calibration)),
                }
            )
            print(f"[wp-v35] fold={fold} skipped: {error}", flush=True)
            continue

        fold_feature_integrity = validate_feature_contract(
            bundle.feature_columns
        )
        feature_integrity &= fold_feature_integrity
        scored_calibration = bundle.predict(calibration)
        policy = calibrate_regime_policy(
            scored_calibration,
            calibration_dates=calibration_dates,
            spec=policy_spec,
        )
        scored_test_slots = bundle.predict(test_slots)
        scored_test_slots["v35_source_fold"] = int(fold)
        licensed_slots = apply_regime_policy_to_slots(
            scored_test_slots,
            policy,
        )
        licensed_slots["v35_source_fold"] = int(fold)
        selected = select_regime_candidates(
            test_candidates,
            licensed_slots,
            policy,
        )
        selected["v35_source_fold"] = int(fold)
        validate_selected_contract(selected, policy)

        scored_slot_frames.append(scored_test_slots)
        if not licensed_slots.empty:
            licensed_slot_frames.append(licensed_slots)
        if not selected.empty:
            selected_frames.append(selected)
        covered_dates.update(test_dates)
        metrics = economic_policy_metrics(
            selected,
            total_days=len(test_dates),
            seed=config.model.random_seed + int(fold) * 89,
            bootstrap_samples=args.bootstrap_samples,
        )
        fold_rows.append(
            {
                **base,
                "scored": True,
                "reason": "fixed_prior_oos_v35_regime_policy_applied",
                "temporal_integrity": fold_temporal,
                "feature_integrity": fold_feature_integrity,
                "model": bundle_metadata(
                    bundle,
                    train_dates=train_dates,
                    calibration_dates=calibration_dates,
                ),
                "policy": policy.as_dict(),
                "licensed_slots": slot_outcome_metrics(licensed_slots),
                "selected": metrics,
            }
        )
        print(
            f"[wp-v35] fold={fold} "
            f"threshold={policy.score_threshold:.6f} "
            f"slots={len(licensed_slots)} "
            f"events={metrics['events']} "
            f"days={metrics['candidate_days']} "
            f"win={metrics['win_rate']:.4f} "
            f"mean={metrics['mean_net_return_pct']}",
            flush=True,
        )

    scored_slots_all = concat_or_empty(scored_slot_frames, slots)
    licensed_slots_all = concat_or_empty(
        licensed_slot_frames,
        scored_slots_all,
    )
    selected_all = concat_or_empty(selected_frames, candidates)
    assert_slot_unique(scored_slots_all, "V35 nested OOS scored slots")
    assert_slot_unique(licensed_slots_all, "V35 nested OOS licensed slots")
    assert_unique(selected_all, "V35 nested OOS selected candidates")
    validate_selected_contract(selected_all, None)

    nested_metrics = economic_policy_metrics(
        selected_all,
        total_days=len(evaluation_dates),
        seed=config.model.random_seed + 35_000,
        bootstrap_samples=max(4_000, args.bootstrap_samples),
    )
    outcome_audit = selected_outcome_audit(
        selected_all,
        total_days=len(evaluation_dates),
    )
    data_integrity &= bool(
        outcome_audit["all_selected_outcomes_verified"]
        and len(slots) > 0
        and not slots.duplicated(
            ["trade_date", "signal_slot"],
            keep=False,
        ).any()
    )
    yearly = yearly_metrics(
        selected_all,
        total_dates=evaluation_dates,
        seed=config.model.random_seed + 35,
        bootstrap_samples=args.bootstrap_samples,
    )
    add_yearly_economic_metrics(yearly, selected_all)
    readiness = v35_research_readiness(
        nested_metrics,
        yearly=yearly,
        temporal_integrity=temporal_integrity,
        source_integrity=bool(
            source["source_integrity"] and feature_integrity
        ),
        data_integrity=data_integrity,
    )

    final_bundle, final_policy, final_model = fit_final_model(
        slots,
        calendar_dates=full_calendar,
        random_seed=config.model.random_seed,
        policy_spec=policy_spec,
    )
    bundle_path = output / "wp_v35_frozen_research_bundle.joblib"
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "research_only": True,
            "production_authorized": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "regime_license": final_bundle,
            "policy": final_policy,
            "source": source,
            "v24_data_manifest": v24_manifest,
            "v34_data_manifest": path_manifest,
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
            "cross-stock agreement of causal V34 full-session paths inside "
            "the immutable V24 source-ranked basket"
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
            "v24_data_run_id": V24_DATA_RUN_ID,
            "v34_data_run_id": args.v34_data_run_id,
            "model_train_days": MODEL_TRAIN_DAYS,
            "model_calibration_days": MODEL_CALIBRATION_DAYS,
            "purge_days": MODEL_PURGE_DAYS,
            "minimum_train_rows": MINIMUM_TRAIN_ROWS,
            "minimum_calibration_rows": MINIMUM_CALIBRATION_ROWS,
            "policy_family_size": 1,
            "fixed_policy": policy_spec.as_dict(),
            "basket_good_label": (
                "mean_net_return_gt_0_and_strict_majority_positive"
            ),
            "margin_target_pct": MARGIN_TARGET_PCT,
            "severe_loss_target_pct": SEVERE_TARGET_PCT,
            "model_features": list(V35_REGIME_FEATURES),
            "selected_stock_identity_used_by_license_model": False,
            "selected_stock_outcome_used_by_license_model": False,
            "first_qualifying_slot_is_immutable": True,
            "signal_price_is_immutable": True,
            "maximum_candidates_per_day": (
                FIXED_MAX_CANDIDATES_PER_DAY
            ),
            "no_signal_allowed": True,
            "future_information_allowed": False,
            "post_result_threshold_search_allowed": False,
        },
        "source": source,
        "v24_data_manifest": v24_manifest,
        "v34_data_manifest": path_manifest,
        "source_candidate_rows": int(len(source_candidates)),
        "joined_candidate_rows": int(len(candidates)),
        "slot_rows": int(len(slots)),
        "folds": fold_rows,
        "nested_oos_metrics": nested_metrics,
        "licensed_slot_metrics": slot_outcome_metrics(licensed_slots_all),
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
    atomic_write_json(output / "wp_v35_research_summary.json", summary)
    atomic_write_csv(
        pd.json_normalize(fold_rows, sep="."),
        output / "wp_v35_folds.csv",
    )
    atomic_write_parquet(
        scored_slots_all,
        output / "wp_v35_nested_oos_scored_slots.parquet",
    )
    atomic_write_csv(
        licensed_slots_all,
        output / "wp_v35_nested_oos_licensed_slots.csv",
    )
    atomic_write_csv(
        selected_all,
        output / "wp_v35_nested_oos_candidates.csv",
    )
    atomic_write_parquet(
        selected_all,
        output / "wp_v35_nested_oos_candidates.parquet",
    )
    atomic_write_csv(
        pd.json_normalize(yearly, sep="."),
        output / "wp_v35_yearly.csv",
    )
    print(
        "WP_V35_RESULT="
        + json.dumps(
            json_safe(
                {
                    "evaluation_days": len(evaluation_dates),
                    "model_covered_days": len(covered_dates),
                    "source_candidate_rows": int(len(source_candidates)),
                    "joined_candidate_rows": int(len(candidates)),
                    "slot_rows": int(len(slots)),
                    "nested_oos_metrics": nested_metrics,
                    "licensed_slot_metrics": slot_outcome_metrics(
                        licensed_slots_all
                    ),
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


def fit_final_model(
    slots: pd.DataFrame,
    *,
    calendar_dates: list[str],
    random_seed: int,
    policy_spec: RegimePolicySpec,
) -> tuple[RegimeLicenseBundle, FrozenRegimePolicy, dict[str, Any]]:
    segments = rolling_regime_segments(
        calendar_dates,
        reserve_final_purge=False,
    )
    if segments is None:
        raise RuntimeError("V35 final model has insufficient OOS history")
    train_dates, calibration_dates = segments
    train = labeled_regime_slots(
        slots.loc[slots["trade_date"].astype(str).isin(train_dates)]
    )
    calibration = labeled_regime_slots(
        slots.loc[slots["trade_date"].astype(str).isin(calibration_dates)]
    )
    bundle = fit_regime_license(
        train,
        calibration,
        random_seed=random_seed + 35_335,
    )
    validate_feature_contract(bundle.feature_columns)
    policy = calibrate_regime_policy(
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
    bundle: RegimeLicenseBundle,
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
    test_candidates: pd.DataFrame,
    test_dates: list[str],
    *,
    reason: str,
    config: Any,
    bootstrap_samples: int,
) -> dict[str, Any]:
    metrics = economic_policy_metrics(
        test_candidates.head(0),
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


def slot_outcome_metrics(slots: pd.DataFrame) -> dict[str, Any]:
    if slots.empty:
        return {
            "slots": 0,
            "days": 0,
            "good_rate": 0.0,
            "mean_basket_net_return_pct": None,
            "median_basket_net_return_pct": None,
            "severe_slot_rate": 0.0,
        }
    net = pd.to_numeric(
        slots["v35_basket_mean_net_return_pct"],
        errors="coerce",
    ).dropna()
    good = pd.to_numeric(
        slots["v35_target_good"],
        errors="coerce",
    ).dropna()
    severe = pd.to_numeric(
        slots["v35_target_severe"],
        errors="coerce",
    ).dropna()
    return {
        "slots": int(len(slots)),
        "days": int(slots["trade_date"].astype(str).nunique()),
        "good_rate": float(good.mean()) if len(good) else 0.0,
        "mean_basket_net_return_pct": (
            float(net.mean()) if len(net) else None
        ),
        "median_basket_net_return_pct": (
            float(net.median()) if len(net) else None
        ),
        "severe_slot_rate": (
            float(severe.mean()) if len(severe) else 0.0
        ),
    }


def assert_slot_unique(frame: pd.DataFrame, name: str) -> None:
    if frame.empty:
        return
    duplicates = int(
        frame.duplicated(["trade_date", "signal_slot"], keep=False).sum()
    )
    if duplicates:
        raise RuntimeError(f"{name} has {duplicates} duplicate slot rows")


if __name__ == "__main__":
    raise SystemExit(main())
