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
from research_wp_v32_public_event import (
    bundle_metadata,
    skipped_fold,
    v32_research_readiness,
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
from wp.v3.v25_ranker import within_slot_rank_diagnostics
from wp.v3.v33_ecology_ranker import (
    EXPECTED_NET_RETURN_MIN_PCT,
    EXPECTED_RETURN_SCORE_WEIGHT,
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
    PAIRWISE_SCORE_MIN,
    PAIRWISE_SCORE_WEIGHT,
    POSITIVE_PROBABILITY_MIN,
    POSITIVE_SCORE_WEIGHT,
    SCHEMA_VERSION,
    SEVERE_PROBABILITY_MAX,
    SEVERE_SCORE_WEIGHT,
    SEVERE_TARGET_PCT,
    EcologyPolicySpec,
    EcologyRankBundle,
    FrozenEcologyPolicy,
    apply_ecology_policy,
    calibrate_ecology_policy,
    fit_ecology_ranker,
    labeled_ecology_rows,
    validate_feature_contract,
    validate_selected_contract,
)
from wp.v3.v33_limit_ecology import (
    V33_LIMIT_ECOLOGY_FEATURE_COLUMNS,
)


V9_SOURCE_RUN_ID = 30_600_193_544
V24_DATA_RUN_ID = 30_635_569_735
DATA_SCHEMA_VERSION = "wp_v33_limit_industry_ecology_features_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run preregistered V33 limit-industry nested-OOS research."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--v24-data-dir", required=True)
    parser.add_argument("--v33-data-dir", required=True)
    parser.add_argument("--v33-data-run-id", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.v33_data_run_id <= 0:
        raise ValueError("V33 immutable data run ID must be positive")
    config = load_v3_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    v24_features, v24_manifest, v24_integrity = load_v24_features(
        args.v24_data_dir
    )
    ecology_features, data_manifest, ecology_integrity = (
        load_v33_ecology_features(args.v33_data_dir)
    )
    validate_v33_v24_source(args.v24_data_dir, data_manifest)
    source_candidates, source = load_v23_research_source(
        args.shard_dir,
        evaluation_end=config.history.evaluation_end_date,
        features=v24_features,
        data_manifest=v24_manifest,
    )
    source = {
        **source,
        "schema_version": "wp_v33_research_source_1",
        "source_candidate_rows": int(len(source_candidates)),
    }
    assert_unique(
        source_candidates,
        "V33 immutable outcome-blind V9 top-five candidates",
    )
    joined = join_features(source_candidates, v24_features)
    joined = join_ecology_features(joined, ecology_features)
    assert_unique(joined, "V33 joined source candidates")

    evaluation_dates = load_evaluation_calendar(
        v24_manifest,
        start_date=config.history.evaluation_start_date,
        end_date=config.history.evaluation_end_date,
    )
    if not evaluation_dates:
        raise RuntimeError("V33 evaluation window has no trade dates")
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
    full_calendar = load_full_trade_calendar(v24_manifest)
    fold_rows: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    covered_dates: set[str] = set()
    temporal_integrity = True
    feature_integrity = True
    data_integrity = bool(v24_integrity and ecology_integrity)
    policy_spec = EcologyPolicySpec(
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
                f"V33 outer test dates overlap: {sorted(overlap)[:5]}"
            )
        test = fold_source.loc[
            fold_source["trade_date"].astype(str).isin(test_dates)
        ].copy()
        prior_calendar_dates = [
            date for date in full_calendar if date < test_start
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
                    reason="insufficient_prior_oos_ecology_history",
                    config=config,
                    bootstrap_samples=args.bootstrap_samples,
                )
            )
            continue

        train_dates, calibration_dates = segments
        history = joined.loc[
            joined["trade_date"].astype(str).lt(test_start)
        ].copy()
        train = labeled_ecology_rows(
            history.loc[
                history["trade_date"].astype(str).isin(train_dates)
            ]
        )
        calibration = labeled_ecology_rows(
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
                f"V33 fold {fold} historical evidence crosses test start"
            )
        try:
            bundle = fit_ecology_ranker(
                train,
                calibration,
                random_seed=(
                    config.model.random_seed + int(fold) * 33_003
                ),
            )
        except ValueError as error:
            fold_rows.append(
                {
                    **skipped_fold(
                        base,
                        test,
                        test_dates,
                        reason="insufficient_complete_prior_ecology_rows",
                        config=config,
                        bootstrap_samples=args.bootstrap_samples,
                    ),
                    "temporal_integrity": fold_temporal,
                    "model_error": str(error),
                    "train_rows": int(len(train)),
                    "calibration_rows": int(len(calibration)),
                }
            )
            print(f"[wp-v33] fold={fold} skipped: {error}", flush=True)
            continue

        feature_integrity &= validate_feature_contract(
            bundle.feature_columns
        )
        scored_calibration = bundle.predict(calibration)
        policy = calibrate_ecology_policy(
            scored_calibration,
            calibration_dates=calibration_dates,
            spec=policy_spec,
        )
        scored_test = bundle.predict(test)
        scored_test["v33_source_fold"] = int(fold)
        selected = apply_ecology_policy(scored_test, policy)
        selected["v33_source_fold"] = int(fold)
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
        rank_diagnostics = ecology_rank_diagnostics(
            scored_test,
            seed=config.model.random_seed + int(fold) * 97,
            bootstrap_samples=args.bootstrap_samples,
        )
        fold_rows.append(
            {
                **base,
                "scored": True,
                "reason": "fixed_prior_oos_v33_policy_applied",
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
            f"[wp-v33] fold={fold} "
            f"events={metrics['events']} "
            f"days={metrics['candidate_days']} "
            f"win={metrics['win_rate']:.4f} "
            f"mean={metrics['mean_net_return_pct']} "
            f"rank_ic={rank_diagnostics['mean_within_slot_ic']:.4f} "
            "rank_spread="
            f"{rank_diagnostics['mean_top_minus_bottom_return_pct']:.4f}",
            flush=True,
        )

    scored_all = concat_or_empty(scored_frames, joined)
    selected_all = concat_or_empty(selected_frames, scored_all)
    assert_unique(scored_all, "V33 nested OOS scored candidates")
    assert_unique(selected_all, "V33 nested OOS selected candidates")
    validate_selected_contract(selected_all, None)
    nested_metrics = economic_policy_metrics(
        selected_all,
        total_days=len(evaluation_dates),
        seed=config.model.random_seed + 33_000,
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
        seed=config.model.random_seed + 33,
        bootstrap_samples=args.bootstrap_samples,
    )
    add_yearly_economic_metrics(yearly, selected_all)
    rank_diagnostics = ecology_rank_diagnostics(
        scored_all,
        seed=config.model.random_seed + 33_033,
        bootstrap_samples=max(4_000, args.bootstrap_samples),
    )
    readiness = v32_research_readiness(
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
        calendar_dates=full_calendar,
        random_seed=config.model.random_seed,
        policy_spec=policy_spec,
    )
    bundle_path = output / "wp_v33_frozen_research_bundle.joblib"
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "research_only": True,
            "production_authorized": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ecology_ranker": final_bundle,
            "policy": final_policy,
            "source": source,
            "v24_data_manifest": v24_manifest,
            "v33_data_manifest": data_manifest,
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
            "causal same-L2 and same-L3 limit-event diffusion measured "
            "at each candidate signal slot"
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
            "v33_data_run_id": args.v33_data_run_id,
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
            "score_weights": {
                "expected_return": EXPECTED_RETURN_SCORE_WEIGHT,
                "positive": POSITIVE_SCORE_WEIGHT,
                "pairwise": PAIRWISE_SCORE_WEIGHT,
                "severe_loss": -SEVERE_SCORE_WEIGHT,
            },
            "absolute_gates": {
                "positive_probability_min": (
                    POSITIVE_PROBABILITY_MIN
                ),
                "expected_net_return_min_pct": (
                    EXPECTED_NET_RETURN_MIN_PCT
                ),
                "severe_probability_max": SEVERE_PROBABILITY_MAX,
                "pairwise_score_min": PAIRWISE_SCORE_MIN,
            },
            "severe_loss_target_pct": SEVERE_TARGET_PCT,
            "model_features": list(MODEL_FEATURES),
            "same_day_same_slot_pairwise_ranker": True,
            "current_ecology_active_required": True,
            "raw_industry_code_as_alpha": False,
            "old_model_outputs_used_as_alpha": False,
            "first_qualifying_signal_is_immutable": True,
            "multiple_candidates_per_slot_allowed": True,
            "no_signal_allowed": True,
            "future_information_allowed": False,
            "post_result_threshold_search_allowed": False,
        },
        "source": source,
        "v24_data_manifest": v24_manifest,
        "v33_data_manifest": data_manifest,
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
    atomic_write_json(output / "wp_v33_research_summary.json", summary)
    atomic_write_csv(
        pd.json_normalize(fold_rows, sep="."),
        output / "wp_v33_folds.csv",
    )
    atomic_write_parquet(
        scored_all,
        output / "wp_v33_nested_oos_scored_candidates.parquet",
    )
    atomic_write_csv(
        selected_all,
        output / "wp_v33_nested_oos_candidates.csv",
    )
    atomic_write_parquet(
        selected_all,
        output / "wp_v33_nested_oos_candidates.parquet",
    )
    atomic_write_csv(
        pd.json_normalize(yearly, sep="."),
        output / "wp_v33_yearly.csv",
    )
    print(
        "WP_V33_RESULT="
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


def load_v33_ecology_features(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    root = Path(data_dir)
    manifests = list(
        root.rglob("wp_v33_limit_industry_ecology_data_manifest.json")
    )
    if len(manifests) != 1:
        raise RuntimeError(
            f"expected one V33 data manifest under {root}; "
            f"found {len(manifests)}"
        )
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATA_SCHEMA_VERSION:
        raise RuntimeError("V33 data manifest schema mismatch")
    if manifest.get("profit_outcomes_read") is not False:
        raise RuntimeError("V33 data build was not outcome blind")
    if manifest.get("future_information_allowed") is not False:
        raise RuntimeError("V33 data build permitted future information")
    if not manifest.get("v33_model_research_authorized"):
        raise RuntimeError("V33 data manifest did not authorize research")
    coverage = manifest.get("coverage_audit") or {}
    candidate_audit = coverage.get("candidate_features") or {}
    full_coverage = coverage.get("full_three_year_coverage") or {}
    parity = coverage.get("probe_feature_parity") or {}
    if not (
        coverage.get("query_contract_passed")
        and coverage.get("date_contract_passed")
        and candidate_audit.get("coverage_passed")
        and full_coverage.get("passed")
        and parity.get("passed")
    ):
        raise RuntimeError("V33 ecology coverage contract failed")

    feature_paths = list(
        root.rglob("wp_v33_limit_industry_ecology_features.parquet")
    )
    index_paths = list(
        root.rglob("wp_v33_outcome_blind_candidate_index.parquet")
    )
    if len(feature_paths) != 1 or len(index_paths) != 1:
        raise RuntimeError(
            "V33 data artifact must contain one feature parquet "
            "and one candidate index"
        )
    artifacts = manifest.get("artifacts") or {}
    expected_feature_sha = str(
        (artifacts.get("ecology_features") or {}).get("sha256") or ""
    )
    expected_index_sha = str(
        (artifacts.get("candidate_index") or {}).get("sha256") or ""
    )
    if (
        not expected_feature_sha
        or file_sha256(feature_paths[0]) != expected_feature_sha
    ):
        raise RuntimeError("V33 ecology feature digest mismatch")
    if (
        not expected_index_sha
        or file_sha256(index_paths[0]) != expected_index_sha
    ):
        raise RuntimeError("V33 candidate-index digest mismatch")

    features = pd.read_parquet(feature_paths[0])
    candidate_index = pd.read_parquet(index_paths[0])
    required = {
        *IDENTITY_COLUMNS,
        "fold",
        "l2_code",
        "l3_code",
        "v33_membership_available",
        "v33_ecology_active_before_signal",
        *V33_LIMIT_ECOLOGY_FEATURE_COLUMNS,
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise RuntimeError(f"V33 feature parquet missing columns: {missing}")
    identity = [*IDENTITY_COLUMNS, "fold"]
    if features.duplicated(list(IDENTITY_COLUMNS), keep=False).any():
        raise RuntimeError("V33 feature parquet has duplicate identities")
    if candidate_index.duplicated(list(IDENTITY_COLUMNS), keep=False).any():
        raise RuntimeError("V33 candidate index has duplicate identities")
    expected_rows = int(candidate_audit.get("candidate_rows", -1))
    if (
        expected_rows <= 0
        or len(features) != expected_rows
        or len(candidate_index) != expected_rows
    ):
        raise RuntimeError("V33 data row count changed candidates")
    left = features.loc[:, identity].copy()
    right = candidate_index.loc[:, identity].copy()
    for column in IDENTITY_COLUMNS:
        left[column] = left[column].astype(str)
        right[column] = right[column].astype(str)
    left["fold"] = pd.to_numeric(left["fold"], errors="coerce")
    right["fold"] = pd.to_numeric(right["fold"], errors="coerce")
    left.sort_values(identity, kind="stable", inplace=True)
    right.sort_values(identity, kind="stable", inplace=True)
    if not left.reset_index(drop=True).equals(right.reset_index(drop=True)):
        raise RuntimeError(
            "V33 feature identities differ from candidate index"
        )
    numeric = features.loc[
        :,
        [
            "v33_membership_available",
            "v33_ecology_active_before_signal",
            *V33_LIMIT_ECOLOGY_FEATURE_COLUMNS,
        ],
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError("V33 feature parquet is incomplete or infinite")
    return features, manifest, True


def join_ecology_features(
    source_candidates: pd.DataFrame,
    ecology_features: pd.DataFrame,
) -> pd.DataFrame:
    identities = list(IDENTITY_COLUMNS)
    ecology = ecology_features.loc[
        :,
        [
            *identities,
            "fold",
            "v33_membership_available",
            "v33_ecology_active_before_signal",
            *V33_LIMIT_ECOLOGY_FEATURE_COLUMNS,
        ],
    ].copy()
    for column in identities:
        ecology[column] = ecology[column].astype(str)
    ecology.rename(columns={"fold": "_v33_ecology_fold"}, inplace=True)
    source = source_candidates.copy()
    for column in identities:
        source[column] = source[column].astype(str)
    result = source.merge(
        ecology,
        on=identities,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not result["_merge"].eq("both").all():
        raise RuntimeError("V33 ecology join missed source identities")
    source_fold = pd.to_numeric(result["fold"], errors="coerce")
    ecology_fold = pd.to_numeric(
        result["_v33_ecology_fold"],
        errors="coerce",
    )
    if not source_fold.equals(ecology_fold):
        raise RuntimeError("V33 ecology folds do not match source")
    numeric = result.loc[
        :,
        [
            "v33_membership_available",
            "v33_ecology_active_before_signal",
            *V33_LIMIT_ECOLOGY_FEATURE_COLUMNS,
        ],
    ].apply(pd.to_numeric, errors="coerce")
    complete = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    if not complete.all():
        raise RuntimeError("V33 ecology join produced incomplete features")
    result["v33_ecology_features_complete"] = complete
    return result.drop(columns=["_v33_ecology_fold", "_merge"])


def validate_v33_v24_source(
    v24_data_dir: str | Path,
    v33_manifest: dict[str, Any],
) -> None:
    manifests = list(
        Path(v24_data_dir).rglob("wp_v24_data_manifest.json")
    )
    if len(manifests) != 1:
        raise RuntimeError("V33 requires one immutable V24 data manifest")
    expected_sha = str(
        (
            (v33_manifest.get("source_contract") or {})
            .get("v24", {})
            .get("manifest_sha256")
        )
        or ""
    )
    if not expected_sha or file_sha256(manifests[0]) != expected_sha:
        raise RuntimeError(
            "V33 ecology data does not match immutable V24 data"
        )


def ecology_rank_diagnostics(
    scored: pd.DataFrame,
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if "v33_ecology_active_before_signal" not in scored:
        raise RuntimeError("V33 rank diagnostics require ecology activity")
    diagnostic_frame = scored.loc[
        pd.to_numeric(
            scored["v33_ecology_active_before_signal"],
            errors="coerce",
        ).gt(0)
    ].rename(
        columns={
            "v33_within_slot_rank_score": (
                "v25_within_slot_rank_score"
            )
        }
    )
    return within_slot_rank_diagnostics(
        diagnostic_frame,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )


def fit_final_model(
    joined: pd.DataFrame,
    *,
    calendar_dates: list[str],
    random_seed: int,
    policy_spec: EcologyPolicySpec,
) -> tuple[EcologyRankBundle, FrozenEcologyPolicy, dict[str, Any]]:
    segments = rolling_cross_section_segments(
        calendar_dates,
        reserve_final_purge=False,
    )
    if segments is None:
        raise RuntimeError("V33 final model has insufficient OOS history")
    train_dates, calibration_dates = segments
    train = labeled_ecology_rows(
        joined.loc[
            joined["trade_date"].astype(str).isin(train_dates)
        ]
    )
    calibration = labeled_ecology_rows(
        joined.loc[
            joined["trade_date"].astype(str).isin(calibration_dates)
        ]
    )
    bundle = fit_ecology_ranker(
        train,
        calibration,
        random_seed=random_seed + 33_333,
    )
    policy = calibrate_ecology_policy(
        bundle.predict(calibration),
        calibration_dates=calibration_dates,
        spec=policy_spec,
    )
    metadata = bundle_metadata(
        bundle,
        train_dates=train_dates,
        calibration_dates=calibration_dates,
    )
    return bundle, policy, metadata


if __name__ == "__main__":
    raise SystemExit(main())
