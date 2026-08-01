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
from research_wp_v32_public_event import skipped_fold
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
from wp.v3.v24_cross_section import (
    rolling_cross_section_segments,
    v24_research_readiness,
)
from wp.v3.v25_ranker import within_slot_rank_diagnostics
from wp.v3.v34_intraday_path import (
    V34_INTRADAY_PATH_FEATURE_COLUMNS,
    V34_QUALITY_COLUMNS,
)
from wp.v3.v34_path_ranker import (
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
    PROBABILITY_SPREAD_MAX,
    RETURN_SPREAD_MAX_PCT,
    SCHEMA_VERSION,
    SEVERE_PROBABILITY_UPPER_MAX,
    SEVERE_TARGET_PCT,
    FrozenPathPolicy,
    PathPolicySpec,
    PathRankBundle,
    add_path_context_features,
    apply_path_policy,
    calibrate_path_policy,
    fit_path_ranker,
    labeled_path_rows,
    validate_feature_contract,
    validate_selected_contract,
)


V9_SOURCE_RUN_ID = 30_600_193_544
V24_DATA_RUN_ID = 30_635_569_735
DATA_SCHEMA_VERSION = "wp_v34_full_session_path_features_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preregistered V34 full-session nested-OOS research."
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
    if args.v34_data_run_id <= 0:
        raise ValueError("V34 immutable data run ID must be positive")
    config = load_v3_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    v24_features, v24_manifest, v24_integrity = load_v24_features(
        args.v24_data_dir
    )
    path_features, data_manifest, path_integrity = load_v34_path_features(
        args.v34_data_dir
    )
    validate_v34_v24_source(args.v24_data_dir, data_manifest)
    source_candidates, source = load_v23_research_source(
        args.shard_dir,
        evaluation_end=config.history.evaluation_end_date,
        features=v24_features,
        data_manifest=v24_manifest,
    )
    source = {
        **source,
        "schema_version": "wp_v34_research_source_1",
        "source_candidate_rows": int(len(source_candidates)),
    }
    assert_unique(source_candidates, "V34 immutable V9 top-five candidates")
    joined = join_features(source_candidates, v24_features)
    joined = join_path_features(joined, path_features)
    joined = add_path_context_features(joined)
    assert_unique(joined, "V34 joined source candidates")

    evaluation_dates = load_evaluation_calendar(
        v24_manifest,
        start_date=config.history.evaluation_start_date,
        end_date=config.history.evaluation_end_date,
    )
    if not evaluation_dates:
        raise RuntimeError("V34 evaluation window has no trade dates")
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
    data_integrity = bool(v24_integrity and path_integrity)
    policy_spec = PathPolicySpec(
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
                f"V34 outer test dates overlap: {sorted(overlap)[:5]}"
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
                    test,
                    test_dates,
                    reason="insufficient_prior_oos_path_history",
                    config=config,
                    bootstrap_samples=args.bootstrap_samples,
                )
            )
            continue
        train_dates, calibration_dates = segments
        history = joined.loc[
            joined["trade_date"].astype(str).lt(test_start)
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
                f"V34 fold {fold} history crosses the outer test"
            )
        try:
            bundle = fit_path_ranker(
                train,
                calibration,
                random_seed=config.model.random_seed + int(fold) * 34_003,
            )
        except ValueError as error:
            fold_rows.append(
                {
                    **skipped_fold(
                        base,
                        test,
                        test_dates,
                        reason="insufficient_complete_prior_path_rows",
                        config=config,
                        bootstrap_samples=args.bootstrap_samples,
                    ),
                    "temporal_integrity": fold_temporal,
                    "model_error": str(error),
                    "train_rows": int(len(train)),
                    "calibration_rows": int(len(calibration)),
                }
            )
            print(f"[wp-v34] fold={fold} skipped: {error}", flush=True)
            continue

        feature_integrity &= validate_feature_contract(
            bundle.feature_columns
        )
        scored_calibration = bundle.predict(calibration)
        policy = calibrate_path_policy(
            scored_calibration,
            calibration_dates=calibration_dates,
            spec=policy_spec,
        )
        scored_test = bundle.predict(test)
        scored_test["v34_source_fold"] = int(fold)
        selected = apply_path_policy(scored_test, policy)
        selected["v34_source_fold"] = int(fold)
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
        rank = path_rank_diagnostics(
            scored_test,
            seed=config.model.random_seed + int(fold) * 97,
            bootstrap_samples=args.bootstrap_samples,
        )
        fold_rows.append(
            {
                **base,
                "scored": True,
                "reason": "fixed_prior_oos_v34_policy_applied",
                "temporal_integrity": fold_temporal,
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
            f"[wp-v34] fold={fold} "
            f"events={metrics['events']} "
            f"days={metrics['candidate_days']} "
            f"win={metrics['win_rate']:.4f} "
            f"mean={metrics['mean_net_return_pct']} "
            f"rank_ic={rank['mean_within_slot_ic']:.4f}",
            flush=True,
        )

    scored_all = concat_or_empty(scored_frames, joined)
    selected_all = concat_or_empty(selected_frames, scored_all)
    assert_unique(scored_all, "V34 nested OOS scored candidates")
    assert_unique(selected_all, "V34 nested OOS selected candidates")
    validate_selected_contract(selected_all, None)
    nested_metrics = economic_policy_metrics(
        selected_all,
        total_days=len(evaluation_dates),
        seed=config.model.random_seed + 34_000,
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
        seed=config.model.random_seed + 34,
        bootstrap_samples=args.bootstrap_samples,
    )
    add_yearly_economic_metrics(yearly, selected_all)
    rank_diagnostics = path_rank_diagnostics(
        scored_all,
        seed=config.model.random_seed + 34_034,
        bootstrap_samples=max(4_000, args.bootstrap_samples),
    )
    readiness = v24_research_readiness(
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
        calendar_dates=full_calendar,
        random_seed=config.model.random_seed,
        policy_spec=policy_spec,
    )
    bundle_path = output / "wp_v34_frozen_research_bundle.joblib"
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "research_only": True,
            "production_authorized": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "path_ranker": final_bundle,
            "policy": final_policy,
            "source": source,
            "v24_data_manifest": v24_manifest,
            "v34_data_manifest": data_manifest,
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
            "full causal intraday path, simultaneous path ranks, and "
            "same-stock candidate-path evolution"
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
            "absolute_gates": {
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
            },
            "margin_target_pct": MARGIN_TARGET_PCT,
            "severe_loss_target_pct": SEVERE_TARGET_PCT,
            "model_features": list(MODEL_FEATURES),
            "first_qualifying_signal_is_immutable": True,
            "maximum_candidates_per_day": FIXED_MAX_CANDIDATES_PER_DAY,
            "no_signal_allowed": True,
            "future_information_allowed": False,
            "post_result_threshold_search_allowed": False,
        },
        "source": source,
        "v24_data_manifest": v24_manifest,
        "v34_data_manifest": data_manifest,
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
    atomic_write_json(output / "wp_v34_research_summary.json", summary)
    atomic_write_csv(
        pd.json_normalize(fold_rows, sep="."),
        output / "wp_v34_folds.csv",
    )
    atomic_write_parquet(
        scored_all,
        output / "wp_v34_nested_oos_scored_candidates.parquet",
    )
    atomic_write_csv(
        selected_all,
        output / "wp_v34_nested_oos_candidates.csv",
    )
    atomic_write_parquet(
        selected_all,
        output / "wp_v34_nested_oos_candidates.parquet",
    )
    atomic_write_csv(
        pd.json_normalize(yearly, sep="."),
        output / "wp_v34_yearly.csv",
    )
    print(
        "WP_V34_RESULT="
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


def load_v34_path_features(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    root = Path(data_dir)
    manifests = list(
        root.rglob("wp_v34_intraday_path_data_manifest.json")
    )
    if len(manifests) != 1:
        raise RuntimeError(
            f"expected one V34 data manifest under {root}; "
            f"found {len(manifests)}"
        )
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATA_SCHEMA_VERSION:
        raise RuntimeError("V34 data manifest schema mismatch")
    if manifest.get("profit_outcomes_read") is not False:
        raise RuntimeError("V34 data build was not outcome blind")
    if manifest.get("future_information_allowed") is not False:
        raise RuntimeError("V34 data build permitted future information")
    if not manifest.get("v34_model_research_authorized"):
        raise RuntimeError("V34 data manifest did not authorize research")
    coverage = manifest.get("coverage_audit") or {}
    parity = manifest.get("probe_feature_parity") or {}
    if not (
        coverage.get("coverage_passed")
        and parity.get("passed")
        and not manifest.get("query_failures")
    ):
        raise RuntimeError("V34 full-data coverage contract failed")

    feature_paths = list(
        root.rglob("wp_v34_intraday_path_features.parquet")
    )
    index_paths = list(
        root.rglob("wp_v34_outcome_blind_candidate_index.parquet")
    )
    if len(feature_paths) != 1 or len(index_paths) != 1:
        raise RuntimeError(
            "V34 data artifact must contain one feature parquet "
            "and one candidate index"
        )
    artifacts = manifest.get("artifacts") or {}
    expected_feature_sha = str(
        (artifacts.get("features") or {}).get("sha256") or ""
    )
    expected_index_sha = str(
        (artifacts.get("candidate_index") or {}).get("sha256") or ""
    )
    if (
        not expected_feature_sha
        or file_sha256(feature_paths[0]) != expected_feature_sha
    ):
        raise RuntimeError("V34 feature digest mismatch")
    if (
        not expected_index_sha
        or file_sha256(index_paths[0]) != expected_index_sha
    ):
        raise RuntimeError("V34 candidate-index digest mismatch")

    features = pd.read_parquet(feature_paths[0])
    candidate_index = pd.read_parquet(index_paths[0])
    required = {
        *IDENTITY_COLUMNS,
        "fold",
        "signal_price",
        *V34_QUALITY_COLUMNS,
        *V34_INTRADAY_PATH_FEATURE_COLUMNS,
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise RuntimeError(f"V34 feature parquet missing columns: {missing}")
    identity = [*IDENTITY_COLUMNS, "fold", "signal_price"]
    if features.duplicated(list(IDENTITY_COLUMNS), keep=False).any():
        raise RuntimeError("V34 feature parquet has duplicate identities")
    if candidate_index.duplicated(list(IDENTITY_COLUMNS), keep=False).any():
        raise RuntimeError("V34 candidate index has duplicate identities")
    expected_rows = int(coverage.get("candidate_rows", -1))
    if (
        expected_rows <= 0
        or len(features) != expected_rows
        or len(candidate_index) != expected_rows
    ):
        raise RuntimeError("V34 data row count changed immutable candidates")
    left = _normalized_identity(features, identity)
    right = _normalized_identity(candidate_index, identity)
    if not left.equals(right):
        raise RuntimeError(
            "V34 feature identities differ from immutable candidate index"
        )
    numeric = features.loc[
        :,
        [*V34_INTRADAY_PATH_FEATURE_COLUMNS],
    ].apply(pd.to_numeric, errors="coerce")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise RuntimeError("V34 feature parquet contains infinite values")
    return features, manifest, True


def join_path_features(
    source_candidates: pd.DataFrame,
    path_features: pd.DataFrame,
) -> pd.DataFrame:
    identities = list(IDENTITY_COLUMNS)
    path = path_features.loc[
        :,
        [
            *identities,
            "fold",
            "signal_price",
            *V34_QUALITY_COLUMNS,
            *V34_INTRADAY_PATH_FEATURE_COLUMNS,
        ],
    ].copy()
    for column in identities:
        path[column] = path[column].astype(str)
    path.rename(
        columns={
            "fold": "_v34_path_fold",
            "signal_price": "_v34_path_signal_price",
        },
        inplace=True,
    )
    source = source_candidates.copy()
    for column in identities:
        source[column] = source[column].astype(str)
    result = source.merge(
        path,
        on=identities,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not result["_merge"].eq("both").all():
        raise RuntimeError("V34 path join missed source identities")
    if not pd.to_numeric(result["fold"], errors="coerce").equals(
        pd.to_numeric(result["_v34_path_fold"], errors="coerce")
    ):
        raise RuntimeError("V34 path folds do not match source")
    source_price = pd.to_numeric(result["signal_price"], errors="coerce")
    path_price = pd.to_numeric(
        result["_v34_path_signal_price"],
        errors="coerce",
    )
    if not np.allclose(
        source_price.to_numpy(dtype=float),
        path_price.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
        equal_nan=False,
    ):
        raise RuntimeError("V34 path signal prices do not match source")
    return result.drop(
        columns=[
            "_v34_path_fold",
            "_v34_path_signal_price",
            "_merge",
        ]
    )


def validate_v34_v24_source(
    v24_data_dir: str | Path,
    v34_manifest: dict[str, Any],
) -> None:
    manifests = list(
        Path(v24_data_dir).rglob("wp_v24_data_manifest.json")
    )
    if len(manifests) != 1:
        raise RuntimeError("V34 requires one immutable V24 data manifest")
    expected_sha = str(
        (
            (v34_manifest.get("source_contract") or {})
            .get("v24_candidate_source", {})
            .get("manifest_sha256")
        )
        or ""
    )
    if not expected_sha or file_sha256(manifests[0]) != expected_sha:
        raise RuntimeError(
            "V34 path data does not match immutable V24 data"
        )


def path_rank_diagnostics(
    scored: pd.DataFrame,
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    complete = scored["v34_path_complete"].fillna(False).astype(bool)
    diagnostic = scored.loc[complete].copy()
    diagnostic["v25_within_slot_rank_score"] = pd.to_numeric(
        diagnostic["v34_path_score"],
        errors="coerce",
    )
    return within_slot_rank_diagnostics(
        diagnostic,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )


def fit_final_model(
    joined: pd.DataFrame,
    *,
    calendar_dates: list[str],
    random_seed: int,
    policy_spec: PathPolicySpec,
) -> tuple[PathRankBundle, FrozenPathPolicy, dict[str, Any]]:
    segments = rolling_cross_section_segments(
        calendar_dates,
        reserve_final_purge=False,
    )
    if segments is None:
        raise RuntimeError("V34 final model has insufficient OOS history")
    train_dates, calibration_dates = segments
    train = labeled_path_rows(
        joined.loc[
            joined["trade_date"].astype(str).isin(train_dates)
        ]
    )
    calibration = labeled_path_rows(
        joined.loc[
            joined["trade_date"].astype(str).isin(calibration_dates)
        ]
    )
    bundle = fit_path_ranker(
        train,
        calibration,
        random_seed=random_seed + 34_334,
    )
    policy = calibrate_path_policy(
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
    }


def _normalized_identity(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    result = frame.loc[:, columns].copy()
    for column in IDENTITY_COLUMNS:
        result[column] = result[column].astype(str)
    result["fold"] = pd.to_numeric(result["fold"], errors="coerce")
    result["signal_price"] = pd.to_numeric(
        result["signal_price"],
        errors="coerce",
    )
    result.sort_values(
        list(IDENTITY_COLUMNS),
        kind="stable",
        inplace=True,
    )
    return result.reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main())
