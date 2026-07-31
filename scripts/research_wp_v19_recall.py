from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from wp.v3.contracts import load_v3_config
from wp.v3.io import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    file_sha256,
)
from wp.v3.meta_alpha import IDENTITY_COLUMNS
from wp.v3.sharding import (
    AGGREGATE_PREDICTION_COLUMNS,
    SHARD_MANIFEST_NAME,
    SHARD_PREDICTIONS_NAME,
)
from wp.v3.v16_policy import policy_metrics
from wp.v3.v16_research import (
    eligible_labeled_rows,
    rolling_model_segments,
)
from wp.v3.v19_recall import (
    DEFAULT_EXPLORATION_PER_SLOT,
    DEFAULT_MAX_TRAIN_STOCKS_PER_DAY,
    DEFAULT_TOP_PER_SOURCE,
    POLICY_CONFIRMATION_DAYS,
    POLICY_DESIGN_DAYS,
    SCHEMA_VERSION,
    RecallPolicySelection,
    apply_recall_policy,
    build_recall_frontier,
    fit_recall_selector,
    recall_frontier_audit,
    recall_policy_grid,
    rolling_recall_policy_segments,
    select_recall_policy,
    v19_research_readiness,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run V19 broad-recall causal selector research over fresh immutable "
            "V9 full-universe walk-forward predictions."
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
        "--max-train-stocks-per-day",
        type=int,
        default=DEFAULT_MAX_TRAIN_STOCKS_PER_DAY,
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
        evaluation_start=config.history.evaluation_start_date,
        evaluation_end=config.history.evaluation_end_date,
        top_per_source=args.top_per_source,
        exploration_per_slot=args.exploration_per_slot,
    )
    folds = sorted(
        pd.to_numeric(frontier["fold"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    fold_rows: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    model_evaluated_dates: set[str] = set()
    policy_evaluated_dates: set[str] = set()
    temporal_integrity = True
    numeric_fold = pd.to_numeric(frontier["fold"], errors="coerce")

    for fold in folds:
        test = frontier.loc[numeric_fold.eq(fold)].copy()
        if test.empty:
            continue
        test_dates = sorted(test["trade_date"].astype(str).unique())
        overlap = model_evaluated_dates.intersection(test_dates)
        if overlap:
            raise RuntimeError(
                f"outer test dates overlap: {sorted(overlap)[:5]}"
            )
        test_start = test_dates[0]
        test_end = test_dates[-1]
        history = frontier.loc[
            frontier["trade_date"].astype(str).lt(test_start)
        ].copy()
        model_segments = rolling_model_segments(
            history["trade_date"].astype(str).unique()
        )
        base = {
            "fold": int(fold),
            "test_start": test_start,
            "test_end": test_end,
            "test_rows": int(len(test)),
            "test_days": int(len(test_dates)),
        }
        if model_segments is None:
            fold_rows.append(
                {
                    **base,
                    "scored": False,
                    "reason": "insufficient_prior_model_history",
                    "policy_authorized": False,
                }
            )
            continue

        train_dates, model_calibration_dates = model_segments
        train = eligible_labeled_rows(
            history.loc[
                history["trade_date"].astype(str).isin(train_dates)
            ]
        )
        model_calibration = eligible_labeled_rows(
            history.loc[
                history["trade_date"]
                .astype(str)
                .isin(model_calibration_dates)
            ]
        )
        model_temporal_ok = bool(
            train_dates[-1] < model_calibration_dates[0]
            and model_calibration_dates[-1] < test_start
        )
        temporal_integrity &= model_temporal_ok
        print(
            f"[wp-v19] fold={fold} "
            f"train={train_dates[0]}..{train_dates[-1]} rows={len(train):,} "
            f"calibration={model_calibration_dates[0]}.."
            f"{model_calibration_dates[-1]} rows={len(model_calibration):,} "
            f"test={test_start}..{test_end} rows={len(test):,}",
            flush=True,
        )
        bundle = fit_recall_selector(
            train,
            model_calibration,
            random_seed=config.model.random_seed + int(fold) * 1_903,
            max_stocks_per_day=args.max_train_stocks_per_day,
        )
        scored_test = bundle.predict(test)
        scored_test["selector_source_fold"] = int(fold)

        prior_scored = concat_prior(scored_frames, scored_test)
        policy_segments = rolling_recall_policy_segments(
            prior_scored["trade_date"].astype(str).unique()
            if not prior_scored.empty
            else []
        )
        selection: RecallPolicySelection | None = None
        selected = scored_test.head(0).copy()
        policy_temporal_ok = True
        reason = "insufficient_prior_oos_policy_history"
        if policy_segments is not None:
            threshold_dates, design_dates, confirmation_dates = policy_segments
            threshold_calibration = prior_scored.loc[
                prior_scored["trade_date"].astype(str).isin(threshold_dates)
            ].copy()
            design = prior_scored.loc[
                prior_scored["trade_date"].astype(str).isin(design_dates)
            ].copy()
            confirmation = prior_scored.loc[
                prior_scored["trade_date"].astype(str).isin(
                    confirmation_dates
                )
            ].copy()
            policy_temporal_ok = bool(
                threshold_dates[-1] < design_dates[0]
                and design_dates[-1] < confirmation_dates[0]
                and confirmation_dates[-1] < test_start
            )
            temporal_integrity &= policy_temporal_ok
            if not policy_temporal_ok:
                raise RuntimeError(
                    f"V19 fold {fold} policy evidence crosses test start"
                )
            policy_evaluated_dates.update(test_dates)
            selection = select_recall_policy(
                threshold_calibration,
                design,
                confirmation,
                threshold_calibration_dates=threshold_dates,
                design_total_days=len(design_dates),
                confirmation_total_days=len(confirmation_dates),
                seed=config.model.random_seed + int(fold) * 19_007,
                bootstrap_samples=args.bootstrap_samples,
            )
            if selection.policy is not None:
                selected = apply_recall_policy(
                    scored_test,
                    selection.policy,
                )
                selected["selector_source_fold"] = int(fold)
                selected["nested_policy_id"] = selection.policy.policy_id
                selected_frames.append(selected)
                reason = "prior_oos_policy_passed_design_and_confirmation"
            else:
                reason = "prior_oos_policy_not_confirmed"

        scored_frames.append(scored_test)
        model_evaluated_dates.update(test_dates)
        test_metrics = policy_metrics(
            selected,
            total_days=len(test_dates),
            seed=config.model.random_seed + int(fold) * 53,
            bootstrap_samples=args.bootstrap_samples,
        )
        fold_rows.append(
            {
                **base,
                "scored": True,
                "reason": reason,
                "model_temporal_integrity": model_temporal_ok,
                "policy_temporal_integrity": policy_temporal_ok,
                "model": {
                    "train_start": train_dates[0],
                    "train_end": train_dates[-1],
                    "train_days": len(train_dates),
                    "source_train_rows": bundle.source_train_rows,
                    "sampled_train_rows": bundle.sampled_train_rows,
                    "calibration_start": model_calibration_dates[0],
                    "calibration_end": model_calibration_dates[-1],
                    "calibration_days": len(model_calibration_dates),
                    "source_calibration_rows": (
                        bundle.source_calibration_rows
                    ),
                    "sampled_calibration_rows": (
                        bundle.sampled_calibration_rows
                    ),
                    "feature_count": len(bundle.feature_columns),
                    "features": list(bundle.feature_columns),
                },
                "policy_authorized": bool(
                    selection is not None and selection.policy is not None
                ),
                "policy_selection": compact_selection(selection),
                "test": test_metrics,
            }
        )
        print(
            f"[wp-v19] fold={fold} policy="
            f"{selection.policy.policy_id if selection and selection.policy else 'NO_SIGNAL'} "
            f"events={test_metrics['events']} "
            f"days={test_metrics['candidate_days']} "
            f"win={test_metrics['win_rate']:.4f} "
            f"mean={test_metrics['mean_net_return_pct']}",
            flush=True,
        )

    scored_all = concat_or_empty(scored_frames, frontier)
    selected_all = concat_or_empty(selected_frames, scored_all)
    assert_unique(scored_all, "V19 scored OOS frontier")
    assert_unique(selected_all, "V19 nested OOS candidates")
    policy_days = sorted(policy_evaluated_dates)
    nested_metrics = policy_metrics(
        selected_all,
        total_days=len(policy_days),
        seed=config.model.random_seed + 19_000,
        bootstrap_samples=max(4_000, args.bootstrap_samples),
    )
    yearly = yearly_metrics(
        selected_all,
        total_dates=policy_days,
        seed=config.model.random_seed + 19,
        bootstrap_samples=args.bootstrap_samples,
    )
    readiness = v19_research_readiness(
        nested_metrics,
        yearly=yearly,
        temporal_integrity=temporal_integrity,
        source_integrity=bool(source["source_integrity"]),
    )
    final_selection = select_final_policy(
        scored_all,
        config_seed=config.model.random_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    final_bundle_path: Path | None = None
    final_model: dict[str, Any] = {"reason": "not_fit"}
    if final_selection is not None and final_selection.policy is not None:
        final_bundle, final_model = fit_final_selector(
            frontier,
            random_seed=config.model.random_seed,
            max_stocks_per_day=args.max_train_stocks_per_day,
        )
        final_bundle_path = output / "wp_v19_frozen_research_bundle.joblib"
        joblib.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "research_only": True,
                "production_authorized": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "selector": final_bundle,
                "policy": final_selection.policy,
                "policy_selection": final_selection.as_dict(),
                "source": source,
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
            "Maximize positive net return probability for executable 14:20-14:50 "
            "signals under the fixed T+1 close exit after all established costs."
        ),
        "same_historical_window_already_explored": True,
        "historical_result_role": (
            "research_screen_only; cannot replace future 150-day shadow evidence"
        ),
        "evaluation_start": config.history.evaluation_start_date,
        "evaluation_end": config.history.evaluation_end_date,
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
            "broad_recall_top_per_source": args.top_per_source,
            "deterministic_exploration_per_slot": (
                args.exploration_per_slot
            ),
            "max_train_stocks_per_day": args.max_train_stocks_per_day,
            "policy_family_size": len(recall_policy_grid()),
            "policy_design_days": POLICY_DESIGN_DAYS,
            "policy_confirmation_days": POLICY_CONFIRMATION_DAYS,
            "first_signal_is_immutable": True,
            "no_trade_allowed": True,
            "future_information_allowed": False,
        },
        "source": source,
        "frontier_rows": int(len(frontier)),
        "folds": fold_rows,
        "nested_oos_metrics": nested_metrics,
        "yearly": yearly,
        "temporal_integrity": temporal_integrity,
        "research_readiness": readiness,
        "final_policy_selection": compact_selection(final_selection),
        "final_model": final_model,
        "frozen_bundle": (
            artifact(final_bundle_path) if final_bundle_path else None
        ),
    }
    atomic_write_json(output / "wp_v19_research_summary.json", summary)
    atomic_write_csv(
        pd.json_normalize(fold_rows, sep="."),
        output / "wp_v19_folds.csv",
    )
    atomic_write_parquet(
        selected_all,
        output / "wp_v19_nested_oos_candidates.parquet",
    )
    atomic_write_csv(
        selected_all,
        output / "wp_v19_nested_oos_candidates.csv",
    )
    atomic_write_csv(
        pd.json_normalize(yearly, sep="."),
        output / "wp_v19_yearly.csv",
    )
    atomic_write_csv(
        pd.DataFrame(source["shards"]),
        output / "wp_v19_source_shards.csv",
    )
    print(
        "WP_V19_RESULT="
        + json.dumps(
            json_safe(
                {
                    "frontier_rows": int(len(frontier)),
                    "nested_oos_metrics": nested_metrics,
                    "yearly": yearly,
                    "research_readiness": readiness,
                    "final_policy_selection": compact_selection(
                        final_selection
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


def load_recall_frontier(
    shard_dir: str | Path,
    *,
    evaluation_start: str,
    evaluation_end: str,
    top_per_source: int,
    exploration_per_slot: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(shard_dir)
    manifest_paths = sorted(root.rglob(SHARD_MANIFEST_NAME))
    if not manifest_paths:
        raise FileNotFoundError(f"no V9 shard manifests under {root}")
    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    produced_folds: set[int] = set()
    expected_folds: set[int] = set()
    model_fingerprints: set[str] = set()
    policy_fingerprints: set[str] = set()
    fold_model_contract_ok = True
    dataset_manifest_digests: set[str] = set()
    source_rows = 0
    source_integrity = True
    prediction_columns = tuple(
        dict.fromkeys(
            (
                *AGGREGATE_PREDICTION_COLUMNS,
                "p_cross_section_top",
            )
        )
    )
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_folds.update(
            int(value) for value in manifest.get("expected_folds", [])
        )
        dataset_manifest_digests.add(
            str(manifest.get("dataset_manifest_sha256") or "")
        )
        prediction_path = manifest_path.with_name(SHARD_PREDICTIONS_NAME)
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        expected_sha = str(manifest.get("prediction_sha256", ""))
        actual_sha = file_sha256(prediction_path)
        digest_ok = expected_sha == actual_sha
        source_integrity &= digest_ok
        if not digest_ok:
            raise RuntimeError(
                f"prediction digest mismatch for {prediction_path}"
            )
        available = set(pq.read_schema(prediction_path).names)
        required = {
            *IDENTITY_COLUMNS,
            "fold",
            "execution_eligible",
            "label_available",
            "target_net_positive",
            "net_return_pct",
            "p_net_positive",
            "p_severe_loss",
            "selection_score",
        }
        missing = sorted(required - available)
        if missing:
            raise RuntimeError(
                f"V9 shard missing required columns {missing}: "
                f"{prediction_path}"
            )
        columns = [
            column for column in prediction_columns if column in available
        ]
        frame = pq.read_table(
            prediction_path,
            columns=columns,
        ).to_pandas()
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame = frame.loc[
            frame["trade_date"].between(
                str(evaluation_start),
                str(evaluation_end),
            )
        ].copy()
        model_fingerprints.update(
            str(value)
            for value in frame.get(
                "model_fingerprint",
                pd.Series(dtype=str),
            )
            .dropna()
            .astype(str)
            .unique()
        )
        policy_fingerprints.update(
            str(value)
            for value in frame.get(
                "policy_fingerprint",
                pd.Series(dtype=str),
            )
            .dropna()
            .astype(str)
            .unique()
        )
        if "model_fingerprint" in frame:
            fold_model_contract_ok &= bool(
                frame.assign(
                    _fold=pd.to_numeric(frame["fold"], errors="coerce")
                )
                .dropna(subset=["_fold"])
                .groupby("_fold", sort=False)["model_fingerprint"]
                .nunique(dropna=True)
                .eq(1)
                .all()
            )
        source_rows += len(frame)
        frontier = build_recall_frontier(
            frame,
            top_per_source=top_per_source,
            exploration_per_slot=exploration_per_slot,
        )
        audit = recall_frontier_audit(frame, frontier)
        folds = {
            int(value)
            for value in pd.to_numeric(frame["fold"], errors="coerce")
            .dropna()
            .astype(int)
        }
        overlap = produced_folds & folds
        if overlap:
            raise RuntimeError(
                f"duplicate prediction folds: {sorted(overlap)}"
            )
        produced_folds.update(folds)
        audit_rows.append(
            {
                "manifest": str(manifest_path.relative_to(root)),
                "prediction_sha256": actual_sha,
                "digest_verified": digest_ok,
                "folds": sorted(folds),
                **audit,
            }
        )
        frames.append(frontier)
        print(
            f"[wp-v19] loaded {prediction_path.name} "
            f"rows={len(frame):,} frontier={len(frontier):,} "
            f"positive_recall={audit['positive_row_recall']:.4f} "
            f"folds={sorted(folds)}",
            flush=True,
        )
    result = pd.concat(frames, ignore_index=True)
    result.sort_values(
        ["fold", *IDENTITY_COLUMNS],
        kind="stable",
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)
    duplicate_rows = int(
        result.duplicated(list(IDENTITY_COLUMNS), keep=False).sum()
    )
    if duplicate_rows:
        raise RuntimeError(
            f"V19 recall frontier has {duplicate_rows} duplicate identities"
        )
    folds_complete = produced_folds == expected_folds and bool(expected_folds)
    source_integrity &= (
        duplicate_rows == 0
        and folds_complete
        and bool(model_fingerprints)
        and bool(policy_fingerprints)
        and fold_model_contract_ok
        and len(dataset_manifest_digests - {""}) == 1
    )
    if not source_integrity:
        raise RuntimeError(
            "V19 immutable source contract is incomplete or inconsistent"
        )
    return result, {
        "schema_version": "wp_v19_v9_shard_source_1",
        "shards": audit_rows,
        "folds": sorted(produced_folds),
        "expected_folds": sorted(expected_folds),
        "folds_complete": folds_complete,
        "model_fingerprints": sorted(model_fingerprints),
        "policy_fingerprints": sorted(policy_fingerprints),
        "one_model_fingerprint_per_fold": fold_model_contract_ok,
        "dataset_manifest_sha256": sorted(
            dataset_manifest_digests - {""}
        ),
        "source_rows": int(source_rows),
        "frontier_rows": int(len(result)),
        "frontier_fraction_of_source": (
            len(result) / source_rows if source_rows else 0.0
        ),
        "duplicate_identity_rows": duplicate_rows,
        "source_integrity": bool(source_integrity),
    }


def select_final_policy(
    scored: pd.DataFrame,
    *,
    config_seed: int,
    bootstrap_samples: int,
) -> RecallPolicySelection | None:
    segments = rolling_recall_policy_segments(
        scored["trade_date"].astype(str).unique(),
        reserve_final_purge=False,
    )
    if segments is None:
        return None
    threshold_dates, design_dates, confirmation_dates = segments
    threshold = scored.loc[
        scored["trade_date"].astype(str).isin(threshold_dates)
    ].copy()
    design = scored.loc[
        scored["trade_date"].astype(str).isin(design_dates)
    ].copy()
    confirmation = scored.loc[
        scored["trade_date"].astype(str).isin(confirmation_dates)
    ].copy()
    return select_recall_policy(
        threshold,
        design,
        confirmation,
        threshold_calibration_dates=threshold_dates,
        design_total_days=len(design_dates),
        confirmation_total_days=len(confirmation_dates),
        seed=config_seed + 190_000,
        bootstrap_samples=bootstrap_samples,
    )


def fit_final_selector(
    frontier: pd.DataFrame,
    *,
    random_seed: int,
    max_stocks_per_day: int,
) -> tuple[Any, dict[str, Any]]:
    segments = rolling_model_segments(
        frontier["trade_date"].astype(str).unique(),
        reserve_final_purge=False,
    )
    if segments is None:
        raise RuntimeError("V19 final selector has insufficient history")
    train_dates, calibration_dates = segments
    train = eligible_labeled_rows(
        frontier.loc[
            frontier["trade_date"].astype(str).isin(train_dates)
        ]
    )
    calibration = eligible_labeled_rows(
        frontier.loc[
            frontier["trade_date"].astype(str).isin(calibration_dates)
        ]
    )
    bundle = fit_recall_selector(
        train,
        calibration,
        random_seed=random_seed + 199_999,
        max_stocks_per_day=max_stocks_per_day,
    )
    return bundle, {
        "train_start": train_dates[0],
        "train_end": train_dates[-1],
        "train_days": len(train_dates),
        "source_train_rows": bundle.source_train_rows,
        "sampled_train_rows": bundle.sampled_train_rows,
        "calibration_start": calibration_dates[0],
        "calibration_end": calibration_dates[-1],
        "calibration_days": len(calibration_dates),
        "source_calibration_rows": bundle.source_calibration_rows,
        "sampled_calibration_rows": bundle.sampled_calibration_rows,
        "feature_count": len(bundle.feature_columns),
        "features": list(bundle.feature_columns),
    }


def yearly_metrics(
    selected: pd.DataFrame,
    *,
    total_dates: Iterable[str],
    seed: int,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    dates = sorted(set(map(str, total_dates)))
    years = sorted({date[:4] for date in dates})
    rows: list[dict[str, Any]] = []
    for offset, year in enumerate(years):
        year_dates = [date for date in dates if date.startswith(year)]
        group = selected.loc[
            selected["trade_date"].astype(str).str.startswith(year)
        ].copy()
        rows.append(
            {
                "year": year,
                **policy_metrics(
                    group,
                    total_days=len(year_dates),
                    seed=seed + offset,
                    bootstrap_samples=bootstrap_samples,
                ),
            }
        )
    return rows


def compact_selection(
    selection: RecallPolicySelection | None,
) -> dict[str, Any]:
    if selection is None:
        return {"reason": "insufficient_prior_oos_policy_history"}
    payload = selection.as_dict()
    design = payload.get("design", {})
    if isinstance(design, dict) and isinstance(
        design.get("policies"),
        list,
    ):
        policies = sorted(
            design["policies"],
            key=lambda row: (
                bool(row.get("design_gate_passed")),
                float(row.get("clustered_mean_lower_pct") or -999.0),
                float(row.get("mean_net_return_pct") or -999.0),
            ),
            reverse=True,
        )
        payload["design"] = {
            "reason": design.get("reason"),
            "best_diagnostics": policies[:5],
        }
    return payload


def concat_prior(
    frames: list[pd.DataFrame],
    template: pd.DataFrame,
) -> pd.DataFrame:
    if not frames:
        return template.head(0).copy()
    return pd.concat(frames, ignore_index=True)


def concat_or_empty(
    frames: list[pd.DataFrame],
    template: pd.DataFrame,
) -> pd.DataFrame:
    if not frames:
        return template.head(0).copy()
    return pd.concat(frames, ignore_index=True)


def assert_unique(frame: pd.DataFrame, label: str) -> None:
    if frame.empty:
        return
    duplicates = frame.duplicated(list(IDENTITY_COLUMNS), keep=False)
    if duplicates.any():
        raise RuntimeError(
            f"{label} has {int(duplicates.sum())} duplicate identities"
        )


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
