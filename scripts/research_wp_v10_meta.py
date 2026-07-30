from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from wp.v3.contracts import load_v3_config
from wp.v3.io import atomic_write_csv, atomic_write_json
from wp.v3.meta_alpha import (
    IDENTITY_COLUMNS,
    apply_meta_policy,
    fast_policy_metrics,
    fit_meta_alpha,
    prune_candidate_universe,
    select_meta_policy,
)
from wp.v3.overlay import performance_summary
from wp.v3.sharding import (
    SHARD_MANIFEST_NAME,
    SHARD_PREDICTIONS_NAME,
)


PREDICTION_COLUMNS = (
    "trade_date",
    "signal_slot",
    "ts_code",
    "name",
    "fold",
    "net_return_pct",
    "ret_from_prev_close_pct",
    "execution_eligible",
    "entry_fillable",
    "exit_fillable",
    "label_available",
    "target_net_positive",
    "target_severe_loss",
    "p_entry_fill",
    "p_exit_fill_given_entry",
    "p_round_trip_fill_lower",
    "p_net_positive",
    "p_net_positive_lower",
    "p_conditional_net_positive",
    "p_severe_loss",
    "selection_score",
    "selection_rank_pct",
    "expected_utility_pct",
    "expected_utility_lower_pct",
    "downside_q10_pct",
    "probability_model_spread",
    "fill_probability_model_spread",
    "selection_rank_spread",
    "expected_return_model_spread",
    "data_age_seconds",
)
REQUIRED_PREDICTION_COLUMNS = tuple(
    column
    for column in PREDICTION_COLUMNS
    if column not in {"name", "data_age_seconds"}
)

META_TRAIN_DAYS = 126
META_CALIBRATION_DAYS = 21
POLICY_DESIGN_DAYS = 42
POLICY_CONFIRMATION_DAYS = 21
PURGE_DAYS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a nested OOS V10 market-state meta-alpha study over immutable "
            "V9 walk-forward predictions."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-per-score", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_v3_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    candidates, shard_audit = _load_pruned_predictions(
        args.shard_dir,
        evaluation_start=config.history.evaluation_start_date,
        evaluation_end=config.history.evaluation_end_date,
        top_per_score=args.top_per_score,
    )
    selected_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    scored_test_frames: list[pd.DataFrame] = []
    folds = sorted(
        pd.to_numeric(candidates["fold"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    for fold in folds:
        current_mask = (
            pd.to_numeric(candidates["fold"], errors="coerce").eq(fold)
        )
        test = candidates.loc[current_mask].copy()
        test_start = str(test["trade_date"].min())
        history = candidates.loc[
            ~current_mask
            & candidates["trade_date"].astype(str).lt(test_start)
        ].copy()
        segments = _rolling_segments(
            sorted(history["trade_date"].astype(str).unique())
        )
        if segments is None:
            fold_rows.append(
                {
                    "fold": fold,
                    "test_start": test_start,
                    "test_end": str(test["trade_date"].max()),
                    "reason": "insufficient_prior_oos_history",
                    "authorized": False,
                    "test_events": 0,
                }
            )
            continue
        train_dates, calibration_dates, design_dates, confirmation_dates = (
            segments
        )
        train = history.loc[
            history["trade_date"].astype(str).isin(train_dates)
        ]
        calibration = history.loc[
            history["trade_date"].astype(str).isin(calibration_dates)
        ]
        design = history.loc[
            history["trade_date"].astype(str).isin(design_dates)
        ]
        confirmation = history.loc[
            history["trade_date"].astype(str).isin(confirmation_dates)
        ]
        print(
            f"[wp-v10] fold={fold} train={train_dates[0]}..{train_dates[-1]} "
            f"rows={len(train):,} calibration={calibration_dates[0]}.."
            f"{calibration_dates[-1]} rows={len(calibration):,} "
            f"design={design_dates[0]}..{design_dates[-1]} rows={len(design):,} "
            f"confirmation={confirmation_dates[0]}.."
            f"{confirmation_dates[-1]} rows={len(confirmation):,} "
            f"test={test_start}..{str(test['trade_date'].max())} "
            f"rows={len(test):,}",
            flush=True,
        )
        bundle = fit_meta_alpha(
            train,
            calibration,
            random_seed=config.model.random_seed + fold * 101,
        )
        scored_design = bundle.predict(design)
        scored_confirmation = bundle.predict(confirmation)
        selection = select_meta_policy(
            scored_design,
            scored_confirmation,
            config,
        )
        scored_test = bundle.predict(test)
        scored_test["meta_fold"] = fold
        scored_test_frames.append(scored_test)
        if selection.policy is None:
            selected = scored_test.head(0).copy()
            reason = "no_policy_passed_design_and_confirmation"
            authorized = False
        else:
            selected = apply_meta_policy(scored_test, selection.policy)
            selected["meta_policy_id"] = selection.policy.policy_id
            selected_frames.append(selected)
            reason = "nested_policy_authorized"
            authorized = True
        test_metrics = fast_policy_metrics(selected, config)
        fold_rows.append(
            {
                "fold": fold,
                "test_start": test_start,
                "test_end": str(test["trade_date"].max()),
                "reason": reason,
                "authorized": authorized,
                "policy": (
                    selection.policy.as_dict()
                    if selection.policy is not None
                    else None
                ),
                "design": selection.design,
                "confirmation": selection.confirmation,
                "search": selection.as_dict()["search"],
                "test": test_metrics,
            }
        )
        print(
            f"[wp-v10] fold={fold} authorized={authorized} "
            f"policy={selection.policy.policy_id if selection.policy else 'none'} "
            f"test_events={test_metrics['events']} "
            f"test_mean={test_metrics['mean_net_return_pct']} "
            f"test_win={test_metrics['win_rate']:.4f}",
            flush=True,
        )

    selected_all = (
        pd.concat(selected_frames, ignore_index=True)
        if selected_frames
        else candidates.head(0).copy()
    )
    selected_all = selected_all.sort_values(
        [*IDENTITY_COLUMNS],
        kind="stable",
    ).reset_index(drop=True)
    selected_all = selected_all.drop_duplicates(
        ["trade_date", "ts_code"],
        keep="first",
    )
    scored_all = (
        pd.concat(scored_test_frames, ignore_index=True)
        if scored_test_frames
        else candidates.head(0).copy()
    )
    metrics = performance_summary(
        selected_all,
        config,
        bootstrap_samples=4_000,
        seed=config.model.random_seed + 10_000,
    )
    yearly: list[dict[str, Any]] = []
    if not selected_all.empty:
        selected_all["year"] = selected_all["trade_date"].astype(str).str[:4]
        for year, group in selected_all.groupby("year", sort=True):
            yearly.append(
                {
                    "year": str(year),
                    **performance_summary(
                        group,
                        config,
                        bootstrap_samples=2_000,
                        seed=config.model.random_seed + int(year),
                    ),
                }
            )
        selected_all.drop(columns="year", inplace=True)

    promotion_readiness = _research_readiness(metrics)
    summary = {
        "schema_version": "wp_v10_market_state_meta_alpha_research_1",
        "research_only": True,
        "production_model_changed": False,
        "source": "immutable_wp_v9_walk_forward_oos_predictions",
        "evaluation_start": config.history.evaluation_start_date,
        "evaluation_end": config.history.evaluation_end_date,
        "contract": {
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
            "meta_train_days": META_TRAIN_DAYS,
            "meta_calibration_days": META_CALIBRATION_DAYS,
            "policy_design_days": POLICY_DESIGN_DAYS,
            "policy_confirmation_days": POLICY_CONFIRMATION_DAYS,
            "purge_days_between_segments": PURGE_DAYS,
            "top_per_base_score": args.top_per_score,
            "policy_selection": (
                "design_pass_then_confirmation_pass_then_frozen_test"
            ),
            "no_trade_allowed": True,
        },
        "shards": shard_audit,
        "candidate_universe_rows": int(len(candidates)),
        "folds": fold_rows,
        "oos_metrics": metrics,
        "yearly": yearly,
        "research_readiness": promotion_readiness,
        "shadow_requirement": {
            "minimum_trading_days": (
                config.promotion.minimum_shadow_trading_days
            ),
            "status": "not_started_for_v10",
        },
    }
    atomic_write_json(output / "wp_v10_meta_summary.json", summary)
    atomic_write_csv(
        pd.json_normalize(fold_rows, sep="."),
        output / "wp_v10_meta_folds.csv",
    )
    atomic_write_csv(
        selected_all,
        output / "wp_v10_meta_oos_candidates.csv",
    )
    atomic_write_csv(
        pd.json_normalize(yearly, sep="."),
        output / "wp_v10_meta_yearly.csv",
    )
    scored_columns = [
        column
        for column in (
            *IDENTITY_COLUMNS,
            "name",
            "fold",
            "meta_fold",
            "net_return_pct",
            "entry_fillable",
            "exit_fillable",
            "meta_p_positive",
            "meta_expected_net_return_pct",
            "meta_p_severe_loss",
            "meta_score",
            "meta_rank_pct",
        )
        if column in scored_all
    ]
    atomic_write_csv(
        scored_all.loc[:, scored_columns],
        output / "wp_v10_meta_scored_test_frontier.csv",
    )
    print(
        "WP_V10_META_RESULT="
        + json.dumps(
            {
                "candidate_universe_rows": int(len(candidates)),
                "folds": [
                    {
                        "fold": row["fold"],
                        "authorized": row["authorized"],
                        "policy": row.get("policy"),
                        "test": row.get("test"),
                    }
                    for row in fold_rows
                ],
                "oos_metrics": metrics,
                "yearly": yearly,
                "research_readiness": promotion_readiness,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_pruned_predictions(
    shard_dir: str | Path,
    *,
    evaluation_start: str,
    evaluation_end: str,
    top_per_score: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(shard_dir)
    manifest_paths = sorted(root.rglob(SHARD_MANIFEST_NAME))
    if not manifest_paths:
        raise FileNotFoundError(f"no V9 shard manifests under {root}")
    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    produced_folds: set[int] = set()
    before_rows = 0
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prediction_path = manifest_path.with_name(SHARD_PREDICTIONS_NAME)
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        expected_sha = str(manifest.get("prediction_sha256", ""))
        actual_sha = _sha256(prediction_path)
        if expected_sha != actual_sha:
            raise RuntimeError(
                f"prediction digest mismatch for {prediction_path}"
            )
        schema = pq.read_schema(prediction_path)
        available = set(schema.names)
        missing = sorted(
            set(REQUIRED_PREDICTION_COLUMNS) - available
        )
        if missing:
            raise RuntimeError(
                f"V9 shard missing required columns {missing}: "
                f"{prediction_path}"
            )
        columns = [
            column for column in PREDICTION_COLUMNS if column in available
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
        before_rows += len(frame)
        pruned = prune_candidate_universe(
            frame,
            top_per_score=top_per_score,
        )
        frames.append(pruned)
        folds = {
            int(value)
            for value in pd.to_numeric(
                frame["fold"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
        }
        overlap = produced_folds & folds
        if overlap:
            raise RuntimeError(f"duplicate prediction folds: {sorted(overlap)}")
        produced_folds.update(folds)
        audit_rows.append(
            {
                "manifest": str(manifest_path.relative_to(root)),
                "prediction_sha256": actual_sha,
                "source_rows": int(len(frame)),
                "pruned_rows": int(len(pruned)),
                "folds": sorted(folds),
            }
        )
        print(
            f"[wp-v10] loaded {prediction_path.name} rows={len(frame):,} "
            f"pruned={len(pruned):,} folds={sorted(folds)}",
            flush=True,
        )
    result = pd.concat(frames, ignore_index=True)
    result.sort_values(
        ["fold", *IDENTITY_COLUMNS],
        kind="stable",
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)
    duplicates = int(result.duplicated(list(IDENTITY_COLUMNS)).sum())
    if duplicates:
        raise RuntimeError(
            f"V9 pruned prediction universe has {duplicates} duplicates"
        )
    return result, {
        "manifests": audit_rows,
        "folds": sorted(produced_folds),
        "source_rows": int(before_rows),
        "pruned_rows": int(len(result)),
        "duplicate_identity_rows": duplicates,
    }


def _rolling_segments(
    prior_dates: list[str],
) -> tuple[list[str], list[str], list[str], list[str]] | None:
    needed = (
        META_TRAIN_DAYS
        + META_CALIBRATION_DAYS
        + POLICY_DESIGN_DAYS
        + POLICY_CONFIRMATION_DAYS
        + 4 * PURGE_DAYS
    )
    if len(prior_dates) < needed:
        return None
    selected = prior_dates[-needed:-PURGE_DAYS]
    cursor = 0
    train = selected[cursor : cursor + META_TRAIN_DAYS]
    cursor += META_TRAIN_DAYS + PURGE_DAYS
    calibration = selected[
        cursor : cursor + META_CALIBRATION_DAYS
    ]
    cursor += META_CALIBRATION_DAYS + PURGE_DAYS
    design = selected[cursor : cursor + POLICY_DESIGN_DAYS]
    cursor += POLICY_DESIGN_DAYS + PURGE_DAYS
    confirmation = selected[
        cursor : cursor + POLICY_CONFIRMATION_DAYS
    ]
    return train, calibration, design, confirmation


def _research_readiness(metrics: dict[str, Any]) -> dict[str, Any]:
    gates = {
        "minimum_oos_candidates": metrics.get("events", 0) >= 250,
        "minimum_oos_win_rate": metrics.get("win_rate", 0.0) >= 0.55,
        "minimum_clustered_win_rate_lower": (
            metrics.get("win_rate_day_clustered_lower", 0.0) >= 0.52
        ),
        "minimum_mean_net_return": (
            (metrics.get("mean_net_return_pct") or -999.0) >= 0.20
        ),
        "clustered_mean_return_lower_positive": (
            (
                metrics.get(
                    "mean_net_return_day_clustered_lower_pct"
                )
                or -999.0
            )
            > 0.0
        ),
        "minimum_profit_factor": (
            (metrics.get("profit_factor") or 0.0) >= 1.20
        ),
        "minimum_entry_fill_rate": (
            metrics.get("entry_fill_rate", 0.0) >= 0.98
        ),
        "minimum_exit_fill_rate": (
            metrics.get("exit_fill_rate_given_entry", 0.0) >= 0.98
        ),
        "50bps_stress_nonnegative": bool(
            metrics.get("stress", {})
            .get("50bps", {})
            .get("positive_total_return", False)
        ),
    }
    return {
        "all_oos_gates_passed": all(gates.values()),
        "gates": gates,
        "failed_gates": [
            name for name, passed in gates.items() if not passed
        ],
        "production_authorized": False,
        "reason": (
            "shadow_150_days_required"
            if all(gates.values())
            else "oos_evidence_insufficient"
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
