from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from wp.v3.contracts import V3Config, load_v3_config
from wp.v3.exit_research import (
    PANEL_EXIT_COLUMNS,
    attach_exit_truth,
    contract_by_id,
    materialize_contract,
)
from wp.v3.history import load_panel_partitions
from wp.v3.io import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    file_sha256,
)
from wp.v3.meta_alpha import (
    IDENTITY_COLUMNS,
    apply_meta_policy,
    fit_meta_alpha,
    prune_candidate_universe,
    select_meta_policy,
)
from wp.v3.overlay import performance_summary
from wp.v3.sharding import SHARD_MANIFEST_NAME, SHARD_PREDICTIONS_NAME


PREDICTION_COLUMNS = (
    "trade_date",
    "signal_slot",
    "ts_code",
    "name",
    "fold",
    "entry_price",
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
            "Train and evaluate an exit-specific nested meta-alpha over "
            "immutable V9 out-of-sample predictions."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--contract-id", required=True)
    parser.add_argument("--top-per-score", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_v3_config(args.config)
    contract = contract_by_id(args.contract_id)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    candidates, shard_audit = load_pruned_predictions(
        args.shard_dir,
        evaluation_start=config.history.evaluation_start_date,
        evaluation_end=config.history.evaluation_end_date,
        top_per_score=args.top_per_score,
    )
    panel = load_panel_partitions(
        args.panel_dir,
        columns=list(PANEL_EXIT_COLUMNS),
        start_date=config.history.evaluation_start_date,
        end_date=config.history.evaluation_end_date,
    )
    candidates = materialize_contract(
        attach_exit_truth(candidates, panel, config),
        contract.contract_id,
    )

    folds = sorted(
        pd.to_numeric(candidates["fold"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    selected_frames: list[pd.DataFrame] = []
    scored_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_mask = pd.to_numeric(
            candidates["fold"],
            errors="coerce",
        ).eq(fold)
        test = candidates.loc[fold_mask].copy()
        test_start = str(test["trade_date"].min())
        history = candidates.loc[
            ~fold_mask & candidates["trade_date"].astype(str).lt(test_start)
        ].copy()
        segments = rolling_segments(
            sorted(history["trade_date"].astype(str).unique())
        )
        base = {
            "contract_id": contract.contract_id,
            "fold": fold,
            "test_start": test_start,
            "test_end": str(test["trade_date"].max()),
        }
        if segments is None:
            fold_rows.append(
                {
                    **base,
                    "authorized": False,
                    "reason": "insufficient_prior_oos_history",
                    "test": empty_metrics(config),
                }
            )
            continue

        train_dates, calibration_dates, design_dates, confirmation_dates = (
            segments
        )
        train = on_dates(history, train_dates)
        calibration = on_dates(history, calibration_dates)
        design = on_dates(history, design_dates)
        confirmation = on_dates(history, confirmation_dates)
        print(
            f"[wp-v12] contract={contract.contract_id} fold={fold} "
            f"train={len(train):,} calibration={len(calibration):,} "
            f"design={len(design):,} confirmation={len(confirmation):,} "
            f"test={len(test):,}",
            flush=True,
        )
        bundle = fit_meta_alpha(
            train,
            calibration,
            random_seed=(
                config.model.random_seed
                + fold * 101
                + sum(ord(character) for character in contract.contract_id)
            ),
        )
        scored_design = bundle.predict(design)
        scored_confirmation = bundle.predict(confirmation)
        selection = select_meta_policy(
            scored_design,
            scored_confirmation,
            config,
        )
        scored_test = bundle.predict(test)
        scored_test["exit_alpha_fold"] = fold
        scored_test["exit_contract_id"] = contract.contract_id
        scored_frames.append(scored_test)

        if selection.policy is None:
            selected = scored_test.head(0).copy()
            authorized = False
            reason = "no_policy_passed_design_and_confirmation"
        else:
            selected = apply_meta_policy(scored_test, selection.policy)
            selected["exit_alpha_policy_id"] = selection.policy.policy_id
            selected_frames.append(selected)
            authorized = True
            reason = "nested_exit_alpha_authorized"
        test_metrics = performance_summary(
            selected,
            config,
            bootstrap_samples=1_000,
            seed=config.model.random_seed + fold,
        )
        fold_rows.append(
            {
                **base,
                "authorized": authorized,
                "reason": reason,
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
            f"[wp-v12] contract={contract.contract_id} fold={fold} "
            f"authorized={authorized} "
            f"events={test_metrics['events']} "
            f"mean={test_metrics['mean_net_return_pct']} "
            f"win={test_metrics['win_rate']:.4f}",
            flush=True,
        )

    selected_all = concatenate(selected_frames, candidates)
    if not selected_all.empty:
        selected_all.sort_values(
            [*IDENTITY_COLUMNS],
            kind="stable",
            inplace=True,
        )
        selected_all = selected_all.drop_duplicates(
            ["trade_date", "ts_code"],
            keep="first",
        ).reset_index(drop=True)
    scored_all = concatenate(scored_frames, candidates)
    oos_metrics = performance_summary(
        selected_all,
        config,
        bootstrap_samples=4_000,
        seed=(
            config.model.random_seed
            + 12_000
            + sum(ord(character) for character in contract.contract_id)
        ),
    )
    yearly = yearly_metrics(selected_all, config)
    readiness = research_readiness(oos_metrics)

    scored_columns = [
        column
        for column in (
            *IDENTITY_COLUMNS,
            "name",
            "fold",
            "exit_alpha_fold",
            "exit_contract_id",
            "entry_price",
            "net_return_pct",
            "entry_fillable",
            "exit_fillable",
            "target_net_positive",
            "target_severe_loss",
            "meta_p_positive",
            "meta_expected_net_return_pct",
            "meta_p_severe_loss",
            "meta_score",
            "meta_rank_pct",
        )
        if column in scored_all
    ]
    scored_path = atomic_write_parquet(
        scored_all.loc[:, scored_columns],
        output / "wp_v12_exit_alpha_scored_oos.parquet",
    )
    atomic_write_csv(
        selected_all,
        output / "wp_v12_exit_alpha_candidates.csv",
    )
    atomic_write_csv(
        pd.json_normalize(fold_rows, sep="."),
        output / "wp_v12_exit_alpha_folds.csv",
    )
    atomic_write_csv(
        pd.json_normalize(yearly, sep="."),
        output / "wp_v12_exit_alpha_yearly.csv",
    )

    summary = {
        "schema_version": "wp_v12_exit_specific_meta_alpha_1",
        "research_only": True,
        "production_model_changed": False,
        "source": "immutable_wp_v9_walk_forward_oos_predictions",
        "contract": contract.as_dict(),
        "evaluation_start": config.history.evaluation_start_date,
        "evaluation_end": config.history.evaluation_end_date,
        "protocol": {
            "top_per_base_score": args.top_per_score,
            "meta_train_days": META_TRAIN_DAYS,
            "meta_calibration_days": META_CALIBRATION_DAYS,
            "policy_design_days": POLICY_DESIGN_DAYS,
            "policy_confirmation_days": POLICY_CONFIRMATION_DAYS,
            "purge_days_between_segments": PURGE_DAYS,
            "label": (
                "positive net return under this exact executable exit "
                "contract after costs and failed-exit penalty"
            ),
            "policy_selection": (
                "design_pass_then_confirmation_pass_then_frozen_outer_test"
            ),
            "no_trade_allowed": True,
        },
        "shards": shard_audit,
        "candidate_frontier_rows": int(len(candidates)),
        "folds_total": len(folds),
        "folds_authorized": int(
            sum(bool(row.get("authorized")) for row in fold_rows)
        ),
        "folds": fold_rows,
        "oos_metrics": oos_metrics,
        "yearly": yearly,
        "research_readiness": readiness,
        "scored_oos_sha256": file_sha256(scored_path),
        "production_authorized": False,
        "shadow_requirement": {
            "minimum_trading_days": (
                config.promotion.minimum_shadow_trading_days
            ),
            "status": "not_started_for_v12",
        },
    }
    atomic_write_json(output / "wp_v12_exit_alpha_summary.json", summary)
    print(
        "WP_V12_EXIT_ALPHA_RESULT="
        + json.dumps(
            {
                "contract_id": contract.contract_id,
                "candidate_frontier_rows": int(len(candidates)),
                "folds_total": len(folds),
                "folds_authorized": summary["folds_authorized"],
                "authorized_fold_results": [
                    {
                        "fold": row["fold"],
                        "policy": row.get("policy"),
                        "test": row.get("test"),
                    }
                    for row in fold_rows
                    if row.get("authorized")
                ],
                "oos_metrics": oos_metrics,
                "yearly": yearly,
                "research_readiness": readiness,
                "scored_oos_sha256": summary["scored_oos_sha256"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
            default=json_default,
        ),
        flush=True,
    )
    return 0


def load_pruned_predictions(
    shard_dir: str | Path,
    *,
    evaluation_start: str,
    evaluation_end: str,
    top_per_score: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(shard_dir)
    manifests = sorted(root.rglob(SHARD_MANIFEST_NAME))
    if not manifests:
        raise FileNotFoundError(f"no V9 shard manifests under {root}")
    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    produced_folds: set[int] = set()
    source_rows = 0
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prediction_path = manifest_path.with_name(SHARD_PREDICTIONS_NAME)
        expected_sha = str(manifest.get("prediction_sha256", ""))
        actual_sha = sha256(prediction_path)
        if expected_sha != actual_sha:
            raise RuntimeError(
                f"prediction digest mismatch for {prediction_path}"
            )
        available = set(pq.read_schema(prediction_path).names)
        missing = sorted(set(REQUIRED_PREDICTION_COLUMNS) - available)
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
        source_rows += len(frame)
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
        "source_rows": int(source_rows),
        "pruned_rows": int(len(result)),
        "duplicate_identity_rows": duplicates,
    }


def rolling_segments(
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
    calibration = selected[cursor : cursor + META_CALIBRATION_DAYS]
    cursor += META_CALIBRATION_DAYS + PURGE_DAYS
    design = selected[cursor : cursor + POLICY_DESIGN_DAYS]
    cursor += POLICY_DESIGN_DAYS + PURGE_DAYS
    confirmation = selected[cursor : cursor + POLICY_CONFIRMATION_DAYS]
    return train, calibration, design, confirmation


def on_dates(frame: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    return frame.loc[frame["trade_date"].astype(str).isin(dates)].copy()


def concatenate(
    frames: list[pd.DataFrame],
    reference: pd.DataFrame,
) -> pd.DataFrame:
    if not frames:
        return reference.head(0).copy()
    return pd.concat(frames, ignore_index=True)


def empty_metrics(config: V3Config) -> dict[str, Any]:
    return performance_summary(
        pd.DataFrame(
            columns=[
                "trade_date",
                "net_return_pct",
                "entry_fillable",
                "exit_fillable",
            ]
        ),
        config,
        bootstrap_samples=100,
        seed=config.model.random_seed,
    )


def yearly_metrics(
    selected: pd.DataFrame,
    config: V3Config,
) -> list[dict[str, Any]]:
    if selected.empty:
        return []
    frame = selected.copy()
    frame["year"] = frame["trade_date"].astype(str).str[:4]
    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby("year", sort=True):
        rows.append(
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
    return rows


def research_readiness(metrics: dict[str, Any]) -> dict[str, Any]:
    stress = metrics.get("stress", {}).get("50bps", {})
    gates = {
        "minimum_oos_candidates": metrics.get("events", 0) >= 250,
        "minimum_oos_candidate_days": metrics.get("trade_days", 0) >= 50,
        "minimum_oos_win_rate": metrics.get("win_rate", 0.0) >= 0.55,
        "minimum_clustered_win_rate_lower": (
            metrics.get("win_rate_day_clustered_lower", 0.0) >= 0.52
        ),
        "minimum_mean_net_return": (
            (metrics.get("mean_net_return_pct") or -999.0) >= 0.20
        ),
        "clustered_mean_return_lower_positive": (
            (
                metrics.get("mean_net_return_day_clustered_lower_pct")
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
            stress.get("positive_total_return", False)
        ),
        "tail_q10_above_minus_3pct": (
            (metrics.get("net_return_q10_pct") or -999.0) >= -3.0
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
            "new_150_day_shadow_required"
            if all(gates.values())
            else "oos_evidence_insufficient"
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not serializable")


if __name__ == "__main__":
    raise SystemExit(main())
