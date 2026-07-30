from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from wp.v3.contracts import load_v3_config
from wp.v3.exit_research import (
    PANEL_EXIT_COLUMNS,
    apply_exit_policy,
    attach_exit_truth,
    exit_contracts,
    fast_exit_metrics,
    materialize_contract,
    select_exit_policy,
)
from wp.v3.history import load_panel_partitions
from wp.v3.io import atomic_write_csv, atomic_write_json, atomic_write_parquet
from wp.v3.meta_alpha import (
    IDENTITY_COLUMNS,
    prune_candidate_universe,
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

POLICY_DESIGN_DAYS = 84
POLICY_CONFIRMATION_DAYS = 42
PURGE_DAYS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit executable T+1 exit contracts and run nested OOS policy "
            "selection over immutable V9 predictions."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--v10-source-dir")
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
    panel = load_panel_partitions(
        args.panel_dir,
        columns=list(PANEL_EXIT_COLUMNS),
        start_date=config.history.evaluation_start_date,
        end_date=config.history.evaluation_end_date,
    )
    candidates = attach_exit_truth(candidates, panel, config)
    v10_candidates = _load_v10_candidates(args.v10_source_dir)
    v10_slice = _match_identity_slice(candidates, v10_candidates)
    frontier_path = atomic_write_parquet(
        candidates,
        output / "wp_v11_exit_frontier.parquet",
    )

    contract_diagnostics: list[dict[str, Any]] = []
    for contract in exit_contracts():
        frontier = materialize_contract(candidates, contract.contract_id)
        contract_diagnostics.append(
            {
                "scope": "causal_candidate_frontier",
                "contract": contract.as_dict(),
                **fast_exit_metrics(frontier, config),
            }
        )
        if not v10_slice.empty:
            selected = materialize_contract(v10_slice, contract.contract_id)
            contract_diagnostics.append(
                {
                    "scope": "exact_v10_selected_candidates",
                    "contract": contract.as_dict(),
                    **performance_summary(
                        selected,
                        config,
                        bootstrap_samples=4_000,
                        seed=config.model.random_seed
                        + 11_000
                        + len(contract_diagnostics),
                    ),
                }
            )

    fixed_contract_runs: dict[str, dict[str, Any]] = {}
    all_selected_frames: list[pd.DataFrame] = []
    fold_audit_rows: list[dict[str, Any]] = []
    for contract in exit_contracts():
        result = _nested_oos(
            candidates,
            config,
            contract_ids=(contract.contract_id,),
            label=f"fixed:{contract.contract_id}",
        )
        fixed_contract_runs[contract.contract_id] = result["summary"]
        fold_audit_rows.extend(result["fold_rows"])
        if not result["selected"].empty:
            all_selected_frames.append(result["selected"])

    adaptive = _nested_oos(
        candidates,
        config,
        contract_ids=tuple(
            contract.contract_id for contract in exit_contracts()
        ),
        label="adaptive_exit_contract",
    )
    fold_audit_rows.extend(adaptive["fold_rows"])
    if not adaptive["selected"].empty:
        all_selected_frames.append(adaptive["selected"])

    summary = {
        "schema_version": "wp_v11_executable_exit_research_1",
        "research_only": True,
        "production_model_changed": False,
        "objective": (
            "positive net T+1 return under a predeclared executable exit "
            "contract after costs and failed-exit penalties"
        ),
        "source": "immutable_wp_v9_walk_forward_oos_predictions",
        "evaluation_start": config.history.evaluation_start_date,
        "evaluation_end": config.history.evaluation_end_date,
        "protocol": {
            "candidate_frontier": (
                "fixed union of top 12 per causal V9 score per slot"
            ),
            "policy_design_days": POLICY_DESIGN_DAYS,
            "policy_confirmation_days": POLICY_CONFIRMATION_DAYS,
            "purge_days": PURGE_DAYS,
            "contract_selection": (
                "design pass then independent confirmation pass then "
                "frozen outer-fold test"
            ),
            "take_profit_fill": (
                "preplaced limit credited at target only when T+1 daily high "
                "trades at least one price tick through; otherwise close "
                "auction fallback"
            ),
            "no_trade_allowed": True,
            "production_requires_new_150_day_shadow": True,
        },
        "contracts": [contract.as_dict() for contract in exit_contracts()],
        "shards": shard_audit,
        "candidate_frontier_rows": int(len(candidates)),
        "candidate_frontier_sha256": _sha256(frontier_path),
        "v10_candidate_rows_matched": int(len(v10_slice)),
        "diagnostics": contract_diagnostics,
        "fixed_contract_nested_oos": fixed_contract_runs,
        "adaptive_contract_nested_oos": adaptive["summary"],
        "best_research_direction": _best_direction(
            fixed_contract_runs,
            adaptive["summary"],
        ),
    }
    atomic_write_json(output / "wp_v11_exit_summary.json", summary)
    atomic_write_csv(
        pd.json_normalize(contract_diagnostics, sep="."),
        output / "wp_v11_exit_contract_diagnostics.csv",
    )
    atomic_write_csv(
        pd.json_normalize(fold_audit_rows, sep="."),
        output / "wp_v11_exit_nested_folds.csv",
    )
    selected_all = (
        pd.concat(all_selected_frames, ignore_index=True)
        if all_selected_frames
        else candidates.head(0).copy()
    )
    selected_columns = [
        column
        for column in (
            "research_run",
            *IDENTITY_COLUMNS,
            "name",
            "fold",
            "exit_contract_id",
            "exit_policy_id",
            "entry_price",
            "net_return_pct",
            "entry_fillable",
            "exit_fillable",
            "p_net_positive",
            "p_severe_loss",
            "expected_utility_pct",
            "selection_rank_pct",
        )
        if column in selected_all
    ]
    atomic_write_csv(
        selected_all.loc[:, selected_columns],
        output / "wp_v11_exit_oos_candidates.csv",
    )
    print(
        "WP_V11_EXIT_RESULT="
        + json.dumps(
            summary,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


def _nested_oos(
    candidates: pd.DataFrame,
    config: Any,
    *,
    contract_ids: tuple[str, ...],
    label: str,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    folds = sorted(
        pd.to_numeric(candidates["fold"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    for fold in folds:
        current = pd.to_numeric(
            candidates["fold"],
            errors="coerce",
        ).eq(fold)
        test = candidates.loc[current].copy()
        test_start = str(test["trade_date"].min())
        history = candidates.loc[
            ~current & candidates["trade_date"].astype(str).lt(test_start)
        ].copy()
        segments = _rolling_segments(
            sorted(history["trade_date"].astype(str).unique())
        )
        base_row: dict[str, Any] = {
            "research_run": label,
            "fold": fold,
            "test_start": test_start,
            "test_end": str(test["trade_date"].max()),
        }
        if segments is None:
            fold_rows.append(
                {
                    **base_row,
                    "authorized": False,
                    "reason": "insufficient_prior_oos_history",
                    "test": fast_exit_metrics(test.head(0), config),
                }
            )
            continue
        design_dates, confirmation_dates = segments
        design = history.loc[
            history["trade_date"].astype(str).isin(design_dates)
        ]
        confirmation = history.loc[
            history["trade_date"].astype(str).isin(confirmation_dates)
        ]
        policy, audit = select_exit_policy(
            design,
            confirmation,
            config,
            contract_ids=contract_ids,
        )
        if policy is None:
            selected = test.head(0).copy()
            authorized = False
            reason = "no_policy_passed_design_and_confirmation"
        else:
            selected = apply_exit_policy(test, policy)
            selected["exit_policy_id"] = policy.policy_id
            selected["research_run"] = label
            selected_frames.append(selected)
            authorized = True
            reason = "nested_policy_authorized"
        test_metrics = fast_exit_metrics(selected, config)
        fold_rows.append(
            {
                **base_row,
                "authorized": authorized,
                "reason": reason,
                "policy": policy.as_dict() if policy else None,
                "search": {
                    key: audit[key]
                    for key in (
                        "tested",
                        "design_passed",
                        "confirmation_passed",
                    )
                },
                "design": audit.get("design", {}),
                "confirmation": audit.get("confirmation", {}),
                "test": test_metrics,
            }
        )
        print(
            f"[wp-v11] {label} fold={fold} authorized={authorized} "
            f"policy={policy.policy_id if policy else 'none'} "
            f"events={test_metrics['events']} "
            f"mean={test_metrics['mean_net_return_pct']} "
            f"win={test_metrics['win_rate']:.4f}",
            flush=True,
        )

    selected_all = (
        pd.concat(selected_frames, ignore_index=True)
        if selected_frames
        else candidates.head(0).copy()
    )
    metrics = performance_summary(
        selected_all,
        config,
        bootstrap_samples=4_000,
        seed=config.model.random_seed
        + sum(ord(character) for character in label),
    )
    readiness = _research_readiness(metrics)
    final_policy, final_audit = _final_policy(
        candidates,
        config,
        contract_ids=contract_ids,
    )
    summary = {
        "label": label,
        "contract_ids": list(contract_ids),
        "folds_total": len(folds),
        "folds_authorized": int(
            sum(bool(row.get("authorized")) for row in fold_rows)
        ),
        "oos_metrics": metrics,
        "research_readiness": readiness,
        "final_shadow_challenger": {
            "policy": final_policy.as_dict() if final_policy else None,
            "selection_audit": final_audit,
            "production_authorized": False,
            "reason": (
                "new_150_day_shadow_required"
                if final_policy is not None
                else "no_current_policy_confirmed"
            ),
        },
    }
    return {
        "summary": summary,
        "fold_rows": fold_rows,
        "selected": selected_all,
    }


def _final_policy(
    candidates: pd.DataFrame,
    config: Any,
    *,
    contract_ids: tuple[str, ...],
) -> tuple[Any, dict[str, Any]]:
    dates = sorted(candidates["trade_date"].astype(str).unique())
    segments = _rolling_segments(dates, reserve_final_purge=False)
    if segments is None:
        return None, {"reason": "insufficient_history"}
    design_dates, confirmation_dates = segments
    design = candidates.loc[
        candidates["trade_date"].astype(str).isin(design_dates)
    ]
    confirmation = candidates.loc[
        candidates["trade_date"].astype(str).isin(confirmation_dates)
    ]
    return select_exit_policy(
        design,
        confirmation,
        config,
        contract_ids=contract_ids,
    )


def _rolling_segments(
    prior_dates: list[str],
    *,
    reserve_final_purge: bool = True,
) -> tuple[list[str], list[str]] | None:
    final_purge = PURGE_DAYS if reserve_final_purge else 0
    needed = (
        POLICY_DESIGN_DAYS
        + PURGE_DAYS
        + POLICY_CONFIRMATION_DAYS
        + final_purge
    )
    if len(prior_dates) < needed:
        return None
    selected = prior_dates[-needed:]
    design = selected[:POLICY_DESIGN_DAYS]
    confirmation_start = POLICY_DESIGN_DAYS + PURGE_DAYS
    confirmation = selected[
        confirmation_start : confirmation_start + POLICY_CONFIRMATION_DAYS
    ]
    return design, confirmation


def _research_readiness(metrics: dict[str, Any]) -> dict[str, Any]:
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


def _best_direction(
    fixed: dict[str, dict[str, Any]],
    adaptive: dict[str, Any],
) -> dict[str, Any]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for contract_id, summary in fixed.items():
        rows.append((contract_id, summary.get("oos_metrics", {})))
    rows.append(
        ("adaptive_exit_contract", adaptive.get("oos_metrics", {}))
    )
    eligible = [
        (name, metrics)
        for name, metrics in rows
        if int(metrics.get("events", 0) or 0) > 0
    ]
    if not eligible:
        return {
            "status": "no_nested_oos_candidates",
            "direction": None,
        }
    eligible.sort(
        key=lambda item: (
            item[1].get("stress", {})
            .get("50bps", {})
            .get("mean_net_return_pct")
            or -999.0,
            item[1].get("mean_net_return_pct") or -999.0,
            item[1].get("profit_factor") or 0.0,
            item[1].get("events") or 0,
        ),
        reverse=True,
    )
    name, metrics = eligible[0]
    return {
        "status": "ranked_research_direction_only",
        "direction": name,
        "metrics": metrics,
        "production_authorized": False,
    }


def _load_v10_candidates(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=IDENTITY_COLUMNS)
    root = Path(path)
    matches = sorted(root.rglob("wp_v10_meta_oos_candidates.csv"))
    if not matches:
        return pd.DataFrame(columns=IDENTITY_COLUMNS)
    frame = pd.read_csv(matches[0], dtype=str)
    missing = sorted(set(IDENTITY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"V10 candidates missing identity columns: {missing}")
    return frame.loc[:, IDENTITY_COLUMNS].drop_duplicates().reset_index(drop=True)


def _match_identity_slice(
    candidates: pd.DataFrame,
    identities: pd.DataFrame,
) -> pd.DataFrame:
    if identities.empty:
        return candidates.head(0).copy()
    marker = identities.copy()
    marker["_selected_v10"] = True
    merged = candidates.merge(
        marker,
        on=list(IDENTITY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(marker):
        raise RuntimeError(
            f"matched {len(merged)} of {len(marker)} V10 candidate identities"
        )
    return merged.drop(columns="_selected_v10")


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not serializable")


if __name__ == "__main__":
    raise SystemExit(main())
