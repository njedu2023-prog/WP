from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wp.v3.contracts import load_v3_config
from wp.v3.exit_research import fast_exit_metrics
from wp.v3.history import load_panel_partitions
from wp.v3.io import atomic_write_csv, atomic_write_json
from wp.v3.meta_alpha import IDENTITY_COLUMNS
from wp.v3.overlay import performance_summary
from wp.v3.tail_exit import (
    TAIL_PANEL_COLUMNS,
    attach_t1_tail_exit_truth,
    materialize_tail_exit_contract,
    tail_exit_contracts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit executable next-five-minute T+1 tail exits over immutable "
            "V11 and V10 out-of-sample evidence."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--v11-source-dir", required=True)
    parser.add_argument("--v10-source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_v3_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frontier, source_audit = load_v11_frontier(args.v11_source_dir)
    identities = load_v10_identities(args.v10_source_dir)
    exact_v10 = match_identities(frontier, identities)
    panel = load_panel_partitions(
        args.panel_dir,
        columns=list(TAIL_PANEL_COLUMNS),
        start_date=config.history.evaluation_start_date,
        end_date=config.history.evaluation_end_date,
    )
    enriched = attach_t1_tail_exit_truth(frontier, panel, config)
    exact_enriched = match_identities(enriched, identities)

    diagnostic_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    baseline_frames = {
        "causal_candidate_frontier": materialize_close_baseline(enriched),
        "exact_v10_selected_candidates": materialize_close_baseline(
            exact_enriched
        ),
    }
    for scope, baseline in baseline_frames.items():
        valid_baseline = baseline.loc[
            baseline["label_available"].fillna(False).astype(bool)
        ].copy()
        diagnostic_rows.append(
            {
                "scope": scope,
                "contract_id": "t1_close_auction",
                "decision_slot": "14:57",
                "benchmark_slot": "close_auction",
                **scope_metrics(valid_baseline, config, scope),
            }
        )

    materialized_exact: list[pd.DataFrame] = []
    for contract_index, contract in enumerate(
        tail_exit_contracts(),
        start=1,
    ):
        for scope, source in (
            ("causal_candidate_frontier", enriched),
            ("exact_v10_selected_candidates", exact_enriched),
        ):
            materialized = materialize_tail_exit_contract(
                source,
                contract.contract_id,
            )
            materialized = materialized.loc[
                materialized["label_available"].fillna(False).astype(bool)
            ].copy()
            if scope == "exact_v10_selected_candidates":
                materialized_exact.append(materialized)
            metrics = scope_metrics(materialized, config, scope)
            diagnostic_rows.append(
                {
                    "scope": scope,
                    **contract.as_dict(),
                    **metrics,
                }
            )
            baseline = baseline_frames[scope]
            paired = paired_contract_delta(
                materialized,
                baseline,
                seed=(
                    config.model.random_seed
                    + 13_000
                    + contract_index
                    + (100 if scope.startswith("exact") else 0)
                ),
                bootstrap_samples=(
                    4_000 if scope.startswith("exact") else 1_000
                ),
            )
            paired_rows.append(
                {
                    "scope": scope,
                    **contract.as_dict(),
                    **paired,
                }
            )
            if scope.startswith("exact"):
                for signal_slot, group in materialized.groupby(
                    "signal_slot",
                    sort=True,
                ):
                    slot_rows.append(
                        {
                            "scope": scope,
                            **contract.as_dict(),
                            "entry_signal_slot": str(signal_slot),
                            **performance_summary(
                                group,
                                config,
                                bootstrap_samples=1_000,
                                seed=(
                                    config.model.random_seed
                                    + contract_index * 100
                                    + sum(ord(value) for value in str(signal_slot))
                                ),
                            ),
                        }
                    )

    diagnostics = pd.DataFrame(diagnostic_rows)
    paired = pd.DataFrame(paired_rows)
    slots = pd.DataFrame(slot_rows)
    best_direction = rank_research_direction(diagnostic_rows, paired_rows)
    summary = {
        "schema_version": "wp_v13_t1_tail_exit_research_1",
        "research_only": True,
        "production_model_changed": False,
        "objective": (
            "find an executable T+1 sell-time direction that improves net "
            "return without changing the immutable T-day entry decision"
        ),
        "source": source_audit,
        "evaluation_start": config.history.evaluation_start_date,
        "evaluation_end": config.history.evaluation_end_date,
        "protocol": {
            "entry_contract": (
                "immutable V9/V10 next-five-minute close plus 10bp slippage"
            ),
            "exit_contract": (
                "fixed T+1 decision slot, execution at the next five-minute "
                "close minus 10bp slippage"
            ),
            "liquidity": (
                "minimum 3m RMB benchmark bar, 100k order <=1% of bar amount"
            ),
            "down_limit": (
                "price at or below down-limit queue is non-fill and receives "
                "the predeclared -10% penalty"
            ),
            "multiple_testing": (
                "four sell times are research hypotheses only; a positive "
                "mean is not production authorization"
            ),
            "production_requires_new_150_day_shadow": True,
        },
        "contracts": [
            contract.as_dict() for contract in tail_exit_contracts()
        ],
        "candidate_frontier_rows": int(len(frontier)),
        "exact_v10_rows": int(len(exact_v10)),
        "diagnostics": diagnostic_rows,
        "paired_vs_t1_close": paired_rows,
        "entry_slot_diagnostics": slot_rows,
        "best_research_direction": best_direction,
    }
    atomic_write_json(output / "wp_v13_t1_tail_exit_summary.json", summary)
    atomic_write_csv(
        diagnostics,
        output / "wp_v13_t1_tail_exit_diagnostics.csv",
    )
    atomic_write_csv(
        paired,
        output / "wp_v13_t1_tail_exit_paired.csv",
    )
    atomic_write_csv(
        slots,
        output / "wp_v13_t1_tail_exit_by_entry_slot.csv",
    )
    exact_outcomes = (
        pd.concat(materialized_exact, ignore_index=True)
        if materialized_exact
        else exact_enriched.head(0).copy()
    )
    exact_columns = [
        column
        for column in (
            *IDENTITY_COLUMNS,
            "name",
            "fold",
            "exit_contract_id",
            "entry_price",
            "net_return_pct",
            "entry_fillable",
            "exit_fillable",
        )
        if column in exact_outcomes
    ]
    atomic_write_csv(
        exact_outcomes.loc[:, exact_columns],
        output / "wp_v13_exact_v10_exit_outcomes.csv",
    )
    print(
        "WP_V13_T1_TAIL_EXIT_RESULT="
        + json.dumps(
            summary,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
            default=json_default,
        ),
        flush=True,
    )
    return 0


def load_v11_frontier(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(path)
    frontier_paths = sorted(root.rglob("wp_v11_exit_frontier.parquet"))
    summary_paths = sorted(root.rglob("wp_v11_exit_summary.json"))
    if len(frontier_paths) != 1 or len(summary_paths) != 1:
        raise FileNotFoundError(
            "V11 source must contain exactly one frontier and one summary"
        )
    frontier_path = frontier_paths[0]
    summary = json.loads(summary_paths[0].read_text(encoding="utf-8"))
    expected_sha = str(summary.get("candidate_frontier_sha256") or "")
    actual_sha = sha256(frontier_path)
    if not expected_sha or actual_sha != expected_sha:
        raise RuntimeError(
            f"V11 frontier digest mismatch: {actual_sha} != {expected_sha}"
        )
    frame = pd.read_parquet(frontier_path)
    missing = sorted(
        {
            *IDENTITY_COLUMNS,
            "entry_price",
            "entry_fillable",
            "net_t1_close_auction_pct",
            "exit_t1_close_auction_fillable",
        }
        - set(frame.columns)
    )
    if missing:
        raise ValueError(f"V11 frontier missing columns: {missing}")
    return frame, {
        "v11_schema_version": summary.get("schema_version"),
        "v11_frontier_sha256": actual_sha,
        "v11_frontier_rows": int(len(frame)),
    }


def load_v10_identities(path: str | Path) -> pd.DataFrame:
    matches = sorted(
        Path(path).rglob("wp_v10_meta_oos_candidates.csv")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            "V10 source must contain exactly one selected-candidate CSV"
        )
    frame = pd.read_csv(matches[0], dtype=str)
    missing = sorted(set(IDENTITY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"V10 selected candidates missing {missing}")
    identities = frame.loc[:, IDENTITY_COLUMNS].drop_duplicates()
    if len(identities) != len(frame):
        raise RuntimeError("V10 selected-candidate identities are not unique")
    return identities.reset_index(drop=True)


def match_identities(
    frame: pd.DataFrame,
    identities: pd.DataFrame,
) -> pd.DataFrame:
    marker = identities.copy()
    marker["_selected_v10"] = True
    matched = frame.merge(
        marker,
        on=list(IDENTITY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    if len(matched) != len(marker):
        raise RuntimeError(
            f"matched {len(matched)} of {len(marker)} V10 identities"
        )
    return matched.drop(columns="_selected_v10")


def materialize_close_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["net_return_pct"] = pd.to_numeric(
        result["net_t1_close_auction_pct"],
        errors="coerce",
    )
    result["exit_fillable"] = boolean(
        result["exit_t1_close_auction_fillable"]
    )
    result["label_available"] = result["net_return_pct"].notna()
    result["exit_contract_id"] = "t1_close_auction"
    return result


def scope_metrics(
    frame: pd.DataFrame,
    config: Any,
    scope: str,
) -> dict[str, Any]:
    if scope.startswith("exact"):
        return performance_summary(
            frame,
            config,
            bootstrap_samples=4_000,
            seed=(
                config.model.random_seed
                + 13_500
                + sum(ord(value) for value in str(
                    frame.get(
                        "exit_contract_id",
                        pd.Series(["unknown"]),
                    ).iloc[0] if not frame.empty else "empty"
                ))
            ),
        )
    return fast_exit_metrics(frame, config)


def paired_contract_delta(
    tail: pd.DataFrame,
    close: pd.DataFrame,
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    tail_values = tail.loc[
        :,
        [*IDENTITY_COLUMNS, "net_return_pct"],
    ].rename(columns={"net_return_pct": "tail_net_return_pct"})
    close_values = close.loc[
        close["label_available"].fillna(False).astype(bool),
        [*IDENTITY_COLUMNS, "net_return_pct"],
    ].rename(columns={"net_return_pct": "close_net_return_pct"})
    paired = tail_values.merge(
        close_values,
        on=list(IDENTITY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    paired["delta_pct"] = (
        pd.to_numeric(paired["tail_net_return_pct"], errors="coerce")
        - pd.to_numeric(paired["close_net_return_pct"], errors="coerce")
    )
    paired = paired.dropna(subset=["delta_pct"])
    if paired.empty:
        return {
            "paired_events": 0,
            "paired_days": 0,
            "mean_delta_pct": None,
            "median_delta_pct": None,
            "improved_event_share": None,
            "day_equal_mean_delta_pct": None,
            "day_clustered_mean_delta_lower_pct": None,
            "day_clustered_mean_delta_upper_pct": None,
        }
    daily = paired.groupby("trade_date", sort=True)["delta_pct"].mean()
    rng = np.random.default_rng(seed)
    samples = np.empty(bootstrap_samples, dtype=float)
    values = daily.to_numpy(dtype=float)
    for index in range(bootstrap_samples):
        samples[index] = rng.choice(
            values,
            size=len(values),
            replace=True,
        ).mean()
    return {
        "paired_events": int(len(paired)),
        "paired_days": int(len(daily)),
        "mean_delta_pct": finite(paired["delta_pct"].mean()),
        "median_delta_pct": finite(paired["delta_pct"].median()),
        "improved_event_share": finite(paired["delta_pct"].gt(0).mean()),
        "day_equal_mean_delta_pct": finite(daily.mean()),
        "day_clustered_mean_delta_lower_pct": finite(
            np.quantile(samples, 0.025)
        ),
        "day_clustered_mean_delta_upper_pct": finite(
            np.quantile(samples, 0.975)
        ),
    }


def rank_research_direction(
    diagnostics: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> dict[str, Any]:
    exact = [
        row
        for row in diagnostics
        if (
            row.get("scope") == "exact_v10_selected_candidates"
            and row.get("contract_id") != "t1_close_auction"
            and int(row.get("events", 0) or 0) > 0
        )
    ]
    if not exact:
        return {
            "status": "no_exact_v10_tail_outcomes",
            "contract_id": None,
            "production_authorized": False,
        }
    exact.sort(
        key=lambda row: (
            (
                row.get("stress", {})
                .get("50bps", {})
                .get("mean_net_return_pct")
                or -999.0
            ),
            row.get("mean_net_return_pct") or -999.0,
            row.get("profit_factor") or 0.0,
        ),
        reverse=True,
    )
    best = exact[0]
    delta = next(
        (
            row
            for row in paired
            if (
                row.get("scope") == "exact_v10_selected_candidates"
                and row.get("contract_id") == best.get("contract_id")
            )
        ),
        {},
    )
    stress = best.get("stress", {}).get("50bps", {})
    positive = bool(
        (best.get("mean_net_return_pct") or -999.0) > 0.0
        and (best.get("profit_factor") or 0.0) > 1.0
        and stress.get("positive_total_return", False)
    )
    confirmed = bool(
        positive
        and int(best.get("events", 0) or 0) >= 250
        and (
            best.get("mean_net_return_day_clustered_lower_pct") or -999.0
        )
        > 0.0
        and (best.get("win_rate_day_clustered_lower") or 0.0) >= 0.52
    )
    return {
        "status": (
            "positive_and_statistically_confirmed"
            if confirmed
            else (
                "positive_mean_unconfirmed"
                if positive
                else "no_positive_exact_v10_tail_contract"
            )
        ),
        "contract_id": best.get("contract_id"),
        "metrics": best,
        "paired_vs_close": delta,
        "production_authorized": False,
        "reason": (
            "new_150_day_shadow_required"
            if confirmed
            else "historical_direction_only"
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boolean(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "qualified", "pass"}
    )


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return finite(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


if __name__ == "__main__":
    raise SystemExit(main())
