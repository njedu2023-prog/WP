from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from wp.v3.io import atomic_write_csv, atomic_write_json, file_sha256


SCHEMA_VERSION = "wp_v20_loss_attribution_1"
STRESS_BPS = 50

CATEGORICAL_DIMENSIONS = (
    "signal_slot",
    "v20_source_fold",
    "v20_stock_rank_in_slot",
    "v20_leader_appearances_so_far",
)

NUMERIC_DIMENSIONS = (
    "v20_gate_p_positive_lower",
    "v20_gate_expected_net_return_pct",
    "v20_gate_return_q25_pct",
    "v20_gate_probability_spread",
    "v20_gate_score",
    "v20_stock_score",
    "p_net_positive_lower",
    "expected_utility_lower_pct",
    "p_severe_loss",
    "ret_from_prev_close_pct",
    "v19_full_context_return_mean_pct",
    "v19_full_context_return_dispersion_pct",
    "v19_full_context_breadth_positive",
    "v19_full_context_breadth_above_5pct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute V20 nested-OOS gains and losses without selecting or "
            "authorizing a successor policy."
        )
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    summary_path = find_one(source, "wp_v20_research_summary.json")
    candidate_path = find_one(
        source,
        "wp_v20_nested_oos_candidates.csv",
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = pd.read_csv(candidate_path)
    diagnosis, groups, correlations = diagnose(summary, candidates)
    groups_path = atomic_write_csv(
        groups,
        output / "wp_v20_loss_attribution_groups.csv",
    )
    correlations_path = atomic_write_csv(
        correlations,
        output / "wp_v20_feature_return_correlations.csv",
    )
    diagnosis["artifacts"] = {
        "source_summary_sha256": file_sha256(summary_path),
        "source_candidates_sha256": file_sha256(candidate_path),
        "groups_sha256": file_sha256(groups_path),
        "correlations_sha256": file_sha256(correlations_path),
    }
    atomic_write_json(
        output / "wp_v20_loss_attribution.json",
        diagnosis,
    )
    print(
        "WP_V20_DIAGNOSIS_RESULT="
        + json.dumps(
            json_safe(
                {
                    "overall": diagnosis["overall"],
                    "loss_concentration": diagnosis["loss_concentration"],
                    "exploratory_positive_stress_groups": (
                        diagnosis["exploratory_positive_stress_groups"]
                    ),
                    "exploratory_worst_groups": (
                        diagnosis["exploratory_worst_groups"]
                    ),
                    "strongest_feature_correlations": (
                        diagnosis["strongest_feature_correlations"]
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


def diagnose(
    summary: dict[str, Any],
    candidates: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    required = {
        "trade_date",
        "signal_slot",
        "ts_code",
        "net_return_pct",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"V20 candidates missing columns: {missing}")
    frame = candidates.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["year"] = frame["trade_date"].str[:4]
    returns = _numeric(frame, "net_return_pct")
    frame = frame.loc[returns.notna()].copy()
    expected_events = int(
        (summary.get("nested_oos_metrics") or {}).get("events") or 0
    )
    if len(frame) != expected_events:
        raise RuntimeError(
            f"V20 candidate count mismatch: {len(frame)} != {expected_events}"
        )

    group_frames: list[pd.DataFrame] = []
    for dimension in ("year", *CATEGORICAL_DIMENSIONS):
        if dimension not in frame:
            continue
        group_frames.append(categorical_groups(frame, dimension))
    for dimension in NUMERIC_DIMENSIONS:
        if dimension not in frame:
            continue
        grouped = quantile_groups(frame, dimension)
        if not grouped.empty:
            group_frames.append(grouped)
    groups = (
        pd.concat(group_frames, ignore_index=True)
        if group_frames
        else pd.DataFrame()
    )
    if not groups.empty:
        groups.sort_values(
            ["dimension", "group_order"],
            kind="stable",
            inplace=True,
        )
        groups.reset_index(drop=True, inplace=True)

    correlations = feature_correlations(frame, NUMERIC_DIMENSIONS)
    large = groups.loc[groups["events"].ge(20)].copy()
    positive_stress = large.loc[
        large["stress_50bps_mean_net_return_pct"].ge(0.0)
    ].sort_values(
        [
            "stress_50bps_mean_net_return_pct",
            "events",
        ],
        ascending=[False, False],
        kind="stable",
    )
    worst = large.sort_values(
        ["mean_net_return_pct", "events"],
        ascending=[True, False],
        kind="stable",
    )
    negative = -_numeric(frame, "net_return_pct").clip(upper=0.0)
    ordered_losses = negative.sort_values(ascending=False)
    total_loss = float(negative.sum())
    top_ten_loss = float(ordered_losses.head(10).sum())
    top_twenty_loss = float(ordered_losses.head(20).sum())

    return (
        {
            "schema_version": SCHEMA_VERSION,
            "research_only": True,
            "production_authorized": False,
            "successor_policy_authorized": False,
            "outcome_driven_threshold_selection_allowed": False,
            "source_schema_version": summary.get("schema_version"),
            "overall": simple_metrics(frame),
            "loss_concentration": {
                "total_negative_return_points": total_loss,
                "top_10_losses_points": top_ten_loss,
                "top_10_loss_share": (
                    top_ten_loss / total_loss if total_loss else 0.0
                ),
                "top_20_losses_points": top_twenty_loss,
                "top_20_loss_share": (
                    top_twenty_loss / total_loss if total_loss else 0.0
                ),
            },
            "groups_evaluated": int(len(groups)),
            "groups_with_at_least_20_events": int(len(large)),
            "exploratory_positive_stress_groups": records(
                positive_stress.head(15)
            ),
            "exploratory_worst_groups": records(worst.head(15)),
            "strongest_feature_correlations": records(
                correlations.head(15)
            ),
            "interpretation_contract": (
                "These are exploratory loss-attribution groups, not candidate "
                "rules. Any V21 mechanism must be independently specified and "
                "frozen before another nested evaluation."
            ),
        },
        groups,
        correlations,
    )


def categorical_groups(
    frame: pd.DataFrame,
    dimension: str,
) -> pd.DataFrame:
    rows = []
    values = frame[dimension].astype(str).fillna("MISSING")
    for order, value in enumerate(sorted(values.unique())):
        group = frame.loc[values.eq(value)]
        rows.append(
            {
                "dimension": dimension,
                "group": value,
                "group_order": order,
                "bin_lower": np.nan,
                "bin_upper": np.nan,
                **simple_metrics(group),
            }
        )
    return pd.DataFrame(rows)


def quantile_groups(
    frame: pd.DataFrame,
    dimension: str,
    *,
    bins: int = 5,
) -> pd.DataFrame:
    values = _numeric(frame, dimension)
    clean = values.dropna()
    if clean.nunique() < 2:
        return pd.DataFrame()
    ranked = clean.rank(method="first")
    quantile = pd.qcut(
        ranked,
        q=min(bins, clean.nunique()),
        labels=False,
        duplicates="drop",
    )
    rows = []
    for bucket in sorted(quantile.dropna().astype(int).unique()):
        indexes = quantile.index[quantile.eq(bucket)]
        group = frame.loc[indexes]
        group_values = values.loc[indexes]
        rows.append(
            {
                "dimension": dimension,
                "group": f"Q{bucket + 1}",
                "group_order": int(bucket),
                "bin_lower": float(group_values.min()),
                "bin_upper": float(group_values.max()),
                **simple_metrics(group),
            }
        )
    return pd.DataFrame(rows)


def simple_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    returns = _numeric(frame, "net_return_pct").dropna()
    events = int(len(returns))
    profits = float(returns.loc[returns > 0].sum())
    losses = float(-returns.loc[returns < 0].sum())
    profit_factor = (
        profits / losses
        if losses > 0
        else (float("inf") if profits > 0 else 0.0)
    )
    return {
        "events": events,
        "candidate_days": int(
            frame.loc[returns.index, "trade_date"].astype(str).nunique()
        ),
        "wins": int(returns.gt(0).sum()),
        "win_rate": float(returns.gt(0).mean()) if events else 0.0,
        "mean_net_return_pct": (
            float(returns.mean()) if events else float("nan")
        ),
        "median_net_return_pct": (
            float(returns.median()) if events else float("nan")
        ),
        "return_p10_pct": (
            float(returns.quantile(0.10)) if events else float("nan")
        ),
        "profit_factor": profit_factor,
        "stress_50bps_mean_net_return_pct": (
            float((returns - STRESS_BPS / 100.0).mean())
            if events
            else float("nan")
        ),
    }


def feature_correlations(
    frame: pd.DataFrame,
    dimensions: Iterable[str],
) -> pd.DataFrame:
    target = _numeric(frame, "net_return_pct")
    rows = []
    for feature in dimensions:
        if feature not in frame:
            continue
        values = _numeric(frame, feature)
        clean = values.notna() & target.notna()
        if clean.sum() < 20 or values.loc[clean].nunique() < 2:
            continue
        correlation = values.loc[clean].corr(
            target.loc[clean],
            method="spearman",
        )
        rows.append(
            {
                "feature": feature,
                "events": int(clean.sum()),
                "spearman_return_correlation": float(correlation),
                "absolute_correlation": float(abs(correlation)),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result.sort_values(
            ["absolute_correlation", "feature"],
            ascending=[False, True],
            kind="stable",
            inplace=True,
        )
        result.reset_index(drop=True, inplace=True)
    return result


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        json_safe(row)
        for row in frame.to_dict(orient="records")
    ]


def find_one(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one {name} under {root}, found {len(matches)}"
        )
    return matches[0]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


if __name__ == "__main__":
    raise SystemExit(main())
