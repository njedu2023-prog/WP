from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .contracts import V3Config
from .dataset import first_crossing_candidates
from .statistics import day_clustered_intervals, wilson_interval


GATE_ORDER = (
    ("execution_eligible", "execution"),
    ("candidate_policy_authorized", "prior_oos_policy_authorized"),
    ("passes_policy", "all_candidate_gates"),
)

SCORE_COLUMNS = {
    "executable_positive_probability": "p_net_positive",
    "conservative_probability": "p_net_positive_lower",
    "entry_fill_probability": "p_entry_fill",
    "exit_fill_probability": "p_exit_fill_given_entry",
    "round_trip_fill_probability": "p_round_trip_fill",
    "conditional_positive_probability": "p_conditional_net_positive",
    "expected_utility": "expected_utility_pct",
    "conditional_expected_return": "conditional_expected_net_return_pct",
    "cross_section_probability": "p_cross_section_top",
    "severe_loss_safety": "_severe_loss_safety",
    "learned_rank": "ranking_score",
    "selection_score": "selection_score",
}


def build_prediction_diagnostics(
    predictions: pd.DataFrame,
    config: V3Config,
) -> dict[str, Any]:
    """Describe OOS discrimination without changing the frozen trading policy."""
    if predictions.empty:
        return {
            "schema_version": "wp_v7_prediction_diagnostics_2",
            "rows": 0,
            "policy_funnel": [],
            "score_quality": {},
            "score_deciles": [],
            "top_n_per_slot": [],
            "extreme_rank_cohorts": [],
            "joint_gate_cohorts": [],
            "slot_quality": [],
        }

    frame = predictions.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["signal_slot"] = frame["signal_slot"].astype(str)
    frame["target_net_positive"] = pd.to_numeric(
        frame.get("target_net_positive"),
        errors="coerce",
    )
    frame["net_return_pct"] = pd.to_numeric(
        frame.get("net_return_pct"),
        errors="coerce",
    )
    for column in {
        *SCORE_COLUMNS.values(),
        "downside_q10_pct",
        "p_entry_fill",
        "p_exit_fill_given_entry",
        "p_round_trip_fill",
        "p_conditional_net_positive",
        "expected_utility_pct",
        "conditional_expected_net_return_pct",
        "p_cross_section_top",
        "p_severe_loss",
        "ranking_score",
        "probability_model_spread",
        "selection_rank_spread",
        "selection_rank_pct",
    }:
        default = pd.Series(np.nan, index=frame.index, dtype=float)
        frame[column] = pd.to_numeric(
            frame.get(column, default),
            errors="coerce",
        )
    frame = frame.loc[frame["target_net_positive"].notna()].copy()
    frame["target_net_positive"] = frame["target_net_positive"].astype(int)
    frame["_composite_rank"] = _composite_rank(frame)
    frame["_severe_loss_safety"] = 1.0 - pd.to_numeric(
        frame.get("p_severe_loss"),
        errors="coerce",
    )
    for column in ("target_entry_fillable", "target_exit_fillable"):
        frame[column] = pd.to_numeric(
            frame.get(column, pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )

    score_columns = {
        **SCORE_COLUMNS,
        "legacy_composite_rank": "_composite_rank",
    }
    score_quality = {
        name: _score_quality(frame, column)
        for name, column in score_columns.items()
    }
    deciles: list[dict[str, Any]] = []
    top_n: list[dict[str, Any]] = []
    for score_name, score_column in score_columns.items():
        deciles.extend(_score_deciles(frame, score_name, score_column))
        top_n.extend(
            _top_n_per_slot(
                frame,
                config,
                score_name=score_name,
                score_column=score_column,
            )
        )

    slot_quality = []
    for slot, group in frame.groupby("signal_slot", sort=True):
        row = {
            "signal_slot": str(slot),
            "rows": int(len(group)),
            "base": _return_summary(group),
        }
        for score_name, score_column in score_columns.items():
            row[score_name] = _score_quality(group, score_column)
        slot_quality.append(row)

    return {
        "schema_version": "wp_v7_prediction_diagnostics_2",
        "rows": int(len(frame)),
        "trade_days": int(frame["trade_date"].nunique()),
        "base": _return_summary(frame),
        "base_by_execution_state": {
            "entry_filled": _return_summary(
                frame.loc[frame["target_entry_fillable"].eq(1)]
            ),
            "round_trip_filled": _return_summary(
                frame.loc[
                    frame["target_entry_fillable"].eq(1)
                    & frame["target_exit_fillable"].eq(1)
                ]
            ),
        },
        "policy_funnel": _policy_funnel(frame),
        "score_distributions": {
            name: _quantiles(frame[column])
            for name, column in {
                **score_columns,
                "downside_q10": "downside_q10_pct",
                "model_spread": "probability_model_spread",
                "selection_rank_spread": "selection_rank_spread",
                "selection_rank_percentile": "selection_rank_pct",
            }.items()
        },
        "score_quality": score_quality,
        "score_deciles": deciles,
        "top_n_per_slot": top_n,
        "extreme_rank_cohorts": _extreme_rank_cohorts(
            frame,
            config,
            score_columns={
                "executable_positive_probability": "p_net_positive",
                "expected_utility": "expected_utility_pct",
                "selection_score": "selection_score",
                "legacy_composite_rank": "_composite_rank",
            },
        ),
        "joint_gate_cohorts": _joint_gate_cohorts(frame, config),
        "slot_quality": slot_quality,
        "interpretation": (
            "Diagnostic cohorts are chronological OOS observations. They do not "
            "alter, tune, or authorize the frozen candidate policy."
        ),
    }


def diagnostics_tables(
    diagnostics: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    return {
        "policy_funnel": pd.DataFrame(diagnostics.get("policy_funnel", [])),
        "score_deciles": pd.DataFrame(diagnostics.get("score_deciles", [])),
        "top_n_per_slot": pd.DataFrame(diagnostics.get("top_n_per_slot", [])),
        "extreme_rank_cohorts": pd.DataFrame(
            diagnostics.get("extreme_rank_cohorts", [])
        ),
        "joint_gate_cohorts": pd.DataFrame(
            diagnostics.get("joint_gate_cohorts", [])
        ),
        "slot_quality": pd.json_normalize(diagnostics.get("slot_quality", [])),
    }


def _policy_funnel(frame: pd.DataFrame) -> list[dict[str, Any]]:
    total = int(len(frame))
    cumulative = pd.Series(True, index=frame.index, dtype=bool)
    rows: list[dict[str, Any]] = []
    for column, label in GATE_ORDER:
        gate = (
            _boolean(frame[column], frame.index)
            if column in frame
            else pd.Series(True, index=frame.index, dtype=bool)
        )
        cumulative &= gate
        rows.append(
            {
                "gate": label,
                "column": column,
                "independent_pass_count": int(gate.sum()),
                "independent_pass_rate": float(gate.mean()) if total else 0.0,
                "cumulative_pass_count": int(cumulative.sum()),
                "cumulative_pass_rate": float(cumulative.mean()) if total else 0.0,
            }
        )
    policy = _boolean(frame.get("passes_policy"), frame.index)
    rows.append(
        {
            "gate": "final_policy",
            "column": "passes_policy",
            "independent_pass_count": int(policy.sum()),
            "independent_pass_rate": float(policy.mean()) if total else 0.0,
            "cumulative_pass_count": int((cumulative & policy).sum()),
            "cumulative_pass_rate": float((cumulative & policy).mean())
            if total
            else 0.0,
        }
    )
    return rows


def _composite_rank(frame: pd.DataFrame) -> pd.Series:
    groups = [frame["trade_date"], frame["signal_slot"]]
    probability_rank = frame["p_net_positive"].groupby(groups, sort=False).rank(
        method="average",
        pct=True,
    )
    expected_rank = frame["expected_utility_pct"].groupby(
        groups,
        sort=False,
    ).rank(method="average", pct=True)
    downside_rank = frame["downside_q10_pct"].groupby(
        groups,
        sort=False,
    ).rank(method="average", pct=True)
    stability_rank = (
        -frame["probability_model_spread"]
    ).groupby(groups, sort=False).rank(method="average", pct=True)
    cross_rank = frame["p_cross_section_top"].groupby(
        groups,
        sort=False,
    ).rank(method="average", pct=True)
    severe_safe_rank = (-frame["p_severe_loss"]).groupby(
        groups,
        sort=False,
    ).rank(method="average", pct=True)
    return (
        0.25 * probability_rank
        + 0.20 * cross_rank
        + 0.20 * frame["ranking_score"]
        + 0.15 * expected_rank
        + 0.10 * downside_rank
        + 0.05 * severe_safe_rank
        + 0.05 * stability_rank
    )


def _score_quality(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    clean = frame.loc[
        frame[score_column].notna()
        & frame["target_net_positive"].notna()
        & frame["net_return_pct"].notna()
    ]
    if clean.empty:
        return {
            "rows": 0,
            "roc_auc": None,
            "average_precision": None,
            "rank_correlation_to_net_return": None,
        }
    target = clean["target_net_positive"].astype(int).to_numpy()
    score = clean[score_column].astype(float).to_numpy()
    auc = (
        float(roc_auc_score(target, score))
        if len(np.unique(target)) > 1 and len(np.unique(score)) > 1
        else None
    )
    average_precision = (
        float(average_precision_score(target, score))
        if len(np.unique(target)) > 1
        else None
    )
    return {
        "rows": int(len(clean)),
        "roc_auc": auc,
        "average_precision": average_precision,
        "rank_correlation_to_net_return": _rank_correlation(
            clean[score_column],
            clean["net_return_pct"],
        ),
    }


def _score_deciles(
    frame: pd.DataFrame,
    score_name: str,
    score_column: str,
) -> list[dict[str, Any]]:
    clean = frame.loc[frame[score_column].notna()].copy()
    if clean.empty:
        return []
    percentile = clean[score_column].groupby(
        [clean["trade_date"], clean["signal_slot"]],
        sort=False,
    ).rank(method="average", pct=True)
    clean["_score_decile"] = np.ceil(percentile * 10.0).clip(1, 10).astype(int)
    rows = []
    for decile, group in clean.groupby("_score_decile", sort=True):
        rows.append(
            {
                "score": score_name,
                "decile": int(decile),
                "direction": "10_is_highest",
                **_return_summary(group),
            }
        )
    return rows


def _top_n_per_slot(
    frame: pd.DataFrame,
    config: V3Config,
    *,
    score_name: str,
    score_column: str,
    values: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> list[dict[str, Any]]:
    clean = frame.loc[
        frame[score_column].notna()
        & _boolean(frame.get("execution_eligible"), frame.index)
    ].copy()
    if clean.empty:
        return []
    rank = clean[score_column].groupby(
        [clean["trade_date"], clean["signal_slot"]],
        sort=False,
    ).rank(method="first", ascending=False)
    rows = []
    for top in values:
        clean["_diagnostic_selected"] = rank.le(top)
        selected = first_crossing_candidates(
            clean,
            config,
            status_column="_diagnostic_selected",
        )
        rows.append(
            {
                "score": score_name,
                "top_n": int(top),
                **_return_summary(selected, clustered=True),
            }
        )
    return rows


def _extreme_rank_cohorts(
    frame: pd.DataFrame,
    config: V3Config,
    *,
    score_columns: dict[str, str],
    thresholds: tuple[float, ...] = (0.99, 0.995, 0.998, 0.999),
) -> list[dict[str, Any]]:
    execution = _boolean(frame.get("execution_eligible"), frame.index)
    rows: list[dict[str, Any]] = []
    for score_name, score_column in score_columns.items():
        clean = frame.loc[frame[score_column].notna() & execution].copy()
        if clean.empty:
            continue
        percentile = clean[score_column].groupby(
            [clean["trade_date"], clean["signal_slot"]],
            sort=False,
        ).rank(method="first", pct=True)
        for threshold in thresholds:
            clean["_diagnostic_selected"] = percentile.ge(threshold)
            selected = first_crossing_candidates(
                clean,
                config,
                status_column="_diagnostic_selected",
            )
            rows.append(
                {
                    "score": score_name,
                    "rank_percentile_min": float(threshold),
                    **_return_summary(selected, clustered=True),
                }
            )
    return rows


def _joint_gate_cohorts(
    frame: pd.DataFrame,
    config: V3Config,
    *,
    rank_thresholds: tuple[float, ...] = (0.99, 0.995),
    utility_thresholds: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20),
    entry_thresholds: tuple[float, ...] = (0.95, 0.97),
    exit_thresholds: tuple[float, ...] = (0.95, 0.98),
) -> list[dict[str, Any]]:
    clean = frame.loc[
        frame["expected_utility_pct"].notna()
        & _boolean(frame.get("execution_eligible"), frame.index)
    ].copy()
    if clean.empty:
        return []
    utility_rank = clean["expected_utility_pct"].groupby(
        [clean["trade_date"], clean["signal_slot"]],
        sort=False,
    ).rank(method="first", pct=True)
    rows: list[dict[str, Any]] = []
    for rank_threshold in rank_thresholds:
        for utility_threshold in utility_thresholds:
            for entry_threshold in entry_thresholds:
                for exit_threshold in exit_thresholds:
                    clean["_diagnostic_selected"] = (
                        utility_rank.ge(rank_threshold)
                        & clean["expected_utility_pct"].ge(utility_threshold)
                        & clean["p_entry_fill"].ge(entry_threshold)
                        & clean["p_exit_fill_given_entry"].ge(exit_threshold)
                    )
                    selected = first_crossing_candidates(
                        clean,
                        config,
                        status_column="_diagnostic_selected",
                    )
                    rows.append(
                        {
                            "rank_percentile_min": float(rank_threshold),
                            "expected_utility_min_pct": float(utility_threshold),
                            "entry_fill_probability_min": float(entry_threshold),
                            "exit_fill_probability_min": float(exit_threshold),
                            **_return_summary(selected, clustered=True),
                        }
                    )
    return rows


def _return_summary(
    frame: pd.DataFrame,
    *,
    clustered: bool = False,
) -> dict[str, Any]:
    returns = pd.to_numeric(frame.get("net_return_pct"), errors="coerce").dropna()
    if returns.empty:
        return {
            "events": 0,
            "trade_days": 0,
            "wins": 0,
            "win_rate": None,
            "win_rate_wilson_lower": None,
            "mean_net_return_pct": None,
            "median_net_return_pct": None,
            "profit_factor": None,
            "win_rate_day_clustered_lower": None,
            "mean_net_return_day_clustered_lower_pct": None,
        }
    selected = frame.loc[returns.index]
    entry = pd.to_numeric(
        selected.get("target_entry_fillable"),
        errors="coerce",
    )
    exit_fill = pd.to_numeric(
        selected.get("target_exit_fillable"),
        errors="coerce",
    )
    entry_filled = entry.eq(1)
    round_trip_filled = entry_filled & exit_fill.eq(1)
    entry_returns = returns.loc[entry_filled]
    round_trip_returns = returns.loc[round_trip_filled]
    wins = int(returns.gt(0).sum())
    lower, _ = wilson_interval(wins, len(returns))
    profits = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    interval = (
        day_clustered_intervals(selected, samples=1_000)
        if clustered
        else None
    )
    return {
        "events": int(len(returns)),
        "trade_days": int(selected["trade_date"].astype(str).nunique()),
        "wins": wins,
        "win_rate": float(returns.gt(0).mean()),
        "win_rate_wilson_lower": float(lower),
        "mean_net_return_pct": float(returns.mean()),
        "median_net_return_pct": float(returns.median()),
        "profit_factor": (
            float(profits / losses)
            if losses > 0
            else (999.0 if profits > 0 else 0.0)
        ),
        "entry_fill_rate": (
            float(entry.mean()) if entry.notna().any() else None
        ),
        "exit_fill_rate_given_entry": (
            float(exit_fill.loc[entry_filled].mean())
            if entry_filled.any()
            else None
        ),
        "round_trip_fill_rate": (
            float(round_trip_filled.mean()) if len(selected) else None
        ),
        "win_rate_given_entry": (
            float(entry_returns.gt(0).mean())
            if not entry_returns.empty
            else None
        ),
        "mean_net_return_given_entry_pct": (
            float(entry_returns.mean()) if not entry_returns.empty else None
        ),
        "mean_net_return_round_trip_pct": (
            float(round_trip_returns.mean())
            if not round_trip_returns.empty
            else None
        ),
        "win_rate_day_clustered_lower": (
            float(interval.win_rate_lower) if interval else None
        ),
        "mean_net_return_day_clustered_lower_pct": (
            float(interval.mean_return_lower_pct) if interval else None
        ),
    }


def _quantiles(values: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0}
    return {
        "count": int(len(clean)),
        "minimum": float(clean.min()),
        "q01": float(clean.quantile(0.01)),
        "q05": float(clean.quantile(0.05)),
        "q10": float(clean.quantile(0.10)),
        "q25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "q75": float(clean.quantile(0.75)),
        "q90": float(clean.quantile(0.90)),
        "q95": float(clean.quantile(0.95)),
        "q99": float(clean.quantile(0.99)),
        "maximum": float(clean.max()),
    }


def _rank_correlation(left: pd.Series, right: pd.Series) -> float | None:
    x = pd.to_numeric(left, errors="coerce")
    y = pd.to_numeric(right, errors="coerce")
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 3:
        return None
    x_rank = x.loc[valid].rank(method="average").to_numpy(dtype=float)
    y_rank = y.loc[valid].rank(method="average").to_numpy(dtype=float)
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _boolean(values: Any, index: pd.Index) -> pd.Series:
    if values is None:
        return pd.Series(False, index=index, dtype=bool)
    if not isinstance(values, pd.Series):
        return pd.Series(bool(values), index=index, dtype=bool)
    if values.dtype == bool:
        return values.reindex(index).fillna(False).astype(bool)
    normalized = values.reindex(index).astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y", "pass", "qualified"})
