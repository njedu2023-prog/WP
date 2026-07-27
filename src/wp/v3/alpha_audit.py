from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS
from .statistics import day_clustered_intervals, wilson_interval


IDENTITY = ("trade_date", "signal_slot", "ts_code")
SCORE_COLUMNS = (
    "p_net_positive",
    "p_net_positive_lower",
    "expected_net_return_pct",
    "downside_q10_pct",
    "ranking_score",
    "selection_score",
)


@dataclass(frozen=True)
class PolicyThresholds:
    probability: float
    probability_lower: float
    expected_return_pct: float
    downside_q10_pct: float
    selection_rank_pct: float

    def as_dict(self) -> dict[str, float]:
        return {
            "probability": self.probability,
            "probability_lower": self.probability_lower,
            "expected_return_pct": self.expected_return_pct,
            "downside_q10_pct": self.downside_q10_pct,
            "selection_rank_pct": self.selection_rank_pct,
        }


def audit_oos_predictions(
    predictions: pd.DataFrame,
    *,
    lockbox_days: int = 150,
) -> dict[str, Any]:
    frame = _validate_predictions(predictions)
    dates = sorted(frame["trade_date"].unique())
    if len(dates) <= lockbox_days:
        raise ValueError(
            f"alpha audit requires more than {lockbox_days} OOS trade days; "
            f"received {len(dates)}"
        )
    lockbox_dates = set(dates[-lockbox_days:])
    development = frame.loc[~frame["trade_date"].isin(lockbox_dates)].copy()
    lockbox = frame.loc[frame["trade_date"].isin(lockbox_dates)].copy()

    policy_development = _policy_grid(development)
    selected_policies = _select_development_policies(policy_development)
    policy_confirmation = [
        {
            "thresholds": item["thresholds"],
            "development": item["metrics"],
            "lockbox": cohort_metrics(
                _policy_candidates(
                    lockbox,
                    PolicyThresholds(**_threshold_constructor_args(item["thresholds"])),
                )
            ),
        }
        for item in selected_policies
    ]

    feature_development = _feature_tail_audit(development)
    selected_features = _select_development_features(feature_development)
    feature_confirmation = [
        {
            "feature": item["feature"],
            "direction": item["direction"],
            "tail_fraction": item["tail_fraction"],
            "development": item["metrics"],
            "lockbox": cohort_metrics(
                _feature_tail_candidates(
                    lockbox,
                    item["feature"],
                    item["direction"],
                    float(item["tail_fraction"]),
                )
            ),
        }
        for item in selected_features
    ]

    return {
        "schema_version": "wp_v4_oos_alpha_audit_1",
        "semantics": (
            "Diagnostic only. Policies and feature tails are selected on the early "
            "OOS development period, then evaluated unchanged on the final "
            f"{lockbox_days}-trade-day historical lockbox. This audit cannot "
            "replace the mandatory future shadow period."
        ),
        "rows": int(len(frame)),
        "trade_days": int(len(dates)),
        "date_start": dates[0],
        "date_end": dates[-1],
        "development": {
            "date_start": str(development["trade_date"].min()),
            "date_end": str(development["trade_date"].max()),
            "trade_days": int(development["trade_date"].nunique()),
            "baseline": cohort_metrics(development),
        },
        "lockbox": {
            "date_start": str(lockbox["trade_date"].min()),
            "date_end": str(lockbox["trade_date"].max()),
            "trade_days": int(lockbox["trade_date"].nunique()),
            "baseline": cohort_metrics(lockbox),
        },
        "score_top_n": {
            "development": _score_top_n(development),
            "lockbox": _score_top_n(lockbox),
        },
        "development_policy_grid": {
            "tested": int(len(policy_development)),
            "eligible_for_confirmation": int(
                sum(_confirmation_eligible(item["metrics"]) for item in policy_development)
            ),
        },
        "policy_confirmation": policy_confirmation,
        "feature_confirmation": feature_confirmation,
    }


def cohort_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    returns = pd.to_numeric(frame.get("net_return_pct"), errors="coerce").dropna()
    total = int(len(returns))
    wins = int(returns.gt(0).sum())
    lower, upper = wilson_interval(wins, total)
    aligned = frame.loc[returns.index].copy() if total else frame.iloc[0:0].copy()
    clustered = day_clustered_intervals(aligned)
    profits = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    profit_factor = (
        profits / losses
        if losses > 0
        else (float("inf") if profits > 0 else 0.0)
    )
    return {
        "events": total,
        "trade_days": int(
            aligned.get("trade_date", pd.Series(dtype=str)).astype(str).nunique()
        ),
        "wins": wins,
        "win_rate": _finite(wins / total if total else None),
        "win_rate_wilson_lower": _finite(lower),
        "win_rate_wilson_upper": _finite(upper),
        "win_rate_day_clustered_lower": _finite(clustered.win_rate_lower),
        "mean_net_return_pct": _finite(returns.mean() if total else None),
        "mean_net_return_day_clustered_lower_pct": _finite(
            clustered.mean_return_lower_pct
        ),
        "median_net_return_pct": _finite(returns.median() if total else None),
        "net_return_q10_pct": _finite(returns.quantile(0.10) if total else None),
        "profit_factor": _finite(profit_factor),
        "total_net_return_pct": _finite(returns.sum() if total else None),
    }


def _policy_grid(frame: pd.DataFrame) -> list[dict[str, Any]]:
    grid = product(
        (0.50, 0.55, 0.60, 0.65),
        (0.45, 0.50),
        (-0.25, 0.20, 0.75),
        (-5.00, -3.50, -2.00),
        (0.980, 0.995, 0.997),
    )
    rows = []
    for values in grid:
        thresholds = PolicyThresholds(*values)
        candidates = _policy_candidates(frame, thresholds)
        rows.append(
            {
                "thresholds": thresholds.as_dict(),
                "metrics": cohort_metrics(candidates),
            }
        )
    return rows


def _policy_candidates(
    frame: pd.DataFrame,
    thresholds: PolicyThresholds,
) -> pd.DataFrame:
    mask = (
        _number(frame, "p_net_positive").ge(thresholds.probability)
        & _number(frame, "p_net_positive_lower").ge(
            thresholds.probability_lower
        )
        & _number(frame, "expected_net_return_pct").ge(
            thresholds.expected_return_pct
        )
        & _number(frame, "downside_q10_pct").ge(thresholds.downside_q10_pct)
        & _number(frame, "selection_rank_pct").ge(
            thresholds.selection_rank_pct
        )
        & _boolean(frame, "execution_eligible", default=True)
    )
    return _first_crossing(frame, mask)


def _score_top_n(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = [frame["trade_date"], frame["signal_slot"]]
    for score in SCORE_COLUMNS:
        if score not in frame:
            continue
        rank = _number(frame, score).groupby(groups, sort=False).rank(
            method="first",
            ascending=False,
        )
        for top_n in (1, 3, 5):
            rows.append(
                {
                    "score": score,
                    "top_n": top_n,
                    "metrics": cohort_metrics(
                        _first_crossing(frame, rank.le(top_n))
                    ),
                }
            )
    return rows


def _feature_tail_audit(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = [frame["trade_date"], frame["signal_slot"]]
    for feature in FEATURE_COLUMNS:
        if feature not in frame:
            continue
        values = _number(frame, feature)
        if values.notna().sum() < 1_000 or values.nunique(dropna=True) < 10:
            continue
        percentile = values.groupby(groups, sort=False).rank(
            method="average",
            pct=True,
        )
        for tail_fraction in (0.01, 0.05):
            for direction, mask in (
                ("high", percentile.ge(1.0 - tail_fraction)),
                ("low", percentile.le(tail_fraction)),
            ):
                rows.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "tail_fraction": tail_fraction,
                        "metrics": cohort_metrics(_first_crossing(frame, mask)),
                    }
                )
    return rows


def _feature_tail_candidates(
    frame: pd.DataFrame,
    feature: str,
    direction: str,
    tail_fraction: float,
) -> pd.DataFrame:
    if feature not in frame:
        return frame.iloc[0:0].copy()
    values = _number(frame, feature)
    percentile = values.groupby(
        [frame["trade_date"], frame["signal_slot"]],
        sort=False,
    ).rank(method="average", pct=True)
    mask = (
        percentile.ge(1.0 - tail_fraction)
        if direction == "high"
        else percentile.le(tail_fraction)
    )
    return _first_crossing(frame, mask)


def _select_development_policies(
    rows: list[dict[str, Any]],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    eligible = [
        item for item in rows if _confirmation_eligible(item["metrics"])
    ]
    return sorted(
        eligible,
        key=lambda item: (
            _sort_number(item["metrics"].get("win_rate_wilson_lower")),
            _sort_number(item["metrics"].get("mean_net_return_pct")),
            int(item["metrics"].get("events", 0)),
        ),
        reverse=True,
    )[:limit]


def _select_development_features(
    rows: list[dict[str, Any]],
    *,
    limit: int = 30,
) -> list[dict[str, Any]]:
    eligible = [
        item
        for item in rows
        if int(item["metrics"].get("events", 0)) >= 250
        and int(item["metrics"].get("trade_days", 0)) >= 50
    ]
    return sorted(
        eligible,
        key=lambda item: (
            _sort_number(item["metrics"].get("win_rate_wilson_lower")),
            _sort_number(item["metrics"].get("mean_net_return_pct")),
        ),
        reverse=True,
    )[:limit]


def _confirmation_eligible(metrics: dict[str, Any]) -> bool:
    return (
        int(metrics.get("events", 0)) >= 100
        and int(metrics.get("trade_days", 0)) >= 30
    )


def _first_crossing(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    selected = frame.loc[mask.fillna(False)].copy()
    if selected.empty:
        return selected
    return (
        selected.sort_values(
            ["trade_date", "ts_code", "signal_slot"],
            kind="stable",
        )
        .drop_duplicates(["trade_date", "ts_code"], keep="first")
        .reset_index(drop=True)
    )


def _validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(
        set((*IDENTITY, "net_return_pct", "target_net_positive"))
        - set(predictions.columns)
    )
    if missing:
        raise ValueError(f"OOS alpha audit is missing columns: {missing}")
    frame = predictions.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["signal_slot"] = frame["signal_slot"].astype(str)
    frame["ts_code"] = frame["ts_code"].astype(str)
    if frame.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("OOS alpha audit received duplicate prediction identities")
    return frame.sort_values(
        ["trade_date", "signal_slot", "ts_code"],
        kind="stable",
    ).reset_index(drop=True)


def _threshold_constructor_args(values: dict[str, float]) -> dict[str, float]:
    return {
        "probability": float(values["probability"]),
        "probability_lower": float(values["probability_lower"]),
        "expected_return_pct": float(values["expected_return_pct"]),
        "downside_q10_pct": float(values["downside_q10_pct"]),
        "selection_rank_pct": float(values["selection_rank_pct"]),
    }


def _number(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(
        frame.get(column, pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )


def _boolean(
    frame: pd.DataFrame,
    column: str,
    *,
    default: bool,
) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _finite(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _sort_number(value: Any) -> float:
    parsed = _finite(value)
    return float(parsed) if parsed is not None else -1e9
