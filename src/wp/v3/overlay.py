from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .contracts import V3Config
from .statistics import day_clustered_intervals, wilson_interval


def previous_trade_date_map(trade_dates: Iterable[str]) -> dict[str, str]:
    ordered = sorted({str(value) for value in trade_dates})
    return {
        current: previous
        for previous, current in zip(ordered, ordered[1:], strict=False)
    }


def build_limit_up_flags(
    daily: pd.DataFrame,
    limits: pd.DataFrame,
    *,
    tick_tolerance: float = 0.005,
) -> pd.DataFrame:
    required_daily = {"trade_date", "ts_code", "close", "high"}
    required_limits = {"trade_date", "ts_code", "up_limit"}
    missing = sorted(
        (required_daily - set(daily.columns))
        | (required_limits - set(limits.columns))
    )
    if missing:
        raise ValueError(f"cannot build prior limit flags; missing columns: {missing}")

    prices = daily.loc[:, sorted(required_daily)].copy()
    bounds = limits.loc[:, sorted(required_limits)].copy()
    for frame in (prices, bounds):
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame["ts_code"] = frame["ts_code"].astype(str)
    for column in ("close", "high"):
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    bounds["up_limit"] = pd.to_numeric(bounds["up_limit"], errors="coerce")
    merged = prices.merge(
        bounds,
        on=["trade_date", "ts_code"],
        how="inner",
        validate="one_to_one",
    )
    valid = merged["up_limit"].gt(0)
    merged["closed_up_limit"] = (
        valid
        & merged["close"].ge(merged["up_limit"] - tick_tolerance)
    )
    merged["touched_up_limit"] = (
        valid
        & merged["high"].ge(merged["up_limit"] - tick_tolerance)
    )
    return merged.loc[
        :,
        [
            "trade_date",
            "ts_code",
            "closed_up_limit",
            "touched_up_limit",
        ],
    ]


def attach_previous_limit_flags(
    predictions: pd.DataFrame,
    day_flags: pd.DataFrame,
    *,
    trade_dates: Iterable[str],
) -> pd.DataFrame:
    required = {"trade_date", "ts_code"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"cannot attach prior limit flags; missing columns: {missing}")
    result = predictions.copy()
    result["trade_date"] = result["trade_date"].astype(str)
    result["ts_code"] = result["ts_code"].astype(str)
    mapping = previous_trade_date_map(trade_dates)
    result["previous_trade_date"] = result["trade_date"].map(mapping)

    flags = day_flags.copy()
    flags["previous_trade_date"] = flags["trade_date"].astype(str)
    flags["ts_code"] = flags["ts_code"].astype(str)
    flags = flags.drop(columns="trade_date")
    flags = flags.rename(
        columns={
            "closed_up_limit": "previous_day_closed_up_limit",
            "touched_up_limit": "previous_day_touched_up_limit",
        }
    )
    result = result.merge(
        flags,
        on=["previous_trade_date", "ts_code"],
        how="left",
        validate="many_to_one",
    )
    for column in (
        "previous_day_closed_up_limit",
        "previous_day_touched_up_limit",
    ):
        result[column] = result[column].fillna(False).astype(bool)
    return result


def overlay_mask(
    frame: pd.DataFrame,
    *,
    signal_slot: str = "14:20",
    intraday_return_min_pct: float = 7.0,
    previous_limit_mode: str = "closed",
) -> pd.Series:
    if previous_limit_mode not in {"closed", "touched"}:
        raise ValueError("previous_limit_mode must be 'closed' or 'touched'")
    prior_column = (
        "previous_day_closed_up_limit"
        if previous_limit_mode == "closed"
        else "previous_day_touched_up_limit"
    )
    execution = _boolean(
        frame.get("execution_eligible", pd.Series(True, index=frame.index))
    )
    return (
        frame["signal_slot"].astype(str).eq(signal_slot)
        & pd.to_numeric(
            frame["ret_from_prev_close_pct"],
            errors="coerce",
        ).gt(intraday_return_min_pct)
        & ~_boolean(frame[prior_column])
        & execution
    ).fillna(False)


def top_n_per_day(
    frame: pd.DataFrame,
    *,
    score_column: str,
    count: int,
) -> pd.DataFrame:
    if count < 1:
        raise ValueError("count must be positive")
    if score_column not in frame:
        raise ValueError(f"missing score column: {score_column}")
    ranked = frame.copy()
    ranked["_score"] = pd.to_numeric(
        ranked[score_column],
        errors="coerce",
    )
    ranked = ranked.dropna(subset=["_score"]).sort_values(
        ["trade_date", "_score", "ts_code"],
        ascending=[True, False, True],
        kind="stable",
    )
    return (
        ranked.groupby("trade_date", sort=False)
        .head(count)
        .drop(columns="_score")
        .reset_index(drop=True)
    )


def performance_summary(
    frame: pd.DataFrame,
    config: V3Config,
    *,
    bootstrap_samples: int = 2_000,
    seed: int = 20_260_730,
) -> dict[str, Any]:
    clean = frame.copy()
    clean["net_return_pct"] = pd.to_numeric(
        clean.get("net_return_pct"),
        errors="coerce",
    )
    clean = clean.dropna(subset=["net_return_pct"])
    returns = clean["net_return_pct"]
    total = int(len(clean))
    wins = int(returns.gt(0).sum())
    lower, upper = wilson_interval(wins, total)
    clustered = day_clustered_intervals(
        clean,
        samples=bootstrap_samples,
        seed=seed,
    )
    profits = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    profit_factor = (
        profits / losses
        if losses > 0
        else (float("inf") if profits > 0 else 0.0)
    )
    entry = _boolean(
        clean.get("entry_fillable", pd.Series(False, index=clean.index))
    )
    exit_fill = _boolean(
        clean.get("exit_fillable", pd.Series(False, index=clean.index))
    )
    entered = int(entry.sum())
    exited = int((entry & exit_fill).sum())
    daily = clean.groupby("trade_date", sort=True)["net_return_pct"].mean()
    equity = (1.0 + daily / 100.0).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    stress: dict[str, dict[str, Any]] = {}
    for cost_bps in config.execution.stress_cost_bps:
        extra_cost_pct = (
            cost_bps - config.execution.baseline_all_in_cost_bps
        ) / 100.0
        stressed = returns - extra_cost_pct * entry.astype(float)
        stress[f"{int(cost_bps)}bps"] = {
            "mean_net_return_pct": _finite(stressed.mean()),
            "win_rate": _finite(stressed.gt(0).mean()),
            "positive_total_return": bool(stressed.sum() > 0),
        }

    return {
        "events": total,
        "trade_days": int(clean["trade_date"].astype(str).nunique()),
        "wins": wins,
        "win_rate": wins / total if total else 0.0,
        "win_rate_wilson_lower": lower,
        "win_rate_wilson_upper": upper,
        "win_rate_day_clustered_lower": clustered.win_rate_lower,
        "win_rate_day_clustered_upper": clustered.win_rate_upper,
        "mean_net_return_pct": _finite(returns.mean()),
        "mean_net_return_day_clustered_lower_pct": _finite(
            clustered.mean_return_lower_pct
        ),
        "mean_net_return_day_clustered_upper_pct": _finite(
            clustered.mean_return_upper_pct
        ),
        "median_net_return_pct": _finite(returns.median()),
        "net_return_q10_pct": _finite(returns.quantile(0.10)),
        "net_return_q90_pct": _finite(returns.quantile(0.90)),
        "profit_factor": _finite(profit_factor),
        "entry_fill_rate": entered / total if total else 0.0,
        "exit_fill_rate_given_entry": exited / entered if entered else 0.0,
        "round_trip_fill_rate": exited / total if total else 0.0,
        "day_equal_weight_mean_net_return_pct": _finite(daily.mean()),
        "day_equal_weight_win_rate": _finite(daily.gt(0).mean()),
        "day_equal_weight_cumulative_return_pct": _finite(
            (equity.iloc[-1] - 1.0) * 100.0 if len(equity) else np.nan
        ),
        "maximum_day_equal_weight_drawdown_pct": _finite(
            drawdown.min() * 100.0 if len(drawdown) else np.nan
        ),
        "stress": stress,
    }


def _boolean(values: pd.Series | bool) -> pd.Series:
    if isinstance(values, bool):
        return pd.Series(dtype=bool)
    if values.dtype == bool:
        return values.fillna(False)
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y", "qualified", "pass"})
    )


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
