from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = (
    "slot_minute",
    "ret_from_prev_close_pct",
    "ret_from_open_pct",
    "ret_5m_pct",
    "ret_10m_pct",
    "ret_20m_pct",
    "tail_range_10m_pct",
    "tail_close_position_10m",
    "slot_amount_log",
    "slot_amount_ratio_20d",
    "tail_amount_acceleration",
    "distance_to_up_limit_pct",
    "distance_to_down_limit_pct",
    "prev_1d_return_pct",
    "prev_2d_return_pct",
    "prev_5d_return_pct",
    "prev_10d_return_pct",
    "prev_20d_return_pct",
    "prev_20d_volatility_pct",
    "prev_5d_amplitude_pct",
    "prev_20d_amount_log",
    "prev_turnover_rate",
    "float_mv_log",
    "total_mv_log",
    "market_return_pct",
    "market_breadth",
    "market_tail_return_pct",
    "industry_return_pct",
    "industry_breadth",
    "up_limit_count_log",
    "down_limit_count_log",
    "relative_market_return_pct",
    "relative_industry_return_pct",
    "tail_relative_market_pct",
    "return_cs_rank",
    "tail_return_cs_rank",
    "slot_amount_ratio_cs_rank",
    "volatility_cs_rank",
    "float_mv_cs_rank",
)

FORBIDDEN_FEATURE_TOKENS = (
    "next_",
    "t1_",
    "future_",
    "target",
    "label",
    "gross_return",
    "net_return",
    "exit_price",
    "truth_",
)


def assert_feature_contract(columns: Iterable[str]) -> None:
    selected = tuple(columns)
    unknown = sorted(set(selected) - set(FEATURE_COLUMNS))
    if unknown:
        raise ValueError(f"unregistered model features: {unknown}")
    contaminated = [
        column
        for column in selected
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if contaminated:
        raise ValueError(f"future-aware features are forbidden: {contaminated}")


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    assert_feature_contract(FEATURE_COLUMNS)
    values = frame.reindex(columns=FEATURE_COLUMNS).copy()
    for column in FEATURE_COLUMNS:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def slot_to_minute(slot: pd.Series) -> pd.Series:
    parsed = slot.astype(str).str.extract(r"(?P<hour>\d{1,2}):(?P<minute>\d{2})")
    absolute = (
        pd.to_numeric(parsed["hour"], errors="coerce") * 60
        + pd.to_numeric(parsed["minute"], errors="coerce")
    )
    return absolute - (14 * 60 + 20)


def enrich_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive stable transformations without consulting any future row."""
    result = frame.copy()
    if "slot_minute" not in result and "signal_slot" in result:
        result["slot_minute"] = slot_to_minute(result["signal_slot"])

    log_sources = {
        "slot_amount_log": "slot_amount",
        "prev_20d_amount_log": "prev_20d_amount",
        "float_mv_log": "float_mv",
        "total_mv_log": "total_mv",
        "up_limit_count_log": "up_limit_count",
        "down_limit_count_log": "down_limit_count",
    }
    for output, source in log_sources.items():
        if output not in result and source in result:
            raw = pd.to_numeric(result[source], errors="coerce").clip(lower=0)
            result[output] = np.log1p(raw)

    if "slot_amount_ratio_20d" not in result:
        numerator = pd.to_numeric(result.get("slot_amount"), errors="coerce")
        denominator = pd.to_numeric(result.get("prev_20d_amount"), errors="coerce")
        result["slot_amount_ratio_20d"] = numerator / denominator.replace(0, np.nan)

    relative_features = {
        "relative_market_return_pct": (
            "ret_from_prev_close_pct",
            "market_return_pct",
        ),
        "relative_industry_return_pct": (
            "ret_from_prev_close_pct",
            "industry_return_pct",
        ),
        "tail_relative_market_pct": (
            "ret_20m_pct",
            "market_tail_return_pct",
        ),
    }
    for output, (left, right) in relative_features.items():
        if output not in result:
            result[output] = (
                pd.to_numeric(result.get(left), errors="coerce")
                - pd.to_numeric(result.get(right), errors="coerce")
            )

    rank_sources = {
        "return_cs_rank": "ret_from_prev_close_pct",
        "tail_return_cs_rank": "ret_20m_pct",
        "slot_amount_ratio_cs_rank": "slot_amount_ratio_20d",
        "volatility_cs_rank": "prev_20d_volatility_pct",
        "float_mv_cs_rank": "float_mv",
    }
    group_columns = [
        column for column in ("trade_date", "signal_slot") if column in result
    ]
    for output, source in rank_sources.items():
        if output in result:
            continue
        if source not in result:
            result[output] = np.nan
            continue
        values = pd.to_numeric(result.get(source), errors="coerce")
        if group_columns:
            result[output] = values.groupby(
                [result[column] for column in group_columns],
                sort=False,
            ).rank(method="average", pct=True)
        else:
            result[output] = values.rank(method="average", pct=True)

    return result
