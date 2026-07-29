from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = (
    "slot_minute",
    "gap_open_pct",
    "ret_from_prev_close_pct",
    "ret_from_open_pct",
    "ret_5m_pct",
    "ret_10m_pct",
    "ret_20m_pct",
    "bar_body_pct",
    "bar_range_pct",
    "bar_upper_wick_pct",
    "bar_lower_wick_pct",
    "tail_range_10m_pct",
    "tail_close_position_10m",
    "tail_return_from_1400_pct",
    "tail_range_since_1400_pct",
    "tail_close_position_since_1400",
    "tail_amount_weighted_price_gap_pct",
    "tail_realized_volatility_pct",
    "tail_mean_abs_return_pct",
    "tail_up_bar_share",
    "tail_down_bar_share",
    "tail_directional_efficiency",
    "tail_trend_slope_pct",
    "tail_trend_r2",
    "tail_max_drawdown_pct",
    "tail_rebound_from_low_pct",
    "tail_cumulative_amount_log",
    "tail_cumulative_amount_ratio_20d",
    "tail_latest_amount_share",
    "tail_amount_concentration",
    "slot_amount_log",
    "slot_amount_ratio_20d",
    "tail_amount_acceleration",
    "distance_to_up_limit_pct",
    "distance_to_down_limit_pct",
    "prev_day_gap_pct",
    "prev_day_intraday_return_pct",
    "prev_day_close_position",
    "prev_day_upper_wick_pct",
    "prev_day_lower_wick_pct",
    "prev_1d_return_pct",
    "prev_2d_return_pct",
    "prev_3d_return_pct",
    "prev_5d_return_pct",
    "prev_10d_return_pct",
    "prev_20d_return_pct",
    "prev_5d_positive_share",
    "prev_10d_positive_share",
    "prev_20d_positive_share",
    "prev_20d_volatility_pct",
    "prev_20d_downside_volatility_pct",
    "prev_20d_drawdown_pct",
    "prev_5d_amplitude_pct",
    "prev_amount_ratio_20d",
    "prev_5d_amount_ratio_20d",
    "prev_20d_amount_log",
    "prev_turnover_rate",
    "prev_turnover_ratio_20d",
    "prev_volume_ratio",
    "prev_pe_ttm",
    "prev_pb",
    "float_mv_log",
    "total_mv_log",
    "market_return_pct",
    "market_breadth",
    "market_breadth_above_2pct",
    "market_breadth_above_5pct",
    "market_return_dispersion_pct",
    "market_gap_pct",
    "market_tail_return_pct",
    "market_tail_breadth",
    "market_tail_dispersion_pct",
    "market_prev_5d_return_pct",
    "market_prev_20d_volatility_pct",
    "industry_return_pct",
    "industry_breadth",
    "industry_tail_return_pct",
    "industry_tail_breadth",
    "up_limit_count_log",
    "down_limit_count_log",
    "relative_market_return_pct",
    "relative_industry_return_pct",
    "tail_relative_market_pct",
    "tail_relative_industry_pct",
    "momentum_reversal_5_20",
    "volatility_adjusted_momentum_5d",
    "liquidity_acceleration",
    "tail_volume_price_confirmation",
    "distance_to_limit_asymmetry",
    "market_regime_strength",
    "industry_regime_strength",
    "relative_strength_alignment",
    "tail_trend_quality",
    "tail_reversal_pressure",
    "tail_breakout_pressure",
    "overnight_intraday_alignment",
    "risk_adjusted_tail_return",
    "market_dispersion_adjusted_return",
    "industry_dispersion_proxy",
    "return_cs_rank",
    "open_return_cs_rank",
    "tail_return_cs_rank",
    "tail_efficiency_cs_rank",
    "tail_price_gap_cs_rank",
    "slot_amount_ratio_cs_rank",
    "tail_amount_ratio_cs_rank",
    "prev_5d_return_cs_rank",
    "prev_amount_ratio_cs_rank",
    "turnover_cs_rank",
    "volatility_cs_rank",
    "float_mv_cs_rank",
    "up_limit_distance_cs_rank",
)

MARKET_FEATURE_COLUMNS = (
    "slot_minute",
    "market_return_pct",
    "market_breadth",
    "market_breadth_above_2pct",
    "market_breadth_above_5pct",
    "market_return_dispersion_pct",
    "market_gap_pct",
    "market_tail_return_pct",
    "market_tail_breadth",
    "market_tail_dispersion_pct",
    "market_prev_5d_return_pct",
    "market_prev_20d_volatility_pct",
    "up_limit_count_log",
    "down_limit_count_log",
    "market_regime_strength",
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
        values[column] = pd.to_numeric(
            values[column],
            errors="coerce",
        ).astype("float32")
    return values.replace([np.inf, -np.inf], np.nan)


def slot_to_minute(slot: pd.Series) -> pd.Series:
    parsed = slot.astype(str).str.extract(r"(?P<hour>\d{1,2}):(?P<minute>\d{2})")
    absolute = (
        pd.to_numeric(parsed["hour"], errors="coerce") * 60
        + pd.to_numeric(parsed["minute"], errors="coerce")
    )
    return absolute - (14 * 60 + 20)


def enrich_feature_frame(
    frame: pd.DataFrame,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Derive stable transformations without consulting any future row."""
    result = frame.copy() if copy else frame
    if "slot_minute" not in result and "signal_slot" in result:
        result["slot_minute"] = slot_to_minute(result["signal_slot"])

    log_sources = {
        "slot_amount_log": "slot_amount",
        "tail_cumulative_amount_log": "tail_cumulative_amount",
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
    if "tail_cumulative_amount_ratio_20d" not in result:
        numerator = pd.to_numeric(
            result.get("tail_cumulative_amount"),
            errors="coerce",
        )
        denominator = pd.to_numeric(result.get("prev_20d_amount"), errors="coerce")
        result["tail_cumulative_amount_ratio_20d"] = numerator / denominator.replace(
            0,
            np.nan,
        )

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
        "tail_relative_industry_pct": (
            "ret_20m_pct",
            "industry_tail_return_pct",
        ),
    }
    for output, (left, right) in relative_features.items():
        if output not in result:
            result[output] = (
                pd.to_numeric(result.get(left), errors="coerce")
                - pd.to_numeric(result.get(right), errors="coerce")
            )

    def numeric(column: str) -> pd.Series:
        if column not in result:
            return pd.Series(np.nan, index=result.index, dtype=float)
        return pd.to_numeric(result[column], errors="coerce")

    result["momentum_reversal_5_20"] = (
        numeric("prev_5d_return_pct") - numeric("prev_20d_return_pct")
    )
    result["volatility_adjusted_momentum_5d"] = (
        numeric("prev_5d_return_pct")
        / numeric("prev_20d_volatility_pct").abs().clip(lower=0.25)
    )
    result["liquidity_acceleration"] = (
        numeric("tail_cumulative_amount_ratio_20d")
        - numeric("prev_5d_amount_ratio_20d")
    )
    result["tail_volume_price_confirmation"] = (
        numeric("ret_20m_pct")
        * np.log1p(numeric("tail_amount_acceleration").clip(lower=0.0))
    )
    result["distance_to_limit_asymmetry"] = (
        numeric("distance_to_down_limit_pct")
        - numeric("distance_to_up_limit_pct")
    )
    result["market_regime_strength"] = (
        0.45 * numeric("market_return_pct")
        + 2.0 * (numeric("market_breadth") - 0.5)
        + 0.35 * numeric("market_tail_return_pct")
        + 1.0 * (numeric("market_tail_breadth") - 0.5)
    )
    result["industry_regime_strength"] = (
        0.55 * numeric("industry_return_pct")
        + 1.5 * (numeric("industry_breadth") - 0.5)
        + 0.45 * numeric("industry_tail_return_pct")
        + 0.75 * (numeric("industry_tail_breadth") - 0.5)
    )
    result["relative_strength_alignment"] = (
        np.sign(numeric("relative_market_return_pct"))
        * np.sign(numeric("relative_industry_return_pct"))
        * np.sqrt(
            numeric("relative_market_return_pct").abs()
            * numeric("relative_industry_return_pct").abs()
        )
    )
    result["tail_trend_quality"] = (
        numeric("tail_directional_efficiency")
        * numeric("tail_trend_r2").clip(lower=0.0, upper=1.0)
        * numeric("tail_trend_slope_pct")
    )
    result["tail_reversal_pressure"] = (
        numeric("tail_rebound_from_low_pct")
        + numeric("bar_lower_wick_pct")
        - numeric("bar_upper_wick_pct")
    )
    result["tail_breakout_pressure"] = (
        numeric("tail_close_position_since_1400") - 0.5
    ) * numeric("tail_range_since_1400_pct")
    result["overnight_intraday_alignment"] = (
        numeric("gap_open_pct") * numeric("ret_from_open_pct")
    )
    result["risk_adjusted_tail_return"] = (
        numeric("ret_20m_pct")
        / numeric("tail_realized_volatility_pct").abs().clip(lower=0.05)
    )
    result["market_dispersion_adjusted_return"] = (
        numeric("relative_market_return_pct")
        / numeric("market_return_dispersion_pct").abs().clip(lower=0.10)
    )
    result["industry_dispersion_proxy"] = (
        numeric("relative_industry_return_pct")
        - numeric("relative_market_return_pct")
    )

    rank_sources = {
        "return_cs_rank": "ret_from_prev_close_pct",
        "open_return_cs_rank": "ret_from_open_pct",
        "tail_return_cs_rank": "ret_20m_pct",
        "tail_efficiency_cs_rank": "tail_directional_efficiency",
        "tail_price_gap_cs_rank": "tail_amount_weighted_price_gap_pct",
        "slot_amount_ratio_cs_rank": "slot_amount_ratio_20d",
        "tail_amount_ratio_cs_rank": "tail_cumulative_amount_ratio_20d",
        "prev_5d_return_cs_rank": "prev_5d_return_pct",
        "prev_amount_ratio_cs_rank": "prev_amount_ratio_20d",
        "turnover_cs_rank": "prev_turnover_rate",
        "volatility_cs_rank": "prev_20d_volatility_pct",
        "float_mv_cs_rank": "float_mv",
        "up_limit_distance_cs_rank": "distance_to_up_limit_pct",
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
