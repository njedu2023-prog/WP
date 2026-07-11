from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import clip, first_existing, numeric_series


def _robust_score(series: pd.Series, higher_is_better: bool = True, neutral: float = 50.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(neutral, index=series.index, dtype="float64")
    low = float(valid.quantile(0.10))
    high = float(valid.quantile(0.90))
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-9:
        return pd.Series(neutral, index=series.index, dtype="float64")
    score = ((values - low) / (high - low) * 100).clip(0, 100)
    if not higher_is_better:
        score = 100 - score
    return score.fillna(neutral)


def _quality_range_score(
    series: pd.Series,
    low: float,
    ideal_low: float,
    ideal_high: float,
    high: float,
    neutral: float = 50.0,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    rising = (values - low) / max(ideal_low - low, 1e-9) * 100
    falling = (high - values) / max(high - ideal_high, 1e-9) * 100
    score = pd.Series(
        np.select(
            [values < low, values < ideal_low, values <= ideal_high, values < high],
            [0.0, rising, 100.0, falling],
            default=0.0,
        ),
        index=series.index,
        dtype="float64",
    )
    return score.where(values.notna(), neutral).clip(0, 100)


def _linear_score(series: pd.Series, low: float, high: float, neutral: float = 50.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    score = ((values - low) / max(high - low, 1e-9) * 100).clip(0, 100)
    return score.where(values.notna(), neutral)


def _weighted_score(parts: list[tuple[pd.Series, float]]) -> pd.Series:
    total_weight = sum(weight for _, weight in parts)
    if total_weight <= 0:
        return pd.Series(dtype="float64")
    result = sum(part * weight for part, weight in parts) / total_weight
    return clip(result)


def _feature_coverage(frame: pd.DataFrame) -> pd.Series:
    core_groups = [
        (["pct_chg", "change_pct", "涨跌幅"], False),
        (["amount", "成交额", "turnover_amount"], True),
        (["volume_ratio", "量比"], True),
        (["amount_ratio_5d", "成交额5日放大"], True),
        (["amount_ratio_20d", "成交额20日放大"], True),
        (["turnover_rate", "换手率"], True),
        (["open", "开盘价"], True),
        (["high", "最高价"], True),
        (["low", "最低价"], True),
        (["close", "price", "收盘价"], True),
        (["pre_close", "昨收"], True),
        (["ret_3d", "三日涨幅"], False),
        (["ret_5d", "五日涨幅"], False),
        (["ret_10d", "十日涨幅"], False),
        (["ret_20d", "二十日涨幅"], False),
        (["close_position"], False),
        (["intraday_pullback_pct"], False),
        (["intraday_vwap_position"], False),
        (["sector_rank", "板块排名"], True),
    ]
    available = []
    for names, must_be_positive in core_groups:
        column = first_existing(frame, names)
        if column is None:
            available.append(pd.Series(0.0, index=frame.index))
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna()
        if must_be_positive:
            valid &= values > 0
        if column in {"sector_rank", "板块排名"}:
            valid &= values.between(1, 50)
        available.append(valid.astype(float))
    return pd.concat(available, axis=1).mean(axis=1).mul(100).clip(0, 100)


def add_feature_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    sector_rank = numeric_series(out, ["sector_rank", "板块排名"], 50)
    sector_gt6 = numeric_series(out, ["sector_gt6_count", "板块8%以上家数", "板块6%以上家数"], 0)
    sector_lu = numeric_series(out, ["sector_limitup_count", "板块涨停家数"], 0)
    sector_amount_ratio = numeric_series(out, ["sector_amount_ratio", "板块成交额放大"], 1)
    sector_net_inflow = numeric_series(out, ["sector_net_inflow"], 0)
    sector_turnover = numeric_series(out, ["sector_turnover"], 0)
    sector_hot_score = numeric_series(out, ["sector_hot_score"], 0)
    volume_ratio = numeric_series(out, ["volume_ratio", "量比"], 1)
    amount_ratio_5d = numeric_series(out, ["amount_ratio_5d", "成交额5日放大"], 1)
    amount_ratio_20d = numeric_series(out, ["amount_ratio_20d", "成交额20日放大", "成交额放大"], amount_ratio_5d)
    amount_ratio = amount_ratio_5d.combine(amount_ratio_20d, max)
    turnover = numeric_series(out, ["turnover_rate", "换手率"], 0)
    turnover_5d = numeric_series(out, ["turnover_rate_5d_avg"], turnover)
    high = numeric_series(out, ["high", "最高价"], out["close"])
    low = numeric_series(out, ["low", "最低价"], out["close"])
    open_price = numeric_series(out, ["open", "开盘价"], out["close"])
    close = numeric_series(out, ["close", "price"], out["close"])
    pre_close = numeric_series(out, ["pre_close", "昨收"], out["pre_close"])
    ret_3d = numeric_series(out, ["ret_3d", "三日涨幅"], out["pct_chg"])
    ret_5d = numeric_series(out, ["ret_5d", "五日涨幅"], out["pct_chg"])
    ret_10d = numeric_series(out, ["ret_10d", "十日涨幅"], ret_5d)
    ret_20d = numeric_series(out, ["ret_20d", "二十日涨幅"], ret_5d)
    ma5_position = numeric_series(out, ["ma5_position"], 0)
    ma10_position = numeric_series(out, ["ma10_position"], 0)
    ma20_position = numeric_series(out, ["ma20_position"], 0)
    high_20d_break = numeric_series(out, ["high_20d_break"], 0)
    platform_break_20d = numeric_series(out, ["platform_break_20d"], 0)
    dragon_tiger_flag = numeric_series(out, ["dragon_tiger_flag"], 0)
    dragon_tiger_net_rate = numeric_series(out, ["dragon_tiger_net_rate"], 0)
    announcement_flag = numeric_series(out, ["announcement_flag"], 0)
    hot_topic_flag = numeric_series(out, ["hot_topic_flag"], 0)
    auction_pct_chg = numeric_series(out, ["auction_pct_chg"], 0)
    auction_amount_ratio = numeric_series(out, ["auction_amount_ratio"], 0)
    auction_strength_score = numeric_series(out, ["auction_strength_score"], 0)
    close_position = numeric_series(out, ["close_position"], np.nan)
    close_position = close_position.where(close_position.notna(), pd.Series(np.where(high > low, (close - low) / (high - low) * 100, 50), index=out.index)).fillna(50)
    pullback_pct = numeric_series(out, ["intraday_pullback_pct"], np.nan)
    pullback_pct = pullback_pct.where(pullback_pct.notna(), pd.Series(np.where(close > 0, (high / close - 1) * 100, 0), index=out.index)).fillna(0)
    open_to_close_pct = numeric_series(out, ["open_to_close_pct"], np.nan)
    open_to_close_pct = open_to_close_pct.where(open_to_close_pct.notna(), pd.Series(np.where(open_price > 0, (close / open_price - 1) * 100, 0), index=out.index)).fillna(0)
    gap_open_pct = numeric_series(out, ["gap_open_pct"], np.nan)
    gap_open_pct = gap_open_pct.where(gap_open_pct.notna(), pd.Series(np.where(pre_close > 0, (open_price / pre_close - 1) * 100, 0), index=out.index)).fillna(0)
    amplitude = numeric_series(out, ["amplitude"], np.nan)
    amplitude = amplitude.where(amplitude.notna(), pd.Series(np.where(pre_close > 0, (high - low) / pre_close * 100, 0), index=out.index)).fillna(0)
    intraday_vwap_position = numeric_series(out, ["intraday_vwap_position"], 0)
    late_pullback_pct = numeric_series(out, ["late_pullback_pct"], pullback_pct)
    late_price_change_pct = numeric_series(out, ["late_price_change_pct"], 0)
    late_volume_ratio = numeric_series(out, ["late_volume_ratio"], 1)
    tail_lift_flag = numeric_series(out, ["tail_lift_flag"], 0)
    high_open_low_walk = ((gap_open_pct >= 3) & (open_to_close_pct <= -2)) | ((gap_open_pct >= 5) & (close_position < 45))
    volume_price_sync = (amount_ratio.between(1.2, 4.5)) & (close_position >= 60) & (open_to_close_pct >= -1.5)

    pct_score = _robust_score(out["pct_chg"])
    amount_score = _robust_score(np.log10(out["amount"].clip(lower=1)))
    amount_quality = _quality_range_score(amount_ratio, 0.6, 1.2, 3.8, 7.0)
    volume_quality = _quality_range_score(volume_ratio, 0.5, 1.1, 3.5, 6.0)
    turnover_quality = _quality_range_score(turnover, 0.3, 2.0, 15.0, 28.0)
    close_quality = _robust_score(close_position)
    vwap_score = _robust_score(intraday_vwap_position)
    pullback_score = _robust_score(pullback_pct, higher_is_better=False)
    late_pullback_score = _robust_score(late_pullback_pct, higher_is_better=False)
    open_to_close_score = _linear_score(open_to_close_pct, -4.0, 6.0)
    sector_rank_score = _robust_score(sector_rank.where(sector_rank.between(1, 50)), higher_is_better=False)
    sector_amount_score = _robust_score(np.log1p(sector_amount_ratio.clip(lower=0)))
    sector_flow_score = _robust_score(sector_net_inflow)
    sector_turnover_score = _robust_score(sector_turnover)
    breakout_score = clip(50 + high_20d_break * 25 + platform_break_20d * 25)
    ret_3d_score = _quality_range_score(ret_3d, -5, 4, 18, 35)
    ret_5d_score = _quality_range_score(ret_5d, -8, 6, 28, 55)
    ret_10d_score = _quality_range_score(ret_10d, -10, 8, 38, 70)
    ret_20d_score = _quality_range_score(ret_20d, -15, 10, 45, 85)
    ma5_score = _quality_range_score(ma5_position, -8, 0, 10, 25)
    ma10_score = _quality_range_score(ma10_position, -10, 0, 15, 30)
    ma20_score = _quality_range_score(ma20_position, -12, 0, 20, 38)
    ma_structure_score = (ma5_score + ma10_score + ma20_score) / 3

    out["sector_strength_score"] = _weighted_score(
        [
            (sector_rank_score, 0.24),
            (_robust_score(sector_gt6), 0.22),
            (_robust_score(sector_lu), 0.14),
            (sector_amount_score, 0.18),
            (_robust_score(sector_hot_score), 0.12),
            (sector_flow_score, 0.07),
            (sector_turnover_score, 0.03),
        ]
    )
    out["stock_strength_score"] = _weighted_score(
        [
            (pct_score, 0.30),
            (close_quality, 0.20),
            (amount_quality, 0.15),
            (volume_quality, 0.10),
            (amount_score, 0.10),
            (breakout_score, 0.10),
            (clip(50 + hot_topic_flag * 25 + auction_strength_score * 0.25), 0.05),
        ]
    )
    out["acceptance_score"] = clip(
        _weighted_score(
            [
                (close_quality, 0.30),
                (amount_quality, 0.20),
                (turnover_quality, 0.10),
                (vwap_score, 0.10),
                (pullback_score, 0.15),
                (late_pullback_score, 0.08),
                (open_to_close_score, 0.07),
            ]
        )
        + volume_price_sync.astype(int) * 4
        - high_open_low_walk.astype(int) * 14
        - tail_lift_flag * 10
        - np.maximum(volume_ratio - 5.0, 0) * 4
    )
    out["momentum_score"] = _weighted_score(
        [
            (pct_score, 0.15),
            (ret_3d_score, 0.15),
            (ret_5d_score, 0.20),
            (ret_10d_score, 0.15),
            (ret_20d_score, 0.10),
            (ma_structure_score, 0.15),
            (breakout_score, 0.10),
        ]
    )
    out["capital_score"] = _weighted_score(
        [
            (amount_score, 0.40),
            (amount_quality, 0.25),
            (sector_amount_score, 0.15),
            (volume_quality, 0.10),
            (_robust_score(sector_net_inflow + dragon_tiger_net_rate), 0.05),
            (_robust_score(auction_amount_ratio), 0.05),
        ]
    )
    out["pattern_score"] = clip(
        _weighted_score(
            [
                (close_quality, 0.25),
                (open_to_close_score, 0.15),
                (amount_quality, 0.15),
                (ret_5d_score, 0.15),
                (ma_structure_score, 0.10),
                (breakout_score, 0.10),
                (clip(50 + volume_price_sync.astype(int) * 50), 0.10),
            ]
        )
        + hot_topic_flag * 3
        + np.where(auction_pct_chg > 0, 2, 0)
        - high_open_low_walk.astype(int) * 12
        - tail_lift_flag * 8
        - np.maximum(amplitude - 18, 0) * 0.6
    )
    out["feature_coverage"] = _feature_coverage(out)
    out["liquidity_score"] = amount_score
    out["close_position"] = close_position
    out["intraday_pullback_pct"] = pullback_pct
    out["open_to_close_pct"] = open_to_close_pct
    out["gap_open_pct"] = gap_open_pct
    out["amplitude"] = amplitude
    out["high_open_low_walk_flag"] = high_open_low_walk.astype(int)
    out["volume_price_sync_flag"] = volume_price_sync.astype(int)
    out["turnover_rate_vs_5d"] = np.where(turnover_5d > 0, turnover / turnover_5d, 1)
    out["intraday_vwap_position"] = intraday_vwap_position
    out["late_pullback_pct"] = late_pullback_pct
    out["late_price_change_pct"] = late_price_change_pct
    out["late_volume_ratio"] = late_volume_ratio
    out["tail_lift_flag"] = tail_lift_flag
    out["volume_ratio"] = volume_ratio
    out["amount_ratio_5d"] = amount_ratio_5d
    out["amount_ratio_20d"] = amount_ratio_20d
    out["turnover_rate"] = turnover
    out["ret_3d"] = ret_3d
    out["ret_5d"] = ret_5d
    out["ret_10d"] = ret_10d
    out["ret_20d"] = ret_20d
    out["ma5_position"] = ma5_position
    out["ma10_position"] = ma10_position
    out["ma20_position"] = ma20_position
    out["auction_pct_chg"] = auction_pct_chg
    out["auction_amount_ratio"] = auction_amount_ratio
    out["auction_strength_score"] = auction_strength_score
    return out
