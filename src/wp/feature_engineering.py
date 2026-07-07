from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import clip, numeric_series


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

    out["sector_strength_score"] = clip(
        100 - sector_rank * 3
        + sector_gt6 * 5
        + sector_lu * 6
        + (sector_amount_ratio - 1) * 20
        + sector_hot_score * 0.25
        + np.maximum(sector_net_inflow, 0).clip(0, 1000000000) / 100000000
        + sector_turnover.clip(0, 20) * 0.5
    )
    out["stock_strength_score"] = clip(out["pct_chg"] * 6.5 + volume_ratio * 6 + amount_ratio_5d * 8 + close_position * 0.25 + high_20d_break * 8 + hot_topic_flag * 4 + auction_strength_score * 0.08)
    out["acceptance_score"] = clip(
        close_position * 0.48
        + amount_ratio.clip(0, 4.5) * 12
        + turnover.clip(0, 18) * 1.15
        + intraday_vwap_position.clip(-5, 8) * 2.0
        + late_price_change_pct.clip(-5, 5) * 1.6
        + auction_strength_score.clip(0, 100) * 0.08
        + volume_price_sync.astype(int) * 10
        - np.maximum(volume_ratio - 4.5, 0) * 7
        - pullback_pct * 4
        - late_pullback_pct.clip(0, 8) * 2.5
        - tail_lift_flag * 12
        - high_open_low_walk.astype(int) * 18
    )
    out["momentum_score"] = clip(
        out["pct_chg"] * 3.5
        + ret_3d * 0.65
        + ret_5d * 0.9
        + ret_10d * 0.35
        + np.maximum(ret_20d, 0) * 0.22
        + ma5_position.clip(-8, 12) * 1.1
        + ma10_position.clip(-10, 16) * 0.8
        + ma20_position.clip(-12, 20) * 0.45
        + high_20d_break * 10
        + platform_break_20d * 12
        - np.maximum(ret_20d - 45, 0) * 1.2
        - np.maximum(ma20_position - 30, 0) * 1.1
    )
    out["capital_score"] = clip(
        np.log10(out["amount"].clip(lower=1)) * 8.5
        + amount_ratio.clip(0, 5) * 9
        + sector_amount_ratio.clip(0, 4) * 7
        + dragon_tiger_flag * 5
        + np.maximum(dragon_tiger_net_rate, 0).clip(0, 20) * 0.35
        + np.maximum(sector_net_inflow, 0).clip(0, 1000000000) / 120000000
        + auction_amount_ratio.clip(0, 0.25) * 80
    )
    out["pattern_score"] = clip(
        close_position * 0.40
        + np.where(close > pre_close, 16, 0)
        + np.where(ret_5d > 0, 8, 0)
        + np.where(ma5_position > 0, 5, 0)
        + np.where(ma10_position > 0, 4, 0)
        + high_20d_break * 12
        + platform_break_20d * 12
        + volume_price_sync.astype(int) * 8
        + announcement_flag * 2
        + hot_topic_flag * 5
        + np.where(auction_pct_chg > 0, 3, 0)
        - high_open_low_walk.astype(int) * 16
        - tail_lift_flag * 10
        - np.maximum(amplitude - 18, 0) * 0.8
    )
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
    return out
