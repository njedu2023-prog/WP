from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import clip, numeric_series
from .utils import load_yaml


DEFAULT_RULES = {
    "high_risk_threshold": 65,
    "medium_risk_threshold": 45,
    "late_pullback_pct": 3.0,
    "high_position_ret_20d": 35.0,
    "excessive_amount_ratio": 4.0,
    "excessive_volume_ratio": 4.0,
    "min_liquidity_amount": 100000000,
    "min_stock_age_days": 10,
    "excessive_ma20_position": 30.0,
}


def risk_rules() -> dict:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    configured = load_yaml(root / "config" / "risk_rules.yml", DEFAULT_RULES)
    rules = DEFAULT_RULES.copy()
    rules.update({key: float(value) for key, value in configured.items() if key in rules})
    return rules


def add_risk_penalty(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    rules = risk_rules()
    close_position = numeric_series(out, ["close_position"], 50)
    volume_ratio = numeric_series(out, ["volume_ratio", "量比"], 1)
    amount_ratio_5d = numeric_series(out, ["amount_ratio_5d", "成交额5日放大"], 1)
    ret_20d = numeric_series(out, ["ret_20d", "二十日涨幅"], out["pct_chg"])
    sector_rank = numeric_series(out, ["sector_rank", "板块排名"], 50)
    amount = numeric_series(out, ["amount", "成交额"], 0)
    pullback_pct = numeric_series(out, ["intraday_pullback_pct"], 0)
    open_to_close_pct = numeric_series(out, ["open_to_close_pct"], 0)
    gap_open_pct = numeric_series(out, ["gap_open_pct"], 0)
    amplitude = numeric_series(out, ["amplitude"], 0)
    sector_strength = numeric_series(out, ["sector_strength_score"], 50)
    stock_strength = numeric_series(out, ["stock_strength_score"], 50)
    late_pullback_pct = numeric_series(out, ["late_pullback_pct"], pullback_pct)
    late_volume_ratio = numeric_series(out, ["late_volume_ratio"], 1)
    tail_lift_flag = numeric_series(out, ["tail_lift_flag"], 0)
    intraday_vwap_position = numeric_series(out, ["intraday_vwap_position"], 0)
    ma20_position = numeric_series(out, ["ma20_position"], 0)
    announcement_flag = numeric_series(out, ["announcement_flag"], 0)
    stock_age_days = numeric_series(out, ["stock_age_days"], 9999)
    suspended_flag = numeric_series(out, ["suspended_flag"], 0)
    delist_flag = numeric_series(out, ["delist_flag"], 0)
    data_quality_flag = numeric_series(out, ["data_quality_flag"], 0)
    auction_pct_chg = numeric_series(out, ["auction_pct_chg"], 0)
    auction_amount_ratio = numeric_series(out, ["auction_amount_ratio"], 0)
    pullback_risk = np.maximum(70 - close_position, 0) * 0.45
    high_position_risk = np.maximum(ret_20d - rules["high_position_ret_20d"], 0) * 0.8
    volume_risk = np.maximum(volume_ratio - rules["excessive_volume_ratio"], 0) * 8 + np.maximum(amount_ratio_5d - rules["excessive_amount_ratio"], 0) * 8
    sector_rank_known = sector_rank.between(1, 50)
    rear_sector_risk = np.where(sector_rank_known, np.maximum(sector_rank - 20, 0) * 0.7, 0.0)
    liquidity_risk = np.where(amount < rules["min_liquidity_amount"], 25, 0)
    high_open_low_walk_risk = np.where(((gap_open_pct >= 3) & (open_to_close_pct <= -2)) | ((gap_open_pct >= 5) & (close_position < 45)), 18, 0)
    intraday_pullback_risk = np.maximum(pullback_pct - rules["late_pullback_pct"], 0) * 6
    late_attack_risk = np.where((tail_lift_flag == 1) | ((late_volume_ratio >= 2.0) & (late_pullback_pct <= 1.0) & (close_position >= 82)), 14, 0)
    vwap_risk = np.maximum(-intraday_vwap_position - 1.0, 0) * 5
    wide_amplitude_risk = np.maximum(amplitude - 18, 0) * 1.2
    trapped_pressure_risk = np.maximum(ret_20d - 55, 0) * 0.9 + np.maximum(ma20_position - rules["excessive_ma20_position"], 0) * 1.3
    sector_lag_risk = np.where((sector_strength >= 70) & (stock_strength < 45), 18, 0)
    announcement_risk = np.where(announcement_flag == 1, 8, 0)
    auction_risk = np.where((auction_pct_chg >= 5) & (auction_amount_ratio < 0.01), 8, 0)
    hard_filter_risk = (
        np.where(stock_age_days < rules["min_stock_age_days"], 30, 0)
        + np.where(suspended_flag == 1, 80, 0)
        + np.where(delist_flag == 1, 80, 0)
        + np.where(data_quality_flag == 1, 35, 0)
    )
    out["risk_rear_sector"] = rear_sector_risk
    out["risk_liquidity"] = liquidity_risk
    out["risk_price_structure"] = (
        pullback_risk
        + high_position_risk
        + high_open_low_walk_risk
        + intraday_pullback_risk
        + vwap_risk
        + wide_amplitude_risk
        + trapped_pressure_risk
    )
    out["risk_volume"] = volume_risk + late_attack_risk
    out["risk_data"] = hard_filter_risk
    out["risk_penalty_score"] = clip(
        pullback_risk
        + high_position_risk
        + volume_risk
        + rear_sector_risk
        + liquidity_risk
        + high_open_low_walk_risk
        + intraday_pullback_risk
        + late_attack_risk
        + vwap_risk
        + wide_amplitude_risk
        + trapped_pressure_risk
        + sector_lag_risk
        + announcement_risk
        + auction_risk
        + hard_filter_risk
    )
    return out
