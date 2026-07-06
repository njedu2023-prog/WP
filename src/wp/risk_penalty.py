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
    pullback_risk = np.maximum(70 - close_position, 0) * 0.45
    high_position_risk = np.maximum(ret_20d - rules["high_position_ret_20d"], 0) * 0.8
    volume_risk = np.maximum(volume_ratio - rules["excessive_volume_ratio"], 0) * 8 + np.maximum(amount_ratio_5d - rules["excessive_amount_ratio"], 0) * 8
    rear_sector_risk = np.maximum(sector_rank - 20, 0) * 0.7
    liquidity_risk = np.where(amount < rules["min_liquidity_amount"], 25, 0)
    high_open_low_walk_risk = np.where(((gap_open_pct >= 3) & (open_to_close_pct <= -2)) | ((gap_open_pct >= 5) & (close_position < 45)), 18, 0)
    intraday_pullback_risk = np.maximum(pullback_pct - rules["late_pullback_pct"], 0) * 6
    wide_amplitude_risk = np.maximum(amplitude - 18, 0) * 1.2
    sector_lag_risk = np.where((sector_strength >= 70) & (stock_strength < 45), 18, 0)
    out["risk_penalty_score"] = clip(
        pullback_risk
        + high_position_risk
        + volume_risk
        + rear_sector_risk
        + liquidity_risk
        + high_open_low_walk_risk
        + intraday_pullback_risk
        + wide_amplitude_risk
        + sector_lag_risk
    )
    return out
