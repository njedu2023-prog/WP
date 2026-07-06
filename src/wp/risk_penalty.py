from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import clip, numeric_series


def add_risk_penalty(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    close_position = numeric_series(out, ["close_position"], 50)
    volume_ratio = numeric_series(out, ["volume_ratio", "量比"], 1)
    ret_20d = numeric_series(out, ["ret_20d", "二十日涨幅"], out["pct_chg"])
    sector_rank = numeric_series(out, ["sector_rank", "板块排名"], 50)
    amount = numeric_series(out, ["amount", "成交额"], 0)
    pullback_risk = np.maximum(70 - close_position, 0) * 0.45
    high_position_risk = np.maximum(ret_20d - 35, 0) * 0.8
    volume_risk = np.maximum(volume_ratio - 4, 0) * 10
    rear_sector_risk = np.maximum(sector_rank - 20, 0) * 0.7
    liquidity_risk = np.where(amount < 100000000, 25, 0)
    out["risk_penalty_score"] = clip(pullback_risk + high_position_risk + volume_risk + rear_sector_risk + liquidity_risk)
    return out
