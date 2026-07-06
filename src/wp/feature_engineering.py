from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import clip, numeric_series


def add_feature_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    sector_rank = numeric_series(out, ["sector_rank", "板块排名"], 50)
    sector_gt6 = numeric_series(out, ["sector_gt6_count", "板块6%以上家数"], 0)
    sector_lu = numeric_series(out, ["sector_limitup_count", "板块涨停家数"], 0)
    sector_amount_ratio = numeric_series(out, ["sector_amount_ratio", "板块成交额放大"], 1)
    volume_ratio = numeric_series(out, ["volume_ratio", "量比"], 1)
    amount_ratio = numeric_series(out, ["amount_ratio_5d", "amount_ratio_20d", "成交额放大"], 1)
    turnover = numeric_series(out, ["turnover_rate", "换手率"], 0)
    high = numeric_series(out, ["high", "最高价"], out["close"])
    low = numeric_series(out, ["low", "最低价"], out["close"])
    close = numeric_series(out, ["close", "price"], out["close"])
    pre_close = numeric_series(out, ["pre_close", "昨收"], out["pre_close"])
    ret_5d = numeric_series(out, ["ret_5d", "五日涨幅"], out["pct_chg"])
    ret_20d = numeric_series(out, ["ret_20d", "二十日涨幅"], ret_5d)
    close_position = pd.Series(np.where(high > low, (close - low) / (high - low) * 100, 50), index=out.index).fillna(50)

    out["sector_strength_score"] = clip(100 - sector_rank * 3 + sector_gt6 * 5 + sector_lu * 6 + (sector_amount_ratio - 1) * 20)
    out["stock_strength_score"] = clip(out["pct_chg"] * 7 + volume_ratio * 8 + close_position * 0.25)
    out["acceptance_score"] = clip(close_position * 0.55 + amount_ratio * 12 + turnover * 1.2 - np.maximum(volume_ratio - 4, 0) * 8)
    out["momentum_score"] = clip(out["pct_chg"] * 4 + ret_5d * 1.2 + np.maximum(ret_20d, 0) * 0.3 - np.maximum(ret_20d - 45, 0) * 1.2)
    out["capital_score"] = clip(np.log10(out["amount"].clip(lower=1)) * 9 + amount_ratio * 10 + sector_amount_ratio * 8)
    out["pattern_score"] = clip(close_position * 0.45 + np.where(close > pre_close, 20, 0) + np.where(ret_5d > 0, 10, 0))
    out["close_position"] = close_position
    return out
