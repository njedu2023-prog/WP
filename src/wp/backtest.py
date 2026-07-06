from __future__ import annotations

import pandas as pd


def build_label_frame(today_rank: pd.DataFrame, next_day: pd.DataFrame) -> pd.DataFrame:
    if today_rank.empty:
        return pd.DataFrame()
    merged = today_rank.merge(next_day, on="ts_code", how="left", suffixes=("", "_next"))
    if "next_day_limitup_price" in merged and "next_day_high" in merged:
        merged["label_t1_limitup"] = (merged["next_day_high"] >= merged["next_day_limitup_price"] * 0.999).astype(int)
    else:
        merged["label_t1_limitup"] = 0
    return merged
