from __future__ import annotations

import pandas as pd


OUTPUT_COLUMNS = [
    "rank", "ts_code", "name", "trade_date", "update_time", "price", "pct_chg",
    "pre_day_limitup", "today_limitup", "sector_name", "sector_strength_score",
    "stock_strength_score", "acceptance_score", "momentum_score", "capital_score",
    "risk_penalty_score", "p_limitup_t1", "wp_score", "model_confidence",
    "signal_level", "core_reason", "risk_reason",
]


def rank_candidates(df: pd.DataFrame, update_time: str, top_n: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        return empty, empty
    out = df.copy()
    out["update_time"] = update_time
    out = out.sort_values(
        ["p_limitup_t1", "wp_score", "sector_strength_score", "acceptance_score", "amount"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    out["rank"] = out.index + 1
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    full = out[OUTPUT_COLUMNS].copy()
    return full.head(top_n).copy(), full
