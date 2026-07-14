from __future__ import annotations

import pandas as pd


OUTPUT_COLUMNS = [
    "rank", "ts_code", "name", "trade_date", "update_time", "price", "pct_chg",
    "amount", "pre_day_limitup", "today_limitup", "sector_name", "sector_strength_score",
    "stock_strength_score", "acceptance_score", "momentum_score", "capital_score",
    "pattern_score", "liquidity_score", "risk_penalty_score", "ranking_score",
    "tail_profit_score", "tail_profit_eligible", "tail_profit_filter_reason",
    "tail_profit_model_version", "tail_rank_pct_chg", "tail_rank_capital",
    "tail_rank_sector", "tail_rank_risk", "amount_ratio_5d", "close_position",
    "p_limitup_t1_raw", "p_limitup_t1",
    "wp_score", "feature_coverage", "model_confidence", "model_version",
    "calibration_method", "calibration_sample_count", "self_learning_adjustment",
    "signal_level", "core_reason", "risk_reason",
]


def build_ranked_pool(scored: pd.DataFrame, full_rank: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if scored.empty or full_rank.empty:
        return scored.iloc[0:0].copy()
    rank_map = full_rank[["ts_code", "rank"]].drop_duplicates("ts_code")
    pool = scored.drop(columns=["rank"], errors="ignore").merge(rank_map, on="ts_code", how="inner")
    pool["rank"] = pd.to_numeric(pool["rank"], errors="coerce")
    return pool[pool["rank"].between(1, top_n)].sort_values("rank", kind="mergesort").reset_index(drop=True)


def rank_candidates(df: pd.DataFrame, update_time: str, top_n: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        return empty, empty
    out = df.copy()
    out["update_time"] = update_time
    numeric_columns = [
        "tail_profit_score", "pct_chg", "capital_score", "sector_strength_score",
        "p_limitup_t1", "wp_score", "acceptance_score", "risk_penalty_score", "amount",
    ]
    for column in numeric_columns:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["ts_code"] = out.get("ts_code", pd.Series("", index=out.index)).fillna("").astype(str)
    if "tail_profit_eligible" in out.columns:
        out["tail_profit_eligible"] = out["tail_profit_eligible"].fillna(False).astype(bool)
        out = out.sort_values(
            [
                "tail_profit_eligible", "tail_profit_score", "pct_chg", "capital_score",
                "sector_strength_score", "risk_penalty_score", "p_limitup_t1", "amount", "ts_code",
            ],
            ascending=[False, False, True, False, False, True, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        out = out.sort_values(
            ["p_limitup_t1", "wp_score", "acceptance_score", "risk_penalty_score", "amount", "ts_code"],
            ascending=[False, False, False, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
    out["rank"] = out.index + 1
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    full = out[OUTPUT_COLUMNS].copy()
    return full.head(top_n).copy(), full
