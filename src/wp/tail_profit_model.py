from __future__ import annotations

import numpy as np
import pandas as pd


TAIL_PROFIT_MODEL_VERSION = "tail_profit_v1"
TAIL_PROFIT_WEIGHTS = {
    "anti_chase": 0.50,
    "capital": 0.25,
    "sector": 0.20,
    "low_risk": 0.05,
}
DEFAULT_TAIL_PROFIT_CONFIG = {
    "tail_profit_min_pct_chg": 8.0,
    "tail_profit_max_pct_chg": 12.0,
    "tail_profit_max_risk_score": 45.0,
    "tail_profit_min_close_position": 50.0,
    "tail_profit_max_amount_ratio_5d": 2.5,
}


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce")


def _flag(frame: pd.DataFrame, name: str) -> pd.Series:
    return _numeric(frame, name).fillna(0).astype(int)


def _config(config: dict | None) -> dict[str, float]:
    values = DEFAULT_TAIL_PROFIT_CONFIG.copy()
    values.update({key: value for key, value in (config or {}).items() if key in values})
    return {key: float(value) for key, value in values.items()}


def _rank_within_trade_date(frame: pd.DataFrame, column: str) -> pd.Series:
    if "trade_date" in frame.columns:
        groups = frame["trade_date"].fillna("").astype(str)
    else:
        groups = pd.Series("", index=frame.index, dtype="object")
    return frame.groupby(groups, dropna=False)[column].rank(method="average", pct=True)


def _filter_reason(row: pd.Series, cfg: dict[str, float]) -> str:
    reasons: list[str] = []
    required = (
        "pct_chg",
        "capital_score",
        "sector_strength_score",
        "risk_penalty_score",
        "close_position",
        "amount_ratio_5d",
    )
    if any(pd.isna(row.get(name)) for name in required):
        return "关键字段缺失"
    pct_chg = float(row["pct_chg"])
    if pct_chg <= cfg["tail_profit_min_pct_chg"]:
        reasons.append("涨幅未过8%")
    if pct_chg > cfg["tail_profit_max_pct_chg"]:
        reasons.append("涨幅过热")
    if float(row["risk_penalty_score"]) > cfg["tail_profit_max_risk_score"]:
        reasons.append("风险偏高")
    if float(row["close_position"]) < cfg["tail_profit_min_close_position"]:
        reasons.append("收盘位置低")
    amount_ratio = float(row["amount_ratio_5d"])
    if amount_ratio <= 0:
        reasons.append("量能数据无效")
    elif amount_ratio > cfg["tail_profit_max_amount_ratio_5d"]:
        reasons.append("放量过度")
    if int(row.get("pre_day_limitup", 0) or 0) == 1:
        reasons.append("昨日涨停")
    if int(row.get("today_limitup", 0) or 0) == 1:
        reasons.append("今日涨停")
    return "，".join(reasons)


def add_tail_profit_scores(frame: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Add a causal 14:35 tail-return score to the full >8% candidate pool."""
    out = frame.copy()
    cfg = _config(config)
    numeric_columns = (
        "pct_chg",
        "capital_score",
        "sector_strength_score",
        "risk_penalty_score",
        "close_position",
        "amount_ratio_5d",
    )
    for column in numeric_columns:
        out[column] = _numeric(out, column)
    out["pre_day_limitup"] = _flag(out, "pre_day_limitup")
    out["today_limitup"] = _flag(out, "today_limitup")

    if out.empty:
        out["tail_profit_score"] = pd.Series(dtype="float64")
        out["tail_profit_eligible"] = pd.Series(dtype="bool")
        out["tail_profit_filter_reason"] = pd.Series(dtype="object")
        out["tail_profit_model_version"] = TAIL_PROFIT_MODEL_VERSION
        return out

    rank_sources = {
        "tail_rank_pct_chg": "pct_chg",
        "tail_rank_capital": "capital_score",
        "tail_rank_sector": "sector_strength_score",
        "tail_rank_risk": "risk_penalty_score",
    }
    for output_column, source_column in rank_sources.items():
        out[output_column] = _rank_within_trade_date(out, source_column)

    score = (
        TAIL_PROFIT_WEIGHTS["anti_chase"] * (1.0 - out["tail_rank_pct_chg"])
        + TAIL_PROFIT_WEIGHTS["capital"] * out["tail_rank_capital"]
        + TAIL_PROFIT_WEIGHTS["sector"] * out["tail_rank_sector"]
        + TAIL_PROFIT_WEIGHTS["low_risk"] * (1.0 - out["tail_rank_risk"])
    )
    out["tail_profit_score"] = (score.clip(0, 1) * 100).round(6).fillna(0.0)
    out["tail_profit_filter_reason"] = out.apply(_filter_reason, axis=1, cfg=cfg)
    out["tail_profit_eligible"] = out["tail_profit_filter_reason"].eq("")
    out["tail_profit_model_version"] = TAIL_PROFIT_MODEL_VERSION
    return out
