from __future__ import annotations

import pandas as pd

from .limitup_probability import add_limitup_probability
from .risk_penalty import add_risk_penalty


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = add_limitup_probability(add_risk_penalty(df))
    out["wp_score"] = (
        0.30 * out["sector_strength_score"]
        + 0.25 * out["stock_strength_score"]
        + 0.20 * out["acceptance_score"]
        + 0.10 * out["momentum_score"]
        + 0.10 * out["capital_score"]
        + 0.05 * out["pattern_score"]
        - 0.25 * out["risk_penalty_score"]
    ).clip(0, 100)
    out["model_confidence"] = (100 - out["risk_penalty_score"] * 0.45).clip(20, 95)
    out["signal_level"] = out.apply(signal_level, axis=1)
    out["core_reason"] = out.apply(core_reason, axis=1)
    out["risk_reason"] = out.apply(risk_reason, axis=1)
    return out


def signal_level(row: pd.Series) -> str:
    p = row["p_limitup_t1"]
    risk = row["risk_penalty_score"]
    if p >= 45 and risk <= 30:
        return "S级"
    if p >= 35 and risk <= 40:
        return "A级"
    if p >= 25 and risk <= 50:
        return "B级"
    if p >= 15:
        return "C级"
    return "D级"


def core_reason(row: pd.Series) -> str:
    reasons = []
    if row["sector_strength_score"] >= 70:
        reasons.append("所属板块强度靠前")
    if row["stock_strength_score"] >= 70:
        reasons.append("个股涨幅和量价强度较高")
    if row["acceptance_score"] >= 65:
        reasons.append("收盘位置较好且资金承接较强")
    if row["momentum_score"] >= 65:
        reasons.append("短线动量保持向上")
    return "，".join(reasons) + "。" if reasons else "满足涨幅和非涨停过滤条件，综合评分进入候选排序。"


def risk_reason(row: pd.Series) -> str:
    risks = []
    if row["risk_penalty_score"] >= 65:
        risks.append("综合风险偏高")
    if row.get("close_position", 50) < 45:
        risks.append("收盘位置不佳，存在冲高回落风险")
    if row.get("volume_ratio", 1) > 4:
        risks.append("量能放大过快")
    if not risks:
        return "当前主要风险可控，但次日仍需观察板块延续性。"
    return "，".join(risks) + "。"
