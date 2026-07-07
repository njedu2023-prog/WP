from __future__ import annotations

import pandas as pd

from .calibration import apply_statistical_calibration
from .limitup_probability import add_limitup_probability
from .risk_penalty import add_risk_penalty
from .utils import load_yaml


DEFAULT_WEIGHTS = {
    "sector_strength_score": 0.30,
    "stock_strength_score": 0.25,
    "acceptance_score": 0.20,
    "momentum_score": 0.10,
    "capital_score": 0.10,
    "pattern_score": 0.05,
    "risk_penalty_score": -0.25,
}


def model_weights() -> dict:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    configured = load_yaml(root / "config" / "model_weights.yml", DEFAULT_WEIGHTS)
    weights = DEFAULT_WEIGHTS.copy()
    weights.update({key: float(value) for key, value in configured.items() if key in weights})
    return weights


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = add_limitup_probability(add_risk_penalty(df))
    weights = model_weights()
    out["wp_score"] = (
        weights["sector_strength_score"] * out["sector_strength_score"]
        + weights["stock_strength_score"] * out["stock_strength_score"]
        + weights["acceptance_score"] * out["acceptance_score"]
        + weights["momentum_score"] * out["momentum_score"]
        + weights["capital_score"] * out["capital_score"]
        + weights["pattern_score"] * out["pattern_score"]
        + weights["risk_penalty_score"] * out["risk_penalty_score"]
    ).clip(0, 100)
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    out = apply_statistical_calibration(out, root)
    out["model_confidence"] = (100 - out["risk_penalty_score"] * 0.45).clip(20, 95)
    if "calibration_sample_count" in out.columns:
        out["model_confidence"] = (out["model_confidence"] + (out["calibration_sample_count"].clip(0, 300) / 300) * 5).clip(20, 98)
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
    if row.get("volume_price_sync_flag", 0) == 1:
        reasons.append("量价结构同步")
    if row.get("high_20d_break", 0) == 1:
        reasons.append("突破阶段高点")
    if row.get("platform_break_20d", 0) == 1:
        reasons.append("突破近20日平台")
    if row.get("dragon_tiger_flag", 0) == 1:
        reasons.append("出现龙虎榜资金线索")
    if row.get("hot_topic_flag", 0) == 1:
        reasons.append("属于当日热门题材")
    if row.get("intraday_vwap_position", 0) > 1:
        reasons.append("收盘强于日内均衡价格")
    if row.get("self_learning_adjustment", 0) > 1:
        reasons.append("历史校准对概率有正向修正")
    return "，".join(reasons) + "。" if reasons else "满足涨幅和非涨停过滤条件，综合评分进入候选排序。"


def risk_reason(row: pd.Series) -> str:
    risks = []
    if row["risk_penalty_score"] >= 65:
        risks.append("综合风险偏高")
    if row.get("close_position", 50) < 45:
        risks.append("收盘位置不佳，存在冲高回落风险")
    if row.get("volume_ratio", 1) > 4:
        risks.append("量能放大过快")
    if row.get("amount_ratio_5d", 1) > 5:
        risks.append("成交额放大过度，存在爆量滞涨风险")
    if row.get("high_open_low_walk_flag", 0) == 1:
        risks.append("高开低走结构偏弱")
    if row.get("intraday_pullback_pct", 0) > 3:
        risks.append("冲高回落幅度偏大")
    if row.get("tail_lift_flag", 0) == 1:
        risks.append("尾盘放量拉升，需防次日承接不足")
    if row.get("intraday_vwap_position", 0) < -1:
        risks.append("收盘弱于日内均衡价格")
    if row.get("sector_strength_score", 50) >= 70 and row.get("stock_strength_score", 50) < 45:
        risks.append("板块强但个股相对掉队")
    if row.get("announcement_flag", 0) == 1:
        risks.append("存在公告或异动信息扰动")
    if row.get("self_learning_adjustment", 0) < -1:
        risks.append("历史校准对概率有负向修正")
    if not risks:
        return "当前主要风险可控，但次日仍需观察板块延续性。"
    return "，".join(risks) + "。"
