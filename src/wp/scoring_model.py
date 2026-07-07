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
    source_penalty = pd.Series(0.0, index=out.index)
    if "realtime_source" in out.columns:
        fallback = out["realtime_source"].fillna("").astype(str).str.lower().str.contains("fallback")
        source_penalty = source_penalty.mask(fallback, 10.0)
    out["data_source_penalty"] = source_penalty
    out["model_confidence"] = (out["model_confidence"] - source_penalty).clip(20, 98)
    out["signal_level"] = out.apply(signal_level, axis=1)
    out["core_reason"] = out.apply(core_reason, axis=1)
    out["risk_reason"] = out.apply(risk_reason, axis=1)
    return out


def signal_level(row: pd.Series) -> str:
    p = row["p_limitup_t1"]
    risk = row["risk_penalty_score"]
    if p >= 8 and risk <= 30:
        return "S级"
    if p >= 6.5 and risk <= 45:
        return "A级"
    if p >= 5 and risk <= 65:
        return "B级"
    if p >= 3:
        return "C级"
    return "D级"


def core_reason(row: pd.Series) -> str:
    reasons = []
    if row["sector_strength_score"] >= 70:
        reasons.append("板块强")
    if row["stock_strength_score"] >= 70:
        reasons.append("个股强")
    if row["acceptance_score"] >= 65:
        reasons.append("承接好")
    if row["momentum_score"] >= 65:
        reasons.append("动量上")
    if row.get("volume_price_sync_flag", 0) == 1:
        reasons.append("量价齐")
    if row.get("high_20d_break", 0) == 1:
        reasons.append("新高")
    if row.get("platform_break_20d", 0) == 1:
        reasons.append("破平台")
    if row.get("dragon_tiger_flag", 0) == 1:
        reasons.append("龙虎榜")
    if row.get("hot_topic_flag", 0) == 1:
        reasons.append("热题材")
    if row.get("intraday_vwap_position", 0) > 1:
        reasons.append("均价上")
    if row.get("auction_strength_score", 0) >= 60:
        reasons.append("竞价强")
    if row.get("self_learning_adjustment", 0) > 1:
        reasons.append("校准+")
    return "、".join(reasons[:3]) if reasons else "基础入选"


def risk_reason(row: pd.Series) -> str:
    risks = []
    if row.get("data_source_penalty", 0) > 0:
        risks.append("分钟缺")
    if row["risk_penalty_score"] >= 65:
        risks.append("风险高")
    if row.get("close_position", 50) < 45:
        risks.append("位置差")
    if row.get("volume_ratio", 1) > 4:
        risks.append("放量急")
    if row.get("amount_ratio_5d", 1) > 5:
        risks.append("爆量")
    if row.get("high_open_low_walk_flag", 0) == 1:
        risks.append("高开低走")
    if row.get("intraday_pullback_pct", 0) > 3:
        risks.append("回落大")
    if row.get("tail_lift_flag", 0) == 1:
        risks.append("尾盘拉")
    if row.get("intraday_vwap_position", 0) < -1:
        risks.append("均价下")
    if row.get("sector_strength_score", 50) >= 70 and row.get("stock_strength_score", 50) < 45:
        risks.append("后排")
    if row.get("announcement_flag", 0) == 1:
        risks.append("公告扰动")
    if row.get("auction_pct_chg", 0) >= 5 and row.get("auction_amount_ratio", 0) < 0.01:
        risks.append("竞价弱")
    if row.get("self_learning_adjustment", 0) < -1:
        risks.append("校准-")
    if not risks:
        return "风险可控"
    return "、".join(risks[:3])
