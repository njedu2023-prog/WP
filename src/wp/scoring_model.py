from __future__ import annotations

from pathlib import Path

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
MODEL_VERSION = "wp_rule_v2_1"


def model_weights() -> dict:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    configured = load_yaml(root / "config" / "model_weights.yml", DEFAULT_WEIGHTS)
    weights = DEFAULT_WEIGHTS.copy()
    weights.update({key: float(value) for key, value in configured.items() if key in weights})
    return weights


def add_scores(df: pd.DataFrame, calibration_enabled: bool = True) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = add_risk_penalty(df)
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
    out["ranking_score"] = (out["capital_score"] * 0.70 + out["wp_score"] * 0.30).clip(0, 100)
    out = add_limitup_probability(out)
    fallback = pd.Series(False, index=out.index)
    if "realtime_source" in out.columns:
        fallback = out["realtime_source"].fillna("").astype(str).str.lower().str.contains("fallback")
    out["data_source_probability_factor"] = fallback.map({True: 0.80, False: 1.00}).astype(float)
    out["p_limitup_t1"] = out["p_limitup_t1"] * out["data_source_probability_factor"]
    root = Path(__file__).resolve().parents[2]
    out["model_version"] = MODEL_VERSION
    as_of_date = None
    if "trade_date" in out.columns:
        dates = out["trade_date"].fillna("").astype(str).str.replace("-", "", regex=False)
        valid_dates = dates[dates.str.fullmatch(r"\d{8}")]
        as_of_date = str(valid_dates.max()) if not valid_dates.empty else None
    if calibration_enabled:
        out = apply_statistical_calibration(
            out,
            root,
            model_version=MODEL_VERSION,
            before_date=as_of_date,
        )
    else:
        out["p_limitup_t1_raw"] = out["p_limitup_t1"]
        out["calibration_sample_count"] = 0
        out["self_learning_adjustment"] = 0.0
        out["calibration_method"] = "disabled"

    coverage = pd.to_numeric(out.get("feature_coverage", pd.Series(70.0, index=out.index)), errors="coerce").fillna(70).clip(0, 100)
    out["model_confidence"] = (45 + coverage * 0.40 - out["risk_penalty_score"] * 0.25).clip(20, 92)
    if "calibration_sample_count" in out.columns:
        support = pd.to_numeric(out["calibration_sample_count"], errors="coerce").fillna(0).clip(0, 500)
        out["model_confidence"] = (out["model_confidence"] + support / 500 * 6).clip(20, 96)
    source_penalty = pd.Series(0.0, index=out.index)
    source_penalty = source_penalty.mask(fallback, 10.0)
    out["data_source_penalty"] = source_penalty
    out["model_confidence"] = (out["model_confidence"] - source_penalty).clip(20, 98)
    out["signal_level"] = out.apply(signal_level, axis=1)
    out["core_reason"] = out.apply(core_reason, axis=1)
    out["risk_reason"] = out.apply(risk_reason, axis=1)
    return out


def signal_level(row: pd.Series) -> str:
    p = _row_float(row, "p_limitup_t1")
    risk = _row_float(row, "risk_penalty_score", 100)
    if p >= 5 and risk <= 30:
        return "S级"
    if p >= 4 and risk <= 45:
        return "A级"
    if p >= 3 and risk <= 60:
        return "B级"
    if p >= 2:
        return "C级"
    return "D级"


def _row_float(row: pd.Series, name: str, default: float = 0.0) -> float:
    value = pd.to_numeric(pd.Series([row.get(name, default)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else float(default)


def core_reason(row: pd.Series) -> str:
    reasons = []
    if _row_float(row, "sector_strength_score") >= 70:
        reasons.append("板块强")
    if _row_float(row, "stock_strength_score") >= 70:
        reasons.append("个股强")
    if _row_float(row, "acceptance_score") >= 65:
        reasons.append("承接好")
    if _row_float(row, "momentum_score") >= 65:
        reasons.append("动量上")
    if _row_float(row, "volume_price_sync_flag") == 1:
        reasons.append("量价齐")
    if _row_float(row, "high_20d_break") == 1:
        reasons.append("新高")
    if _row_float(row, "platform_break_20d") == 1:
        reasons.append("破平台")
    if _row_float(row, "dragon_tiger_flag") == 1:
        reasons.append("龙虎榜")
    if _row_float(row, "hot_topic_flag") == 1:
        reasons.append("热题材")
    if _row_float(row, "intraday_vwap_position") > 1:
        reasons.append("均价上")
    if _row_float(row, "auction_strength_score") >= 60:
        reasons.append("竞价强")
    if _row_float(row, "self_learning_adjustment") > 1:
        reasons.append("校准+")
    return "、".join(reasons[:3]) if reasons else "基础入选"


def risk_reason(row: pd.Series) -> str:
    risks = []
    if _row_float(row, "data_source_penalty") > 0:
        risks.append("分钟缺")
    if _row_float(row, "risk_penalty_score") >= 65:
        risks.append("风险高")
    if _row_float(row, "close_position", 50) < 45:
        risks.append("位置差")
    if _row_float(row, "volume_ratio", 1) > 4:
        risks.append("放量急")
    if _row_float(row, "amount_ratio_5d", 1) > 5:
        risks.append("爆量")
    if _row_float(row, "high_open_low_walk_flag") == 1:
        risks.append("高开低走")
    if _row_float(row, "intraday_pullback_pct") > 3:
        risks.append("回落大")
    if _row_float(row, "tail_lift_flag") == 1:
        risks.append("尾盘拉")
    if _row_float(row, "intraday_vwap_position") < -1:
        risks.append("均价下")
    if _row_float(row, "sector_strength_score", 50) >= 70 and _row_float(row, "stock_strength_score", 50) < 45:
        risks.append("后排")
    if _row_float(row, "announcement_flag") == 1:
        risks.append("公告扰动")
    if _row_float(row, "auction_pct_chg") >= 5 and _row_float(row, "auction_amount_ratio") < 0.01:
        risks.append("竞价弱")
    if _row_float(row, "self_learning_adjustment") < -1:
        risks.append("校准-")
    if not risks:
        return "风险可控"
    return "、".join(risks[:3])
