from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


DEFAULT_BUY_CONFIG = {
    "buy_max_count": 5,
    "buy_target_total_position_pct": 50.0,
    "buy_max_single_position_pct": 14.0,
    "buy_min_single_position_pct": 6.0,
    "buy_max_sector_positions": 2,
    "buy_max_risk_score": 65.0,
    "buy_min_probability": 35.0,
    "buy_min_wp_score": 45.0,
    "buy_min_acceptance_score": 50.0,
    "buy_min_model_confidence": 55.0,
    "buy_min_close_position": 55.0,
    "buy_max_intraday_pullback_pct": 8.0,
}


BUY_COLUMNS = [
    "buy_rank",
    "portfolio_group",
    "suggest_position_pct",
    "ts_code",
    "name",
    "rank",
    "price",
    "pct_chg",
    "sector_name",
    "p_limitup_t1",
    "wp_score",
    "decision_score",
    "model_confidence",
    "risk_penalty_score",
    "acceptance_score",
    "momentum_score",
    "close_position",
    "intraday_pullback_pct",
    "intraday_vwap_position",
    "confirm_before_buy",
    "reject_if",
    "buy_reason",
]


DECISION_COLUMNS = BUY_COLUMNS + ["buy_flag", "decision_action", "skip_reason"]


@dataclass
class BuyDecisionResult:
    buy_plan: pd.DataFrame
    decision_table: pd.DataFrame
    summary: dict


def _num(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce").fillna(default)


def _txt(frame: pd.DataFrame, name: str, default: str = "") -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="object")
    return frame[name].fillna(default).astype(str)


def _clip(series: pd.Series, low: float = 0.0, high: float = 100.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(low, high)


def _sort_for_buy(frame: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["p_limitup_t1", "wp_score", "decision_score", "acceptance_score", "amount"]
    for col in sort_cols:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame.sort_values(sort_cols, ascending=[False, False, False, False, False]).reset_index(drop=True)


def _reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if float(row.get("p_limitup_t1", 0) or 0) >= 70:
        reasons.append("次日涨停概率靠前")
    if float(row.get("sector_strength_score", 0) or 0) >= 70:
        reasons.append("板块强度靠前")
    if float(row.get("acceptance_score", 0) or 0) >= 65:
        reasons.append("收盘承接较好")
    if float(row.get("close_position", 50) or 50) >= 75:
        reasons.append("收盘位置靠近高位")
    if float(row.get("intraday_vwap_position", 0) or 0) > 0:
        reasons.append("收盘强于日内均衡价格")
    if float(row.get("risk_penalty_score", 0) or 0) <= 45:
        reasons.append("风险惩罚较低")
    return "，".join(reasons) or "综合评分进入买入观察池"


def _skip_reason(row: pd.Series, cfg: dict) -> str:
    reasons: list[str] = []
    if float(row.get("risk_penalty_score", 0) or 0) > cfg["buy_max_risk_score"]:
        reasons.append("风险分过高")
    if float(row.get("p_limitup_t1", 0) or 0) < cfg["buy_min_probability"]:
        reasons.append("次日涨停概率不足")
    if float(row.get("wp_score", 0) or 0) < cfg["buy_min_wp_score"]:
        reasons.append("综合评分不足")
    if float(row.get("acceptance_score", 0) or 0) < cfg["buy_min_acceptance_score"]:
        reasons.append("承接分不足")
    if float(row.get("model_confidence", 0) or 0) < cfg["buy_min_model_confidence"]:
        reasons.append("模型置信度不足")
    if float(row.get("close_position", 50) or 50) < cfg["buy_min_close_position"]:
        reasons.append("收盘位置偏低")
    if float(row.get("intraday_pullback_pct", 0) or 0) > cfg["buy_max_intraday_pullback_pct"]:
        reasons.append("日内回撤偏大")
    if int(float(row.get("today_limitup", 0) or 0)) == 1:
        reasons.append("今日已涨停")
    if int(float(row.get("pre_day_limitup", 0) or 0)) == 1:
        reasons.append("昨日涨停")
    return "，".join(reasons)


def _confirm_text() -> str:
    return "14:50前仍满足涨幅>8%、未涨停、尾盘不明显回落、成交承接不塌、同板块强势不退潮"


def _reject_text() -> str:
    return "跌破8%、冲高回落扩大、收盘位置明显下降、风险分升高、同板块快速转弱或临近涨停无法成交"


def _portfolio_total(selected: pd.DataFrame, cfg: dict) -> float:
    if selected.empty:
        return 0.0
    avg_score = float(selected["decision_score"].mean())
    avg_risk = float(selected["risk_penalty_score"].mean())
    target = float(cfg["buy_target_total_position_pct"])
    if avg_score >= 70 and avg_risk <= 45 and len(selected) >= 4:
        return target
    if avg_score >= 60 and avg_risk <= 55:
        return target * 0.8
    return target * 0.55


def _assign_positions(selected: pd.DataFrame, cfg: dict) -> pd.Series:
    if selected.empty:
        return pd.Series(dtype="float64")
    total = _portfolio_total(selected, cfg)
    scores = _clip(selected["decision_score"], 1, 100)
    weights = scores / scores.sum() if float(scores.sum()) > 0 else pd.Series(1 / len(selected), index=selected.index)
    raw = weights * total
    raw = raw.clip(float(cfg["buy_min_single_position_pct"]), float(cfg["buy_max_single_position_pct"]))
    if float(raw.sum()) > total and float(raw.sum()) > 0:
        raw = raw / raw.sum() * total
    return raw.round(1)


def build_buy_decision(ranked_input: pd.DataFrame, config: dict | None = None) -> BuyDecisionResult:
    cfg = DEFAULT_BUY_CONFIG.copy()
    cfg.update({key: value for key, value in (config or {}).items() if key in cfg})
    for key in DEFAULT_BUY_CONFIG:
        cfg[key] = float(cfg[key])
    max_count = int(cfg["buy_max_count"])
    max_sector_positions = int(cfg["buy_max_sector_positions"])

    if ranked_input.empty:
        empty_plan = pd.DataFrame(columns=BUY_COLUMNS)
        empty_decision = pd.DataFrame(columns=DECISION_COLUMNS)
        return BuyDecisionResult(empty_plan, empty_decision, {"buy_count": 0, "target_total_position_pct": 0.0})

    out = ranked_input.copy()
    if "rank" not in out.columns:
        out = _sort_for_buy(out)
        out["rank"] = out.index + 1

    out["sector_name"] = _txt(out, "sector_name", "未知板块")
    out["p_limitup_t1"] = _num(out, "p_limitup_t1")
    out["wp_score"] = _num(out, "wp_score")
    out["sector_strength_score"] = _num(out, "sector_strength_score")
    out["stock_strength_score"] = _num(out, "stock_strength_score")
    out["acceptance_score"] = _num(out, "acceptance_score")
    out["momentum_score"] = _num(out, "momentum_score")
    out["capital_score"] = _num(out, "capital_score")
    out["model_confidence"] = _num(out, "model_confidence")
    out["risk_penalty_score"] = _num(out, "risk_penalty_score")
    out["pct_chg"] = _num(out, "pct_chg")
    out["price"] = _num(out, "price")
    out["close_position"] = _num(out, "close_position", 50)
    out["intraday_pullback_pct"] = _num(out, "intraday_pullback_pct")
    out["intraday_vwap_position"] = _num(out, "intraday_vwap_position")
    out["pre_day_limitup"] = _num(out, "pre_day_limitup").astype(int)
    out["today_limitup"] = _num(out, "today_limitup").astype(int)
    out["decision_score"] = (
        out["p_limitup_t1"] * 0.26
        + out["wp_score"] * 0.20
        + out["acceptance_score"] * 0.16
        + out["sector_strength_score"] * 0.12
        + out["momentum_score"] * 0.10
        + out["model_confidence"] * 0.08
        + out["capital_score"] * 0.08
        - out["risk_penalty_score"] * 0.22
    ).clip(0, 100)
    out["skip_reason"] = out.apply(_skip_reason, axis=1, cfg=cfg)
    out["buy_eligible"] = out["skip_reason"].eq("")
    out = _sort_for_buy(out)

    selected_indexes: list[int] = []
    sector_counts: dict[str, int] = {}
    for idx, row in out.iterrows():
        if not bool(row["buy_eligible"]):
            continue
        sector = str(row["sector_name"])
        if sector_counts.get(sector, 0) >= max_sector_positions:
            continue
        selected_indexes.append(idx)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected_indexes) >= max_count:
            break

    out["buy_flag"] = 0
    out["decision_action"] = "跳过"
    out.loc[out["buy_eligible"], "decision_action"] = "备选观察"
    if selected_indexes:
        out.loc[selected_indexes, "buy_flag"] = 1
        out.loc[selected_indexes, "decision_action"] = "买入观察"

    selected = out.loc[selected_indexes].copy()
    if not selected.empty:
        selected["buy_rank"] = range(1, len(selected) + 1)
        selected["suggest_position_pct"] = _assign_positions(selected, cfg).values
        selected["portfolio_group"] = selected["buy_rank"].map(lambda rank: "核心" if rank <= 2 else "标准")
        selected["confirm_before_buy"] = _confirm_text()
        selected["reject_if"] = _reject_text()
        selected["buy_reason"] = selected.apply(_reason, axis=1)
    for col in BUY_COLUMNS:
        if col not in selected.columns:
            selected[col] = ""
    buy_plan = selected[BUY_COLUMNS].copy()

    decision = out.copy()
    decision["buy_rank"] = ""
    decision.loc[selected_indexes, "buy_rank"] = range(1, len(selected_indexes) + 1)
    decision["suggest_position_pct"] = 0.0
    if selected_indexes:
        decision.loc[selected_indexes, "suggest_position_pct"] = buy_plan["suggest_position_pct"].values
    decision["portfolio_group"] = ""
    decision.loc[selected_indexes, "portfolio_group"] = buy_plan["portfolio_group"].values
    decision["confirm_before_buy"] = _confirm_text()
    decision["reject_if"] = _reject_text()
    decision["buy_reason"] = decision.apply(_reason, axis=1)
    for col in DECISION_COLUMNS:
        if col not in decision.columns:
            decision[col] = ""
    decision_table = decision[DECISION_COLUMNS].copy()
    summary = {
        "buy_count": int(len(buy_plan)),
        "target_total_position_pct": round(float(buy_plan["suggest_position_pct"].sum()), 1) if not buy_plan.empty else 0.0,
        "max_buy_count": max_count,
        "max_sector_positions": max_sector_positions,
        "selection_rule": "14:20生成最多5支买入观察池；14:50前人工确认尾盘承接后再执行。",
    }
    return BuyDecisionResult(buy_plan, decision_table, summary)
