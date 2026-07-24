from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd

from .tail_window import (
    TAIL_PHASE_CLOSED,
    TAIL_PHASE_FROZEN,
    TAIL_WINDOW_START,
    tail_window_phase,
)


DECISION_SUPPORT_VERSION = "human_tail_decision_v2"

DEFAULT_GUIDANCE_CONFIG = {
    "guidance_wait_until": "14:30",
    "guidance_final_time": "14:50",
    "guidance_min_stable_runs": 2,
    "guidance_min_leader_runs": 2,
    "guidance_min_score": 70.0,
    "guidance_min_score_lead": 4.0,
    "guidance_min_confidence": 35.0,
    "guidance_min_utility": 0.0,
}

DECISION_COLUMNS = [
    "support_rank",
    "support_action",
    "is_current_choice",
    "ts_code",
    "name",
    "sector_name",
    "price",
    "pct_chg",
    "tail_profit_score",
    "risk_penalty_score",
    "quality_lead",
    "qualified_runs",
    "leader_runs",
    "forecast_mode",
    "forecast_confidence",
    "forecast_open_q10_pct",
    "forecast_open_q50_pct",
    "forecast_open_q90_pct",
    "forecast_high_q10_pct",
    "forecast_high_q50_pct",
    "forecast_high_q90_pct",
    "forecast_low_q10_pct",
    "forecast_low_q50_pct",
    "forecast_low_q90_pct",
    "forecast_close_q10_pct",
    "forecast_close_q50_pct",
    "forecast_close_q90_pct",
    "forecast_profit_probability",
    "forecast_touch_plus3_probability",
    "forecast_touch_minus3_probability",
    "forecast_expected_net_return_pct",
    "forecast_risk_adjusted_utility",
    "checks_passed",
    "checks_failed",
    "decision_reason",
    "invalidation_rule",
    "next_checkpoint",
    "manual_execution_only",
    "order_routing_enabled",
]


@dataclass
class DecisionSupportResult:
    table: pd.DataFrame
    summary: dict


def _num(value: object, default: float = 0.0) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else float(default)


def _parse_clock(value: object) -> time | None:
    parsed = pd.to_datetime(str(value or "").strip(), errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime().time()


def _config(config: dict | None) -> dict:
    values = DEFAULT_GUIDANCE_CONFIG.copy()
    values.update({key: value for key, value in (config or {}).items() if key in values})
    return values


def _empty_summary(market_regime: dict, action: str, reason: str) -> dict:
    return {
        "version": DECISION_SUPPORT_VERSION,
        "action": action,
        "candidate_code": "",
        "candidate_name": "",
        "market_state": str(market_regime.get("state") or "数据不足"),
        "market_score": _num(market_regime.get("score")),
        "reason": reason,
        "next_checkpoint": "下一次数据刷新后复核",
        "manual_execution_only": True,
        "order_routing_enabled": False,
        "broker_connection": "disabled",
    }


def build_decision_support(
    observation_pool: pd.DataFrame,
    market_regime: dict,
    market_data_time: str,
    config: dict | None = None,
) -> DecisionSupportResult:
    """Choose one human-review candidate, or explicitly recommend waiting/no trade."""
    cfg = _config(config)
    current_time = _parse_clock(market_data_time)
    phase = tail_window_phase(market_data_time)
    wait_until = _parse_clock(cfg["guidance_wait_until"]) or time(14, 30)
    final_time = _parse_clock(cfg["guidance_final_time"]) or time(14, 50)
    state = str(market_regime.get("state") or "数据不足")

    if phase == TAIL_PHASE_CLOSED:
        summary = _empty_summary(
            market_regime,
            "已收盘",
            "15:00已收盘，停止生成尾盘名单和新开仓建议",
        )
        summary["next_checkpoint"] = "下一交易日14:20"
        return DecisionSupportResult(pd.DataFrame(columns=DECISION_COLUMNS), summary)
    if phase == TAIL_PHASE_FROZEN:
        summary = _empty_summary(
            market_regime,
            "建议空仓",
            "14:50候选生成窗口已结束，15:00前仅保留最后一次观察记录",
        )
        summary["next_checkpoint"] = "停止新开仓"
        return DecisionSupportResult(pd.DataFrame(columns=DECISION_COLUMNS), summary)
    if current_time is None or current_time < TAIL_WINDOW_START:
        summary = _empty_summary(market_regime, "非尾盘时段", "14:20后才启动尾盘人工决策辅助")
        return DecisionSupportResult(pd.DataFrame(columns=DECISION_COLUMNS), summary)
    if state in {"回避", "数据不足"}:
        reason = str(market_regime.get("reason") or "市场环境不支持新开仓")
        summary = _empty_summary(market_regime, "建议空仓", reason)
        return DecisionSupportResult(pd.DataFrame(columns=DECISION_COLUMNS), summary)

    pool = observation_pool.copy() if observation_pool is not None else pd.DataFrame()
    if pool.empty:
        action = "建议空仓" if current_time >= final_time else "继续观察"
        summary = _empty_summary(market_regime, action, "当前没有保持内在资格的尾盘观察票")
        return DecisionSupportResult(pd.DataFrame(columns=DECISION_COLUMNS), summary)
    qualification = pool.get("qualification_status", pd.Series("", index=pool.index)).fillna("").astype(str)
    pool = pool[qualification.eq("合格")].copy()
    if pool.empty:
        action = "建议空仓" if current_time >= final_time else "继续观察"
        summary = _empty_summary(market_regime, action, "现有记录均已封板或正在资格复核")
        return DecisionSupportResult(pd.DataFrame(columns=DECISION_COLUMNS), summary)

    for column in ("tail_profit_score", "risk_penalty_score", "qualified_runs", "leader_runs", "forecast_confidence", "forecast_risk_adjusted_utility"):
        pool[column] = pd.to_numeric(pool.get(column, 0), errors="coerce").fillna(0.0)
    pool = pool.sort_values(
        ["tail_profit_score", "forecast_risk_adjusted_utility", "risk_penalty_score", "ts_code"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    pool["support_rank"] = range(1, len(pool) + 1)
    top_score = _num(pool.iloc[0].get("tail_profit_score"))
    second_score = _num(pool.iloc[1].get("tail_profit_score")) if len(pool) > 1 else 0.0
    quality_lead = round(top_score - second_score, 2) if len(pool) > 1 else round(top_score, 2)

    top = pool.iloc[0]
    required_confidence = float(cfg["guidance_min_confidence"]) + (10.0 if state == "谨慎" else 0.0)
    required_utility = float(cfg["guidance_min_utility"]) + (0.30 if state == "谨慎" else 0.0)
    checks = {
        "质量分": top_score >= float(cfg["guidance_min_score"]),
        "连续合格": _num(top.get("qualified_runs")) >= int(cfg["guidance_min_stable_runs"]),
        "连续领先": _num(top.get("leader_runs")) >= int(cfg["guidance_min_leader_runs"]),
        "领先幅度": len(pool) == 1 or quality_lead >= float(cfg["guidance_min_score_lead"]),
        "预测样本": str(top.get("forecast_mode") or "") != "样本不足",
        "预测置信": _num(top.get("forecast_confidence")) >= required_confidence,
        "风险收益": _num(top.get("forecast_risk_adjusted_utility"), -999.0) >= required_utility,
    }
    passed = [name for name, ok in checks.items() if ok]
    failed = [name for name, ok in checks.items() if not ok]
    all_passed = not failed
    if current_time < wait_until:
        action = "继续观察"
        reason = f"尚处等待窗口；当前领先{quality_lead:.2f}分，等待候选稳定"
    elif all_passed:
        action = "建议关注买入"
        reason = "、".join(passed) + "均通过；仍须人工核对盘口后下单"
    elif current_time >= final_time:
        action = "建议空仓"
        reason = "最终检查未通过：" + "、".join(failed)
    else:
        action = "等待更优票" if len(pool) > 1 else "继续观察"
        reason = "暂未通过：" + "、".join(failed)

    rows: list[dict] = []
    for idx, candidate in pool.iterrows():
        row = {column: candidate.get(column, "") for column in DECISION_COLUMNS}
        row.update(
            {
                "support_rank": int(idx + 1),
                "support_action": action if idx == 0 else "备选观察",
                "is_current_choice": bool(idx == 0),
                "quality_lead": quality_lead if idx == 0 else "",
                "checks_passed": "、".join(passed) if idx == 0 else "",
                "checks_failed": "、".join(failed) if idx == 0 else "",
                "decision_reason": reason if idx == 0 else "排名低于当前首选",
                "invalidation_rule": "跌破8%/承接转弱/板块转弱/资格消失/数据过期",
                "next_checkpoint": "14:50最终确认" if current_time < final_time else "停止新开仓",
                "manual_execution_only": True,
                "order_routing_enabled": False,
            }
        )
        rows.append(row)
    table = pd.DataFrame(rows, columns=DECISION_COLUMNS)
    summary = {
        "version": DECISION_SUPPORT_VERSION,
        "action": action,
        "candidate_code": str(top.get("ts_code") or ""),
        "candidate_name": str(top.get("name") or ""),
        "candidate_count": int(len(pool)),
        "market_state": state,
        "market_score": _num(market_regime.get("score")),
        "quality_lead": quality_lead,
        "qualified_runs": int(_num(top.get("qualified_runs"))),
        "leader_runs": int(_num(top.get("leader_runs"))),
        "forecast_mode": str(top.get("forecast_mode") or ""),
        "forecast_confidence": _num(top.get("forecast_confidence")),
        "forecast_warning": str(top.get("forecast_warning") or ""),
        "reason": reason,
        "failed_checks": failed,
        "next_checkpoint": "14:50最终确认" if current_time < final_time else "停止新开仓",
        "manual_execution_only": True,
        "order_routing_enabled": False,
        "broker_connection": "disabled",
    }
    for key in (
        "forecast_open_q10_pct", "forecast_open_q50_pct", "forecast_open_q90_pct",
        "forecast_high_q10_pct", "forecast_high_q50_pct", "forecast_high_q90_pct",
        "forecast_low_q10_pct", "forecast_low_q50_pct", "forecast_low_q90_pct",
        "forecast_close_q10_pct", "forecast_close_q50_pct", "forecast_close_q90_pct",
        "forecast_profit_probability", "forecast_touch_plus3_probability",
        "forecast_touch_minus3_probability", "forecast_expected_net_return_pct",
        "forecast_risk_adjusted_utility",
    ):
        summary[key] = top.get(key, "")
    return DecisionSupportResult(table, summary)
