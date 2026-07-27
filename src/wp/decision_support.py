from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd

from .tail_window import (
    OFFICIAL_DECISION_START,
    TAIL_PHASE_CLOSED,
    TAIL_PHASE_FROZEN,
    TAIL_WINDOW_START,
    tail_window_phase,
)


DECISION_SUPPORT_VERSION = "t1_net_profit_decision_v3"

DEFAULT_GUIDANCE_CONFIG = {
    "guidance_wait_until": "14:45",
    "guidance_final_time": "14:55",
    "guidance_min_stable_runs": 2,
    "guidance_min_leader_runs": 2,
    "guidance_min_live_samples": 30,
    "guidance_min_live_days": 30,
    "guidance_min_effective_samples": 15,
    "guidance_min_profit_probability": 58.0,
    "guidance_min_profit_probability_lower": 50.0,
    "guidance_min_expected_net_return_pct": 0.30,
    "guidance_max_downside_q10_pct": -4.50,
    "guidance_max_data_age_seconds": 120,
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
    "qualified_runs",
    "leader_runs",
    "forecast_mode",
    "forecast_confidence",
    "forecast_live_sample_count",
    "forecast_live_day_count",
    "forecast_effective_sample_count",
    "forecast_profit_probability",
    "forecast_profit_probability_lower",
    "forecast_expected_net_return_pct",
    "forecast_downside_q10_pct",
    "forecast_actionable",
    "checks_passed",
    "checks_failed",
    "decision_reason",
    "entry_deadline",
    "exit_contract",
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


def _parse_datetime(value: object) -> pd.Timestamp | None:
    parsed = pd.to_datetime(str(value or "").strip(), errors="coerce")
    return None if pd.isna(parsed) else parsed


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def _config(config: dict | None) -> dict:
    values = DEFAULT_GUIDANCE_CONFIG.copy()
    values.update({key: value for key, value in (config or {}).items() if key in values})
    return values


def _summary(
    market_regime: dict,
    *,
    action: str,
    action_code: str,
    reason: str,
    is_final: bool,
    next_checkpoint: str,
) -> dict:
    return {
        "version": DECISION_SUPPORT_VERSION,
        "objective": "最大化T日尾盘可成交买入后T+1收盘净盈利概率",
        "action": action,
        "action_code": action_code,
        "is_final": is_final,
        "candidate_code": "",
        "candidate_name": "",
        "market_state": str(market_regime.get("state") or "数据不足"),
        "market_score": _num(market_regime.get("score")),
        "reason": reason,
        "entry_deadline": "14:55",
        "exit_contract": "T+1收盘卖出",
        "next_checkpoint": next_checkpoint,
        "manual_execution_only": True,
        "order_routing_enabled": False,
        "broker_connection": "disabled",
    }


def _empty_result(summary: dict) -> DecisionSupportResult:
    return DecisionSupportResult(pd.DataFrame(columns=DECISION_COLUMNS), summary)


def _data_age_seconds(market_data_time: str, decision_time: str | None) -> float | None:
    if not decision_time:
        return None
    market_ts = _parse_datetime(market_data_time)
    decision_ts = _parse_datetime(decision_time)
    if market_ts is None or decision_ts is None:
        return None
    return max(0.0, float((decision_ts - market_ts).total_seconds()))


def build_decision_support(
    observation_pool: pd.DataFrame,
    market_regime: dict,
    market_data_time: str,
    config: dict | None = None,
    *,
    decision_time: str | None = None,
) -> DecisionSupportResult:
    """Return one locked BUY decision or an explicit final NO_TRADE."""
    cfg = _config(config)
    current_time = _parse_clock(market_data_time)
    phase = tail_window_phase(market_data_time)
    wait_until = _parse_clock(cfg["guidance_wait_until"]) or OFFICIAL_DECISION_START
    final_time = _parse_clock(cfg["guidance_final_time"]) or time(14, 55)
    state = str(market_regime.get("state") or "数据不足")

    if phase == TAIL_PHASE_CLOSED:
        return _empty_result(
            _summary(
                market_regime,
                action="已收盘",
                action_code="NO_TRADE",
                reason="15:00已收盘，当日不再生成或修改买入决策",
                is_final=True,
                next_checkpoint="下一交易日14:20",
            )
        )
    if phase == TAIL_PHASE_FROZEN:
        return _empty_result(
            _summary(
                market_regime,
                action="NO_TRADE",
                action_code="NO_TRADE",
                reason="已过14:55成交安全截止线，当日禁止新开仓",
                is_final=True,
                next_checkpoint="下一交易日14:20",
            )
        )
    if current_time is None or current_time < TAIL_WINDOW_START:
        return _empty_result(
            _summary(
                market_regime,
                action="观察未开始",
                action_code="WATCH",
                reason="14:20后开始收集尾盘因果快照",
                is_final=False,
                next_checkpoint="14:20",
            )
        )

    pool = observation_pool.copy() if observation_pool is not None else pd.DataFrame()
    if not pool.empty:
        qualification = pool.get("qualification_status", pd.Series("", index=pool.index)).fillna("").astype(str)
        pool = pool[qualification.eq("合格")].copy()
    for column in (
        "tail_profit_score",
        "risk_penalty_score",
        "qualified_runs",
        "leader_runs",
        "forecast_confidence",
        "forecast_live_sample_count",
        "forecast_effective_sample_count",
        "forecast_profit_probability",
        "forecast_profit_probability_lower",
        "forecast_expected_net_return_pct",
        "forecast_downside_q10_pct",
    ):
        if not pool.empty:
            values = pool[column] if column in pool.columns else pd.Series(0.0, index=pool.index)
            pool[column] = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if not pool.empty:
        pool = pool.sort_values(
            [
                "forecast_profit_probability_lower",
                "forecast_profit_probability",
                "forecast_expected_net_return_pct",
                "tail_profit_score",
                "risk_penalty_score",
                "ts_code",
            ],
            ascending=[False, False, False, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        pool["support_rank"] = range(1, len(pool) + 1)

    before_decision = current_time < wait_until
    final_checkpoint = current_time >= final_time
    if state in {"回避", "数据不足"} or pool.empty:
        reason = (
            str(market_regime.get("reason") or "市场环境不允许新开仓")
            if state in {"回避", "数据不足"}
            else "当前没有满足流动性与尾盘资格的候选"
        )
        if final_checkpoint:
            return _empty_result(
                _summary(
                    market_regime,
                    action="NO_TRADE",
                    action_code="NO_TRADE",
                    reason=reason,
                    is_final=True,
                    next_checkpoint="下一交易日14:20",
                )
            )
        return _empty_result(
            _summary(
                market_regime,
                action="继续观察",
                action_code="WATCH",
                reason=reason,
                is_final=False,
                next_checkpoint="14:45正式决策" if before_decision else "14:55最终决策",
            )
        )

    top = pool.iloc[0]
    data_age = _data_age_seconds(market_data_time, decision_time)
    checks = {
        "市场允许": state not in {"回避", "数据不足"},
        "数据新鲜": data_age is None or data_age <= float(cfg["guidance_max_data_age_seconds"]),
        "真实样本模型": str(top.get("forecast_mode") or "") == "实时因果样本"
        and _truthy(top.get("forecast_actionable")),
        "样本数量": _num(top.get("forecast_live_sample_count")) >= float(cfg["guidance_min_live_samples"]),
        "独立交易日": _num(top.get("forecast_live_day_count")) >= float(cfg["guidance_min_live_days"]),
        "有效样本": _num(top.get("forecast_effective_sample_count")) >= float(cfg["guidance_min_effective_samples"]),
        "盈利概率": _num(top.get("forecast_profit_probability")) >= float(cfg["guidance_min_profit_probability"]),
        "概率下界": _num(top.get("forecast_profit_probability_lower"))
        >= float(cfg["guidance_min_profit_probability_lower"]),
        "成本后期望": _num(top.get("forecast_expected_net_return_pct"), -999.0)
        >= float(cfg["guidance_min_expected_net_return_pct"]),
        "下行约束": _num(top.get("forecast_downside_q10_pct"), -999.0)
        >= float(cfg["guidance_max_downside_q10_pct"]),
        "连续合格": _num(top.get("qualified_runs")) >= int(cfg["guidance_min_stable_runs"]),
        "连续领先": _num(top.get("leader_runs")) >= int(cfg["guidance_min_leader_runs"]),
    }
    passed = [name for name, ok in checks.items() if ok]
    failed = [name for name, ok in checks.items() if not ok]

    if before_decision:
        action = "继续观察"
        action_code = "WATCH"
        is_final = False
        reason = "14:45前只收集样本，不形成正式买入决策"
    elif final_checkpoint:
        action = "NO_TRADE"
        action_code = "NO_TRADE"
        is_final = True
        reason = "已到14:55成交安全截止线，未锁定交易不得再开仓"
    elif not failed:
        action = "买入"
        action_code = "BUY"
        is_final = True
        reason = "成本后盈利概率、置信下界、期望收益、风险和可成交条件全部通过"
    else:
        action = "继续观察"
        action_code = "WATCH"
        is_final = False
        reason = "暂未通过：" + "、".join(failed)

    rows: list[dict] = []
    for idx, candidate in pool.iterrows():
        row = {column: candidate.get(column, "") for column in DECISION_COLUMNS}
        row.update(
            {
                "support_rank": int(idx + 1),
                "support_action": action if idx == 0 else "研究候选",
                "is_current_choice": bool(idx == 0),
                "checks_passed": "、".join(passed) if idx == 0 else "",
                "checks_failed": "、".join(failed) if idx == 0 else "",
                "decision_reason": reason if idx == 0 else "不是当前最高保守盈利概率候选",
                "entry_deadline": "14:55",
                "exit_contract": "T+1收盘卖出",
                "next_checkpoint": "停止新开仓" if is_final else "14:55最终决策",
                "manual_execution_only": True,
                "order_routing_enabled": False,
            }
        )
        rows.append(row)
    table = pd.DataFrame(rows, columns=DECISION_COLUMNS)
    summary = _summary(
        market_regime,
        action=action,
        action_code=action_code,
        reason=reason,
        is_final=is_final,
        next_checkpoint="停止新开仓" if is_final else "14:55最终决策",
    )
    summary.update(
        {
            "candidate_code": str(top.get("ts_code") or ""),
            "candidate_name": str(top.get("name") or ""),
            "candidate_count": int(len(pool)),
            "data_age_seconds": data_age,
            "checks_passed": passed,
            "failed_checks": failed,
        }
    )
    for key in (
        "forecast_mode",
        "forecast_confidence",
        "forecast_live_sample_count",
        "forecast_live_day_count",
        "forecast_effective_sample_count",
        "forecast_profit_probability",
        "forecast_profit_probability_lower",
        "forecast_expected_net_return_pct",
        "forecast_downside_q10_pct",
    ):
        summary[key] = top.get(key, "")
    return DecisionSupportResult(table, summary)
