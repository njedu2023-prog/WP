from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .tail_window import (
    TAIL_PHASE_CLOSED,
    TAIL_PHASE_FROZEN,
    TAIL_WINDOW_START,
    tail_window_phase,
)


DECISION_SUPPORT_VERSION = "t1_qualified_cohort_v5"

DEFAULT_GUIDANCE_CONFIG = {
    "guidance_min_stable_runs": 2,
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
    "candidate_deadline",
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
    candidate_count: int = 0,
    qualified_count: int = 0,
) -> dict:
    return {
        "version": DECISION_SUPPORT_VERSION,
        "objective": "最大化14:20-14:50合格候选在T+1收盘扣费后取得正收益的概率",
        "action": action,
        "action_code": action_code,
        "is_final": is_final,
        "candidate_code": "",
        "candidate_name": "",
        "candidate_count": int(candidate_count),
        "qualified_count": int(qualified_count),
        "market_state": str(market_regime.get("state") or "数据不足"),
        "market_score": _num(market_regime.get("score")),
        "reason": reason,
        "candidate_deadline": "14:50",
        "entry_deadline": "15:00",
        "exit_contract": "T+1收盘卖出",
        "next_checkpoint": next_checkpoint,
        "selection_contract": "发布全部合格票，由人工决定买哪一支",
        "validation_contract": "每支首次合格信号均做T+1收盘净收益验证",
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


def _candidate_checks(
    candidate: pd.Series,
    market_state: str,
    data_age: float | None,
    cfg: dict,
) -> dict[str, bool]:
    return {
        "市场允许": market_state not in {"回避", "数据不足"},
        "数据新鲜": data_age is None
        or data_age <= float(cfg["guidance_max_data_age_seconds"]),
        "真实样本模型": str(candidate.get("forecast_mode") or "") == "实时因果样本"
        and _truthy(candidate.get("forecast_actionable")),
        "样本数量": _num(candidate.get("forecast_live_sample_count"))
        >= float(cfg["guidance_min_live_samples"]),
        "独立交易日": _num(candidate.get("forecast_live_day_count"))
        >= float(cfg["guidance_min_live_days"]),
        "有效样本": _num(candidate.get("forecast_effective_sample_count"))
        >= float(cfg["guidance_min_effective_samples"]),
        "盈利概率": _num(candidate.get("forecast_profit_probability"))
        >= float(cfg["guidance_min_profit_probability"]),
        "概率下界": _num(candidate.get("forecast_profit_probability_lower"))
        >= float(cfg["guidance_min_profit_probability_lower"]),
        "成本后期望": _num(candidate.get("forecast_expected_net_return_pct"), -999.0)
        >= float(cfg["guidance_min_expected_net_return_pct"]),
        "下行约束": _num(candidate.get("forecast_downside_q10_pct"), -999.0)
        >= float(cfg["guidance_max_downside_q10_pct"]),
        "连续合格": _num(candidate.get("qualified_runs"))
        >= int(cfg["guidance_min_stable_runs"]),
    }


def build_decision_support(
    observation_pool: pd.DataFrame,
    market_regime: dict,
    market_data_time: str,
    config: dict | None = None,
    *,
    decision_time: str | None = None,
) -> DecisionSupportResult:
    """Evaluate every tail candidate independently and return the qualified set."""
    cfg = _config(config)
    phase = tail_window_phase(market_data_time)
    state = str(market_regime.get("state") or "数据不足")

    if phase == TAIL_PHASE_CLOSED:
        return _empty_result(
            _summary(
                market_regime,
                action="已收盘",
                action_code="CLOSED",
                reason="15:00已收盘，不再生成可买候选；历史信号继续等待T+1真值",
                is_final=True,
                next_checkpoint="下一交易日14:20",
            )
        )
    if phase == TAIL_PHASE_FROZEN:
        return _empty_result(
            _summary(
                market_regime,
                action="候选已冻结",
                action_code="FROZEN",
                reason="14:50后禁止新增合格票，已入账信号保持首次时间和价格不变",
                is_final=True,
                next_checkpoint="15:00停止可买展示",
            )
        )
    market_ts = _parse_datetime(market_data_time)
    if market_ts is None or market_ts.time() < TAIL_WINDOW_START:
        return _empty_result(
            _summary(
                market_regime,
                action="观察未开始",
                action_code="WATCH",
                reason="14:20后开始逐票判定",
                is_final=False,
                next_checkpoint="14:20",
            )
        )
    pool = observation_pool.copy() if observation_pool is not None else pd.DataFrame()
    if not pool.empty:
        qualification = (
            pool.get("qualification_status", pd.Series("", index=pool.index))
            .fillna("")
            .astype(str)
        )
        pool = pool[qualification.eq("合格")].copy()
    if pool.empty:
        return _empty_result(
            _summary(
                market_regime,
                action="继续观察",
                action_code="WATCH",
                reason="当前没有满足基础流动性与尾盘形态要求的候选",
                is_final=False,
                next_checkpoint="下一次5分钟快照",
            )
        )

    numeric_columns = (
        "tail_profit_score",
        "risk_penalty_score",
        "qualified_runs",
        "leader_runs",
        "forecast_confidence",
        "forecast_live_sample_count",
        "forecast_live_day_count",
        "forecast_effective_sample_count",
        "forecast_profit_probability",
        "forecast_profit_probability_lower",
        "forecast_expected_net_return_pct",
        "forecast_downside_q10_pct",
    )
    for column in numeric_columns:
        values = pool[column] if column in pool.columns else pd.Series(0.0, index=pool.index)
        pool[column] = pd.to_numeric(values, errors="coerce").fillna(0.0)
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

    data_age = _data_age_seconds(market_data_time, decision_time)
    rows: list[dict] = []
    qualified_count = 0
    failed_union: set[str] = set()
    for idx, candidate in pool.iterrows():
        checks = _candidate_checks(candidate, state, data_age, cfg)
        passed = [name for name, ok in checks.items() if ok]
        failed = [name for name, ok in checks.items() if not ok]
        is_qualified = not failed
        if is_qualified:
            qualified_count += 1
            support_action = "合格"
            reason = "全部概率、样本、风险、稳定性和数据新鲜度门槛通过"
        else:
            support_action = "观察"
            reason = "暂未通过：" + "、".join(failed)
            failed_union.update(failed)
        row = {column: candidate.get(column, "") for column in DECISION_COLUMNS}
        row.update(
            {
                "support_rank": int(idx + 1),
                "support_action": support_action,
                "is_current_choice": False,
                "checks_passed": "、".join(passed),
                "checks_failed": "、".join(failed),
                "decision_reason": reason,
                "candidate_deadline": "14:50",
                "entry_deadline": "15:00",
                "exit_contract": "T+1收盘卖出",
                "next_checkpoint": "首次合格即锁定" if is_qualified else "下一次5分钟快照",
                "manual_execution_only": True,
                "order_routing_enabled": False,
            }
        )
        rows.append(row)

    table = pd.DataFrame(rows, columns=DECISION_COLUMNS)
    if qualified_count:
        action = f"{qualified_count}支合格票"
        action_code = "QUALIFIED_SET"
        reason = "系统发布全部合格票，人工决定是否买入及买哪一支"
    else:
        action = "继续观察"
        action_code = "WATCH"
        reason = "当前没有逐票通过全部门槛的标的"
        if failed_union:
            reason += "；主要未通过：" + "、".join(sorted(failed_union))
    summary = _summary(
        market_regime,
        action=action,
        action_code=action_code,
        reason=reason,
        is_final=False,
        next_checkpoint=(
            "14:50候选集合冻结"
            if market_ts.hour == 14 and market_ts.minute == 50
            else "下一次5分钟快照"
        ),
        candidate_count=len(pool),
        qualified_count=qualified_count,
    )
    summary["data_age_seconds"] = data_age
    summary["qualified_codes"] = table.loc[
        table["support_action"].eq("合格"), "ts_code"
    ].astype(str).tolist()
    return DecisionSupportResult(table, summary)
