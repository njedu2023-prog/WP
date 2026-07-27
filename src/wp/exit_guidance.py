from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd


EXIT_GUIDANCE_VERSION = "fixed_t1_close_exit_v2"

DEFAULT_EXIT_CONFIG = {
    "exit_time": "14:50",
}

EXIT_COLUMNS = [
    "target_trade_date",
    "ts_code",
    "name",
    "holding_confirmation",
    "plan_time",
    "plan_price",
    "current_price",
    "open_return_pct",
    "high_return_pct",
    "low_return_pct",
    "current_return_pct",
    "sealed_limit_up",
    "guidance_action",
    "guidance_reason",
    "next_checkpoint",
    "forecast_mode",
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
    "manual_execution_only",
    "order_routing_enabled",
]


@dataclass
class ExitGuidanceResult:
    table: pd.DataFrame
    summary: dict


def _num(value: object) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else None


def _return(value: object, entry: float | None) -> float | None:
    price = _num(value)
    if price is None or entry is None or entry <= 0:
        return None
    return round((price / entry - 1) * 100, 4)


def _clock(value: object) -> time | None:
    parsed = pd.to_datetime(str(value or "").strip(), errors="coerce")
    return None if pd.isna(parsed) else parsed.to_pydatetime().time()


def _is_sealed(row: pd.Series) -> bool:
    today_limitup = str(row.get("today_limitup") or "").strip().lower() in {"1", "true", "yes"}
    current = _num(row.get("price")) or _num(row.get("close"))
    up_limit = _num(row.get("limit_up_price")) or _num(row.get("up_limit"))
    return bool(today_limitup or (current is not None and up_limit is not None and up_limit > 0 and current >= up_limit * 0.999))


def build_exit_guidance(
    validation_history: pd.DataFrame,
    market_universe: pd.DataFrame,
    trade_date: str,
    market_data_time: str,
    config: dict | None = None,
) -> ExitGuidanceResult:
    """Apply the same fixed T+1 close contract used by training and validation."""
    cfg = DEFAULT_EXIT_CONFIG.copy()
    cfg.update({key: value for key, value in (config or {}).items() if key in cfg})
    history = validation_history.copy() if validation_history is not None else pd.DataFrame()
    universe = market_universe.copy() if market_universe is not None else pd.DataFrame()
    normalized_date = str(trade_date or "").replace("-", "")
    if history.empty or "target_trade_date" not in history.columns or "ts_code" not in history.columns:
        return ExitGuidanceResult(pd.DataFrame(columns=EXIT_COLUMNS), _summary(0))
    dates = history["target_trade_date"].fillna("").astype(str).str.replace("-", "", regex=False).str.replace(r"\.0$", "", regex=True)
    history = history[dates.eq(normalized_date)].copy()
    if history.empty:
        return ExitGuidanceResult(pd.DataFrame(columns=EXIT_COLUMNS), _summary(0))
    history["_plan_dt"] = pd.to_datetime(history.get("plan_time", ""), errors="coerce")
    history = history.sort_values("_plan_dt", kind="mergesort").drop_duplicates("ts_code", keep="last")
    if "ts_code" not in universe.columns:
        universe = pd.DataFrame(columns=["ts_code"])
    universe["ts_code"] = universe["ts_code"].fillna("").astype(str).str.strip()
    universe = universe.drop_duplicates("ts_code", keep="last").set_index("ts_code", drop=False)
    current_time = _clock(market_data_time)
    exit_time = _clock(cfg["exit_time"]) or time(14, 50)

    rows: list[dict] = []
    for _, plan in history.iterrows():
        code = str(plan.get("ts_code") or "").strip()
        market = universe.loc[code] if code in universe.index else pd.Series(dtype="object")
        entry = _num(plan.get("plan_price"))
        current_price = _num(market.get("price")) or _num(market.get("close"))
        open_return = _return(market.get("open"), entry)
        high_return = _return(market.get("high"), entry)
        low_return = _return(market.get("low"), entry)
        current_return = _return(current_price, entry)
        sealed = _is_sealed(market)

        has_market_price = any(
            _num(market.get(column)) is not None
            for column in ("price", "close", "open", "high", "low")
        )
        if not has_market_price:
            action = "行情数据不足"
            reason = "当前全市场行情中没有该股票，禁止据此生成卖出结论"
            checkpoint = "人工核对实时行情与实际持仓"
        elif current_time is None:
            action = "等待T+1收盘"
            reason = "无法解析行情时间，禁止临时改变固定卖出合同"
            checkpoint = "14:50复核"
        elif current_time >= time(15, 0):
            action = "候选验证合同已结束"
            reason = "正式验证使用T+1收盘价并扣除统一往返成本"
            checkpoint = "等待收盘真值入库"
        elif current_time >= exit_time:
            action = "如实际持有，执行T+1收盘卖出"
            reason = "候选验证统一按T+1收盘；是否实际持有由人工成交记录确认"
            checkpoint = "实际持仓需在15:00前人工完成卖出"
        else:
            action = "按合同持有"
            reason = "训练、决策和验证统一使用T+1收盘退出，盘中不临时改规则"
            checkpoint = "14:50开始卖出"

        row = {column: plan.get(column, "") for column in EXIT_COLUMNS}
        row.update(
            {
                "target_trade_date": normalized_date,
                "ts_code": code,
                "name": str(plan.get("name") or market.get("name") or ""),
                "holding_confirmation": "待人工确认是否实际买入",
                "plan_time": plan.get("plan_time", ""),
                "plan_price": "" if entry is None else entry,
                "current_price": "" if current_price is None else current_price,
                "open_return_pct": "" if open_return is None else open_return,
                "high_return_pct": "" if high_return is None else high_return,
                "low_return_pct": "" if low_return is None else low_return,
                "current_return_pct": "" if current_return is None else current_return,
                "sealed_limit_up": sealed,
                "guidance_action": action,
                "guidance_reason": reason,
                "next_checkpoint": checkpoint,
                "manual_execution_only": True,
                "order_routing_enabled": False,
            }
        )
        rows.append(row)
    table = pd.DataFrame(rows, columns=EXIT_COLUMNS)
    return ExitGuidanceResult(table, _summary(len(table), table))


def _summary(count: int, table: pd.DataFrame | None = None) -> dict:
    actions = {}
    if table is not None and not table.empty:
        actions = table["guidance_action"].value_counts().to_dict()
    return {
        "version": EXIT_GUIDANCE_VERSION,
        "record_count": int(count),
        "action_counts": actions,
        "position_basis": "全部合格候选的假设退出路径；实际持仓与成交价须人工确认",
        "exit_contract": "T+1_close",
        "manual_execution_only": True,
        "order_routing_enabled": False,
        "broker_connection": "disabled",
    }
