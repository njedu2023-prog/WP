from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd


EXIT_GUIDANCE_VERSION = "manual_t1_exit_v1"

DEFAULT_EXIT_CONFIG = {
    "exit_open_stop_pct": -3.0,
    "exit_time": "10:40",
    "exit_profit_protect_pct": 3.0,
    "exit_pullback_pct": 2.0,
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
    """Produce T+1 manual sell guidance without assuming or routing an order."""
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
    exit_time = _clock(cfg["exit_time"]) or time(10, 40)

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
        open_board_count = _num(market.get("open_board_count"))
        stop = float(cfg["exit_open_stop_pct"])
        profit_protect = float(cfg["exit_profit_protect_pct"])
        pullback_limit = float(cfg["exit_pullback_pct"])
        pullback = (high_return - current_return) if high_return is not None and current_return is not None else 0.0

        has_market_price = any(
            _num(market.get(column)) is not None
            for column in ("price", "close", "open", "high", "low")
        )
        if not has_market_price:
            action = "行情数据不足"
            reason = "当前全市场行情中没有该股票，禁止据此生成卖出结论"
            checkpoint = "人工核对实时行情与实际持仓"
        elif current_time is None or current_time < time(9, 31):
            action = "等待开盘确认"
            reason = "9:31前不生成卖出结论"
            checkpoint = "09:31复核"
        elif (open_return is not None and open_return <= stop) or (low_return is not None and low_return <= stop):
            action = "建议风险退出"
            reason = f"开盘/盘中已触及{stop:.1f}%风险线"
            checkpoint = "由人工立即确认盘口并处理"
        elif sealed and open_board_count is not None and open_board_count > 0:
            action = "涨停已炸板，建议保护利润"
            reason = f"当前虽回封，但已记录炸板{int(open_board_count)}次，不满足未炸板持有条件"
            checkpoint = "人工核对封单与流动性后处理"
        elif sealed and current_time >= time(15, 0) and open_board_count == 0:
            action = "确认未炸板，建议继续持有"
            reason = "收盘封板且炸板次数为0，满足T+2例外条件"
            checkpoint = "T+2 09:31人工卖出评估"
        elif sealed:
            action = "当前涨停，继续观察"
            reason = "尚未到收盘或缺少明确炸板次数，不能提前确认T+2持有"
            checkpoint = "收盘人工确认封板和炸板记录"
        elif current_time >= exit_time:
            action = "建议人工择机卖出"
            reason = "10:40后仍未封住涨停，按基准退出规则处理"
            checkpoint = "人工核对流动性后执行"
        elif high_return is not None and high_return >= profit_protect and pullback >= pullback_limit:
            action = "建议保护利润"
            reason = f"最高收益后回撤{pullback:.2f}%，达到保护阈值"
            checkpoint = "10:40前持续复核"
        else:
            action = "继续观察"
            reason = "尚未触发风险退出、利润保护或10:40退出条件"
            checkpoint = "10:40复核"

        row = {column: plan.get(column, "") for column in EXIT_COLUMNS}
        row.update(
            {
                "target_trade_date": normalized_date,
                "ts_code": code,
                "name": str(plan.get("name") or market.get("name") or ""),
                "holding_confirmation": "待人工确认实际持仓",
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
        "position_basis": "系统历史观察记录，实际持仓须人工确认",
        "manual_execution_only": True,
        "order_routing_enabled": False,
        "broker_connection": "disabled",
    }
