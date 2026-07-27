from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .calendar import next_trading_day_str


STRATEGY_VERSION = "t1_net_profit_strategy_v1"
ENTRY_CONTRACT = "signal_last_price"
EXIT_CONTRACT = "T+1_close"

STRATEGY_COLUMNS = [
    "strategy_version",
    "decision_id",
    "plan_trade_date",
    "plan_time",
    "published_at",
    "action",
    "decision_status",
    "ts_code",
    "name",
    "sector_name",
    "plan_price",
    "entry_contract",
    "entry_deadline",
    "target_trade_date",
    "exit_contract",
    "forecast_profit_probability",
    "forecast_profit_probability_lower",
    "forecast_expected_net_return_pct",
    "forecast_live_sample_count",
    "forecast_live_day_count",
    "forecast_effective_sample_count",
    "decision_reason",
    "actual_trade_date",
    "actual_close",
    "gross_return_pct",
    "assumed_cost_pct",
    "net_return_pct",
    "is_net_profit",
    "truth_status",
    "truth_error",
    "truth_updated_at",
]


@dataclass
class StrategyLedgerResult:
    table: pd.DataFrame
    summary: dict


def _number(value: object) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else None


def _series(frame: pd.DataFrame, column: str, default: object = "") -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=STRATEGY_COLUMNS)
    try:
        frame = pd.read_csv(
            path,
            keep_default_na=False,
            dtype={
                "decision_id": str,
                "plan_trade_date": str,
                "target_trade_date": str,
                "actual_trade_date": str,
                "ts_code": str,
            },
        )
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=STRATEGY_COLUMNS)
    for column in STRATEGY_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[STRATEGY_COLUMNS].copy()


def _find_truth(
    validation: pd.DataFrame,
    plan_trade_date: str,
    ts_code: str,
) -> pd.Series | None:
    if validation is None or validation.empty:
        return None
    view = validation.copy()
    plan_dates = _series(view, "plan_trade_date").fillna("").astype(str).str.replace("-", "", regex=False)
    codes = _series(view, "ts_code").fillna("").astype(str)
    matches = view[plan_dates.eq(plan_trade_date) & codes.eq(ts_code)].copy()
    if matches.empty:
        return None
    matches["_verified"] = _series(matches, "truth_status").fillna("").astype(str).eq("verified")
    matches["_plan_time"] = pd.to_datetime(_series(matches, "plan_time"), errors="coerce")
    matches = matches.sort_values(["_verified", "_plan_time"], ascending=[False, False])
    return matches.iloc[0]


def _new_decision_row(
    buy_plan: pd.DataFrame,
    decision: dict,
    health: dict,
    current: datetime,
    cost_pct: float,
) -> dict | None:
    if not bool(decision.get("is_final")):
        return None
    action_code = str(decision.get("action_code") or "")
    if action_code not in {"BUY", "NO_TRADE"}:
        return None
    plan_trade_date = str(health.get("data_trade_date") or "").replace("-", "")
    if len(plan_trade_date) != 8:
        return None
    plan_time = str(health.get("market_data_time") or health.get("data_time") or "")
    row = buy_plan.iloc[0] if action_code == "BUY" and buy_plan is not None and not buy_plan.empty else pd.Series(dtype="object")
    target_trade_date = next_trading_day_str(plan_trade_date) if action_code == "BUY" else ""
    return {
        "strategy_version": STRATEGY_VERSION,
        "decision_id": f"{STRATEGY_VERSION}:{plan_trade_date}",
        "plan_trade_date": plan_trade_date,
        "plan_time": plan_time,
        "published_at": current.strftime("%Y-%m-%d %H:%M:%S"),
        "action": action_code,
        "decision_status": "locked",
        "ts_code": row.get("ts_code", ""),
        "name": row.get("name", ""),
        "sector_name": row.get("sector_name", ""),
        "plan_price": row.get("price", ""),
        "entry_contract": ENTRY_CONTRACT if action_code == "BUY" else "",
        "entry_deadline": f"{plan_trade_date} 14:55:00" if action_code == "BUY" else "",
        "target_trade_date": target_trade_date,
        "exit_contract": EXIT_CONTRACT if action_code == "BUY" else "",
        "forecast_profit_probability": row.get(
            "forecast_profit_probability",
            decision.get("forecast_profit_probability", ""),
        ),
        "forecast_profit_probability_lower": row.get(
            "forecast_profit_probability_lower",
            decision.get("forecast_profit_probability_lower", ""),
        ),
        "forecast_expected_net_return_pct": row.get(
            "forecast_expected_net_return_pct",
            decision.get("forecast_expected_net_return_pct", ""),
        ),
        "forecast_live_sample_count": row.get(
            "forecast_live_sample_count",
            decision.get("forecast_live_sample_count", ""),
        ),
        "forecast_live_day_count": row.get(
            "forecast_live_day_count",
            decision.get("forecast_live_day_count", ""),
        ),
        "forecast_effective_sample_count": row.get(
            "forecast_effective_sample_count",
            decision.get("forecast_effective_sample_count", ""),
        ),
        "decision_reason": str(decision.get("reason") or ""),
        "actual_trade_date": target_trade_date,
        "actual_close": "",
        "gross_return_pct": "",
        "assumed_cost_pct": cost_pct if action_code == "BUY" else "",
        "net_return_pct": "",
        "is_net_profit": "",
        "truth_status": "pending" if action_code == "BUY" else "not_applicable",
        "truth_error": "",
        "truth_updated_at": "",
    }


def _fill_truth(
    table: pd.DataFrame,
    validation: pd.DataFrame,
    current: datetime,
    cost_pct: float,
) -> pd.DataFrame:
    if table.empty:
        return table
    out = table.copy()
    buy_mask = out["action"].fillna("").astype(str).eq("BUY")
    for idx, row in out.loc[buy_mask].iterrows():
        truth = _find_truth(
            validation,
            str(row.get("plan_trade_date") or "").replace("-", ""),
            str(row.get("ts_code") or ""),
        )
        if truth is None:
            continue
        if str(truth.get("truth_status") or "") != "verified":
            out.at[idx, "truth_status"] = "pending"
            out.at[idx, "truth_error"] = str(truth.get("truth_error") or "")
            continue
        entry = _number(row.get("plan_price"))
        actual_close = _number(truth.get("actual_close"))
        if entry is None or entry <= 0 or actual_close is None:
            continue
        gross = (actual_close / entry - 1) * 100
        cost = _number(row.get("assumed_cost_pct"))
        cost = float(cost_pct if cost is None else cost)
        net = round(gross - cost, 4)
        out.at[idx, "actual_trade_date"] = truth.get("actual_trade_date", row.get("target_trade_date", ""))
        out.at[idx, "actual_close"] = actual_close
        out.at[idx, "gross_return_pct"] = round(gross, 4)
        out.at[idx, "assumed_cost_pct"] = cost
        out.at[idx, "net_return_pct"] = net
        out.at[idx, "is_net_profit"] = bool(net > 0)
        out.at[idx, "truth_status"] = "verified"
        out.at[idx, "truth_error"] = ""
        out.at[idx, "truth_updated_at"] = current.strftime("%Y-%m-%d %H:%M:%S")
    return out


def locked_decision_for_date(table: pd.DataFrame, trade_date: str) -> dict | None:
    if table is None or table.empty:
        return None
    normalized = str(trade_date or "").replace("-", "")
    dates = _series(table, "plan_trade_date").fillna("").astype(str).str.replace("-", "", regex=False)
    rows = table[dates.eq(normalized)].copy()
    if rows.empty:
        return None
    rows["_published"] = pd.to_datetime(_series(rows, "published_at"), errors="coerce")
    return rows.sort_values("_published", kind="mergesort").iloc[0].drop(labels=["_published"]).to_dict()


def summarize_strategy(table: pd.DataFrame) -> dict:
    empty = {
        "metric_scope": "strategy",
        "strategy_version": STRATEGY_VERSION,
        "objective": "maximize_probability_of_positive_T1_net_return",
        "entry_contract": ENTRY_CONTRACT,
        "exit_contract": EXIT_CONTRACT,
        "decision_days": 0,
        "trade_days": 0,
        "no_trade_days": 0,
        "verified_trades": 0,
        "pending_trades": 0,
        "net_profit_trades": 0,
        "net_win_rate": 0.0,
        "average_net_return_pct": 0.0,
        "median_net_return_pct": 0.0,
        "cumulative_net_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
    }
    if table is None or table.empty:
        return empty
    scoped = table[_series(table, "strategy_version").fillna("").astype(str).eq(STRATEGY_VERSION)].copy()
    if scoped.empty:
        return empty
    buys = scoped[scoped["action"].fillna("").astype(str).eq("BUY")].copy()
    verified = buys[buys["truth_status"].fillna("").astype(str).eq("verified")].copy()
    net = pd.to_numeric(_series(verified, "net_return_pct"), errors="coerce").dropna()
    equity = (1 + net / 100).cumprod() if len(net) else pd.Series(dtype="float64")
    drawdown = (equity / equity.cummax() - 1) * 100 if len(equity) else pd.Series(dtype="float64")
    result = {
        **empty,
        "decision_days": int(scoped["plan_trade_date"].nunique()),
        "trade_days": int(buys["plan_trade_date"].nunique()),
        "no_trade_days": int(scoped[scoped["action"].eq("NO_TRADE")]["plan_trade_date"].nunique()),
        "verified_trades": int(len(net)),
        "pending_trades": int(len(buys) - len(net)),
        "net_profit_trades": int(net.gt(0).sum()),
        "net_win_rate": round(float(net.gt(0).mean() * 100), 2) if len(net) else 0.0,
        "average_net_return_pct": round(float(net.mean()), 4) if len(net) else 0.0,
        "median_net_return_pct": round(float(net.median()), 4) if len(net) else 0.0,
        "cumulative_net_return_pct": round(float((equity.iloc[-1] - 1) * 100), 4) if len(equity) else 0.0,
        "max_drawdown_pct": round(float(drawdown.min()), 4) if len(drawdown) else 0.0,
    }
    return result


def update_strategy_ledger(
    buy_plan: pd.DataFrame,
    decision: dict,
    validation: pd.DataFrame,
    health: dict,
    output_root: Path,
    current: datetime,
    config: dict | None = None,
) -> StrategyLedgerResult:
    cost_pct = float((config or {}).get("forecast_round_trip_cost_pct", 0.25))
    path = output_root / "csv" / "wp_strategy_ledger.csv"
    table = _read(path)
    new_row = _new_decision_row(buy_plan, decision, health, current, cost_pct)
    if new_row is not None:
        decision_id = str(new_row["decision_id"])
        if not table["decision_id"].fillna("").astype(str).eq(decision_id).any():
            new_frame = pd.DataFrame([new_row], columns=STRATEGY_COLUMNS)
            table = new_frame if table.empty else pd.concat([table, new_frame], ignore_index=True)
    table = _fill_truth(table, validation, current, cost_pct)
    if not table.empty:
        table = table.drop_duplicates(["decision_id"], keep="first")
        table = table.sort_values(["plan_trade_date", "published_at"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return StrategyLedgerResult(table=table, summary=summarize_strategy(table))


def strategy_validation_rows(ledger: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    if ledger is None or ledger.empty or validation is None or validation.empty:
        return pd.DataFrame(columns=validation.columns if validation is not None else [])
    buys = ledger[ledger["action"].fillna("").astype(str).eq("BUY")].copy()
    if buys.empty:
        return validation.iloc[0:0].copy()
    key_set = {
        (str(row.plan_trade_date).replace("-", ""), str(row.ts_code))
        for row in buys.itertuples()
    }
    view = validation.copy()
    keys = list(
        zip(
            _series(view, "plan_trade_date").fillna("").astype(str).str.replace("-", "", regex=False),
            _series(view, "ts_code").fillna("").astype(str),
        )
    )
    view = view[[key in key_set for key in keys]].copy()
    if view.empty:
        return view
    view["_plan_time"] = pd.to_datetime(_series(view, "plan_time"), errors="coerce")
    view = view.sort_values("_plan_time").drop_duplicates(["plan_trade_date", "ts_code"], keep="last")
    return view.drop(columns=["_plan_time"])
