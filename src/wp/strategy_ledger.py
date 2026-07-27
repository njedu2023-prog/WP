from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .calendar import next_trading_day_str


STRATEGY_VERSION = "t1_qualified_cohort_v2"
ENTRY_CONTRACT = "first_qualified_signal_last_price"
EXIT_CONTRACT = "T+1_close"

STRATEGY_COLUMNS = [
    "strategy_version",
    "decision_id",
    "plan_trade_date",
    "plan_time",
    "first_signal_time",
    "last_signal_time",
    "published_at",
    "first_published_at",
    "last_published_at",
    "appearance_count",
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
    plan_dates = (
        _series(view, "plan_trade_date")
        .fillna("")
        .astype(str)
        .str.replace("-", "", regex=False)
    )
    codes = _series(view, "ts_code").fillna("").astype(str)
    matches = view[plan_dates.eq(plan_trade_date) & codes.eq(ts_code)].copy()
    if matches.empty:
        return None
    matches["_verified"] = (
        _series(matches, "truth_status").fillna("").astype(str).eq("verified")
    )
    matches["_plan_time"] = pd.to_datetime(_series(matches, "plan_time"), errors="coerce")
    matches = matches.sort_values(
        ["_verified", "_plan_time"],
        ascending=[False, True],
        kind="mergesort",
    )
    return matches.iloc[0]


def _candidate_row(
    row: pd.Series,
    decision: dict,
    health: dict,
    current: datetime,
    cost_pct: float,
) -> dict | None:
    plan_trade_date = str(health.get("data_trade_date") or "").replace("-", "")
    code = str(row.get("ts_code") or "").strip()
    if len(plan_trade_date) != 8 or not code:
        return None
    plan_time = str(health.get("market_data_time") or health.get("data_time") or "")
    published_at = current.strftime("%Y-%m-%d %H:%M:%S")
    target_trade_date = next_trading_day_str(plan_trade_date)
    return {
        "strategy_version": STRATEGY_VERSION,
        "decision_id": f"{STRATEGY_VERSION}:{plan_trade_date}:{code}",
        "plan_trade_date": plan_trade_date,
        "plan_time": plan_time,
        "first_signal_time": plan_time,
        "last_signal_time": plan_time,
        "published_at": published_at,
        "first_published_at": published_at,
        "last_published_at": published_at,
        "appearance_count": 1,
        "action": "QUALIFIED",
        "decision_status": "candidate_locked",
        "ts_code": code,
        "name": row.get("name", ""),
        "sector_name": row.get("sector_name", ""),
        "plan_price": row.get("price", ""),
        "entry_contract": ENTRY_CONTRACT,
        "entry_deadline": f"{plan_trade_date} 14:50:00",
        "target_trade_date": target_trade_date,
        "exit_contract": EXIT_CONTRACT,
        "forecast_profit_probability": row.get("forecast_profit_probability", ""),
        "forecast_profit_probability_lower": row.get(
            "forecast_profit_probability_lower", ""
        ),
        "forecast_expected_net_return_pct": row.get(
            "forecast_expected_net_return_pct", ""
        ),
        "forecast_live_sample_count": row.get("forecast_live_sample_count", ""),
        "forecast_live_day_count": row.get("forecast_live_day_count", ""),
        "forecast_effective_sample_count": row.get(
            "forecast_effective_sample_count", ""
        ),
        "decision_reason": str(
            row.get("decision_reason")
            or decision.get("reason")
            or "逐票通过全部固定门槛"
        ),
        "actual_trade_date": target_trade_date,
        "actual_close": "",
        "gross_return_pct": "",
        "assumed_cost_pct": cost_pct,
        "net_return_pct": "",
        "is_net_profit": "",
        "truth_status": "pending",
        "truth_error": "",
        "truth_updated_at": "",
    }


def _no_signal_row(
    decision: dict,
    health: dict,
    current: datetime,
) -> dict | None:
    if not bool(decision.get("is_final")):
        return None
    if str(decision.get("action_code") or "") not in {"FROZEN", "CLOSED", "NO_SIGNAL"}:
        return None
    plan_trade_date = str(health.get("data_trade_date") or "").replace("-", "")
    if len(plan_trade_date) != 8:
        return None
    plan_time = str(health.get("market_data_time") or health.get("data_time") or "")
    published_at = current.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "strategy_version": STRATEGY_VERSION,
        "decision_id": f"{STRATEGY_VERSION}:{plan_trade_date}:NO_SIGNAL",
        "plan_trade_date": plan_trade_date,
        "plan_time": plan_time,
        "first_signal_time": "",
        "last_signal_time": "",
        "published_at": published_at,
        "first_published_at": published_at,
        "last_published_at": published_at,
        "appearance_count": 0,
        "action": "NO_SIGNAL",
        "decision_status": "day_frozen",
        "ts_code": "",
        "name": "",
        "sector_name": "",
        "plan_price": "",
        "entry_contract": "",
        "entry_deadline": "",
        "target_trade_date": "",
        "exit_contract": "",
        "forecast_profit_probability": "",
        "forecast_profit_probability_lower": "",
        "forecast_expected_net_return_pct": "",
        "forecast_live_sample_count": "",
        "forecast_live_day_count": "",
        "forecast_effective_sample_count": "",
        "decision_reason": str(
            decision.get("reason") or "14:20-14:50没有股票通过全部固定门槛"
        ),
        "actual_trade_date": "",
        "actual_close": "",
        "gross_return_pct": "",
        "assumed_cost_pct": "",
        "net_return_pct": "",
        "is_net_profit": "",
        "truth_status": "not_applicable",
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
    qualified_mask = (
        out["strategy_version"].fillna("").astype(str).eq(STRATEGY_VERSION)
        & out["action"].fillna("").astype(str).eq("QUALIFIED")
    )
    for idx, row in out.loc[qualified_mask].iterrows():
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
        out.at[idx, "actual_trade_date"] = truth.get(
            "actual_trade_date", row.get("target_trade_date", "")
        )
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
    dates = (
        _series(table, "plan_trade_date")
        .fillna("")
        .astype(str)
        .str.replace("-", "", regex=False)
    )
    rows = table[
        dates.eq(normalized)
        & _series(table, "strategy_version").fillna("").astype(str).eq(STRATEGY_VERSION)
        & _series(table, "action").fillna("").astype(str).eq("QUALIFIED")
    ].copy()
    if rows.empty:
        return None
    rows["_first_signal"] = pd.to_datetime(
        _series(rows, "first_signal_time"), errors="coerce"
    )
    return (
        rows.sort_values(["_first_signal", "ts_code"], kind="mergesort")
        .iloc[0]
        .drop(labels=["_first_signal"])
        .to_dict()
    )


def summarize_strategy(table: pd.DataFrame) -> dict:
    empty = {
        "metric_scope": "qualified_candidate_cohort",
        "strategy_version": STRATEGY_VERSION,
        "objective": "maximize_probability_of_positive_T1_net_return_for_each_qualified_candidate",
        "entry_contract": ENTRY_CONTRACT,
        "exit_contract": EXIT_CONTRACT,
        "candidate_days": 0,
        "decision_days": 0,
        "signal_count": 0,
        "trade_days": 0,
        "no_signal_days": 0,
        "no_trade_days": 0,
        "verified_signals": 0,
        "verified_trades": 0,
        "pending_signals": 0,
        "pending_trades": 0,
        "net_profit_signals": 0,
        "net_profit_trades": 0,
        "net_win_rate": 0.0,
        "average_net_return_pct": 0.0,
        "median_net_return_pct": 0.0,
        "verified_candidate_days": 0,
        "positive_equal_weight_days": 0,
        "equal_weight_day_win_rate": 0.0,
        "average_equal_weight_day_return_pct": 0.0,
        "cumulative_equal_weight_day_return_pct": 0.0,
        "cumulative_net_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
    }
    if table is None or table.empty:
        return empty
    scoped = table[
        _series(table, "strategy_version")
        .fillna("")
        .astype(str)
        .eq(STRATEGY_VERSION)
    ].copy()
    if scoped.empty:
        return empty
    candidates = scoped[scoped["action"].fillna("").astype(str).eq("QUALIFIED")].copy()
    verified = candidates[
        candidates["truth_status"].fillna("").astype(str).eq("verified")
    ].copy()
    verified["_net"] = pd.to_numeric(
        _series(verified, "net_return_pct"), errors="coerce"
    )
    net = verified["_net"].dropna()

    complete_day_returns: list[float] = []
    for _, day in candidates.groupby("plan_trade_date", sort=True):
        day_net = pd.to_numeric(
            day.loc[
                day["truth_status"].fillna("").astype(str).eq("verified"),
                "net_return_pct",
            ],
            errors="coerce",
        ).dropna()
        if len(day) and len(day_net) == len(day):
            complete_day_returns.append(float(day_net.mean()))
    daily = pd.Series(complete_day_returns, dtype="float64")
    equity = (1 + daily / 100).cumprod() if len(daily) else pd.Series(dtype="float64")
    drawdown = (
        (equity / equity.cummax() - 1) * 100
        if len(equity)
        else pd.Series(dtype="float64")
    )
    candidate_days = int(candidates["plan_trade_date"].nunique())
    no_signal_days = int(
        scoped[scoped["action"].fillna("").astype(str).eq("NO_SIGNAL")][
            "plan_trade_date"
        ].nunique()
    )
    cumulative = (
        round(float((equity.iloc[-1] - 1) * 100), 4) if len(equity) else 0.0
    )
    return {
        **empty,
        "candidate_days": candidate_days,
        "decision_days": int(scoped["plan_trade_date"].nunique()),
        "signal_count": int(len(candidates)),
        "trade_days": candidate_days,
        "no_signal_days": no_signal_days,
        "no_trade_days": no_signal_days,
        "verified_signals": int(len(net)),
        "verified_trades": int(len(net)),
        "pending_signals": int(len(candidates) - len(net)),
        "pending_trades": int(len(candidates) - len(net)),
        "net_profit_signals": int(net.gt(0).sum()),
        "net_profit_trades": int(net.gt(0).sum()),
        "net_win_rate": round(float(net.gt(0).mean() * 100), 2) if len(net) else 0.0,
        "average_net_return_pct": round(float(net.mean()), 4) if len(net) else 0.0,
        "median_net_return_pct": round(float(net.median()), 4) if len(net) else 0.0,
        "verified_candidate_days": int(len(daily)),
        "positive_equal_weight_days": int(daily.gt(0).sum()),
        "equal_weight_day_win_rate": round(float(daily.gt(0).mean() * 100), 2)
        if len(daily)
        else 0.0,
        "average_equal_weight_day_return_pct": round(float(daily.mean()), 4)
        if len(daily)
        else 0.0,
        "cumulative_equal_weight_day_return_pct": cumulative,
        "cumulative_net_return_pct": cumulative,
        "max_drawdown_pct": round(float(drawdown.min()), 4)
        if len(drawdown)
        else 0.0,
    }


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
    candidates = buy_plan.copy() if buy_plan is not None else pd.DataFrame()
    for _, candidate in candidates.iterrows():
        new_row = _candidate_row(candidate, decision, health, current, cost_pct)
        if new_row is None:
            continue
        decision_id = str(new_row["decision_id"])
        matches = table["decision_id"].fillna("").astype(str).eq(decision_id)
        if matches.any():
            idx = table.index[matches][0]
            count = _number(table.at[idx, "appearance_count"]) or 1
            table.at[idx, "appearance_count"] = int(count) + 1
            table.at[idx, "last_signal_time"] = new_row["last_signal_time"]
            table.at[idx, "last_published_at"] = new_row["last_published_at"]
        else:
            frame = pd.DataFrame([new_row], columns=STRATEGY_COLUMNS)
            table = frame if table.empty else pd.concat([table, frame], ignore_index=True)

    plan_trade_date = str(health.get("data_trade_date") or "").replace("-", "")
    if len(plan_trade_date) == 8:
        current_version = table[
            table["strategy_version"].fillna("").astype(str).eq(STRATEGY_VERSION)
            & table["plan_trade_date"].fillna("").astype(str).eq(plan_trade_date)
        ]
        has_candidate = current_version["action"].fillna("").astype(str).eq(
            "QUALIFIED"
        ).any()
        if has_candidate:
            no_signal = (
                table["strategy_version"].fillna("").astype(str).eq(STRATEGY_VERSION)
                & table["plan_trade_date"].fillna("").astype(str).eq(plan_trade_date)
                & table["action"].fillna("").astype(str).eq("NO_SIGNAL")
            )
            table = table[~no_signal].copy()
        else:
            no_signal_row = _no_signal_row(decision, health, current)
            if no_signal_row is not None:
                decision_id = str(no_signal_row["decision_id"])
                if not table["decision_id"].fillna("").astype(str).eq(decision_id).any():
                    frame = pd.DataFrame([no_signal_row], columns=STRATEGY_COLUMNS)
                    table = (
                        frame
                        if table.empty
                        else pd.concat([table, frame], ignore_index=True)
                    )

    table = _fill_truth(table, validation, current, cost_pct)
    if not table.empty:
        table = table.drop_duplicates(["decision_id"], keep="first")
        table = table.sort_values(
            ["plan_trade_date", "plan_time", "ts_code"], kind="mergesort"
        ).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return StrategyLedgerResult(table=table, summary=summarize_strategy(table))


def strategy_validation_rows(
    ledger: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    if ledger is None or ledger.empty or validation is None or validation.empty:
        return pd.DataFrame(
            columns=validation.columns if validation is not None else []
        )
    actions = ledger["action"].fillna("").astype(str)
    candidates = ledger[actions.isin({"QUALIFIED", "BUY"})].copy()
    if candidates.empty:
        return validation.iloc[0:0].copy()
    key_set = {
        (str(row.plan_trade_date).replace("-", ""), str(row.ts_code))
        for row in candidates.itertuples()
    }
    view = validation.copy()
    keys = list(
        zip(
            _series(view, "plan_trade_date")
            .fillna("")
            .astype(str)
            .str.replace("-", "", regex=False),
            _series(view, "ts_code").fillna("").astype(str),
        )
    )
    view = view[[key in key_set for key in keys]].copy()
    if view.empty:
        return view
    view["_plan_time"] = pd.to_datetime(_series(view, "plan_time"), errors="coerce")
    view = view.sort_values("_plan_time", kind="mergesort").drop_duplicates(
        ["plan_trade_date", "ts_code"], keep="first"
    )
    candidate_by_key = {
        (str(row.plan_trade_date).replace("-", ""), str(row.ts_code)): row
        for row in candidates.itertuples()
    }
    for idx, row in view.iterrows():
        key = (
            str(row.get("plan_trade_date") or "").replace("-", ""),
            str(row.get("ts_code") or ""),
        )
        candidate = candidate_by_key.get(key)
        if candidate is None:
            continue
        view.at[idx, "plan_time"] = (
            getattr(candidate, "first_signal_time", "")
            or getattr(candidate, "plan_time", "")
        )
        view.at[idx, "market_data_time"] = view.at[idx, "plan_time"]
        view.at[idx, "plan_price"] = getattr(candidate, "plan_price", "")
        view.at[idx, "target_trade_date"] = getattr(
            candidate, "target_trade_date", row.get("target_trade_date", "")
        )
        if "portfolio_group" in view.columns:
            view.at[idx, "portfolio_group"] = "合格候选"
    return view.drop(columns=["_plan_time"])
