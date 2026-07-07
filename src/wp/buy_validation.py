from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path

import pandas as pd

from .calendar import CN_TZ, next_trading_day_str


VALIDATION_COLUMNS = [
    "plan_trade_date",
    "plan_time",
    "market_data_time",
    "target_trade_date",
    "buy_rank",
    "portfolio_group",
    "ts_code",
    "name",
    "pct_chg_plan",
    "sector_name",
    "p_limitup_t1",
    "wp_score",
    "decision_score",
    "risk_penalty_score",
    "actual_trade_date",
    "actual_close",
    "actual_pct_chg",
    "is_limit_up_t1",
    "truth_status",
    "truth_error",
    "truth_updated_at",
]


@dataclass
class BuyValidationResult:
    table: pd.DataFrame
    summary: dict


def _parse_dt(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    result = parsed.to_pydatetime()
    return result if result.tzinfo else result.replace(tzinfo=CN_TZ)


def _in_tail_window(value: object) -> bool:
    parsed = _parse_dt(value)
    if parsed is None:
        return False
    return time(14, 20) <= parsed.time() <= time(14, 50)


def _limit_threshold(ts_code: str) -> float:
    code = str(ts_code).split(".", 1)[0]
    if str(ts_code).endswith(".BJ") or code.startswith(("8", "9")):
        return 29.5
    if code.startswith(("300", "301", "688")):
        return 19.5
    return 9.5


def _is_truth_due(target_trade_date: str, current: datetime) -> bool:
    today = current.strftime("%Y%m%d")
    if target_trade_date < today:
        return True
    return target_trade_date == today and current.time() >= time(15, 5)


def _existing_table(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            frame = pd.read_csv(path, dtype={"plan_trade_date": str, "target_trade_date": str})
            for col in VALIDATION_COLUMNS:
                if col not in frame.columns:
                    frame[col] = ""
            return frame[VALIDATION_COLUMNS].copy()
        except Exception:
            pass
    return pd.DataFrame(columns=VALIDATION_COLUMNS)


def _new_snapshot_rows(buy_plan: pd.DataFrame, health: dict, current: datetime) -> pd.DataFrame:
    if buy_plan.empty:
        return pd.DataFrame(columns=VALIDATION_COLUMNS)
    market_data_time = str(health.get("market_data_time") or health.get("data_time") or "")
    if not _in_tail_window(market_data_time):
        return pd.DataFrame(columns=VALIDATION_COLUMNS)
    plan_trade_date = str(health.get("data_trade_date") or "").replace("-", "")
    if len(plan_trade_date) != 8:
        return pd.DataFrame(columns=VALIDATION_COLUMNS)
    target_trade_date = next_trading_day_str(plan_trade_date)
    rows = []
    for _, row in buy_plan.iterrows():
        rows.append(
            {
                "plan_trade_date": plan_trade_date,
                "plan_time": market_data_time,
                "market_data_time": market_data_time,
                "target_trade_date": target_trade_date,
                "buy_rank": row.get("buy_rank", ""),
                "portfolio_group": row.get("portfolio_group", ""),
                "ts_code": row.get("ts_code", ""),
                "name": row.get("name", ""),
                "pct_chg_plan": row.get("pct_chg", ""),
                "sector_name": row.get("sector_name", ""),
                "p_limitup_t1": row.get("p_limitup_t1", ""),
                "wp_score": row.get("wp_score", ""),
                "decision_score": row.get("decision_score", ""),
                "risk_penalty_score": row.get("risk_penalty_score", ""),
                "actual_trade_date": target_trade_date,
                "actual_close": "",
                "actual_pct_chg": "",
                "is_limit_up_t1": "",
                "truth_status": "pending",
                "truth_error": "",
                "truth_updated_at": "",
            }
        )
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def _fetch_truth_by_date(trade_date: str) -> tuple[pd.DataFrame | None, str]:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        return None, "TUSHARE_TOKEN not configured"
    try:
        import tushare as ts

        ts.set_token(token)
        pro = ts.pro_api()
        frame = pro.daily(trade_date=trade_date)
        if frame is None or frame.empty:
            return None, f"no daily truth for {trade_date}"
        return frame, ""
    except Exception as exc:
        return None, str(exc)


def _fill_truth(table: pd.DataFrame, current: datetime) -> pd.DataFrame:
    if table.empty:
        return table
    out = table.copy()
    pending = out["truth_status"].fillna("").astype(str).ne("verified")
    due_dates = sorted(
        {
            str(value)
            for value in out.loc[pending, "target_trade_date"].dropna().tolist()
            if len(str(value)) == 8 and _is_truth_due(str(value), current)
        }
    )
    if not due_dates:
        return out
    truth_cache: dict[str, tuple[pd.DataFrame | None, str]] = {}
    for trade_date in due_dates:
        truth_cache[trade_date] = _fetch_truth_by_date(trade_date)
    for idx, row in out.loc[pending].iterrows():
        target = str(row.get("target_trade_date", ""))
        if target not in truth_cache:
            continue
        truth, error = truth_cache[target]
        if truth is None:
            out.at[idx, "truth_status"] = "pending"
            out.at[idx, "truth_error"] = error
            continue
        match = truth[truth["ts_code"].astype(str).eq(str(row.get("ts_code", "")))]
        if match.empty:
            out.at[idx, "truth_status"] = "pending"
            out.at[idx, "truth_error"] = f"missing {row.get('ts_code')} on {target}"
            continue
        actual = match.iloc[0]
        pct_chg = pd.to_numeric(actual.get("pct_chg"), errors="coerce")
        out.at[idx, "actual_close"] = actual.get("close", "")
        out.at[idx, "actual_pct_chg"] = "" if pd.isna(pct_chg) else round(float(pct_chg), 4)
        out.at[idx, "is_limit_up_t1"] = bool(not pd.isna(pct_chg) and float(pct_chg) >= _limit_threshold(str(row.get("ts_code", ""))))
        out.at[idx, "truth_status"] = "verified"
        out.at[idx, "truth_error"] = ""
        out.at[idx, "truth_updated_at"] = current.strftime("%Y-%m-%d %H:%M:%S")
    return out


def _summary(table: pd.DataFrame) -> dict:
    if table.empty:
        return {"total_records": 0, "verified_records": 0, "limit_up_records": 0, "limit_up_rate": 0.0}
    verified = table[table["truth_status"].fillna("").astype(str).eq("verified")]
    limit_up = verified[verified["is_limit_up_t1"].astype(str).str.lower().isin({"true", "1", "yes"})]
    rate = round(len(limit_up) / len(verified) * 100, 2) if len(verified) else 0.0
    return {
        "total_records": int(len(table)),
        "verified_records": int(len(verified)),
        "limit_up_records": int(len(limit_up)),
        "limit_up_rate": rate,
    }


def update_buy_plan_validation(buy_plan: pd.DataFrame, health: dict, output_root: Path, current: datetime) -> BuyValidationResult:
    csv_path = output_root / "csv" / "wp_buy_plan_validation.csv"
    existing = _existing_table(csv_path)
    snapshot = _new_snapshot_rows(buy_plan, health, current)
    if not snapshot.empty:
        plan_dates = set(snapshot["plan_trade_date"].astype(str).tolist())
        locked_dates = set(
            existing.loc[
                existing["plan_trade_date"].astype(str).isin(plan_dates)
                & existing["truth_status"].fillna("").astype(str).eq("verified"),
                "plan_trade_date",
            ].astype(str).tolist()
        )
        snapshot = snapshot[~snapshot["plan_trade_date"].astype(str).isin(locked_dates)].copy()
    if not snapshot.empty:
        plan_dates = set(snapshot["plan_trade_date"].astype(str).tolist())
        # The 14:20 buy list is dynamic. Keep only the latest/final list for each
        # plan trade date until the next-day truth has been locked.
        existing = existing[
            ~(
                existing["plan_trade_date"].astype(str).isin(plan_dates)
                & existing["truth_status"].fillna("").astype(str).ne("verified")
            )
        ].copy()
        table = snapshot.copy() if existing.empty else pd.concat([existing, snapshot], ignore_index=True)
    else:
        table = existing
    if not table.empty:
        key_cols = ["plan_trade_date", "plan_time", "ts_code"]
        table = table.drop_duplicates(key_cols, keep="last")
        table["_buy_rank_sort"] = pd.to_numeric(table["buy_rank"], errors="coerce").fillna(999)
        table = table.sort_values(["plan_trade_date", "plan_time", "_buy_rank_sort"]).drop(columns=["_buy_rank_sort"])
    table = _fill_truth(table, current)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return BuyValidationResult(table=table, summary=_summary(table))
