from __future__ import annotations

from datetime import datetime
import re

import pandas as pd

from .calendar import is_a_share_trading_day, is_trading_time


def _parse_time(value) -> datetime | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    try:
        if re.fullmatch(r"\d{8}\s+\d{2}:\d{2}:\d{2}", text):
            return datetime.strptime(text, "%Y%m%d %H:%M:%S")
        if re.fullmatch(r"\d{8}\s+\d{2}:\d{2}", text):
            return datetime.strptime(text, "%Y%m%d %H:%M")
        if re.fullmatch(r"\d{14}", text):
            return datetime.strptime(text, "%Y%m%d%H%M%S")
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()
    except Exception:
        return None


def resolve_market_data_time(raw: pd.DataFrame, source_metadata: dict, fallback: str = "") -> str:
    times: list[datetime] = []
    if "update_time" in raw.columns:
        for value in raw["update_time"].dropna().tolist():
            parsed = _parse_time(value)
            if parsed is not None:
                times.append(parsed)
    if times:
        return max(times).strftime("%Y-%m-%d %H:%M:%S")
    generated_at = source_metadata.get("generated_at", "")
    parsed_generated = _parse_time(generated_at)
    if parsed_generated is not None:
        return parsed_generated.strftime("%Y-%m-%d %H:%M:%S")
    return str(generated_at or fallback)


def build_healthcheck(
    raw: pd.DataFrame,
    candidates: pd.DataFrame,
    top50: pd.DataFrame,
    load_ok: bool,
    load_error: str,
    fallback_used: bool,
    update_time: str,
    expected_trade_date: str | None = None,
    source_metadata: dict | None = None,
) -> dict:
    required = {
        "涨幅字段": ["pct_chg", "change_pct", "涨跌幅"],
        "昨日涨停字段": ["pre_day_limitup", "prev_is_limit_up", "is_limit_up_yesterday", "前一日涨停"],
        "今日涨停字段": ["today_limitup", "is_limit_up_today", "is_limit_up", "今日涨停"],
        "板块字段": ["sector_name", "industry", "板块", "所属板块"],
        "成交额字段": ["amount", "成交额", "turnover_amount"],
    }
    columns = set(raw.columns)
    missing = [name for name, choices in required.items() if not any(item in columns for item in choices)]
    data_trade_date = ""
    if "trade_date" in raw.columns:
        dates = raw["trade_date"].dropna().astype(str).str.replace("-", "", regex=False)
        dates = dates[dates.str.len() == 8]
        if not dates.empty:
            data_trade_date = str(sorted(dates.unique())[-1])
    realtime_sources: list[str] = []
    if "realtime_source" in raw.columns:
        realtime_sources = sorted(
            {
                str(value).strip()
                for value in raw["realtime_source"].dropna().tolist()
                if str(value).strip()
            }
        )
    realtime_fallback_used = any("fallback" in item.lower() for item in realtime_sources)
    source_metadata = source_metadata or {}
    market_data_time = resolve_market_data_time(raw, source_metadata, update_time)
    status = "ok"
    if source_metadata.get("status") == "stale_data":
        data_trade_date = str(source_metadata.get("source_trade_date") or data_trade_date)
        expected_trade_date = str(source_metadata.get("expected_trade_date") or expected_trade_date or "")
        status = "数据日期过期"
    if not load_ok:
        status = "数据异常"
    elif status == "数据日期过期":
        pass
    elif expected_trade_date and data_trade_date and data_trade_date != expected_trade_date:
        status = "数据日期过期"
    elif missing:
        status = "数据不完整"
    elif candidates.empty:
        status = "无符合条件股票"
    return {
        "status": status,
        "is_trading_day": is_a_share_trading_day(),
        "is_trading_time": is_trading_time(),
        "data_time": market_data_time,
        "market_data_time": market_data_time,
        "wp_run_time": update_time,
        "data_trade_date": data_trade_date,
        "expected_trade_date": expected_trade_date or "",
        "raw_count": int(len(raw)),
        "candidate_count": int(len(candidates)),
        "top50_count": int(len(top50)),
        "missing_fields": missing,
        "fallback_used": bool(fallback_used),
        "data_load_fallback_used": bool(fallback_used),
        "realtime_sources": realtime_sources,
        "realtime_fallback_used": bool(realtime_fallback_used),
        "load_ok": bool(load_ok),
        "load_error": load_error,
        "source_status": source_metadata.get("status", ""),
        "source_generated_at": source_metadata.get("generated_at", ""),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def assert_top50_rules(top50: pd.DataFrame) -> list[str]:
    errors = []
    if top50.empty:
        return errors
    if (top50["pct_chg"].astype(float) <= 8).any():
        errors.append("Top50 contains pct_chg <= 8")
    if (top50["pre_day_limitup"].astype(int) == 1).any():
        errors.append("Top50 contains previous-day limit-up stocks")
    if (top50["today_limitup"].astype(int) == 1).any():
        errors.append("Top50 contains today limit-up stocks")
    return errors
