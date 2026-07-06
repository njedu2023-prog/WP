from __future__ import annotations

from datetime import datetime

import pandas as pd

from .calendar import is_a_share_trading_day, is_trading_time


def build_healthcheck(
    raw: pd.DataFrame,
    candidates: pd.DataFrame,
    top50: pd.DataFrame,
    load_ok: bool,
    load_error: str,
    fallback_used: bool,
    update_time: str,
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
    status = "ok"
    if not load_ok:
        status = "数据异常"
    elif missing:
        status = "数据不完整"
    elif candidates.empty:
        status = "无符合条件股票"
    return {
        "status": status,
        "is_trading_day": is_a_share_trading_day(),
        "is_trading_time": is_trading_time(),
        "data_time": update_time,
        "raw_count": int(len(raw)),
        "candidate_count": int(len(candidates)),
        "top50_count": int(len(top50)),
        "missing_fields": missing,
        "fallback_used": bool(fallback_used),
        "load_ok": bool(load_ok),
        "load_error": load_error,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def assert_top50_rules(top50: pd.DataFrame) -> list[str]:
    errors = []
    if top50.empty:
        return errors
    if (top50["pct_chg"].astype(float) <= 6).any():
        errors.append("Top50 contains pct_chg <= 6")
    if (top50["pre_day_limitup"].astype(int) == 1).any():
        errors.append("Top50 contains previous-day limit-up stocks")
    if (top50["today_limitup"].astype(int) == 1).any():
        errors.append("Top50 contains today limit-up stocks")
    return errors
