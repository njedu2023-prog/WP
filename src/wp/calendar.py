from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo


CN_TZ = ZoneInfo("Asia/Shanghai")

A_SHARE_HOLIDAYS = {
    # 2026 mainland China exchange holidays. Weekends remain closed even if
    # they are working days for ordinary offices.
    "20260101",
    "20260216", "20260217", "20260218", "20260219", "20260220",
    "20260406",
    "20260501", "20260504", "20260505",
    "20260619",
    "20260925",
    "20261001", "20261002", "20261005", "20261006", "20261007", "20261008",
}


def now_cn() -> datetime:
    return datetime.now(CN_TZ)


def is_a_share_trading_day(dt: datetime | None = None) -> bool:
    current = dt or now_cn()
    return current.weekday() < 5 and current.strftime("%Y%m%d") not in A_SHARE_HOLIDAYS


def is_trading_time(dt: datetime | None = None) -> bool:
    current = dt or now_cn()
    if not is_a_share_trading_day(current):
        return False
    current_time = current.time()
    return time(9, 25) <= current_time <= time(11, 35) or time(12, 55) <= current_time <= time(15, 10)


def trade_date_str(dt: datetime | None = None) -> str:
    return (dt or now_cn()).strftime("%Y%m%d")


def next_trading_day_str(trade_date: str) -> str:
    current = datetime.strptime(str(trade_date), "%Y%m%d").replace(tzinfo=CN_TZ)
    while True:
        current = current + timedelta(days=1)
        if is_a_share_trading_day(current):
            return current.strftime("%Y%m%d")
