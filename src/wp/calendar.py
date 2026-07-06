from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo


CN_TZ = ZoneInfo("Asia/Shanghai")


def now_cn() -> datetime:
    return datetime.now(CN_TZ)


def is_a_share_trading_day(dt: datetime | None = None) -> bool:
    current = dt or now_cn()
    return current.weekday() < 5


def is_trading_time(dt: datetime | None = None) -> bool:
    current = dt or now_cn()
    if not is_a_share_trading_day(current):
        return False
    current_time = current.time()
    return time(9, 25) <= current_time <= time(11, 35) or time(12, 55) <= current_time <= time(15, 10)


def trade_date_str(dt: datetime | None = None) -> str:
    return (dt or now_cn()).strftime("%Y%m%d")
