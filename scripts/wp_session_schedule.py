from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo


CN_TZ = ZoneInfo("Asia/Shanghai")
SCHEDULE_GRACE_SECONDS = int(os.environ.get("WP_SCHEDULE_GRACE_SECONDS", "120"))
TAIL_SLOTS = (
    (14, 0),
    (14, 5),
    (14, 10),
    (15, 0),
)


def scheduled_slots(day: date) -> list[datetime]:
    return [
        datetime.combine(day, time(hour, minute), CN_TZ)
        for hour, minute in TAIL_SLOTS
    ]


def latest_due_slot(current: datetime) -> datetime | None:
    local = current.astimezone(CN_TZ)
    window_start = datetime.combine(local.date(), time(14, 0), CN_TZ)
    window_end = datetime.combine(local.date(), time(15, 0), CN_TZ)
    if not (
        window_start
        <= local
        <= window_end + timedelta(seconds=SCHEDULE_GRACE_SECONDS)
    ):
        return None
    due = [
        slot
        for slot in scheduled_slots(local.date())
        if slot + timedelta(seconds=SCHEDULE_GRACE_SECONDS) <= local
    ]
    return due[-1] if due else None
