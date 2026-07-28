from datetime import datetime
from zoneinfo import ZoneInfo

from scripts.wp_session_schedule import latest_due_slot, scheduled_slots


CN_TZ = ZoneInfo("Asia/Shanghai")


def test_tail_schedule_has_seven_signal_slots_then_freeze_and_close():
    slots = scheduled_slots(datetime(2026, 7, 27).date())

    assert [slot.strftime("%H:%M") for slot in slots] == [
        "14:20",
        "14:25",
        "14:30",
        "14:35",
        "14:40",
        "14:45",
        "14:50",
        "14:55",
        "15:00",
    ]


def test_latest_due_slot_stays_inside_tail_window():
    assert (
        latest_due_slot(datetime(2026, 7, 27, 14, 46, tzinfo=CN_TZ))
        .strftime("%H:%M")
        == "14:45"
    )
    assert (
        latest_due_slot(datetime(2026, 7, 27, 15, 1, tzinfo=CN_TZ))
        .strftime("%H:%M")
        == "15:00"
    )
    assert latest_due_slot(datetime(2026, 7, 27, 11, 46, tzinfo=CN_TZ)) is None
    assert latest_due_slot(datetime(2026, 7, 27, 15, 3, tzinfo=CN_TZ)) is None
