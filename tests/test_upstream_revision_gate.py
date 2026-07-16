from datetime import datetime
from zoneinfo import ZoneInfo

from scripts.check_upstream_revision import resolve_decision


CN_TZ = ZoneInfo("Asia/Shanghai")


def test_schedule_skips_target_slot_already_processed():
    upstream = {"status": "ok", "generated_at": "2026-07-16 14:35:20"}
    local = {
        "source_trade_date": "20260716",
        "source_scheduled_slot": "2026-07-16 14:35:00",
        "source_generated_at": "2026-07-16 14:35:20",
    }

    should_run, reason = resolve_decision(
        "schedule", upstream, local, datetime(2026, 7, 16, 14, 39, tzinfo=CN_TZ)
    )

    assert should_run is False
    assert "already covered" in reason


def test_schedule_runs_for_due_slot_without_waiting_for_upstream():
    upstream = {"status": "stale_data", "generated_at": "2026-07-16 14:35:18"}
    local = {
        "source_trade_date": "20260716",
        "source_scheduled_slot": "2026-07-16 14:35:00",
    }

    should_run, reason = resolve_decision(
        "schedule", upstream, local, datetime(2026, 7, 16, 14, 46, tzinfo=CN_TZ)
    )

    assert should_run is True
    assert "14:45:00" in reason


def test_repository_dispatch_does_not_process_bad_upstream_status():
    upstream = {"status": "stale_data", "generated_at": "2026-07-15 15:10:00"}

    should_run, reason = resolve_decision("repository_dispatch", upstream, {})

    assert should_run is False
    assert "upstream status" in reason


def test_manual_and_code_push_runs_are_explicit():
    for event_name in ("push", "workflow_dispatch"):
        should_run, reason = resolve_decision(event_name, {}, {})
        assert should_run is True
        assert event_name in reason


def test_schedule_skips_outside_market_window():
    should_run, reason = resolve_decision(
        "schedule", {}, {}, datetime(2026, 7, 16, 12, 10, tzinfo=CN_TZ)
    )

    assert should_run is False
    assert "outside" in reason
