from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from scripts import monitor_wp_system as monitor


CN_TZ = ZoneInfo("Asia/Shanghai")


def _manifest(slot: str = "2026-07-16 14:35:00") -> dict:
    return {
        "health_status": "ok",
        "source_mode": "direct_tushare",
        "source_trade_date": "20260716",
        "source_scheduled_slot": slot,
        "source_generated_at": "2026-07-16 14:37:00",
        "market_data_time": "2026-07-16 14:35:00",
        "wp_run_time": "2026-07-16 14:38:00",
        "report_revision": "2026-07-16 14:38:00",
        "direct_fallback_used": False,
    }


def test_direct_manifest_covers_due_slot():
    slot = datetime(2026, 7, 16, 14, 35, tzinfo=CN_TZ)

    ok, reasons = monitor.wp_health(_manifest(), slot)

    assert ok is True
    assert reasons == []


def test_monitor_rejects_legacy_fallback_even_when_timestamp_is_fresh():
    slot = datetime(2026, 7, 16, 14, 35, tzinfo=CN_TZ)
    manifest = _manifest()
    manifest["source_mode"] = "upstream_fallback"
    manifest["direct_fallback_used"] = True

    ok, reasons = monitor.wp_health(manifest, slot)

    assert ok is False
    assert "source_mode=upstream_fallback" in reasons
    assert "direct_fallback_used=true" in reasons


def test_monitor_dispatches_direct_repair_for_missed_slot(monkeypatch):
    current = datetime(2026, 7, 16, 14, 37, tzinfo=CN_TZ)
    stale = _manifest("2026-07-16 14:25:00")
    dispatched = []
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("TUSHARE_TOKEN", "tushare")
    monkeypatch.setattr(monitor, "is_trade_day", lambda *args: True)
    monkeypatch.setattr(monitor, "read_github_file", lambda *args, **kwargs: stale)
    monkeypatch.setattr(monitor, "workflow_runs", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        monitor,
        "dispatch_workflow",
        lambda repo, workflow, token: dispatched.append((repo, workflow, token)),
    )

    result = monitor.monitor(current)

    assert result == 0
    assert dispatched == [(monitor.WP_REPO, monitor.WP_WORKFLOW, "token")]


def test_monitor_accepts_matching_pages_revision(monkeypatch):
    current = datetime(2026, 7, 16, 14, 37, tzinfo=CN_TZ)
    manifest = _manifest()
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("TUSHARE_TOKEN", "tushare")
    monkeypatch.setattr(monitor, "is_trade_day", lambda *args: True)
    monkeypatch.setattr(monitor, "read_github_file", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(monitor, "workflow_runs", lambda *args, **kwargs: [])
    monkeypatch.setattr(monitor, "read_public_json", lambda *args: manifest)

    assert monitor.monitor(current) == 0
