from scripts.check_upstream_revision import resolve_decision


def test_schedule_skips_revision_already_processed():
    upstream = {"status": "ok", "generated_at": "2026-07-16 14:35:20"}
    local = {"source_generated_at": "2026-07-16 14:35:20"}

    should_run, reason = resolve_decision("schedule", upstream, local)

    assert should_run is False
    assert "already processed" in reason


def test_schedule_runs_when_upstream_revision_changes():
    upstream = {"status": "ok", "generated_at": "2026-07-16 14:45:18"}
    local = {"source_generated_at": "2026-07-16 14:35:20"}

    should_run, reason = resolve_decision("schedule", upstream, local)

    assert should_run is True
    assert "new upstream revision" in reason


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
