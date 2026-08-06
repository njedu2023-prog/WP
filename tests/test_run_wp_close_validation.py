import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from scripts import run_wp_close_validation


def test_v3_close_validation_commits_only_when_truth_state_changes(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    ledger = tmp_path / "outputs" / "json" / "wp_v3_candidate_ledger.json"
    registry = tmp_path / "outputs" / "json" / "wp_model_registry_v3.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"sessions":[]}\n', encoding="utf-8")
    registry.write_text('{"models":[]}\n', encoding="utf-8")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[:3] == [
            run_wp_close_validation.sys.executable,
            "-m",
            "wp.close_validation",
        ]:
            ledger.write_text(
                '{"sessions":[{"truth_status":"verified"}]}\n',
                encoding="utf-8",
            )
            payload = ledger.with_name("wp_buy_plan_validation.json")
            payload.write_text(
                json.dumps(
                    {
                        "summary": {
                            "pending_count": 3,
                            "pending_due_count": 0,
                        }
                    }
                ),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_wp_close_validation.subprocess, "run", fake_run)

    assert run_wp_close_validation.run_once() == 0
    assert len(commands) == 2
    assert "Validate WP V41 next-day close" in commands[1]


def test_delayed_scheduled_validation_still_runs_once(monkeypatch):
    calls = []
    current = datetime(
        2026,
        7,
        28,
        17,
        43,
        tzinfo=run_wp_close_validation.now_cn().tzinfo,
    )
    monkeypatch.setenv("WP_CLOSE_RUN_MODE", "session")
    monkeypatch.setenv("WP_CLOSE_END_TIME", "16:10")
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setattr(run_wp_close_validation, "now_cn", lambda: current)
    monkeypatch.setattr(
        run_wp_close_validation,
        "run_once",
        lambda: calls.append("run") or 0,
    )

    run_wp_close_validation.main()

    assert calls == ["run"]


def test_delayed_scheduled_validation_fails_only_after_attempt(monkeypatch):
    calls = []
    current = datetime(
        2026,
        7,
        28,
        17,
        43,
        tzinfo=run_wp_close_validation.now_cn().tzinfo,
    )
    monkeypatch.setenv("WP_CLOSE_RUN_MODE", "session")
    monkeypatch.setenv("WP_CLOSE_END_TIME", "16:10")
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setattr(run_wp_close_validation, "now_cn", lambda: current)
    monkeypatch.setattr(
        run_wp_close_validation,
        "run_once",
        lambda: calls.append("run") or 2,
    )

    with pytest.raises(SystemExit, match="2 due record"):
        run_wp_close_validation.main()

    assert calls == ["run"]


def test_close_session_retries_due_truth_until_verified(monkeypatch):
    calls = []
    current = datetime(
        2026,
        8,
        6,
        16,
        7,
        tzinfo=run_wp_close_validation.now_cn().tzinfo,
    )
    clock = iter(
        [
            current,
            current,
            current,
            current,
            current,
        ]
    )
    pending = iter([5, 0])
    monkeypatch.setenv("WP_CLOSE_RUN_MODE", "session")
    monkeypatch.setenv("WP_CLOSE_END_TIME", "19:00")
    monkeypatch.setenv("WP_CLOSE_INTERVAL_SECONDS", "1")
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setattr(
        run_wp_close_validation,
        "now_cn",
        lambda: next(clock),
    )
    monkeypatch.setattr(
        run_wp_close_validation,
        "run_once",
        lambda: calls.append("run") or next(pending),
    )
    monkeypatch.setattr(
        run_wp_close_validation.time_module,
        "sleep",
        lambda _: None,
    )

    run_wp_close_validation.main()

    assert calls == ["run", "run"]


def test_close_session_end_time_rejects_invalid_value(monkeypatch):
    current = datetime(
        2026,
        8,
        6,
        16,
        7,
        tzinfo=run_wp_close_validation.now_cn().tzinfo,
    )
    monkeypatch.setenv("WP_CLOSE_END_TIME", "invalid")

    with pytest.raises(ValueError, match="must use HH:MM"):
        run_wp_close_validation._session_end(current)
