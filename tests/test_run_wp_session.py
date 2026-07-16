from types import SimpleNamespace

from scripts import run_wp_session


def test_push_uses_the_approved_direct_run(monkeypatch):
    calls = []
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("WP_MODE", "live")
    monkeypatch.setattr(run_wp_session, "run_once", lambda: calls.append("run_once"))
    monkeypatch.setattr(
        run_wp_session,
        "run_once_if_due",
        lambda: calls.append("run_once_if_due"),
    )

    run_wp_session.main()

    assert calls == ["run_once"]


def test_direct_source_date_is_passed_to_core_engine(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WP_MODE", "live")
    monkeypatch.setenv("WP_DIRECT_SOURCE_ENABLED", "1")
    monkeypatch.delenv("WP_SOURCE_CSV", raising=False)
    monkeypatch.setattr(
        run_wp_session,
        "build_direct_rank_input",
        lambda **kwargs: SimpleNamespace(
            attempted=True,
            ok=True,
            source_path="data/direct/latest/wp_latest_rank_input.csv",
            manifest_path="data/direct/latest/wp_manifest.json",
            source_trade_date="20260716",
            error="",
        ),
    )

    def fake_run(command, **kwargs):
        if command[:3] == [run_wp_session.sys.executable, "-m", "wp.main"]:
            captured.update(kwargs["env"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_wp_session.subprocess, "run", fake_run)

    run_wp_session.run_once()

    assert captured["WP_EXPECTED_TRADE_DATE"] == "20260716"
