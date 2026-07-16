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
