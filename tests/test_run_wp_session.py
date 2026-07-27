from types import SimpleNamespace
from datetime import datetime

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


def test_v3_causal_source_contract_is_passed_to_core_engine(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setattr(
        run_wp_session,
        "now_cn",
        lambda: datetime(2026, 7, 27, 14, 25, tzinfo=run_wp_session.CN_TZ),
    )
    source_path = tmp_path / "data" / "v3" / "latest" / "wp_v3_live_features.csv"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("ts_code\n600001.SH\n", encoding="utf-8")
    monkeypatch.setattr(
        run_wp_session,
        "build_live_input",
        lambda **kwargs: (
            source_path,
            {
                "trade_date": "20260727",
                "signal_slot": "14:25",
            },
        ),
    )

    def fake_run(command, **kwargs):
        if command[:3] == [run_wp_session.sys.executable, "-m", "wp.main"]:
            captured.update(kwargs["env"])
            manifest = tmp_path / "outputs" / "json" / "wp_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{"report_revision":"new"}\n', encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_wp_session.subprocess, "run", fake_run)

    run_wp_session.run_once()

    assert captured["WP_EXPECTED_TRADE_DATE"] == "20260727"
    assert captured["WP_V3_SIGNAL_SLOT"] == "14:25"
    assert captured["WP_V3_SOURCE_CSV"] == source_path.as_posix()
    assert captured["WP_MODE"] == "live"
