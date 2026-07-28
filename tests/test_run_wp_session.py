from types import SimpleNamespace
from datetime import datetime

from scripts import run_wp_session


def test_auto_repair_before_final_slot_runs_continuous_session(monkeypatch):
    calls = []
    monkeypatch.setenv("WP_RUN_MODE", "auto")
    monkeypatch.setenv("WP_MODE", "live")
    monkeypatch.setattr(
        run_wp_session,
        "now_cn",
        lambda: datetime(
            2026,
            7,
            27,
            14,
            23,
            tzinfo=run_wp_session.CN_TZ,
        ),
    )
    monkeypatch.setattr(
        run_wp_session,
        "run_session",
        lambda: calls.append("run_session"),
    )
    monkeypatch.setattr(
        run_wp_session,
        "run_once_if_due",
        lambda: calls.append("run_once_if_due"),
    )

    run_wp_session.main()

    assert calls == ["run_session"]


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


def test_1450_signal_is_built_at_1452_after_completed_bar_grace(
    monkeypatch,
    tmp_path,
):
    captured = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        run_wp_session,
        "now_cn",
        lambda: datetime(
            2026,
            7,
            27,
            14,
            52,
            tzinfo=run_wp_session.CN_TZ,
        ),
    )
    source_path = (
        tmp_path / "data" / "v3" / "latest" / "wp_v3_live_features.csv"
    )
    source_path.parent.mkdir(parents=True)
    source_path.write_text("ts_code\n600001.SH\n", encoding="utf-8")

    def fake_build_live_input(**kwargs):
        captured["built"] = True
        return source_path, {
            "trade_date": "20260727",
            "signal_slot": "14:50",
        }

    def fake_run(command, **kwargs):
        if command[:3] == [run_wp_session.sys.executable, "-m", "wp.main"]:
            manifest = tmp_path / "outputs" / "json" / "wp_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                '{"report_revision":"1450"}\n',
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        run_wp_session,
        "build_live_input",
        fake_build_live_input,
    )
    monkeypatch.setattr(run_wp_session.subprocess, "run", fake_run)

    run_wp_session.run_once("14:50")

    assert captured["built"] is True


def test_signal_iteration_can_settle_previous_candidates_and_score_new_slot(
    monkeypatch,
    tmp_path,
):
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        run_wp_session,
        "now_cn",
        lambda: datetime(
            2026,
            7,
            27,
            14,
            27,
            tzinfo=run_wp_session.CN_TZ,
        ),
    )
    live_path = (
        tmp_path / "data" / "v3" / "latest" / "wp_v3_live_features.csv"
    )
    live_path.parent.mkdir(parents=True)
    live_path.write_text("ts_code\n600001.SH\n", encoding="utf-8")
    settlement_path = live_path.with_name("wp_v3_entry_settlement.csv")
    settlement_path.write_text("ts_code\n600001.SH\n", encoding="utf-8")

    monkeypatch.setattr(
        run_wp_session,
        "capture_entry_settlement_input",
        lambda **kwargs: (
            calls.append(("settle", kwargs["settlement_slot"]))
            or (
                settlement_path,
                {
                    "trade_date": "20260727",
                    "requested_symbols": 1,
                    "observed_symbols": 1,
                },
            )
        ),
    )
    monkeypatch.setattr(
        run_wp_session,
        "build_live_input",
        lambda **kwargs: (
            calls.append(("signal", "14:25"))
            or (
                live_path,
                {
                    "trade_date": "20260727",
                    "signal_slot": "14:25",
                },
            )
        ),
    )

    def fake_run(command, **kwargs):
        if command[:3] == [run_wp_session.sys.executable, "-m", "wp.main"]:
            manifest = tmp_path / "outputs" / "json" / "wp_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                '{"report_revision":"1425"}\n',
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_wp_session.subprocess, "run", fake_run)

    run_wp_session.run_once("14:25", settlement_slot="14:25")

    assert calls == [("settle", "14:25"), ("signal", "14:25")]
