from types import SimpleNamespace
from datetime import datetime, timedelta

import pytest

from scripts import run_wp_session


def test_auto_start_before_warmup_runs_continuous_session(monkeypatch):
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
            13,
            0,
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


def test_auto_start_has_no_signal_capture_deadline(monkeypatch):
    calls = []
    monkeypatch.setenv("WP_RUN_MODE", "auto")
    monkeypatch.setattr(
        run_wp_session,
        "now_cn",
        lambda: datetime(
            2026,
            8,
            6,
            14,
            45,
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


def test_exact_session_runs_cutoff_publish_entry_freeze_and_close(
    monkeypatch,
    tmp_path,
):
    calls = []
    clock = {
        "now": datetime(
            2026,
            8,
            5,
            13,
            54,
            tzinfo=run_wp_session.CN_TZ,
        )
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setattr(
        run_wp_session,
        "now_cn",
        lambda: clock["now"],
    )
    monkeypatch.setattr(
        run_wp_session.time_module,
        "sleep",
        lambda seconds: clock.__setitem__(
            "now",
            clock["now"] + timedelta(seconds=seconds),
        ),
    )
    monkeypatch.setattr(
        run_wp_session,
        "_has_fixed_signal_session",
        lambda trade_date: True,
    )
    monkeypatch.setattr(
        run_wp_session,
        "_assert_required_daily_list",
        lambda trade_date: None,
    )
    monkeypatch.setattr(
        run_wp_session,
        "run_once",
        lambda signal_slot=None, **kwargs: calls.append(
            (signal_slot, kwargs)
        ),
    )

    run_wp_session.run_session()

    assert calls == [
        ("14:00", {}),
        (None, {"settlement_slot": "14:05"}),
        (None, {}),
        (None, {}),
    ]


def test_exact_session_prewarms_compact_references_before_signal(
    monkeypatch,
    tmp_path,
):
    events = []
    clock = {
        "now": datetime(
            2026,
            8,
            5,
            13,
            0,
            tzinfo=run_wp_session.CN_TZ,
        )
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    monkeypatch.setattr(run_wp_session, "now_cn", lambda: clock["now"])
    monkeypatch.setattr(run_wp_session, "is_trade_day", lambda *_: True)
    monkeypatch.setattr(
        run_wp_session,
        "warm_live_reference_input",
        lambda **_: events.append(("warm", clock["now"])) or {
            "prior_trade_date_start": "20260617",
            "prior_trade_date_end": "20260804",
            "prior_trade_date_count": 35,
            "stock_basic_rows": 5_000,
        },
    )
    monkeypatch.setattr(
        run_wp_session,
        "capture_warmup_input",
        lambda observation_slot, **_: events.append(
            ("warmup", observation_slot)
        )
        or {"row_count": 3_000},
    )
    monkeypatch.setattr(
        run_wp_session.time_module,
        "sleep",
        lambda seconds: clock.__setitem__(
            "now",
            clock["now"] + timedelta(seconds=seconds),
        ),
    )
    monkeypatch.setattr(
        run_wp_session,
        "_has_fixed_signal_session",
        lambda trade_date: True,
    )
    monkeypatch.setattr(
        run_wp_session,
        "_assert_required_daily_list",
        lambda trade_date: None,
    )
    monkeypatch.setattr(
        run_wp_session,
        "run_once",
        lambda signal_slot=None, **kwargs: events.append(
            (signal_slot, kwargs)
        ),
    )

    run_wp_session.run_session()

    assert events[0][0] == "warm"
    assert events[1:6] == [
        ("warmup", "13:30"),
        ("warmup", "13:35"),
        ("warmup", "13:40"),
        ("warmup", "13:45"),
        ("warmup", "13:50"),
    ]
    assert events[6:] == [
        ("14:00", {}),
        (None, {"settlement_slot": "14:05"}),
        (None, {}),
        (None, {}),
    ]


def test_exact_session_completes_delayed_signal_as_recovery(
    monkeypatch,
    tmp_path,
):
    calls = []
    clock = {
        "now": datetime(
            2026,
            8,
            5,
            14,
            3,
            tzinfo=run_wp_session.CN_TZ,
        )
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setattr(
        run_wp_session,
        "now_cn",
        lambda: clock["now"],
    )
    monkeypatch.setattr(
        run_wp_session.time_module,
        "sleep",
        lambda seconds: clock.__setitem__(
            "now",
            clock["now"] + timedelta(seconds=seconds),
        ),
    )
    monkeypatch.setattr(
        run_wp_session,
        "_assert_required_daily_list",
        lambda trade_date: calls.append(("verified", trade_date)),
    )
    monkeypatch.setattr(
        run_wp_session,
        "_has_fixed_signal_session",
        lambda trade_date: True,
    )
    monkeypatch.setattr(
        run_wp_session,
        "run_once",
        lambda signal_slot=None, **kwargs: calls.append(
            (signal_slot, kwargs)
        ),
    )

    run_wp_session.run_session()

    assert calls[:5] == [
        ("14:00", {"late_recovery": True}),
        ("verified", "20260805"),
        (
            None,
            {
                "settlement_slot": "14:05",
                "late_recovery": True,
            },
        ),
        ("verified", "20260805"),
        (None, {}),
    ]


def test_required_daily_list_rejects_observation_shortfall(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        run_wp_session,
        "load_shadow_ledger",
        lambda path: {
            "sessions": [
                {
                    "trade_date": "20260806",
                    "observation_target_count": 5,
                    "observations": [
                        {"ts_code": f"60000{index}.SH"}
                        for index in range(4)
                    ],
                }
            ]
        },
    )

    with pytest.raises(RuntimeError, match="observation list is incomplete"):
        run_wp_session._assert_required_daily_list("20260806")


def test_required_daily_list_rejects_non_five_declared_target(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        run_wp_session,
        "load_shadow_ledger",
        lambda path: {
            "sessions": [
                {
                    "trade_date": "20260806",
                    "observation_target_count": 4,
                    "observations": [
                        {"ts_code": f"60000{index}.SH"}
                        for index in range(4)
                    ],
                }
            ]
        },
    )

    with pytest.raises(RuntimeError, match="expected=5"):
        run_wp_session._assert_required_daily_list("20260806")


def test_exact_session_still_freezes_and_closes_after_settlement_failure(
    monkeypatch,
    tmp_path,
):
    calls = []
    clock = {
        "now": datetime(
            2026,
            8,
            5,
            13,
            56,
            tzinfo=run_wp_session.CN_TZ,
        )
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setattr(run_wp_session, "now_cn", lambda: clock["now"])
    monkeypatch.setattr(
        run_wp_session.time_module,
        "sleep",
        lambda seconds: clock.__setitem__(
            "now",
            clock["now"] + timedelta(seconds=seconds),
        ),
    )
    monkeypatch.setattr(
        run_wp_session,
        "_has_fixed_signal_session",
        lambda trade_date: True,
    )
    monkeypatch.setattr(
        run_wp_session,
        "_assert_required_daily_list",
        lambda trade_date: None,
    )

    def fake_run_once(signal_slot=None, **kwargs):
        calls.append((signal_slot, kwargs))
        if kwargs.get("settlement_slot") == "14:05":
            raise RuntimeError("settlement unavailable")

    monkeypatch.setattr(run_wp_session, "run_once", fake_run_once)

    with pytest.raises(RuntimeError, match="settlement unavailable"):
        run_wp_session.run_session()

    assert calls == [
        ("14:00", {}),
        (None, {"settlement_slot": "14:05"}),
        (None, {}),
        (None, {}),
    ]


def test_push_after_missed_slot_runs_same_day_recovery(monkeypatch, tmp_path):
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setattr(
        run_wp_session,
        "now_cn",
        lambda: datetime(
            2026,
            8,
            3,
            14,
            45,
            tzinfo=run_wp_session.CN_TZ,
        ),
    )
    monkeypatch.setattr(
        run_wp_session,
        "_assert_required_daily_list",
        lambda trade_date: None,
    )
    monkeypatch.setattr(
        run_wp_session,
        "run_once",
        lambda signal_slot=None, **kwargs: calls.append(
            (signal_slot, kwargs)
        ),
    )

    run_wp_session.run_once_if_due()

    assert calls == [
        ("14:00", {"late_recovery": True}),
        (
            None,
            {
                "settlement_slot": "14:05",
                "late_recovery": True,
            },
        ),
    ]


def test_push_after_market_close_still_runs_same_day_recovery(
    monkeypatch,
    tmp_path,
):
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setattr(
        run_wp_session,
        "now_cn",
        lambda: datetime(
            2026,
            8,
            6,
            20,
            0,
            tzinfo=run_wp_session.CN_TZ,
        ),
    )
    monkeypatch.setattr(
        run_wp_session,
        "_assert_required_daily_list",
        lambda trade_date: None,
    )
    monkeypatch.setattr(
        run_wp_session,
        "run_once",
        lambda signal_slot=None, **kwargs: calls.append(
            (signal_slot, kwargs)
        ),
    )

    run_wp_session.run_once_if_due()

    assert calls == [
        ("14:00", {"late_recovery": True}),
        (
            None,
            {
                "settlement_slot": "14:05",
                "late_recovery": True,
            },
        ),
    ]


def test_post_close_recovers_missing_settlement_without_rewriting_signal(
    monkeypatch,
    tmp_path,
):
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setattr(
        run_wp_session,
        "now_cn",
        lambda: datetime(
            2026,
            8,
            6,
            20,
            0,
            tzinfo=run_wp_session.CN_TZ,
        ),
    )
    monkeypatch.setattr(
        run_wp_session,
        "_has_required_daily_list",
        lambda trade_date: True,
    )
    monkeypatch.setattr(
        run_wp_session,
        "_has_fixed_entry_settlement",
        lambda trade_date: False,
    )
    monkeypatch.setattr(
        run_wp_session,
        "_assert_required_daily_list",
        lambda trade_date: None,
    )
    monkeypatch.setattr(
        run_wp_session,
        "run_once",
        lambda signal_slot=None, **kwargs: calls.append(
            (signal_slot, kwargs)
        ),
    )

    run_wp_session.run_once_if_due()

    assert calls == [
        (
            None,
            {
                "settlement_slot": "14:05",
                "late_recovery": True,
            },
        ),
    ]


def test_post_close_skips_when_signal_and_settlement_are_complete(
    monkeypatch,
    tmp_path,
):
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.setattr(
        run_wp_session,
        "now_cn",
        lambda: datetime(
            2026,
            8,
            6,
            20,
            0,
            tzinfo=run_wp_session.CN_TZ,
        ),
    )
    monkeypatch.setattr(
        run_wp_session,
        "_has_required_daily_list",
        lambda trade_date: True,
    )
    monkeypatch.setattr(
        run_wp_session,
        "_has_fixed_entry_settlement",
        lambda trade_date: True,
    )
    monkeypatch.setattr(
        run_wp_session,
        "run_once",
        lambda signal_slot=None, **kwargs: calls.append(
            (signal_slot, kwargs)
        ),
    )

    run_wp_session.run_once_if_due()

    assert calls == []


def test_v3_causal_source_contract_is_passed_to_core_engine(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setattr(
        run_wp_session,
        "now_cn",
        lambda: datetime(2026, 7, 27, 14, 2, tzinfo=run_wp_session.CN_TZ),
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
                "signal_slot": "14:00",
            },
        ),
    )

    def fake_run(command, **kwargs):
        if command[:3] == [run_wp_session.sys.executable, "-m", "wp.main"]:
            captured.update(kwargs["env"])
            manifest = tmp_path / "outputs" / "json" / "wp_manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text('{"report_revision":"new"}\n', encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_wp_session.subprocess, "run", fake_run)

    run_wp_session.run_once()

    assert captured["WP_EXPECTED_TRADE_DATE"] == "20260727"
    assert captured["WP_V3_SIGNAL_SLOT"] == "14:00"
    assert captured["WP_V3_MARKET_DATA_CUTOFF_SLOT"] == "13:55"
    assert captured["WP_V3_SOURCE_CSV"] == source_path.as_posix()
    assert captured["WP_MODE"] == "live"


def test_1400_decision_uses_the_completed_1355_cutoff(
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
            13,
            56,
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
            "signal_slot": "14:00",
            "market_data_cutoff_slot": "13:55",
        }

    def fake_run(command, **kwargs):
        if command[:3] == [run_wp_session.sys.executable, "-m", "wp.main"]:
            manifest = tmp_path / "outputs" / "json" / "wp_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                '{"report_revision":"1400"}\n',
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        run_wp_session,
        "build_live_input",
        fake_build_live_input,
    )
    monkeypatch.setattr(run_wp_session.subprocess, "run", fake_run)

    run_wp_session.run_once("14:00")

    assert captured["built"] is True


def test_1405_settlement_reuses_immutable_1400_snapshot(
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
            7,
            tzinfo=run_wp_session.CN_TZ,
        ),
    )
    live_path = (
        tmp_path / "data" / "v3" / "latest" / "wp_v3_live_features.csv"
    )
    live_path.parent.mkdir(parents=True)
    live_path.write_text("ts_code\n600001.SH\n", encoding="utf-8")
    ledger_path = (
        tmp_path / "outputs" / "json" / "wp_v3_candidate_ledger.json"
    )
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(
        (
            '{"schema_version":"wp_candidate_ledger_v4",'
            '"sessions":[{"trade_date":"20260727",'
            '"covered_slots":["14:00"],"candidates":[],'
            '"observations":[]}]}\n'
        ),
        encoding="utf-8",
    )
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
        lambda **kwargs: calls.append(("unexpected_signal_rebuild", "14:00")),
    )

    def fake_run(command, **kwargs):
        if command[:3] == [run_wp_session.sys.executable, "-m", "wp.main"]:
            manifest = tmp_path / "outputs" / "json" / "wp_manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                '{"report_revision":"1405"}\n',
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_wp_session.subprocess, "run", fake_run)

    run_wp_session.run_once(settlement_slot="14:05")

    assert calls == [("settle", "14:05")]


def test_1405_late_recovery_builds_signal_before_settlement(
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
            7,
            tzinfo=run_wp_session.CN_TZ,
        ),
    )
    live_path = (
        tmp_path / "data" / "v3" / "latest" / "wp_v3_live_features.csv"
    )
    settlement_path = live_path.with_name("wp_v3_entry_settlement.csv")

    def fake_build_live_input(**kwargs):
        calls.append("build-1400")
        live_path.parent.mkdir(parents=True, exist_ok=True)
        live_path.write_text("ts_code\n600001.SH\n", encoding="utf-8")
        return live_path, {
            "trade_date": "20260727",
            "signal_slot": "14:00",
        }

    def fake_settlement(**kwargs):
        calls.append("settle-1405")
        settlement_path.write_text(
            "ts_code\n600001.SH\n",
            encoding="utf-8",
        )
        return settlement_path, {
            "trade_date": "20260727",
            "requested_symbols": 1,
            "observed_symbols": 1,
        }

    engine_calls = 0

    def fake_run(command, **kwargs):
        nonlocal engine_calls
        if command[:3] == [run_wp_session.sys.executable, "-m", "wp.main"]:
            engine_calls += 1
            calls.append(f"engine-{engine_calls}")
            if engine_calls == 1:
                ledger = (
                    tmp_path
                    / "outputs"
                    / "json"
                    / "wp_v3_candidate_ledger.json"
                )
                ledger.parent.mkdir(parents=True, exist_ok=True)
                ledger.write_text(
                    (
                        '{"schema_version":"wp_candidate_ledger_v4",'
                        '"sessions":[{"trade_date":"20260727",'
                        '"covered_slots":["14:00"],"candidates":[],'
                        '"observations":[]}]}\n'
                    ),
                    encoding="utf-8",
                )
            manifest = (
                tmp_path / "outputs" / "json" / "wp_manifest.json"
            )
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                f'{{"report_revision":"recovery-{engine_calls}"}}\n',
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        run_wp_session,
        "build_live_input",
        fake_build_live_input,
    )
    monkeypatch.setattr(
        run_wp_session,
        "capture_entry_settlement_input",
        fake_settlement,
    )
    monkeypatch.setattr(run_wp_session.subprocess, "run", fake_run)

    run_wp_session.run_once(settlement_slot="14:05")

    assert calls[:4] == [
        "build-1400",
        "engine-1",
        "settle-1405",
        "engine-2",
    ]
