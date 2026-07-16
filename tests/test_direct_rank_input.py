from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.build_direct_rank_input import build_direct_rank_input


def _prepare_processor(root: Path, *, status: str = "ok") -> Path:
    processor = root / "upstream"
    (processor / "scripts" / "wp").mkdir(parents=True)
    (processor / "scripts" / "fetch_daily_snapshots.py").write_text("# test\n", encoding="utf-8")
    (processor / "scripts" / "wp" / "wp_pipeline.py").write_text("# test\n", encoding="utf-8")
    latest = processor / "data" / "wp" / "latest"
    latest.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "trade_date": "20260716",
                "update_time": "2026-07-16 14:35:10",
                "ts_code": "000001.SZ",
                "name": "平安银行",
                "price": 12.50,
                "pre_close": 11.50,
                "pct_chg": 8.5,
                "amount": 200_000_000,
                "sector_name": "银行",
                "pre_day_limitup": 0,
                "today_limitup": 0,
                "today_limit_up_price": 12.65,
                "ret_5d": 15.0,
                "ret_20d": 25.0,
                "amount_ratio_5d": 1.8,
                "amount_ratio_20d": 1.5,
                "ma5_position": 5.0,
                "ma20_position": 10.0,
                "close_position": 92.0,
                "realtime_source": "rt_min",
                "stock_age_days": 1000,
            }
        ]
    ).to_csv(latest / "wp_latest_rank_input.csv", index=False, encoding="utf-8-sig")
    (latest / "wp_manifest.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-07-16 14:35:20",
                "status": status,
                "source_trade_date": "20260716",
            }
        ),
        encoding="utf-8",
    )
    return processor


def test_direct_builder_archives_validated_rank_input(tmp_path):
    processor = _prepare_processor(tmp_path)
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))

    result = build_direct_rank_input(
        root=tmp_path,
        upstream_root=processor,
        env={
            "TUSHARE_TOKEN": "secret",
            "TRADE_DATE": "20260716",
            "WP_TARGET_SLOT": "2026-07-16 14:35:00",
            "WP_UPSTREAM_PROCESSOR_REVISION": "abc123",
        },
        command_runner=fake_runner,
    )

    assert result.ok is True
    assert result.source_trade_date == "20260716"
    assert len(calls) == 2
    assert Path(result.source_path).is_file()
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["source_mode"] == "direct_tushare"
    assert manifest["scheduled_slot"] == "2026-07-16 14:35:00"
    assert manifest["processor_revision"] == "abc123"
    assert manifest["direct_row_count"] == 1
    assert manifest["direct_quality"]["unique_codes"] == 1
    assert manifest["direct_quality"]["required_columns_complete"] is True
    assert manifest["direct_quality"]["history_feature_coverage_pct"] == 100.0


def test_direct_builder_uses_fetcher_resolved_trade_date(tmp_path):
    processor = _prepare_processor(tmp_path)
    metadata = processor / "data" / "latest" / "_meta.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps({"resolved_trade_date": "20260716"}),
        encoding="utf-8",
    )
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs["env"].copy()))

    result = build_direct_rank_input(
        root=tmp_path,
        upstream_root=processor,
        env={"TUSHARE_TOKEN": "secret"},
        command_runner=fake_runner,
    )

    assert result.ok is True
    assert "TRADE_DATE" not in calls[0][1]
    assert calls[1][1]["TRADE_DATE"] == "20260716"


def test_direct_builder_passes_slot_date_to_fetcher(tmp_path):
    processor = _prepare_processor(tmp_path)
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs["env"].copy()))

    result = build_direct_rank_input(
        root=tmp_path,
        upstream_root=processor,
        env={
            "TUSHARE_TOKEN": "secret",
            "WP_TARGET_SLOT": "2026-07-16 14:35:00",
        },
        command_runner=fake_runner,
    )

    assert result.ok is True
    assert calls[0][1]["TRADE_DATE"] == "20260716"
    assert calls[1][1]["TRADE_DATE"] == "20260716"


def test_direct_builder_falls_back_when_token_is_missing(tmp_path):
    result = build_direct_rank_input(root=tmp_path, env={})

    assert result.attempted is True
    assert result.ok is False
    assert "TUSHARE_TOKEN" in result.error


def test_direct_builder_rejects_stale_processor_output(tmp_path):
    processor = _prepare_processor(tmp_path, status="stale_data")

    result = build_direct_rank_input(
        root=tmp_path,
        upstream_root=processor,
        env={"TUSHARE_TOKEN": "secret", "TRADE_DATE": "20260716"},
        command_runner=lambda *args, **kwargs: None,
    )

    assert result.ok is False
    assert "stale_data" in result.error


def test_direct_builder_uses_one_total_timeout_budget(tmp_path):
    processor = _prepare_processor(tmp_path)
    timeouts = []
    clock = iter([0.0, 0.0, 40.0])

    def fake_runner(command, **kwargs):
        timeouts.append(kwargs["timeout"])

    result = build_direct_rank_input(
        root=tmp_path,
        upstream_root=processor,
        env={
            "TUSHARE_TOKEN": "secret",
            "TRADE_DATE": "20260716",
            "WP_DIRECT_TIMEOUT_SECONDS": "100",
        },
        command_runner=fake_runner,
        monotonic=lambda: next(clock),
    )

    assert result.ok is True
    assert timeouts == [100.0, 60.0]


def test_direct_builder_rejects_incomplete_model_inputs(tmp_path):
    processor = _prepare_processor(tmp_path)
    csv_path = processor / "data" / "wp" / "latest" / "wp_latest_rank_input.csv"
    frame = pd.read_csv(csv_path)
    frame.loc[0, "amount_ratio_5d"] = None
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    result = build_direct_rank_input(
        root=tmp_path,
        upstream_root=processor,
        env={"TUSHARE_TOKEN": "secret", "TRADE_DATE": "20260716"},
        command_runner=lambda *args, **kwargs: None,
    )

    assert result.ok is False
    assert "amount_ratio_5d" in result.error


def test_direct_builder_rejects_synthetic_historical_features(tmp_path):
    processor = _prepare_processor(tmp_path)
    csv_path = processor / "data" / "wp" / "latest" / "wp_latest_rank_input.csv"
    seed = pd.read_csv(csv_path)
    frames = []
    for index in range(3):
        row = seed.copy()
        row["ts_code"] = f"00000{index + 1}.SZ"
        row["ret_5d"] = row["pct_chg"]
        row["amount_ratio_5d"] = 1.0
        row["ma5_position"] = 0.0
        row["ma20_position"] = 0.0
        frames.append(row)
    pd.concat(frames, ignore_index=True).to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    result = build_direct_rank_input(
        root=tmp_path,
        upstream_root=processor,
        env={"TUSHARE_TOKEN": "secret", "TRADE_DATE": "20260716"},
        command_runner=lambda *args, **kwargs: None,
    )

    assert result.ok is False
    assert "historical features are degraded" in result.error
