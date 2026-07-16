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
                "pct_chg": 8.5,
                "amount": 200_000_000,
                "pre_day_limitup": 0,
                "today_limitup": 0,
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
    assert len(calls) == 2
    assert Path(result.source_path).is_file()
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["source_mode"] == "direct_tushare"
    assert manifest["scheduled_slot"] == "2026-07-16 14:35:00"
    assert manifest["processor_revision"] == "abc123"
    assert manifest["direct_row_count"] == 1


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
