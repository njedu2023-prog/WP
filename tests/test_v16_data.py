from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wp.v3.v16_data import (
    candidate_exit_pairs,
    normalize_full_day_minutes,
    write_exit_path_dataset,
)


def test_candidate_exit_pairs_deduplicates_slots() -> None:
    frontier = pd.DataFrame(
        [
            {
                "trade_date": "20260720",
                "signal_slot": "14:20",
                "ts_code": "600001.SH",
            },
            {
                "trade_date": "20260720",
                "signal_slot": "14:25",
                "ts_code": "600001.SH",
            },
        ]
    )
    panel = frontier.assign(target_trade_date="20260721")

    pairs = candidate_exit_pairs(frontier, panel)

    assert pairs.to_dict(orient="records") == [
        {"target_trade_date": "20260721", "ts_code": "600001.SH"}
    ]


def test_normalize_full_day_minutes_reports_pair_quality() -> None:
    bars = pd.date_range(
        "2026-07-21 09:35:00",
        periods=4,
        freq="5min",
    )
    raw = pd.DataFrame(
        {
            "ts_code": "600001.SH",
            "trade_time": bars,
            "open": 10.0,
            "high": 10.1,
            "low": 9.9,
            "close": 10.0,
            "vol": 1000.0,
            "amount": 1_000_000.0,
        }
    )
    raw = pd.concat(
        [
            raw,
            pd.DataFrame(
                {
                    "ts_code": ["600001.SH"],
                    "trade_time": [pd.Timestamp("2026-07-21 15:00:00")],
                    "open": [10.0],
                    "high": [10.1],
                    "low": [9.9],
                    "close": [10.0],
                    "vol": [1000.0],
                    "amount": [1_000_000.0],
                }
            ),
        ],
        ignore_index=True,
    )
    pairs = pd.DataFrame(
        [{"target_trade_date": "20260721", "ts_code": "600001.SH"}]
    )

    minutes, quality = normalize_full_day_minutes(
        raw,
        pairs,
        minimum_bars=5,
    )

    assert len(minutes) == 5
    assert quality.loc[0, "covered"]
    assert quality.loc[0, "last_slot"] == "15:00"


def test_write_exit_path_dataset_enforces_coverage(tmp_path: Path) -> None:
    quality = pd.DataFrame(
        [
            {
                "target_trade_date": "20260721",
                "ts_code": "600001.SH",
                "covered": True,
                "bars": 48,
                "valid_bars": 48,
                "last_slot": "15:00",
            },
            {
                "target_trade_date": "20260721",
                "ts_code": "600002.SH",
                "covered": False,
                "bars": 0,
                "valid_bars": 0,
                "last_slot": None,
            },
        ]
    )
    minutes = pd.DataFrame(
        [
            {
                "target_trade_date": "20260721",
                "ts_code": "600001.SH",
                "trade_time": pd.Timestamp("2026-07-21 15:00:00"),
                "bar_slot": "15:00",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            }
        ]
    )

    with pytest.raises(RuntimeError, match="coverage"):
        write_exit_path_dataset(
            minutes,
            quality,
            tmp_path,
            source={"test": True},
            minimum_pair_coverage=0.98,
        )


def test_write_exit_path_dataset_is_fingerprinted(tmp_path: Path) -> None:
    quality = pd.DataFrame(
        [
            {
                "target_trade_date": "20260721",
                "ts_code": "600001.SH",
                "covered": True,
                "bars": 48,
                "valid_bars": 48,
                "last_slot": "15:00",
            }
        ]
    )
    minutes = pd.DataFrame(
        [
            {
                "target_trade_date": "20260721",
                "ts_code": "600001.SH",
                "trade_time": pd.Timestamp("2026-07-21 15:00:00"),
                "bar_slot": "15:00",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "vol": 1000.0,
                "amount": 1_000_000.0,
            }
        ]
    )

    manifest = write_exit_path_dataset(
        minutes,
        quality,
        tmp_path,
        source={"test": True},
    )

    assert manifest["pair_coverage"] == 1.0
    assert len(manifest["dataset_fingerprint"]) == 64
    assert (tmp_path / "wp_v16_t1_path_manifest.json").exists()
