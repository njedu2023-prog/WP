from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.probe_wp_v26_crowd_confirmation_data import (
    HOT_FIELDS,
    MEMBER_FIELDS,
    SIGNAL_TIMES,
    SW_MINUTE_FIELDS,
    hot_snapshot_record,
    membership_probe_record,
    normalize_membership,
    parse_rank_time,
    select_index_codes,
    sw_minute_probe_record,
)


def test_parse_rank_time_accepts_point_in_time_formats() -> None:
    assert parse_rank_time("14:30:00", "20260723") == pd.Timestamp(
        "2026-07-23 14:30:00"
    )
    assert parse_rank_time("20260723143000", "20260723") == pd.Timestamp(
        "2026-07-23 14:30:00"
    )
    assert parse_rank_time("2026-07-23 14:30:00", "20260723") == pd.Timestamp(
        "2026-07-23 14:30:00"
    )
    assert parse_rank_time(
        np.array(["20260723143009"]),
        "20260723",
    ) == pd.Timestamp("2026-07-23 14:30:09")


def hot_frame(rank_time: str = "20260723143000") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": "20260723",
                "data_type": "热股",
                "ts_code": f"{index:06d}.SZ",
                "ts_name": f"S{index}",
                "rank": index,
                "pct_change": 1.0,
                "current_price": 10.0,
                "rank_time": rank_time,
            }
            for index in range(1, 101)
        ],
        columns=HOT_FIELDS.split(","),
    )


def test_hot_probe_accepts_tail_snapshot_and_rejects_post_close_only() -> None:
    accepted = hot_snapshot_record(
        hot_frame(),
        api_name="ths_hot",
        trade_date="20260723",
    )
    rejected = hot_snapshot_record(
        hot_frame("20260723153000"),
        api_name="ths_hot",
        trade_date="20260723",
    )

    assert accepted["coverage_pass"]
    assert accepted["unique_a_share_codes"] == 100
    assert not rejected["coverage_pass"]
    assert rejected["usable_snapshot_time"] is None


def test_hot_probe_groups_one_fetch_batch_by_minute() -> None:
    frame = hot_frame()
    frame["rank_time"] = [
        f"202607231430{index % 60:02d}"
        for index in range(len(frame))
    ]
    record = hot_snapshot_record(
        frame,
        api_name="ths_hot",
        trade_date="20260723",
    )

    assert record["coverage_pass"]
    assert record["usable_snapshot_rows"] == 100
    assert record["usable_snapshot_time"] == "2026-07-23T14:30:00"


def test_hot_probe_rejects_duplicate_codes() -> None:
    frame = hot_frame()
    frame.loc[1, "ts_code"] = frame.loc[0, "ts_code"]
    record = hot_snapshot_record(
        frame,
        api_name="dc_hot",
        trade_date="20260723",
    )

    assert not record["coverage_pass"]
    assert record["duplicate_codes_in_snapshot"]


def membership_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "l1_code": "801010.SI",
                "l1_name": "L1",
                "l2_code": "801016.SI",
                "l2_name": "L2",
                "l3_code": "850111.SI",
                "l3_name": "L3",
                "ts_code": f"{index:06d}.SZ",
                "name": f"S{index}",
                "in_date": "20210101",
                "out_date": "",
                "is_new": "Y",
            }
            for index in range(1, 3_601)
        ],
        columns=MEMBER_FIELDS.split(","),
    )


def test_membership_probe_requires_large_complete_point_in_time_universe() -> None:
    normalized = normalize_membership(membership_frame())
    record = membership_probe_record(normalized, trade_date="20260723")

    assert record["coverage_pass"]
    assert record["unique_active_codes"] == 3_600


def test_select_index_codes_is_deterministic_and_published_only() -> None:
    frame = pd.DataFrame(
        {
            "index_code": [f"80{index:04d}.SI" for index in range(10)],
            "is_pub": ["1"] * 9 + ["0"],
        }
    )

    assert select_index_codes(frame, count=4) == [
        "800000.SI",
        "800002.SI",
        "800005.SI",
        "800008.SI",
    ]


def test_sw_minute_probe_requires_all_signal_slots() -> None:
    frame = pd.DataFrame(
        [
            {
                "ts_code": "801016.SI",
                "trade_time": f"2026-07-23 {clock}",
                "open": 100.0,
                "close": 101.0,
                "high": 102.0,
                "low": 99.0,
                "amount": 1_000_000.0,
                "vol": 100_000.0,
            }
            for clock in SIGNAL_TIMES
        ],
        columns=SW_MINUTE_FIELDS.split(","),
    )
    accepted = sw_minute_probe_record(
        frame,
        ts_code="801016.SI",
        trade_date="20260723",
    )
    rejected = sw_minute_probe_record(
        frame.iloc[:-1],
        ts_code="801016.SI",
        trade_date="20260723",
    )

    assert accepted["coverage_pass"]
    assert not rejected["coverage_pass"]


def test_probe_source_does_not_read_profit_outcomes() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "probe_wp_v26_crowd_confirmation_data.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "gross_return",
        "net_return",
        "profit_label",
        "target_return",
        "t1_close",
    )

    assert not any(token in source for token in forbidden)
