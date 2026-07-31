from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.probe_wp_v28_fine_industry_peer_data import probe_date
from wp.v3.v28_industry_peer import (
    active_membership,
    leave_one_out_peer_features,
    normalize_membership,
)


def membership_frame() -> pd.DataFrame:
    rows = []
    for index in range(10):
        rows.append(
            {
                "l1_code": "L1",
                "l1_name": "一级",
                "l2_code": "L2",
                "l2_name": "二级",
                "l3_code": "L3A" if index < 5 else "L3B",
                "l3_name": "三级",
                "ts_code": f"{index + 1:06d}.SZ",
                "name": f"S{index}",
                "in_date": "20200101",
                "out_date": "",
                "is_new": "Y",
            }
        )
    return pd.DataFrame(rows)


def minute_frame() -> pd.DataFrame:
    rows = []
    slots = (
        "14:00",
        "14:05",
        "14:10",
        "14:15",
        "14:20",
        "14:25",
        "14:30",
        "14:35",
        "14:40",
        "14:45",
        "14:50",
    )
    for index in range(10):
        for position, slot in enumerate(slots):
            rows.append(
                {
                    "ts_code": f"{index + 1:06d}.SZ",
                    "trade_date": "20260723",
                    "trade_time": f"2026-07-23 {slot}:00",
                    "close": 10.0 + index * 0.1 + position * 0.01,
                    "amount": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [f"{index + 1:06d}.SZ" for index in range(10)],
            "trade_date": ["20260723"] * 10,
            "pre_close": [10.0 + index * 0.1 for index in range(10)],
        }
    )


def test_membership_interval_is_start_inclusive_and_end_exclusive() -> None:
    raw = membership_frame()
    raw.loc[0, "out_date"] = "20260723"
    membership = normalize_membership(raw)

    active = active_membership(membership, trade_date="20260723")

    assert "000001.SZ" not in set(active["ts_code"])
    assert active["ts_code"].nunique() == 9


def test_probe_requires_complete_peer_groups_at_every_slot() -> None:
    record = probe_date(
        minute_frame(),
        daily_frame(),
        normalize_membership(membership_frame()),
        trade_date="20260723",
    )

    assert not record["duplicate_identity"]
    assert record["date_consistent"]
    assert record["slot_count"] == 7
    assert all(slot["rows"] == 10 for slot in record["slot_records"])
    assert all(slot["tail_20m_coverage"] == 1.0 for slot in record["slot_records"])


def test_leave_one_out_features_exclude_candidate_itself() -> None:
    membership = normalize_membership(membership_frame())
    from wp.v3.v28_industry_peer import build_stock_slot_frame

    stock_slots = build_stock_slot_frame(
        minute_frame(),
        daily_frame(),
        membership,
        trade_date="20260723",
    )
    candidate = stock_slots.loc[
        stock_slots["signal_slot"].eq("14:20")
        & stock_slots["ts_code"].eq("000001.SZ"),
        ["trade_date", "signal_slot", "ts_code"],
    ]
    features = leave_one_out_peer_features(
        stock_slots.loc[stock_slots["signal_slot"].eq("14:20")],
        candidate,
        level="l2_code",
    )

    assert features.loc[0, "v28_l2_peer_count"] == 9
    expected = stock_slots.loc[
        stock_slots["signal_slot"].eq("14:20")
        & stock_slots["ts_code"].ne("000001.SZ"),
        "ret_from_prev_close_pct",
    ].median()
    assert features.loc[0, "v28_l2_peer_return_median_pct"] == expected


def test_probe_source_does_not_read_return_outcomes() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "probe_wp_v28_fine_industry_peer_data.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "gross_return",
        "net_return",
        "t1_close",
        "target_net",
        "label_available",
    )

    assert not any(token in source for token in forbidden)
