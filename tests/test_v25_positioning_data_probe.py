from __future__ import annotations

import pandas as pd

from scripts.probe_wp_v25_positioning_data import (
    CYQ_FIELDS,
    cyq_probe_record,
    normalize_cyq,
)


def valid_cyq() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "600000.SH",
                "trade_date": "20260723",
                "his_low": 5.0,
                "his_high": 15.0,
                "cost_5pct": 8.0,
                "cost_15pct": 8.5,
                "cost_50pct": 9.5,
                "cost_85pct": 10.5,
                "cost_95pct": 11.0,
                "weight_avg": 9.6,
                "winner_rate": 63.0,
            }
        ],
        columns=CYQ_FIELDS.split(","),
    )


def test_cyq_probe_accepts_ordered_complete_distribution() -> None:
    normalized = normalize_cyq(valid_cyq())
    record = cyq_probe_record(
        normalized,
        ts_code="600000.SH",
        trade_date="20260723",
    )

    assert record["coverage_pass"]
    assert record["cost_percentiles_ordered"]
    assert record["winner_rate_bounded"]


def test_cyq_probe_rejects_reversed_cost_percentiles() -> None:
    frame = valid_cyq()
    frame.loc[0, "cost_85pct"] = 8.8
    normalized = normalize_cyq(frame)
    record = cyq_probe_record(
        normalized,
        ts_code="600000.SH",
        trade_date="20260723",
    )

    assert not record["coverage_pass"]
    assert not record["cost_percentiles_ordered"]


def test_cyq_probe_requires_exactly_one_stock_date() -> None:
    frame = pd.concat([valid_cyq(), valid_cyq()], ignore_index=True)
    normalized = normalize_cyq(frame)
    record = cyq_probe_record(
        normalized,
        ts_code="600000.SH",
        trade_date="20260723",
    )

    assert not record["coverage_pass"]
    assert not record["unique_stock_date"]

