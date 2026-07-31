from __future__ import annotations

import pandas as pd

from scripts.probe_wp_v23_point_in_time_data import (
    finite_positive_share,
    normalize_minute_probe,
    ohlc_consistent,
)


def test_minute_probe_normalizes_and_checks_valid_ohlc() -> None:
    raw = pd.DataFrame(
        [
            {
                "ts_code": "600000.SH",
                "trade_time": "2026-07-24 14:01:00",
                "open": "10.00",
                "high": "10.10",
                "low": "9.98",
                "close": "10.05",
                "vol": "1000",
                "amount": "10050",
            },
            {
                "ts_code": "600000.SH",
                "trade_time": "2026-07-24 14:02:00",
                "open": "10.05",
                "high": "10.08",
                "low": "10.01",
                "close": "10.02",
                "vol": "800",
                "amount": "8020",
            },
            {
                "ts_code": "600000.SH",
                "trade_time": "2026-07-23 14:02:00",
                "open": "9.90",
                "high": "9.95",
                "low": "9.88",
                "close": "9.93",
                "vol": "900",
                "amount": "8937",
            },
        ]
    )

    normalized = normalize_minute_probe(raw, trade_date="20260724")

    assert len(normalized) == 2
    assert ohlc_consistent(normalized)
    assert finite_positive_share(normalized, "amount") == 1.0
    assert finite_positive_share(normalized, "vol") == 1.0


def test_minute_probe_rejects_inconsistent_ohlc() -> None:
    invalid = pd.DataFrame(
        [
            {
                "ts_code": "600000.SH",
                "trade_time": pd.Timestamp("2026-07-24 14:01:00"),
                "open": 10.00,
                "high": 9.99,
                "low": 9.98,
                "close": 10.05,
                "vol": 1000.0,
                "amount": 10050.0,
            }
        ]
    )

    assert not ohlc_consistent(invalid)
