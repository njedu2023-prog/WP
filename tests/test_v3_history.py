from __future__ import annotations

import threading
import time

import pandas as pd
import pytest

from wp.v3.contracts import V3Config
from wp.v3.history import (
    TushareHistoryClient,
    _industry_at,
    _minute_universe_quality,
    _normalize_historical_minutes,
    _ordered_bounded_map,
    _slot_features,
)


class _CappedPro:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def query(self, _api_name: str, **params):
        offset = int(params.get("offset", 0))
        requested = int(params.get("limit", 8_000))
        return pd.DataFrame(self.rows[offset : offset + min(requested, 2)])


def test_pagination_continues_when_api_cap_is_below_requested_page_size(tmp_path):
    rows = [{"ts_code": f"60000{index}.SH", "value": index} for index in range(5)]
    client = TushareHistoryClient(_CappedPro(rows), tmp_path, page_size=8_000)
    result = client.query(
        "capped",
        cache_key="all",
        paged=True,
        fields="ts_code,value",
    )
    assert result["value"].tolist() == [0, 1, 2, 3, 4]


def test_bounded_parallel_map_preserves_order_and_uses_multiple_workers():
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def work(value: int) -> int:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02 if value == 0 else 0.01)
        with lock:
            active -= 1
        return value * 10

    result = list(_ordered_bounded_map(work, list(range(8)), workers=3))

    assert result == [value * 10 for value in range(8)]
    assert maximum_active >= 2


def test_industry_membership_is_resolved_at_the_signal_date():
    intervals = {
        "600001.SH": [
            ("20200101", "20241231", "旧行业"),
            ("20250101", "29991231", "新行业"),
        ]
    }
    assert _industry_at("600001.SH", "20240701", intervals) == "旧行业"
    assert _industry_at("600001.SH", "20260701", intervals) == "新行业"


def test_slot_features_record_per_symbol_bar_lag():
    bars = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "trade_time": "2026-07-27 14:15:00",
                "open": 10.0,
                "high": 10.1,
                "low": 9.9,
                "close": 10.0,
                "amount": 5_000_000,
            }
        ]
    )
    bars["trade_time"] = pd.to_datetime(bars["trade_time"])
    features = _slot_features(bars, "14:20")
    assert features.loc[0, "slot_bar_lag_minutes"] == 5


def test_market_minute_coverage_below_contract_fails_closed():
    expected = {f"600{index:03d}.SH" for index in range(1_000)}
    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        _minute_universe_quality(
            expected,
            expected,
            set(sorted(expected)[:899]),
            V3Config(),
            trade_date="20260727",
        )


def test_historical_minutes_keep_warmup_and_signal_bars_without_redefining_ohlcv():
    frame = pd.DataFrame(
        {
            "ts_code": ["600000.SH"] * 4,
            "trade_time": [
                "2026-07-21 09:35:00",
                "2026-07-21 14:15:00",
                "2026-07-21 14:20:00",
                "2026-07-21 14:25:00",
            ],
            "open": [10.0, 10.1, 10.2, 10.3],
            "high": [10.1, 10.2, 10.4, 10.5],
            "low": [9.9, 10.0, 10.1, 10.2],
            "close": [10.0, 10.1, 10.3, 10.4],
            "vol": [100.0, 200.0, 300.0, 400.0],
            "amount": [1_000.0, 2_000.0, 3_000.0, 4_000.0],
        }
    )
    result = _normalize_historical_minutes(
        frame,
        signal_slots=("14:20", "14:25"),
    )
    assert result["trade_time"].dt.strftime("%H:%M").tolist() == [
        "14:15",
        "14:20",
        "14:25",
    ]
    assert result["day_open"].tolist() == [10.0, 10.0, 10.0]
    assert result["high"].tolist() == [10.2, 10.4, 10.5]
    assert result["slot_amount"].tolist() == [2_000.0, 3_000.0, 4_000.0]
