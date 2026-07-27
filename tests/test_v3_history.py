from __future__ import annotations

import threading
import time

import pandas as pd
import pytest

from wp.v3.contracts import V3Config
from wp.v3.history import (
    TushareHistoryClient,
    _day,
    _index_by_trade_date,
    _industry_at,
    _load_daily_history,
    _minute_universe_quality,
    _normalize_historical_minutes,
    _ordered_bounded_map,
    _slot_features,
    _slot_features_for_slots,
)


class _CappedPro:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def query(self, _api_name: str, **params):
        offset = int(params.get("offset", 0))
        requested = int(params.get("limit", 8_000))
        return pd.DataFrame(self.rows[offset : offset + min(requested, 2)])


class _DailyPro:
    def query(self, _api_name: str, **params):
        trade_date = str(params["trade_date"])
        fields = str(params["fields"]).split(",")
        row = {
            column: (
                trade_date
                if column == "trade_date"
                else "600000.SH"
                if column == "ts_code"
                else 1.0
            )
            for column in fields
        }
        time.sleep(0.005)
        return pd.DataFrame([row])


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


def test_daily_history_parallel_fetch_retains_trade_date_order(tmp_path):
    client = TushareHistoryClient(
        _DailyPro(),
        tmp_path,
        requests_per_minute=100_000,
    )
    daily, basic, limits, adjustments = _load_daily_history(
        client,
        ["20260721", "20260722", "20260723"],
        workers=3,
    )

    expected = ["20260721", "20260722", "20260723"]
    assert daily["trade_date"].tolist() == expected
    assert basic["trade_date"].tolist() == expected
    assert limits["trade_date"].tolist() == expected
    assert adjustments["trade_date"].tolist() == expected


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


def test_vectorized_slot_features_match_the_signal_contract():
    bars = pd.DataFrame(
        {
            "ts_code": ["600001.SH"] * 5 + ["600002.SH"] * 2,
            "trade_time": pd.to_datetime(
                [
                    "2026-07-27 14:00:00",
                    "2026-07-27 14:05:00",
                    "2026-07-27 14:10:00",
                    "2026-07-27 14:15:00",
                    "2026-07-27 14:20:00",
                    "2026-07-27 14:15:00",
                    "2026-07-27 14:20:00",
                ]
            ),
            "open": [10, 11, 12, 13, 14, 20, 21],
            "high": [10.5, 11.5, 12.5, 13.5, 14.5, 20.5, 21.5],
            "low": [9.5, 10.5, 11.5, 12.5, 13.5, 19.5, 20.5],
            "close": [10, 11, 12, 13, 14, 20, 21],
            "amount": [10, 20, 30, 40, 60, 100, 120],
            "slot_amount": [10, 20, 30, 40, 60, 100, 120],
        }
    )

    features = _slot_features(bars, "14:20").set_index("ts_code")
    first = features.loc["600001.SH"]

    assert first["intraday_snapshot_count"] == 5
    assert first["ret_5m_pct"] == pytest.approx((14 / 13 - 1) * 100)
    assert first["ret_10m_pct"] == pytest.approx((14 / 12 - 1) * 100)
    assert first["ret_20m_pct"] == pytest.approx(40.0)
    assert first["tail_range_10m_pct"] == pytest.approx((14.5 / 12.5 - 1) * 100)
    assert first["tail_close_position_10m"] == pytest.approx(0.75)
    assert first["tail_amount_acceleration"] == pytest.approx(2.0)
    assert features.loc["600002.SH", "intraday_snapshot_count"] == 2
    assert pd.isna(features.loc["600002.SH", "ret_10m_pct"])


def test_multi_slot_features_compute_once_and_keep_full_session_count():
    times = pd.date_range("2026-07-27 14:00:00", periods=11, freq="5min")
    bars = pd.DataFrame(
        {
            "ts_code": ["600001.SH"] * len(times),
            "trade_time": times,
            "open": range(10, 21),
            "high": [value + 0.5 for value in range(10, 21)],
            "low": [value - 0.5 for value in range(10, 21)],
            "close": range(10, 21),
            "amount": range(100, 1_200, 100),
        }
    )

    features = _slot_features_for_slots(
        bars,
        ("14:20", "14:35", "14:50"),
    ).set_index("signal_slot")

    assert features.loc["14:20", "intraday_snapshot_count"] == 5
    assert features.loc["14:35", "intraday_snapshot_count"] == 8
    assert features.loc["14:50", "intraday_snapshot_count"] == 11
    assert features.loc["14:50", "ret_20m_pct"] == pytest.approx(
        (20 / 16 - 1) * 100
    )


def test_date_index_uses_direct_partition_lookup_and_preserves_missing_schema():
    frame = pd.DataFrame(
        {
            "trade_date": ["20260721", "20260722", "20260722"],
            "ts_code": ["600001.SH", "600001.SH", "600002.SH"],
        }
    )
    indexed = _index_by_trade_date(frame)

    selected = _day(indexed, "20260722")
    missing = _day(indexed, "20260723")

    assert selected["ts_code"].tolist() == ["600001.SH", "600002.SH"]
    assert missing.empty
    assert missing.columns.tolist() == frame.columns.tolist()


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
