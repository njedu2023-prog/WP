from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict

import pandas as pd
import pytest

from wp.v3.contracts import V3Config
from wp.v3.history import (
    PANEL_SCHEMA_VERSION,
    TUSHARE_CACHE_SCHEMA_VERSION,
    TushareHistoryClient,
    _build_prior_day_features,
    _day,
    _index_by_trade_date,
    _industry_at,
    _load_daily_history,
    load_panel_partitions,
    _minute_universe_quality,
    _normalize_historical_minutes,
    _ordered_bounded_map,
    _panel_builder_fingerprint,
    _query_historical_minutes_incremental,
    _reusable_panel_manifest,
    _minute_normalizer_fingerprint,
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


class _MinuteGapPro:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def query(self, _api_name: str, **params):
        self.calls.append(dict(params))
        if int(params.get("offset", 0)) > 0:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "ts_code": params["ts_code"],
                    "trade_time": params["start_date"],
                    "open": 8.0,
                    "high": 8.1,
                    "low": 7.9,
                    "close": 8.0,
                    "vol": 100.0,
                    "amount": 1_000.0,
                }
            ]
        )


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


def test_minute_history_reuses_existing_suffix_and_fetches_only_missing_prefix(
    tmp_path,
):
    cache_dir = tmp_path / "stk_mins"
    cache_dir.mkdir()
    pd.DataFrame(
        [
            {
                "ts_code": "600000.SH",
                "trade_time": "2023-07-27 14:20:00",
                "open": 10.0,
                "high": 10.1,
                "low": 9.9,
                "close": 10.0,
                "vol": 100.0,
                "amount": 1_000.0,
            }
        ]
    ).to_parquet(
        cache_dir / "600000_SH_20230727_20260724_5min.parquet",
        index=False,
    )
    pro = _MinuteGapPro()
    client = TushareHistoryClient(
        pro,
        tmp_path,
        requests_per_minute=100_000,
    )

    result = _query_historical_minutes_incremental(
        client,
        ts_code="600000.SH",
        start_date="20210726",
        end_date="20260724",
    )

    first_page = [
        call for call in pro.calls if int(call.get("offset", 0)) == 0
    ]
    assert len(first_page) == 1
    assert first_page[0]["start_date"] == "2021-07-26 09:30:00"
    assert first_page[0]["end_date"] == "2023-07-26 15:00:00"
    assert result["trade_time"].tolist() == [
        "2021-07-26 09:30:00",
        "2023-07-27 14:20:00",
    ]
    assert (
        cache_dir / "600000_SH_20210726_20260724_5min.parquet"
    ).exists()


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
    assert first["tail_return_from_1400_pct"] == pytest.approx(40.0)
    assert first["tail_cumulative_amount"] == pytest.approx(160.0)
    assert first["tail_latest_amount_share"] == pytest.approx(60.0 / 160.0)
    assert first["tail_amount_concentration"] == pytest.approx(60.0 / 160.0)
    assert first["tail_directional_efficiency"] > 0
    assert 0 <= first["tail_trend_r2"] <= 1
    assert features.loc["600002.SH", "intraday_snapshot_count"] == 2
    assert pd.isna(features.loc["600002.SH", "ret_10m_pct"])


def test_tail_features_cannot_change_when_a_future_bar_is_appended():
    times = pd.date_range("2026-07-27 14:00:00", periods=6, freq="5min")
    bars = pd.DataFrame(
        {
            "ts_code": ["600001.SH"] * 6,
            "trade_time": times,
            "open": [10.0, 10.1, 10.2, 10.3, 10.4, 99.0],
            "high": [10.2, 10.3, 10.4, 10.5, 10.6, 120.0],
            "low": [9.9, 10.0, 10.1, 10.2, 10.3, 1.0],
            "close": [10.1, 10.2, 10.3, 10.4, 10.5, 110.0],
            "amount": [10, 20, 30, 40, 50, 9_999_999],
        }
    )
    feature_columns = [
        column
        for column in _slot_features(bars.iloc[:5], "14:20").columns
        if column != "slot_bar_time"
    ]
    before = _slot_features(bars.iloc[:5], "14:20")[feature_columns].iloc[0]
    after = _slot_features(bars, "14:20")[feature_columns].iloc[0]

    pd.testing.assert_series_equal(before, after)


def test_prior_day_features_are_strictly_lagged():
    dates = pd.bdate_range("2026-01-01", periods=25)
    daily = pd.DataFrame(
        {
            "ts_code": ["600001.SH"] * len(dates),
            "trade_date": dates.strftime("%Y%m%d"),
            "open": [10.0 + index for index in range(len(dates))],
            "high": [10.5 + index for index in range(len(dates))],
            "low": [9.5 + index for index in range(len(dates))],
            "close": [10.2 + index for index in range(len(dates))],
            "pre_close": [9.8 + index for index in range(len(dates))],
            "pct_chg": [1.0] * len(dates),
            "amount": [100.0 + index for index in range(len(dates))],
            "adj_factor": [1.0] * len(dates),
        }
    )
    basic = pd.DataFrame(
        {
            "ts_code": ["600001.SH"] * len(dates),
            "trade_date": dates.strftime("%Y%m%d"),
            "turnover_rate": range(1, len(dates) + 1),
            "volume_ratio": [1.0] * len(dates),
            "pe_ttm": [12.0] * len(dates),
            "pb": [1.5] * len(dates),
            "total_mv": [1_000.0] * len(dates),
            "circ_mv": [800.0] * len(dates),
        }
    )
    features = _build_prior_day_features(daily, basic)
    last = features.iloc[-1]
    prior = daily.iloc[-2]

    assert last["prev_day_gap_pct"] == pytest.approx(
        (prior["open"] / prior["pre_close"] - 1.0) * 100.0
    )
    assert last["prev_turnover_rate"] == basic.iloc[-2]["turnover_rate"]
    assert last["prev_volume_ratio"] == basic.iloc[-2]["volume_ratio"]


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


def test_verified_panel_cache_is_reused_only_for_the_same_contract(tmp_path):
    config = V3Config()
    partition_dir = tmp_path / "panel"
    partition_dir.mkdir()
    partition = partition_dir / "wp_v3_panel_202607.parquet"
    partition.write_bytes(b"verified-panel")
    manifest = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "strategy_id": config.strategy.strategy_id,
        "feature_version": config.model.feature_version,
        "panel_builder_fingerprint": _panel_builder_fingerprint(),
        "minute_normalizer_fingerprint": _minute_normalizer_fingerprint(),
        "tushare_cache_schema_version": TUSHARE_CACHE_SCHEMA_VERSION,
        "requested_start": config.history.start_date,
        "requested_end": config.history.end_date,
        "execution_contract": asdict(config.execution),
        "signal_slots": list(config.strategy.signal_slots),
        "exit_contract": config.strategy.exit_contract,
        "coverage": 1.0,
        "failed_trade_days": [],
        "partitions": [
            {
                "path": "artifacts/wp_v3_history/panel/wp_v3_panel_202607.parquet",
                "sha256": hashlib.sha256(b"verified-panel").hexdigest(),
            }
        ],
    }
    manifest_path = tmp_path / "wp_v3_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reused = _reusable_panel_manifest(
        manifest_path,
        partition_dir=partition_dir,
        config=config,
    )
    assert reused is not None
    assert reused["requested_start"] == manifest["requested_start"]
    assert reused["partitions"] == manifest["partitions"]

    incompatible = {
        **manifest,
        "strategy_id": "wp_t1_net_profit_v8",
        "feature_version": "wp_v8_causal_features_1",
        "panel_builder_fingerprint": "legacy-builder",
    }
    manifest_path.write_text(json.dumps(incompatible), encoding="utf-8")
    assert (
        _reusable_panel_manifest(
            manifest_path,
            partition_dir=partition_dir,
            config=config,
        )
        is None
    )

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    partition.write_bytes(b"tampered")
    assert (
        _reusable_panel_manifest(
            manifest_path,
            partition_dir=partition_dir,
            config=config,
        )
        is None
    )


def test_panel_loader_prunes_months_dates_and_columns(tmp_path):
    first = pd.DataFrame(
        {
            "trade_date": ["20260130", "20260131"],
            "ts_code": ["600001.SH", "600002.SH"],
            "unused": [1.0, 2.0],
        }
    )
    second = pd.DataFrame(
        {
            "trade_date": ["20260202", "20260203"],
            "ts_code": ["600003.SH", "600004.SH"],
            "unused": [3.0, 4.0],
        }
    )
    first.to_parquet(tmp_path / "wp_v3_panel_202601.parquet", index=False)
    second.to_parquet(tmp_path / "wp_v3_panel_202602.parquet", index=False)

    loaded = load_panel_partitions(
        tmp_path,
        columns=["trade_date", "ts_code"],
        start_date="20260202",
        end_date="20260202",
    )

    assert loaded.to_dict(orient="records") == [
        {"trade_date": "20260202", "ts_code": "600003.SH"}
    ]
