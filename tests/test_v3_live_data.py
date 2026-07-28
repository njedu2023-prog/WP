from pathlib import Path

import pandas as pd

from wp.v3.history import _normalize_historical_minutes, _slot_features
from wp.v3.live_data import (
    capture_entry_settlement_frame,
    _load_rt_min_session_snapshots,
    _normalize_rt_k_day,
    _normalize_rt_min,
)
from wp.v3.contracts import V3Config


class Client:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir


class SettlementClient(Client):
    def query(self, api_name: str, **params):
        if api_name == "rt_min":
            return _rt_min_bar(10.2, 20_000_000, "14:55:00")
        if api_name == "stk_limit":
            return pd.DataFrame(
                [{"ts_code": "600000.SH", "up_limit": 11.0}]
            )
        raise AssertionError(api_name)


def _rt_min_bar(
    close: float,
    amount: float,
    trade_time: str,
    *,
    high: float | None = None,
    low: float | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["600000.SH"],
            "time": [trade_time],
            "open": [close - 0.02],
            "high": [high if high is not None else close + 0.05],
            "low": [low if low is not None else close - 0.05],
            "close": [close],
            "vol": [100_000],
            "amount": [amount],
        }
    )


def test_rt_min_session_uses_cached_true_five_minute_bars(tmp_path):
    cache = tmp_path / "rt_min_session"
    cache.mkdir()
    prior = _normalize_rt_min(
        _rt_min_bar(10.2, 10_000_000, "14:20:00"),
        trade_date="20260721",
    )
    prior.to_parquet(
        cache / "20260721_1420_main.parquet",
        index=False,
    )
    current = _normalize_rt_min(
        _rt_min_bar(10.3, 11_000_000, "14:25:01"),
        trade_date="20260721",
    )
    result = _load_rt_min_session_snapshots(
        Client(tmp_path),
        trade_date="20260721",
        observation_slot="14:25",
        current=current,
        observation_slots=("14:20", "14:25", "14:30"),
    )
    assert result["close"].tolist() == [10.2, 10.3]
    assert result["slot_amount"].tolist() == [10_000_000, 11_000_000]


def test_live_and_historical_slot_features_have_identical_bar_semantics():
    times = ("14:00", "14:05", "14:10", "14:15", "14:20")
    realtime = pd.concat(
        [
            _normalize_rt_min(
                _rt_min_bar(
                    10.0 + index * 0.1,
                    8_000_000 + index * 1_000_000,
                    f"{slot}:00",
                ),
                trade_date="20260721",
            )
            for index, slot in enumerate(times)
        ],
        ignore_index=True,
    )
    historical_raw = realtime.rename(
        columns={"trade_time": "trade_time"}
    ).drop(columns="slot_amount")
    historical_raw["ts_code"] = "600000.SH"
    historical = _normalize_historical_minutes(
        historical_raw,
        signal_slots=("14:20",),
    )

    live_features = _slot_features(realtime, "14:20")
    historical_features = _slot_features(historical, "14:20")
    columns = [
        "slot_close",
        "slot_amount",
        "intraday_snapshot_count",
        "ret_5m_pct",
        "ret_10m_pct",
        "ret_20m_pct",
        "tail_range_10m_pct",
        "tail_close_position_10m",
        "tail_amount_acceleration",
    ]
    pd.testing.assert_frame_equal(
        live_features[columns],
        historical_features[columns],
    )


def test_realtime_day_contract_carries_authoritative_previous_close():
    normalized = _normalize_rt_k_day(
        pd.DataFrame(
            [
                {
                    "ts_code": "600000.SH",
                    "pre_close": 9.85,
                    "open": 9.90,
                    "trade_time": "2026-07-21 14:20:00",
                }
            ]
        )
    )
    assert normalized.to_dict(orient="records") == [
        {
            "ts_code": "600000.SH",
            "rt_pre_close": 9.85,
            "day_open": 9.9,
        }
    ]


def test_entry_settlement_uses_only_the_exact_requested_bar(tmp_path):
    frame, manifest = capture_entry_settlement_frame(
        SettlementClient(tmp_path),
        trade_date="20260721",
        settlement_slot="14:55",
        ts_codes=["600000.SH"],
        config=V3Config(),
    )

    assert manifest["observed_symbols"] == 1
    assert frame.loc[0, "entry_benchmark_slot"] == "14:55"
    assert frame.loc[0, "entry_benchmark_price"] == 10.2
    assert frame.loc[0, "entry_benchmark_amount"] == 20_000_000
