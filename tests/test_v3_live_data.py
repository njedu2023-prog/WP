from pathlib import Path

import pandas as pd

from wp.v3.live_data import (
    _completed_five_minute_bars,
    _load_rt_k_session_snapshots,
    _normalize_rt_k,
)


class Client:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir


def _snapshot(close: float, amount: float, trade_time: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["600000.SH"],
            "name": ["浦发银行"],
            "pre_close": [10.0],
            "open": [10.1],
            "high": [10.5],
            "low": [9.9],
            "close": [close],
            "vol": [1_000_000],
            "amount": [amount],
            "trade_time": [trade_time],
        }
    )


def test_rt_k_session_uses_cached_prior_slot_and_average_bar_amount(tmp_path):
    cache = tmp_path / "rt_k"
    cache.mkdir()
    _snapshot(10.2, 400_000_000, "2026-07-21 14:20:02").to_parquet(
        cache / "20260721_1420_main.parquet",
        index=False,
    )
    current = _normalize_rt_k(
        _snapshot(10.3, 450_000_000, "2026-07-21 14:25:01")
    )
    result = _load_rt_k_session_snapshots(
        Client(tmp_path),
        trade_date="20260721",
        signal_slot="14:25",
        current=current,
        signal_slots=("14:20", "14:25", "14:30"),
    )
    assert result["close"].tolist() == [10.2, 10.3]
    assert result["amount"].round(2).tolist() == [
        10_000_000.0,
        round(450_000_000 / 41, 2),
    ]


def test_completed_five_minute_bar_count_matches_a_share_sessions():
    assert _completed_five_minute_bars("14:20") == 40
    assert _completed_five_minute_bars("14:50") == 46
