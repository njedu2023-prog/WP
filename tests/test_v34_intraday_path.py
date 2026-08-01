from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from wp.v3.v34_intraday_path import (
    V34_INTRADAY_PATH_FEATURE_COLUMNS,
    audit_intraday_path_coverage,
    build_intraday_path_features,
    expected_minute_rows,
    normalize_historical_minutes,
    normalize_rt_min_daily,
)


def _minutes(trade_date: str = "20260723") -> pd.DataFrame:
    morning = pd.date_range(
        f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 09:30:00",
        f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 11:30:00",
        freq="1min",
    )
    afternoon = pd.date_range(
        f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 13:01:00",
        f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 14:50:00",
        freq="1min",
    )
    times = morning.append(afternoon)
    sequence = np.arange(len(times), dtype=float)
    close = 10.0 + sequence * 0.001 + np.sin(sequence / 7.0) * 0.01
    return pd.DataFrame(
        {
            "ts_code": "600000.SH",
            "trade_time": times,
            "open": close - 0.002,
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
            "vol": 1000.0 + sequence * 3.0,
            "amount": (1000.0 + sequence * 3.0) * close,
        }
    )


def _candidate(slot: str, price: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20260723"],
            "signal_slot": [slot],
            "ts_code": ["600000.SH"],
            "fold": [22],
            "signal_price": [price],
        }
    )


def test_expected_rows_follow_a_share_sessions() -> None:
    assert expected_minute_rows("14:20") == 201
    assert expected_minute_rows("14:50") == 231


def test_realtime_and_historical_minutes_normalize_to_same_schema() -> None:
    historical = _minutes().head(3)
    realtime = historical.rename(
        columns={"ts_code": "code", "trade_time": "time"}
    ).assign(freq="1MIN")
    left = normalize_historical_minutes(historical)
    right = normalize_rt_min_daily(realtime)
    pdt.assert_frame_equal(left, right)


def test_post_signal_mutation_cannot_change_earlier_features() -> None:
    minutes = _minutes()
    signal_close = float(
        minutes.loc[
            minutes["trade_time"].dt.strftime("%H:%M").eq("14:20"),
            "close",
        ].iloc[-1]
    )
    candidate = _candidate("14:20", signal_close)
    baseline = build_intraday_path_features(candidate, minutes)
    changed = minutes.copy()
    after = changed["trade_time"].dt.strftime("%H:%M").gt("14:20")
    changed.loc[after, ["open", "high", "low", "close", "amount"]] *= 100.0
    replay = build_intraday_path_features(candidate, changed)
    pdt.assert_frame_equal(
        baseline[list(V34_INTRADAY_PATH_FEATURE_COLUMNS)],
        replay[list(V34_INTRADAY_PATH_FEATURE_COLUMNS)],
    )
    assert baseline.loc[0, "v34_latest_time"].endswith("14:20:00")
    assert bool(baseline.loc[0, "v34_causal_ok"])


def test_complete_path_has_finite_features_and_price_parity() -> None:
    minutes = _minutes()
    signal_close = float(minutes["close"].iloc[-1])
    candidate = _candidate("14:50", signal_close)
    features = build_intraday_path_features(candidate, minutes)
    values = features[list(V34_INTRADAY_PATH_FEATURE_COLUMNS)].to_numpy(
        dtype=float
    )
    assert np.isfinite(values).all()
    assert bool(features.loc[0, "v34_path_complete"])
    assert features.loc[0, "v34_coverage_ratio"] == 1.0
    assert features.loc[0, "v34_signal_price_error_bps"] == 0.0


def test_audit_rejects_forbidden_or_incomplete_data() -> None:
    minutes = _minutes()
    candidate = _candidate("14:50", float(minutes["close"].iloc[-1]))
    features = build_intraday_path_features(candidate, minutes)
    repeated = pd.concat(
        [
            features.assign(
                trade_date=f"202{i:04d}",
                ts_code=f"{600000 + i:06d}.SH",
            )
            for i in range(10)
        ],
        ignore_index=True,
    )
    candidates = repeated[
        ["trade_date", "signal_slot", "ts_code", "fold", "signal_price"]
    ].copy()
    audit = audit_intraday_path_coverage(
        repeated,
        candidates,
        query_failures=0,
    )
    assert not audit["coverage_passed"]
    contaminated = repeated.assign(target_net_positive=1)
    contaminated_audit = audit_intraday_path_coverage(
        contaminated,
        candidates,
        query_failures=0,
    )
    assert "target_net_positive" in contaminated_audit["forbidden_columns"]
