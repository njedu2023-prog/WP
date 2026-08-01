from __future__ import annotations

import numpy as np
import pandas as pd

from wp.v3.v36_entry_confirmation import (
    BASE_ALERT_SLOTS,
    MODEL_FEATURES,
    POST_ALERT_FEATURES,
    EntryConfirmationPolicySpec,
    FrozenEntryConfirmationPolicy,
    apply_entry_confirmation_policy,
    audit_confirmation_feature_coverage,
    build_post_alert_confirmation_features,
    calibrate_entry_confirmation_policy,
    confirmation_timing,
    fit_entry_confirmation_gate,
    rolling_confirmation_segments,
    validate_feature_contract,
)


def test_confirmation_timing_stays_inside_legal_window() -> None:
    assert confirmation_timing("14:20") == ("14:24", "14:25")
    assert confirmation_timing("14:45") == ("14:49", "14:50")


def test_post_alert_features_stop_before_entry_bar() -> None:
    candidates = pd.DataFrame(
        [
            _candidate("20260727", "14:20", "000001.SZ", entry_close=10.10),
            _candidate("20260727", "14:50", "000002.SZ", entry_close=11.10),
        ]
    )
    minutes = _minute_frame("20260727", "000001.SZ", signal_price=10.0)
    first = build_post_alert_confirmation_features(
        candidates,
        minutes,
        entry_slippage_bps=10.0,
    )
    changed = minutes.copy()
    changed.loc[
        changed["trade_time"].astype(str).str.endswith("14:25:00"),
        ["open", "high", "low", "close"],
    ] = 99.0
    second = build_post_alert_confirmation_features(
        candidates,
        changed,
        entry_slippage_bps=10.0,
    )

    assert len(first) == 1
    assert first.iloc[0]["signal_slot"] == "14:20"
    pd.testing.assert_series_equal(
        first.iloc[0][list(POST_ALERT_FEATURES)],
        second.iloc[0][list(POST_ALERT_FEATURES)],
        check_names=False,
    )
    assert bool(first.iloc[0]["v36_path_complete"])
    assert not bool(second.iloc[0]["v36_path_complete"])
    assert first.iloc[0]["v36_confirmation_latest_time"].endswith("14:24:00")


def test_confirmation_coverage_uses_only_legal_base_alerts() -> None:
    candidates = pd.DataFrame(
        [
            _candidate("20260727", "14:20", "000001.SZ", entry_close=10.10),
            _candidate("20260727", "14:50", "000001.SZ", entry_close=10.20),
        ]
    )
    features = build_post_alert_confirmation_features(
        candidates,
        _minute_frame("20260727", "000001.SZ", signal_price=10.0),
        entry_slippage_bps=10.0,
    )
    audit = audit_confirmation_feature_coverage(features, candidates)

    assert audit["expected_rows"] == 1
    assert audit["feature_rows"] == 1
    assert audit["coverage_passed"]


def test_model_uses_only_preregistered_features() -> None:
    frame = _model_frame(days=75, stocks=10, seed=36)
    dates = sorted(frame["trade_date"].unique())
    train = frame.loc[frame["trade_date"].isin(dates[:55])]
    calibration = frame.loc[frame["trade_date"].isin(dates[55:67])]
    test = frame.loc[frame["trade_date"].isin(dates[67:])]

    bundle = fit_entry_confirmation_gate(
        train,
        calibration,
        random_seed=36,
        minimum_train_rows=500,
        minimum_calibration_rows=100,
    )
    scored = bundle.predict(
        test.drop(columns=["net_return_pct", "label_available"])
    )

    assert validate_feature_contract(bundle.feature_columns)
    assert set(bundle.feature_columns).issubset(MODEL_FEATURES)
    assert scored["v36_p_positive_lower"].between(0.001, 0.999).all()
    assert scored["v36_p_severe_loss_upper"].between(0.001, 0.999).all()
    assert np.isfinite(scored["v36_confirmation_score"]).all()


def test_policy_threshold_cannot_read_outcomes() -> None:
    dates = [
        value.strftime("%Y%m%d")
        for value in pd.bdate_range("2026-06-01", periods=10)
    ]
    scored = pd.DataFrame(
        [
            _scored(date, "14:20", f"{index:06d}.SZ", index / 10.0)
            for index, date in enumerate(dates, start=1)
        ]
    )
    spec = EntryConfirmationPolicySpec(target_candidate_day_rate=0.30)
    first = calibrate_entry_confirmation_policy(
        scored,
        calibration_dates=dates,
        spec=spec,
    )
    changed = scored.copy()
    changed["net_return_pct"] = np.linspace(-100.0, 100.0, len(changed))
    changed["target_net_positive"] = 1.0
    second = calibrate_entry_confirmation_policy(
        changed,
        calibration_dates=dates,
        spec=spec,
    )

    assert first.score_threshold == 0.8
    assert second.score_threshold == 0.8


def test_policy_is_causal_and_keeps_first_confirmation() -> None:
    policy = FrozenEntryConfirmationPolicy(
        spec=EntryConfirmationPolicySpec(max_candidates_per_day=3),
        score_threshold=0.60,
        calibration_start="20260501",
        calibration_end="20260630",
        calibration_days=42,
        eligible_days=20,
    )
    scored = pd.DataFrame(
        [
            _scored("20260727", "14:20", "000001.SZ", 0.70),
            _scored("20260727", "14:20", "000002.SZ", 0.80),
            _scored("20260727", "14:20", "000003.SZ", 0.65),
            _scored("20260727", "14:20", "000004.SZ", 0.95),
            _scored("20260727", "14:25", "000001.SZ", 0.99),
        ]
    )
    selected = apply_entry_confirmation_policy(scored, policy)

    assert selected["v36_confirmation_time"].eq("14:24").all()
    assert selected["ts_code"].tolist() == [
        "000004.SZ",
        "000002.SZ",
        "000001.SZ",
    ]


def test_confirmation_segments_are_double_purged() -> None:
    dates = [
        value.strftime("%Y%m%d")
        for value in pd.bdate_range("2024-01-02", periods=298)
    ]
    segments = rolling_confirmation_segments(dates)
    assert segments is not None
    train, calibration = segments

    assert len(train) == 252
    assert len(calibration) == 42
    assert dates.index(calibration[0]) - dates.index(train[-1]) == 3
    assert dates.index(dates[-1]) - dates.index(calibration[-1]) == 2


def _candidate(
    trade_date: str,
    signal_slot: str,
    ts_code: str,
    *,
    entry_close: float,
) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "signal_slot": signal_slot,
        "ts_code": ts_code,
        "fold": 1,
        "signal_price": 10.0 if ts_code == "000001.SZ" else 11.0,
        "entry_price": entry_close * 1.001,
    }


def _minute_frame(
    trade_date: str,
    ts_code: str,
    *,
    signal_price: float,
) -> pd.DataFrame:
    date = pd.Timestamp(
        f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    )
    times = [
        date + pd.Timedelta(f"{14 * 60 + minute}min")
        for minute in range(1, 26)
    ]
    closes = np.linspace(9.90, signal_price, 20).tolist()
    closes.extend([10.02, 10.04, 10.03, 10.08, 10.10])
    rows = []
    for index, (time, close) in enumerate(zip(times, closes, strict=True)):
        rows.append(
            {
                "ts_code": ts_code,
                "trade_time": time,
                "open": close - 0.01,
                "high": close + 0.02,
                "low": close - 0.02,
                "close": close,
                "vol": 1000.0 + index,
                "amount": 1_000_000.0 + index * 10_000.0,
            }
        )
    return pd.DataFrame(rows)


def _model_frame(
    *,
    days: int,
    stocks: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-06", periods=days)
    rows: list[dict[str, object]] = []
    for day_index, date in enumerate(dates):
        regime = np.sin(day_index / 6.0)
        for stock in range(stocks):
            path = rng.normal(0.0, 1.0, len(MODEL_FEATURES))
            net = (
                0.45 * regime
                + 0.30 * path[0]
                + 0.20 * path[-1]
                + rng.normal(0.0, 0.55)
            )
            rows.append(
                {
                    "trade_date": date.strftime("%Y%m%d"),
                    "signal_slot": BASE_ALERT_SLOTS[
                        stock % len(BASE_ALERT_SLOTS)
                    ],
                    "ts_code": f"{stock:06d}.SZ",
                    "v36_path_complete": True,
                    "label_available": True,
                    "net_return_pct": net,
                    **{
                        column: value
                        for column, value in zip(
                            MODEL_FEATURES,
                            path,
                            strict=True,
                        )
                    },
                }
            )
    return pd.DataFrame(rows)


def _scored(
    trade_date: str,
    signal_slot: str,
    ts_code: str,
    score: float,
) -> dict[str, object]:
    confirmation, entry = confirmation_timing(signal_slot)
    return {
        "trade_date": trade_date,
        "signal_slot": signal_slot,
        "ts_code": ts_code,
        "fold": 1,
        "signal_price": 10.0,
        "entry_price": 10.1,
        "v36_confirmation_time": confirmation,
        "v36_entry_benchmark_time": entry,
        "v36_public_signal_price": 10.05,
        "v36_path_complete": True,
        "v36_confirmation_score": score,
        "v36_p_positive_lower": 0.60,
        "v36_p_margin_lower": 0.30,
        "v36_p_severe_loss_upper": 0.15,
        "v36_expected_net_return_lower_pct": 0.20,
        "v36_positive_model_spread": 0.05,
        "v36_margin_model_spread": 0.05,
        "v36_severe_model_spread": 0.05,
        "v36_expected_return_model_spread_pct": 0.10,
        "p_round_trip_fill_lower": 0.98,
        "p_severe_loss": 0.10,
        "execution_eligible": True,
        "data_age_seconds": 30.0,
    }
