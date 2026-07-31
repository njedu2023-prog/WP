from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wp.v3.features import MARKET_FEATURE_COLUMNS
from wp.v3.v22_market_license import (
    MARKET_AGGREGATE_FEATURES,
    FrozenMarketLicensePolicy,
    MarketLicensePolicySpec,
    apply_market_license_policy,
    build_market_slot_leaders,
    calibrate_market_license_policy,
    fit_market_license,
    rolling_market_license_segments,
    v22_research_readiness,
)


def test_market_slot_leaders_do_not_use_future_truth() -> None:
    source = _source_frame(days=3, stocks=8, seed=22)
    first = build_market_slot_leaders(source)
    changed = source.copy()
    changed["target_net_positive"] = 1 - changed["target_net_positive"]
    changed["net_return_pct"] = -changed["net_return_pct"] + 100.0
    second = build_market_slot_leaders(changed)

    identity = ["trade_date", "signal_slot", "ts_code"]
    pd.testing.assert_frame_equal(first[identity], second[identity])
    assert first.groupby(["trade_date", "signal_slot"]).size().eq(1).all()
    assert first["v22_eligible_count"].eq(8).all()
    assert first["v22_score_top_margin"].ge(0.0).all()


def test_market_license_model_uses_only_market_context() -> None:
    leaders = build_market_slot_leaders(
        _source_frame(days=75, stocks=10, seed=2_222)
    )
    dates = sorted(leaders["trade_date"].unique())
    train = leaders.loc[leaders["trade_date"].isin(dates[:52])]
    calibration = leaders.loc[
        leaders["trade_date"].isin(dates[52:64])
    ]
    test = leaders.loc[leaders["trade_date"].isin(dates[64:])]

    bundle = fit_market_license(
        train,
        calibration,
        random_seed=22,
        minimum_train_rows=300,
        minimum_calibration_rows=70,
    )
    prediction_input = test.drop(
        columns=["net_return_pct", "target_net_positive"],
    )
    scored = bundle.predict(prediction_input)

    assert len(bundle.feature_columns) >= 12
    assert all(
        feature.startswith("v22_market_")
        or feature in MARKET_AGGREGATE_FEATURES
        for feature in bundle.feature_columns
    )
    assert not any(
        token in feature
        for feature in bundle.feature_columns
        for token in ("target", "net_return", "future", "truth")
    )
    assert scored["v22_license_probability"].between(0.001, 0.999).all()
    assert scored["v22_license_probability_lower"].between(
        0.001,
        0.999,
    ).all()


def test_policy_locks_first_signal_and_daily_maximum() -> None:
    policy = FrozenMarketLicensePolicy(
        spec=MarketLicensePolicySpec(
            target_candidate_day_rate=0.12,
            max_candidates_per_day=2,
        ),
        license_probability_lower_threshold=0.60,
        threshold_calibration_start="20260701",
        threshold_calibration_end="20260720",
        threshold_calibration_days=20,
        threshold_eligible_days=10,
    )
    scored = pd.DataFrame(
        [
            _scored_row("20260727", "14:20", "000001.SZ", 0.65),
            _scored_row("20260727", "14:25", "000002.SZ", 0.70),
            _scored_row("20260727", "14:30", "000001.SZ", 0.99),
            _scored_row("20260727", "14:45", "000003.SZ", 0.95),
        ]
    )

    selected = apply_market_license_policy(scored, policy)

    assert len(selected) == 2
    assert set(selected["ts_code"]) == {"000001.SZ", "000002.SZ"}
    first = selected.set_index("ts_code").loc["000001.SZ"]
    assert first["signal_slot"] == "14:20"
    assert first["v22_license_probability_lower"] == 0.65


def test_policy_threshold_uses_fixed_unlabelled_day_rate() -> None:
    dates = [f"202607{day:02d}" for day in range(1, 11)]
    scored = pd.DataFrame(
        [
            _scored_row(
                date,
                "14:20",
                f"{index:06d}.SZ",
                index / 10.0,
            )
            for index, date in enumerate(dates, start=1)
        ]
    )
    spec = MarketLicensePolicySpec(
        target_candidate_day_rate=0.20,
        max_candidates_per_day=2,
    )
    first = calibrate_market_license_policy(
        scored,
        calibration_dates=dates,
        spec=spec,
    )
    changed = scored.copy()
    changed["net_return_pct"] = np.linspace(-100.0, 100.0, len(changed))
    second = calibrate_market_license_policy(
        changed,
        calibration_dates=dates,
        spec=spec,
    )

    assert first.license_probability_lower_threshold == 0.9
    assert second.license_probability_lower_threshold == 0.9
    assert first.threshold_calibration_days == 10


def test_market_license_segments_are_ordered_and_purged() -> None:
    dates = [
        date.strftime("%Y%m%d")
        for date in pd.bdate_range("2025-01-02", periods=180)
    ]

    segments = rolling_market_license_segments(dates)

    assert segments is not None
    train, calibration = segments
    assert len(train) == 126
    assert len(calibration) == 42
    assert train[-1] < calibration[0]
    assert dates.index(calibration[0]) - dates.index(train[-1]) == 3


def test_policy_rejects_incomplete_model_spread_contract() -> None:
    row = _scored_row("20260727", "14:20", "000001.SZ", 0.80)
    row.pop("v22_license_model_spread")
    policy = FrozenMarketLicensePolicy(
        spec=MarketLicensePolicySpec(),
        license_probability_lower_threshold=0.60,
        threshold_calibration_start="20260701",
        threshold_calibration_end="20260720",
        threshold_calibration_days=20,
        threshold_eligible_days=10,
    )

    with pytest.raises(ValueError, match="v22_license_model_spread"):
        apply_market_license_policy(pd.DataFrame([row]), policy)


def test_readiness_requires_frequency_economics_and_stability() -> None:
    metrics = {
        "events": 100,
        "candidate_days": 70,
        "candidate_day_rate": 0.12,
        "win_rate": 0.58,
        "win_rate_wilson_lower": 0.50,
        "clustered_win_rate_lower": 0.50,
        "margin_hit_rate": 0.48,
        "tail_loss_rate": 0.12,
        "mean_net_return_pct": 0.40,
        "clustered_mean_lower_pct": 0.05,
        "profit_factor": 1.35,
        "stress_50bps_mean_net_return_pct": 0.01,
        "return_p10_pct": -2.5,
    }
    yearly = [
        {"year": "2024", "events": 30, "mean_net_return_pct": 0.30},
        {"year": "2025", "events": 30, "mean_net_return_pct": 0.20},
        {"year": "2026", "events": 40, "mean_net_return_pct": 0.50},
    ]

    ready = v22_research_readiness(
        metrics,
        yearly=yearly,
        temporal_integrity=True,
        source_integrity=True,
    )
    assert ready["all_historical_gates_passed"]

    failed = dict(metrics)
    failed["stress_50bps_mean_net_return_pct"] = -0.01
    assert not v22_research_readiness(
        failed,
        yearly=yearly,
        temporal_integrity=True,
        source_integrity=True,
    )["all_historical_gates_passed"]


def _source_frame(
    *,
    days: int,
    stocks: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-06", periods=days)
    slots = (
        "14:20",
        "14:25",
        "14:30",
        "14:35",
        "14:40",
        "14:45",
        "14:50",
    )
    rows: list[dict[str, object]] = []
    for date_index, date in enumerate(dates):
        regime = rng.normal(0.0, 0.8)
        quality = rng.normal(0.0, 1.0, stocks)
        returns = (
            0.80 * regime
            + 0.35 * quality
            + rng.normal(0.0, 1.0, stocks)
        )
        for slot_index, slot in enumerate(slots):
            market_features = {
                column: (
                    regime
                    + 0.03 * slot_index
                    + 0.002 * date_index
                    + 0.01 * feature_index
                )
                for feature_index, column in enumerate(
                    MARKET_FEATURE_COLUMNS
                )
            }
            market_features["slot_minute"] = float(slot_index * 5)
            for stock in range(stocks):
                probability = np.clip(
                    0.50
                    + 0.07 * quality[stock]
                    + 0.02 * regime
                    + 0.003 * slot_index
                    + rng.normal(0.0, 0.02),
                    0.05,
                    0.95,
                )
                intraday = (
                    1.5
                    + quality[stock]
                    + 0.05 * slot_index
                    + regime
                    + rng.normal(0.0, 0.12)
                )
                rows.append(
                    {
                        "trade_date": date.strftime("%Y%m%d"),
                        "signal_slot": slot,
                        "ts_code": f"{stock:06d}.SZ",
                        "fold": date_index // 42 + 1,
                        "execution_eligible": True,
                        "label_available": True,
                        "entry_fillable": True,
                        "exit_fillable": True,
                        "net_return_pct": returns[stock],
                        "target_net_positive": float(returns[stock] > 0),
                        "ret_from_prev_close_pct": intraday,
                        "p_round_trip_fill_lower": 0.98,
                        "p_net_positive_lower": probability - 0.03,
                        "p_severe_loss": np.clip(
                            0.40 - probability,
                            0.02,
                            0.35,
                        ),
                        "selection_score": probability + intraday * 0.01,
                        "expected_utility_lower_pct": probability - 0.50,
                        "probability_model_spread": 0.05,
                        "expected_return_model_spread": 0.10,
                        "data_age_seconds": 30.0,
                        **market_features,
                    }
                )
    return pd.DataFrame(rows)


def _scored_row(
    trade_date: str,
    slot: str,
    ts_code: str,
    license_probability_lower: float,
) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "signal_slot": slot,
        "ts_code": ts_code,
        "net_return_pct": 1.0,
        "v22_license_probability_lower": float(
            license_probability_lower
        ),
        "v22_license_model_spread": 0.05,
    }
