from __future__ import annotations

import numpy as np
import pandas as pd

from wp.v3.v35_regime_gate import (
    MINIMUM_CALIBRATION_ROWS,
    MINIMUM_TRAIN_ROWS,
    MODEL_CALIBRATION_DAYS,
    MODEL_TRAIN_DAYS,
    PATH_MEDIAN_BASE_FEATURES,
    V35_REGIME_FEATURES,
    FrozenRegimePolicy,
    RegimePolicySpec,
    apply_regime_policy_to_slots,
    build_regime_slot_frame,
    calibrate_regime_policy,
    fit_regime_license,
    rolling_regime_segments,
    select_regime_candidates,
    validate_feature_contract,
)


def test_minimum_rows_follow_frozen_day_windows() -> None:
    assert MINIMUM_TRAIN_ROWS == MODEL_TRAIN_DAYS
    assert MINIMUM_CALIBRATION_ROWS == MODEL_CALIBRATION_DAYS


def test_regime_features_are_outcome_blind() -> None:
    source = _candidate_frame(days=4, stocks=5, seed=35)
    first = build_regime_slot_frame(source)
    changed = source.copy()
    changed["net_return_pct"] = -changed["net_return_pct"] + 100.0
    changed["target_net_positive"] = (
        pd.to_numeric(changed["net_return_pct"]) > 0.0
    ).astype(float)
    second = build_regime_slot_frame(changed)

    columns = [
        "trade_date",
        "signal_slot",
        "fold",
        "v35_basket_member_codes",
        *V35_REGIME_FEATURES,
    ]
    pd.testing.assert_frame_equal(first[columns], second[columns])
    assert not first["v35_target_good"].equals(second["v35_target_good"])


def test_regime_model_uses_only_preregistered_aggregates() -> None:
    source = _candidate_frame(days=75, stocks=5, seed=3_535)
    slots = build_regime_slot_frame(source)
    dates = sorted(slots["trade_date"].unique())
    train = slots.loc[slots["trade_date"].isin(dates[:55])]
    calibration = slots.loc[slots["trade_date"].isin(dates[55:67])]
    test = slots.loc[slots["trade_date"].isin(dates[67:])]

    bundle = fit_regime_license(
        train,
        calibration,
        random_seed=35,
        minimum_train_rows=300,
        minimum_calibration_rows=70,
    )
    scored = bundle.predict(
        test.drop(
            columns=[
                "v35_basket_mean_net_return_pct",
                "v35_basket_min_net_return_pct",
                "v35_basket_positive_share",
                "v35_target_good",
                "v35_target_margin",
                "v35_target_severe",
            ]
        )
    )

    assert validate_feature_contract(bundle.feature_columns)
    assert set(bundle.feature_columns).issubset(V35_REGIME_FEATURES)
    assert scored["v35_p_good_lower"].between(0.001, 0.999).all()
    assert scored["v35_p_severe_upper"].between(0.001, 0.999).all()
    assert np.isfinite(scored["v35_regime_score"]).all()


def test_policy_threshold_cannot_read_returns_or_labels() -> None:
    dates = [
        date.strftime("%Y%m%d")
        for date in pd.bdate_range("2026-06-01", periods=10)
    ]
    scored = pd.DataFrame(
        [
            _scored_slot(date, "14:20", score=index / 10.0)
            for index, date in enumerate(dates, start=1)
        ]
    )
    spec = RegimePolicySpec(target_candidate_day_rate=0.20)
    first = calibrate_regime_policy(
        scored,
        calibration_dates=dates,
        spec=spec,
    )
    changed = scored.copy()
    changed["v35_basket_mean_net_return_pct"] = np.linspace(
        -100.0,
        100.0,
        len(changed),
    )
    changed["v35_target_good"] = 1.0
    second = calibrate_regime_policy(
        changed,
        calibration_dates=dates,
        spec=spec,
    )

    assert first.score_threshold == 0.9
    assert second.score_threshold == 0.9


def test_policy_uses_first_licensed_slot_and_preserves_signal_price() -> None:
    policy = FrozenRegimePolicy(
        spec=RegimePolicySpec(),
        score_threshold=0.60,
        calibration_start="20260501",
        calibration_end="20260630",
        calibration_days=42,
        eligible_days=10,
    )
    scored = pd.DataFrame(
        [
            _scored_slot("20260727", "14:20", score=0.70),
            _scored_slot("20260727", "14:25", score=0.95),
        ]
    )
    licensed = apply_regime_policy_to_slots(scored, policy)
    source = _candidate_frame(days=1, stocks=5, seed=350)
    source["trade_date"] = "20260727"
    source = source.loc[
        source["signal_slot"].isin(["14:20", "14:25"])
    ].copy()
    original = source.set_index(
        ["trade_date", "signal_slot", "ts_code"]
    )["signal_price"]

    selected = select_regime_candidates(source, licensed, policy)

    assert licensed["signal_slot"].tolist() == ["14:20"]
    assert len(selected) == 3
    assert selected["signal_slot"].eq("14:20").all()
    for row in selected.itertuples(index=False):
        key = (row.trade_date, row.signal_slot, row.ts_code)
        assert row.signal_price == original.loc[key]


def test_regime_segments_are_ordered_and_double_purged() -> None:
    dates = [
        date.strftime("%Y%m%d")
        for date in pd.bdate_range("2024-01-02", periods=298)
    ]
    segments = rolling_regime_segments(dates)
    assert segments is not None
    train, calibration = segments
    assert len(train) == 252
    assert len(calibration) == 42
    assert dates.index(calibration[0]) - dates.index(train[-1]) == 3
    assert dates.index(dates[-1]) - dates.index(calibration[-1]) == 2


def _candidate_frame(
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
        regime = float(np.sin(date_index / 4.0))
        for slot_index, slot in enumerate(slots):
            for stock in range(stocks):
                stock_noise = rng.normal(0.0, 0.10)
                net = (
                    0.70 * regime
                    + 0.04 * slot_index
                    + 0.03 * (stocks - stock)
                    + rng.normal(0.0, 0.35)
                )
                if (date_index + slot_index) % 11 == 0 and stock == 2:
                    net = -2.50
                path = {
                    column: (
                        regime
                        + 0.03 * slot_index
                        + 0.02 * stock
                        + 0.001 * feature_index
                        + stock_noise
                    )
                    for feature_index, column in enumerate(
                        PATH_MEDIAN_BASE_FEATURES
                    )
                }
                rows.append(
                    {
                        "trade_date": date.strftime("%Y%m%d"),
                        "signal_slot": slot,
                        "ts_code": f"{stock:06d}.SZ",
                        "fold": date_index // 42 + 1,
                        "signal_price": 10.0 + stock + slot_index * 0.01,
                        "v20_stock_rank_in_slot": stock + 1,
                        "selection_score": 1.0 - 0.05 * stock,
                        "p_net_positive_lower": 0.60 - 0.01 * stock,
                        "execution_eligible": True,
                        "p_round_trip_fill_lower": 0.98,
                        "p_severe_loss": 0.10,
                        "probability_model_spread": 0.05,
                        "expected_return_model_spread": 0.10,
                        "data_age_seconds": 30.0,
                        "v23_point_in_time_complete": True,
                        "v34_path_complete": True,
                        "label_available": True,
                        "net_return_pct": net,
                        "target_net_positive": float(net > 0.0),
                        **path,
                    }
                )
    return pd.DataFrame(rows)


def _scored_slot(
    trade_date: str,
    signal_slot: str,
    *,
    score: float,
) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "signal_slot": signal_slot,
        "fold": 1,
        "v35_basket_member_count": 3,
        "v35_regime_score": score,
        "v35_good_model_spread": 0.05,
        "v35_margin_model_spread": 0.05,
        "v35_severe_model_spread": 0.05,
        "v35_return_model_spread_pct": 0.10,
        "v35_p_good_lower": 0.60,
        "v35_p_margin_lower": 0.30,
        "v35_p_severe_upper": 0.15,
        "v35_expected_basket_mean_lower_pct": 0.20,
    }
