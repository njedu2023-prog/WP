from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wp.v3.v22_market_license import MARKET_AGGREGATE_FEATURES
from wp.v3.v39_derivatives_daily import (
    V39_FUTURE_FEATURE_COLUMNS,
    V39_OPTION_FEATURE_COLUMNS,
)
from wp.v3.v39_derivatives_license import (
    DERIVATIVE_FEATURES,
    MODEL_FEATURES,
    DerivativesPolicySpec,
    FrozenDerivativesPolicy,
    apply_policy,
    fit_derivatives_license,
    join_derivative_features,
    rolling_segments,
    validate_feature_contract,
)


def test_join_derivatives_requires_strictly_prior_source_date() -> None:
    leaders = pd.DataFrame(
        {
            "trade_date": ["20260105"],
            "signal_slot": ["14:20"],
            "ts_code": ["000001.SZ"],
        }
    )
    derivatives = _derivatives(["20260105"], ["20260105"])
    with pytest.raises(RuntimeError, match="T-1 causality"):
        join_derivative_features(leaders, derivatives)


def test_join_derivatives_is_many_to_one_and_complete() -> None:
    leaders = pd.DataFrame(
        {
            "trade_date": ["20260105", "20260105"],
            "signal_slot": ["14:20", "14:25"],
            "ts_code": ["000001.SZ", "000002.SZ"],
        }
    )
    derivatives = _derivatives(["20260105"], ["20251231"])
    joined = join_derivative_features(leaders, derivatives)
    assert len(joined) == 2
    assert joined["v39_derivatives_complete"].all()
    assert joined["source_trade_date"].eq("20251231").all()
    assert joined["v39_signal_slot_minute"].tolist() == [860, 865]


def test_feature_contract_requires_both_information_families() -> None:
    assert validate_feature_contract(tuple(MODEL_FEATURES))
    incomplete = tuple(
        column
        for column in MODEL_FEATURES
        if column not in V39_OPTION_FEATURE_COLUMNS
    )
    with pytest.raises(RuntimeError, match="feature contract"):
        validate_feature_contract(incomplete)


def test_rolling_segments_reserve_both_purges() -> None:
    dates = [f"{index:08d}" for index in range(1, 400)]
    train, calibration = rolling_segments(dates) or ([], [])
    assert len(train) == 252
    assert len(calibration) == 42
    assert int(calibration[0]) - int(train[-1]) == 3
    assert int(dates[-1]) - int(calibration[-1]) == 2


def test_policy_keeps_first_signal_and_daily_maximum() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20260105"] * 5,
            "signal_slot": ["14:20", "14:25", "14:30", "14:35", "14:40"],
            "ts_code": [
                "000001.SZ",
                "000001.SZ",
                "000002.SZ",
                "000003.SZ",
                "000004.SZ",
            ],
            "signal_price": [10.0, 10.1, 20.0, 30.0, 40.0],
            "v39_derivatives_complete": [True] * 5,
            "v39_positive_model_spread": [0.1] * 5,
            "v39_margin_model_spread": [0.1] * 5,
            "v39_severe_model_spread": [0.1] * 5,
            "v39_return_model_spread_pct": [0.2] * 5,
            "v39_derivatives_score": [0.7, 0.9, 0.8, 0.75, 0.74],
        }
    )
    spec = DerivativesPolicySpec(max_candidates_per_day=3)
    policy = FrozenDerivativesPolicy(
        spec=spec,
        score_threshold=0.5,
        calibration_start="20250101",
        calibration_end="20251231",
        calibration_days=42,
        eligible_days=42,
    )
    selected = apply_policy(frame, policy)
    assert selected["ts_code"].tolist() == [
        "000001.SZ",
        "000002.SZ",
        "000003.SZ",
    ]
    assert selected.iloc[0]["signal_slot"] == "14:20"
    assert selected.iloc[0]["signal_price"] == 10.0


def test_fixed_model_fits_and_scores_without_future_columns() -> None:
    frame = _model_frame(240)
    train = frame.iloc[:180].copy()
    calibration = frame.iloc[180:].copy()
    bundle = fit_derivatives_license(
        train,
        calibration,
        random_seed=39,
        minimum_train_rows=100,
        minimum_calibration_rows=40,
    )
    scored = bundle.predict(calibration)
    assert validate_feature_contract(bundle.feature_columns)
    assert np.isfinite(scored["v39_derivatives_score"]).all()
    assert scored["v39_p_positive_lower"].between(0.001, 0.999).all()


def _derivatives(
    dates: list[str],
    previous_dates: list[str],
) -> pd.DataFrame:
    rows = []
    for index, (date, previous) in enumerate(
        zip(dates, previous_dates, strict=True)
    ):
        row: dict[str, object] = {
            "trade_date": date,
            "source_trade_date": previous,
            "v39_tminus1_causal": previous < date,
            "v39_futures_complete": True,
            "v39_options_complete": True,
            "v39_futures_finite": True,
            "v39_options_finite": True,
        }
        for offset, column in enumerate(DERIVATIVE_FEATURES):
            row[column] = float(index + offset / 100.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _model_frame(rows: int) -> pd.DataFrame:
    records = []
    for index in range(rows):
        date_index = index // 6
        date = (
            pd.Timestamp("2024-01-02")
            + pd.Timedelta(int(date_index), unit="D")
        )
        slot = 20 + (index % 6) * 5
        phase = index / 11.0
        record: dict[str, object] = {
            "trade_date": date.strftime("%Y%m%d"),
            "signal_slot": f"14:{slot:02d}",
            "ts_code": f"{index % 31:06d}.SZ",
            "signal_price": 10.0 + index / 100.0,
            "label_available": True,
            "v39_derivatives_complete": True,
            "net_return_pct": (
                2.8 * np.sin(phase)
                + 1.2 * np.cos(index / 7.0)
                - 0.10
            ),
            "v39_signal_slot_minute": 14 * 60 + slot,
        }
        for offset, column in enumerate(MARKET_AGGREGATE_FEATURES):
            record[column] = np.sin(phase + offset / 13.0)
        for offset, column in enumerate(DERIVATIVE_FEATURES):
            record[column] = (
                np.cos(date_index / 9.0 + offset / 17.0)
                + 0.01 * (date_index % (offset % 7 + 2))
            )
        records.append(record)
    return pd.DataFrame(records)
