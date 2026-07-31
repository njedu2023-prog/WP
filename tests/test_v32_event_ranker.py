from __future__ import annotations

import numpy as np
import pandas as pd

from wp.v3.v32_event_features import (
    ADMITTED_SOURCES,
    V32_EVENT_FEATURE_COLUMNS,
)
from wp.v3.v32_event_ranker import (
    EventPolicySpec,
    FrozenEventPolicy,
    apply_event_policy,
    event_any,
    fit_event_ranker,
    validate_feature_contract,
    validate_selected_contract,
)


def synthetic_rows(count: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = pd.DataFrame(
        {
            "trade_date": [
                (
                    f"2025{1 + (index // (28 * 6)) % 12:02d}"
                    f"{1 + (index // 6) % 28:02d}"
                )
                for index in range(count)
            ],
            "signal_slot": [
                "14:20" if index % 6 < 3 else "14:25"
                for index in range(count)
            ],
            "ts_code": [
                f"{index % 3:06d}.SZ" for index in range(count)
            ],
            "label_available": True,
            "v23_point_in_time_complete": True,
            "v32_event_features_complete": True,
            "p_round_trip_fill_lower": 0.99,
            "p_severe_loss": 0.10,
            "data_age_seconds": 30.0,
        }
    )
    for offset, column in enumerate(V32_EVENT_FEATURE_COLUMNS):
        rows[column] = rng.normal(
            loc=(offset % 4) * 0.05,
            scale=1.0,
            size=count,
        )
    for source in ADMITTED_SOURCES:
        count_column = f"v32_{source}_event_count_5d"
        active_column = f"v32_{source}_active_5d"
        rows[count_column] = (rng.random(count) > 0.45).astype(float)
        rows[active_column] = rows[count_column].gt(0).astype(float)
    signal = (
        rows["v32_forecast_p_change_mid_mean"]
        - 0.5 * rows["v32_block_trade_latest_price_to_signal_pct"]
        + rng.normal(0, 0.8, count)
    )
    rows["net_return_pct"] = signal
    rows["target_net_positive"] = signal.gt(0).astype(float)
    rows["v32_event_any"] = event_any(rows)
    return rows


def test_event_ranker_fits_frozen_feature_family() -> None:
    train = synthetic_rows(180, seed=1)
    calibration = synthetic_rows(90, seed=2)

    bundle = fit_event_ranker(
        train,
        calibration,
        random_seed=7,
        minimum_train_rows=100,
        minimum_calibration_rows=50,
        minimum_train_pair_rows=100,
        minimum_calibration_pair_rows=40,
    )
    scored = bundle.predict(calibration)

    assert validate_feature_contract(bundle.feature_columns)
    assert scored["v32_p_positive"].between(0, 1).all()
    assert scored["v32_p_margin"].between(0, 1).all()
    assert scored["v32_p_severe_loss"].between(0, 1).all()
    assert scored["v32_within_slot_rank_score"].between(0, 1).all()


def test_policy_keeps_first_event_signal_and_daily_cap() -> None:
    frame = synthetic_rows(12, seed=4)
    frame["trade_date"] = "20260723"
    frame["ts_code"] = [
        "000001.SZ",
        "000001.SZ",
        "000002.SZ",
        "000003.SZ",
    ] * 3
    frame["signal_slot"] = [
        "14:20",
        "14:25",
        "14:20",
        "14:20",
    ] * 3
    frame["v32_p_positive"] = 0.80
    frame["v32_p_margin"] = 0.60
    frame["v32_p_severe_loss"] = 0.10
    frame["v32_within_slot_rank_score"] = 0.70
    frame["v32_event_score"] = np.linspace(1.0, 0.1, len(frame))
    frame["v32_event_any"] = True
    policy = FrozenEventPolicy(
        spec=EventPolicySpec(max_candidates_per_day=2),
        score_threshold=0.0,
        calibration_start="20260101",
        calibration_end="20260301",
        calibration_days=42,
        eligible_days=20,
    )

    selected = apply_event_policy(frame, policy)
    validate_selected_contract(selected, policy)

    assert len(selected) <= 2
    assert not selected.duplicated(["trade_date", "ts_code"]).any()
    first = selected.loc[selected["ts_code"].eq("000001.SZ")]
    if not first.empty:
        assert first.iloc[0]["signal_slot"] == "14:20"
