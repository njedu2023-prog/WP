from __future__ import annotations

import numpy as np
import pandas as pd

from wp.v3.v16_specialists import (
    aggregate_expert_predictions,
    day_temporal_weights,
    fit_specialist,
    prepare_specialist_frame,
    specialist_specs,
)


def base_row(code: str, slot: str = "14:20") -> dict[str, object]:
    return {
        "trade_date": "20260720",
        "signal_slot": slot,
        "ts_code": code,
        "relative_market_return_pct": 1.0,
        "relative_industry_return_pct": 0.5,
        "return_cs_rank": 0.80,
        "tail_trend_slope_pct": 0.2,
        "tail_directional_efficiency": 0.5,
        "tail_close_position_since_1400": 0.8,
        "tail_max_drawdown_pct": -1.0,
        "tail_rebound_from_low_pct": 0.8,
        "ret_5m_pct": 0.3,
        "ret_20m_pct": 1.0,
        "tail_amount_acceleration": 0.5,
        "tail_volume_price_confirmation": 0.4,
        "slot_amount_ratio_cs_rank": 0.8,
        "ret_from_prev_close_pct": 3.0,
        "p_entry_fill": 0.99,
        "p_exit_fill_given_entry": 0.99,
        "p_round_trip_fill_lower": 0.98,
        "p_net_positive": 0.60,
        "p_net_positive_lower": 0.55,
        "p_conditional_net_positive": 0.62,
        "p_severe_loss": 0.10,
        "selection_score": 1.0,
        "selection_rank_pct": 0.98,
        "expected_utility_pct": 0.20,
        "expected_utility_lower_pct": 0.05,
        "downside_q10_pct": -1.0,
        "probability_model_spread": 0.05,
        "fill_probability_model_spread": 0.02,
        "selection_rank_spread": 0.05,
        "expected_return_model_spread": 0.10,
    }


def test_specialist_memberships_are_overlapping_and_causal() -> None:
    frame = prepare_specialist_frame(
        pd.DataFrame(
            [
                base_row("EARLY", "14:20"),
                base_row("LATE", "14:45"),
            ]
        )
    )
    memberships = {
        spec.expert_id: spec.predicate(frame).tolist()
        for spec in specialist_specs()
    }

    assert memberships["early_structure"] == [True, False]
    assert memberships["late_confirmation"] == [False, True]
    assert memberships["market_industry_leader"] == [True, True]
    assert memberships["trend_persistence"] == [True, True]
    assert memberships["pullback_recovery"] == [True, True]
    assert memberships["liquidity_breakout"] == [True, True]


def test_aggregate_expert_predictions_uses_conservative_bounds() -> None:
    source = pd.DataFrame([base_row("A")])
    long = pd.DataFrame(
        [
            {
                "trade_date": "20260720",
                "signal_slot": "14:20",
                "ts_code": "A",
                "expert_id": "one",
                "expert_p_positive": 0.62,
                "expert_p_severe": 0.10,
                "expert_expected_net_return_pct": 0.30,
                "expert_score": 0.40,
            },
            {
                "trade_date": "20260720",
                "signal_slot": "14:20",
                "ts_code": "A",
                "expert_id": "two",
                "expert_p_positive": 0.54,
                "expert_p_severe": 0.20,
                "expert_expected_net_return_pct": 0.10,
                "expert_score": 0.20,
            },
        ]
    )

    result = aggregate_expert_predictions(source, long).iloc[0]

    assert result["expert_count"] == 2
    assert np.isclose(result["expert_p_positive"], 0.58)
    assert np.isclose(result["expert_p_positive_lower"], 0.54)
    assert np.isclose(result["expert_probability_spread"], 0.08)
    assert np.isclose(result["expert_p_severe"], 0.20)
    assert result["expert_expected_return_lower_pct"] < 0.10


def test_day_temporal_weights_equalize_each_day() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20260720", "20260720", "20260721"],
        }
    )

    weights = day_temporal_weights(frame, half_life_days=1000)

    first_day = weights[:2].sum()
    second_day = weights[2]
    assert first_day < second_day
    assert np.isclose(weights.mean(), 1.0)


def test_specialist_rejects_missing_label_contract() -> None:
    frame = pd.DataFrame([base_row("A")])

    with np.testing.assert_raises_regex(
        ValueError,
        "specialist labels missing columns",
    ):
        fit_specialist(
            frame,
            frame,
            specialist_specs()[0],
            random_seed=7,
            minimum_train_rows=1,
            minimum_calibration_rows=1,
        )


def test_prepare_specialist_frame_preserves_identity() -> None:
    frame = pd.DataFrame([base_row("A")])
    prepared = prepare_specialist_frame(frame)

    assert prepared[["trade_date", "signal_slot", "ts_code"]].equals(
        frame[["trade_date", "signal_slot", "ts_code"]]
    )
