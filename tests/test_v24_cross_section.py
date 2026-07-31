from __future__ import annotations

import numpy as np
import pandas as pd

from wp.v3.v24_cross_section import (
    FIXED_MAX_CANDIDATES_PER_DAY,
    FIXED_TARGET_CANDIDATE_DAY_RATE,
    MODEL_FEATURES,
    CrossSectionPolicySpec,
    FrozenCrossSectionPolicy,
    add_cross_section_score,
    apply_cross_section_policy,
    stock_day_equalized_temporal_weights,
    validate_feature_contract,
)


def scored_rows() -> pd.DataFrame:
    slots = ("14:20", "14:25", "14:20", "14:30", "14:35")
    codes = (
        "600000.SH",
        "600000.SH",
        "600001.SH",
        "600002.SH",
        "600003.SH",
    )
    return pd.DataFrame(
        {
            "trade_date": "20260724",
            "signal_slot": slots,
            "ts_code": codes,
            "v23_point_in_time_complete": True,
            "p_round_trip_fill_lower": 0.99,
            "p_severe_loss": 0.20,
            "data_age_seconds": 30.0,
            "v23_p_positive": [0.70, 0.90, 0.65, 0.62, 0.61],
            "v23_p_margin": [0.60, 0.80, 0.55, 0.52, 0.51],
            "v23_p_severe_loss": 0.10,
            "v23_positive_model_spread": 0.05,
            "v23_margin_model_spread": 0.05,
            "v23_severe_model_spread": 0.05,
            "v23_expected_net_return_pct": [1.0, 2.0, 0.8, 0.7, 0.6],
            "v23_expected_return_model_spread_pct": 0.20,
        }
    )


def test_v24_contract_is_fixed_before_research() -> None:
    assert FIXED_TARGET_CANDIDATE_DAY_RATE == 0.25
    assert FIXED_MAX_CANDIDATES_PER_DAY == 3
    assert CrossSectionPolicySpec().policy_id == "v24-rate0.25-k3-top5"
    assert validate_feature_contract(MODEL_FEATURES)


def test_v24_model_features_exclude_outcomes() -> None:
    forbidden = (
        "target",
        "truth",
        "future",
        "gross_return",
        "net_return",
        "t1_",
        "exit_price",
    )
    assert len(MODEL_FEATURES) >= 50
    assert not [
        column
        for column in MODEL_FEATURES
        if any(token in column.lower() for token in forbidden)
    ]


def test_stock_day_weight_is_not_multiplied_by_repeated_slots() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20260723"] * 4,
            "ts_code": [
                "600000.SH",
                "600000.SH",
                "600000.SH",
                "600001.SH",
            ],
        }
    )
    weights = stock_day_equalized_temporal_weights(frame)
    first_total = float(weights[:3].sum())
    second_total = float(weights[3])

    assert np.isclose(first_total, second_total)


def test_v24_soft_score_and_first_signal_are_deterministic() -> None:
    scored = scored_rows()
    prepared = add_cross_section_score(scored)
    assert prepared["v24_cross_section_score"].notna().all()
    policy = FrozenCrossSectionPolicy(
        spec=CrossSectionPolicySpec(),
        score_threshold=-100.0,
        calibration_start="20260501",
        calibration_end="20260630",
        calibration_days=42,
        eligible_days=42,
    )

    first = apply_cross_section_policy(scored, policy)
    second = apply_cross_section_policy(scored, policy)

    assert first[
        ["trade_date", "signal_slot", "ts_code", "v24_cross_section_score"]
    ].equals(
        second[
            ["trade_date", "signal_slot", "ts_code", "v24_cross_section_score"]
        ]
    )
    assert len(first) == 3
    assert not first.duplicated(["trade_date", "ts_code"]).any()
    assert first.loc[first["ts_code"].eq("600000.SH"), "signal_slot"].item() == (
        "14:20"
    )
