from __future__ import annotations

import numpy as np
import pandas as pd

from wp.v3.v25_ranker import (
    FIXED_MAX_CANDIDATES_PER_DAY,
    FIXED_TARGET_CANDIDATE_DAY_RATE,
    MODEL_FEATURES,
    FrozenPositioningPolicy,
    PositioningPolicySpec,
    apply_positioning_policy,
    build_pairwise_examples,
    fit_positioning_ranker,
    validate_feature_contract,
    within_slot_rank_diagnostics,
)


def feature_frame() -> pd.DataFrame:
    rows = []
    for index, (code, net) in enumerate(
        (
            ("600000.SH", 2.0),
            ("600001.SH", 0.5),
            ("600002.SH", -1.0),
        )
    ):
        row = {
            "trade_date": "20260724",
            "signal_slot": "14:20",
            "ts_code": code,
            "net_return_pct": net,
        }
        for offset, feature in enumerate(MODEL_FEATURES):
            row[feature] = float(index + offset / 100.0)
        rows.append(row)
    return pd.DataFrame(rows)


def scored_rows() -> pd.DataFrame:
    slots = ("14:20", "14:20", "14:25", "14:30", "14:35")
    codes = (
        "600000.SH",
        "600001.SH",
        "600000.SH",
        "600002.SH",
        "600003.SH",
    )
    return pd.DataFrame(
        {
            "trade_date": "20260724",
            "signal_slot": slots,
            "ts_code": codes,
            "v23_point_in_time_complete": True,
            "v25_positioning_core_complete": True,
            "p_round_trip_fill_lower": 0.99,
            "p_severe_loss": 0.20,
            "data_age_seconds": 30.0,
            "v25_p_positive": 0.65,
            "v25_p_severe_loss": 0.10,
            "v25_within_slot_rank_score": [0.80, 0.55, 0.75, 0.70, 0.65],
            "v25_positioning_score": [1.0, 0.2, 0.9, 0.8, 0.7],
        }
    )


def training_rows(start_day: int, days: int) -> pd.DataFrame:
    rows = []
    returns = (2.0, 1.0, 0.2, -1.0, -3.0)
    for day_offset in range(days):
        date = f"202601{start_day + day_offset:02d}"
        for rank, net_return in enumerate(returns):
            row = {
                "trade_date": date,
                "signal_slot": "14:20",
                "ts_code": f"600{rank:03d}.SH",
                "net_return_pct": net_return + 0.01 * day_offset,
                "target_net_positive": float(
                    net_return + 0.01 * day_offset > 0.0
                ),
                "label_available": True,
                "v23_point_in_time_complete": True,
                "v25_positioning_core_complete": True,
            }
            for offset, feature in enumerate(MODEL_FEATURES):
                row[feature] = (
                    -float(rank)
                    + 0.01 * day_offset
                    + 0.001 * offset
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_v25_contract_is_frozen_and_outcome_blind() -> None:
    assert FIXED_TARGET_CANDIDATE_DAY_RATE == 0.25
    assert FIXED_MAX_CANDIDATES_PER_DAY == 3
    assert PositioningPolicySpec().policy_id == (
        "v25-positioning-rate0.25-k3-top5"
    )
    assert validate_feature_contract(MODEL_FEATURES)
    forbidden = (
        "target",
        "truth",
        "future",
        "gross_return",
        "net_return",
        "t1_",
        "exit_",
    )
    assert not [
        feature
        for feature in MODEL_FEATURES
        if any(token in feature.lower() for token in forbidden)
    ]


def test_pairwise_examples_are_symmetric_and_group_equalized() -> None:
    frame = feature_frame()
    pairs, targets, weights = build_pairwise_examples(
        frame,
        MODEL_FEATURES,
    )

    assert len(pairs) == 6
    assert np.allclose(
        pairs.iloc[0].to_numpy(dtype=float),
        -pairs.iloc[1].to_numpy(dtype=float),
        equal_nan=True,
    )
    assert targets[0] == 1 - targets[1]
    assert np.isclose(float(weights.mean()), 1.0)


def test_policy_keeps_one_slot_leader_and_immutable_first_signal() -> None:
    policy = FrozenPositioningPolicy(
        spec=PositioningPolicySpec(),
        score_threshold=-100.0,
        calibration_start="20260501",
        calibration_end="20260630",
        calibration_days=42,
        eligible_days=42,
    )
    selected = apply_positioning_policy(scored_rows(), policy)

    assert len(selected) == 3
    assert not selected.duplicated(["trade_date", "ts_code"]).any()
    assert selected.loc[
        selected["ts_code"].eq("600000.SH"),
        "signal_slot",
    ].item() == "14:20"
    assert selected.groupby(["trade_date", "signal_slot"]).size().max() == 1


def test_within_slot_diagnostics_measure_actual_stock_ranking() -> None:
    frame = feature_frame().copy()
    frame["v25_within_slot_rank_score"] = [0.9, 0.6, 0.2]
    diagnostics = within_slot_rank_diagnostics(
        frame,
        seed=25,
        bootstrap_samples=100,
    )

    assert diagnostics["groups"] == 1
    assert diagnostics["mean_within_slot_ic"] == 1.0
    assert diagnostics["mean_top_minus_bottom_return_pct"] == 3.0


def test_ranker_fits_calibrates_and_scores_same_slot_candidates() -> None:
    train = training_rows(1, 12)
    calibration = training_rows(13, 6)
    bundle = fit_positioning_ranker(
        train,
        calibration,
        random_seed=25,
        minimum_train_rows=50,
        minimum_calibration_rows=25,
        minimum_train_pair_rows=200,
        minimum_calibration_pair_rows=100,
    )
    scored = bundle.predict(calibration)

    assert scored["v25_p_positive"].between(0.001, 0.999).all()
    assert scored["v25_p_severe_loss"].between(0.001, 0.999).all()
    assert scored["v25_within_slot_rank_score"].between(0.0, 1.0).all()
    first_day = scored.loc[scored["trade_date"].eq("20260113")]
    best = first_day.loc[first_day["ts_code"].eq("600000.SH")].iloc[0]
    worst = first_day.loc[first_day["ts_code"].eq("600004.SH")].iloc[0]
    assert (
        best["v25_within_slot_rank_score"]
        > worst["v25_within_slot_rank_score"]
    )
