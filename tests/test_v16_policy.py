from __future__ import annotations

import numpy as np
import pandas as pd

from wp.v3.v16_policy import (
    ExpertPolicy,
    apply_expert_policy,
    benjamini_hochberg,
    clustered_mean_bootstrap,
    clustered_mean_significance,
    policy_metrics,
)


POLICY = ExpertPolicy(
    probability_lower_min=0.56,
    expected_return_lower_min_pct=0.10,
    severe_loss_max=0.25,
    round_trip_fill_min=0.95,
    minimum_experts=2,
    probability_spread_max=0.10,
    score_rank_min=0.50,
    max_candidates_per_day=2,
    slot_group="all",
)


def row(
    code: str,
    *,
    score: float,
    slot: str = "14:20",
    probability: float = 0.60,
    expected: float = 0.20,
    severe: float = 0.10,
    experts: int = 2,
    spread: float = 0.05,
    fill: float = 0.99,
    result: float = 1.0,
) -> dict[str, object]:
    return {
        "trade_date": "20260720",
        "signal_slot": slot,
        "ts_code": code,
        "expert_count": experts,
        "expert_p_positive_lower": probability,
        "expert_expected_return_lower_pct": expected,
        "expert_p_severe": severe,
        "expert_probability_spread": spread,
        "expert_score": score,
        "p_round_trip_fill_lower": fill,
        "net_return_pct": result,
    }


def test_policy_applies_consensus_and_daily_top_k() -> None:
    frame = pd.DataFrame(
        [
            row("A", score=5.0),
            row("B", score=4.0),
            row("C", score=3.0),
            row("LOW", score=2.0, probability=0.55),
        ]
    )

    selected = apply_expert_policy(frame, POLICY)

    assert selected["ts_code"].tolist() == ["A", "B"]
    assert selected["expert_policy_id"].nunique() == 1


def test_policy_deduplicates_first_signal_for_same_symbol() -> None:
    frame = pd.DataFrame(
        [
            row("A", score=5.0, slot="14:20"),
            row("A", score=6.0, slot="14:25"),
            row("B", score=4.0, slot="14:20"),
        ]
    )

    selected = apply_expert_policy(frame, POLICY)

    assert selected[["ts_code", "signal_slot"]].to_dict(orient="records") == [
        {"ts_code": "A", "signal_slot": "14:20"},
        {"ts_code": "B", "signal_slot": "14:20"},
    ]


def test_benjamini_hochberg_is_monotone_in_rank() -> None:
    adjusted = benjamini_hochberg([0.01, 0.03, 0.20])

    assert adjusted[0] <= adjusted[1] <= adjusted[2]
    assert np.allclose(adjusted, [0.03, 0.045, 0.20])


def test_clustered_bootstrap_uses_day_means() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": [
                "20260720",
                "20260720",
                "20260721",
                "20260721",
            ],
            "net_return_pct": [1.0, 1.0, 0.5, 0.5],
        }
    )

    result = clustered_mean_bootstrap(frame, seed=7, samples=200)

    assert result["clustered_mean_lower_pct"] > 0
    assert result["clustered_win_rate_lower"] == 1.0


def test_clustered_significance_uses_day_means() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": [f"202607{day:02d}" for day in range(1, 21)],
            "net_return_pct": np.linspace(0.2, 0.8, 20),
        }
    )

    result = clustered_mean_significance(frame)

    assert result["mean_return_p_value"] < 0.001
    assert result["mean_return_test_days"] == 20
    assert result["mean_return_hac_lags"] == 5


def test_constant_positive_day_returns_use_finite_sign_test_floor() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": [f"202607{day:02d}" for day in range(1, 21)],
            "net_return_pct": 0.5,
        }
    )

    result = clustered_mean_significance(frame)

    assert 0.0 < result["mean_return_p_value"] <= 0.5**20


def test_50bps_stress_really_subtracts_half_a_percent() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20260701", "20260702"],
            "signal_slot": ["14:20", "14:20"],
            "ts_code": ["A", "B"],
            "net_return_pct": [0.4, 0.8],
        }
    )

    metrics = policy_metrics(
        frame,
        total_days=2,
        seed=7,
        bootstrap_samples=20,
    )

    assert np.isclose(metrics["stress_50bps_mean_net_return_pct"], 0.1)
