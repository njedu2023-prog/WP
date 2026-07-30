from __future__ import annotations

import pandas as pd
import pytest

from wp.v3.forward_risk import (
    SAFE_RISK_RANK_MAX,
    assert_strictly_forward,
    frozen_meta_policy,
    select_forward_candidates,
)


def candidate(
    code: str,
    *,
    score: float,
    risk_rank: float,
    slot: str = "14:20",
    fold: int = 16,
    trade_date: str = "20250528",
) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "signal_slot": slot,
        "ts_code": code,
        "fold": fold,
        "meta_p_positive": 0.60,
        "meta_expected_net_return_pct": 0.20,
        "meta_p_severe_loss": 0.10,
        "p_round_trip_fill_lower": 0.99,
        "meta_rank_pct": 0.99,
        "meta_score": score,
        "risk_failure_rank_pct": risk_rank,
    }


def test_frozen_policy_matches_v14_discovery_contract() -> None:
    policy = frozen_meta_policy()

    assert policy.policy_id == "p0.54-e0.00-s0.35-f0.95-r0.95-k3-early"
    assert policy.max_candidates_per_day == 3
    assert SAFE_RISK_RANK_MAX == 0.50


def test_risk_gate_precedes_daily_top_three_selection() -> None:
    frame = pd.DataFrame(
        [
            candidate("UNSAFE", score=4.0, risk_rank=0.90),
            candidate("SAFE1", score=3.0, risk_rank=0.10),
            candidate("SAFE2", score=2.0, risk_rank=0.20),
            candidate("SAFE3", score=1.0, risk_rank=0.30),
        ]
    )

    baseline = select_forward_candidates(
        frame,
        apply_exit_risk_gate=False,
    )
    challenger = select_forward_candidates(
        frame,
        apply_exit_risk_gate=True,
    )

    assert baseline["ts_code"].tolist() == ["UNSAFE", "SAFE1", "SAFE2"]
    assert challenger["ts_code"].tolist() == ["SAFE1", "SAFE2", "SAFE3"]


def test_frozen_policy_rejects_late_slot_and_failed_threshold() -> None:
    frame = pd.DataFrame(
        [
            candidate("PASS", score=3.0, risk_rank=0.20),
            candidate("LATE", score=2.0, risk_rank=0.20, slot="14:40"),
            {
                **candidate("LOW_PROB", score=1.0, risk_rank=0.20),
                "meta_p_positive": 0.53,
            },
        ]
    )

    selected = select_forward_candidates(
        frame,
        apply_exit_risk_gate=True,
    )

    assert selected["ts_code"].tolist() == ["PASS"]


def test_forward_guard_rejects_discovery_period() -> None:
    valid = pd.DataFrame(
        [candidate("VALID", score=1.0, risk_rank=0.20)]
    )
    assert_strictly_forward(valid)

    invalid = pd.DataFrame(
        [
            candidate(
                "INVALID",
                score=1.0,
                risk_rank=0.20,
                fold=15,
                trade_date="2025-05-27",
            )
        ]
    )
    with pytest.raises(ValueError, match="unexpected folds"):
        assert_strictly_forward(invalid)
