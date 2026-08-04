from __future__ import annotations

import pandas as pd
import pytest

from wp.v3.contracts import V3Config
from wp.v3.ledger import (
    assert_ledger_invariants,
    empty_shadow_ledger,
    freeze_shadow_session,
    record_shadow_slot,
    settle_entry_benchmarks,
)


def _prediction(
    price: float,
    fingerprint: str = "abc",
    *,
    code: str = "600001.SH",
    passes_policy: bool = True,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": code,
                "name": "测试",
                "target_trade_date": "20260724",
                "signal_price": price,
                "passes_policy": passes_policy,
                "p_entry_fill": 0.99,
                "p_exit_fill_given_entry": 0.995,
                "p_round_trip_fill": 0.985,
                "p_conditional_net_positive": 0.68,
                "p_net_positive": 0.67,
                "p_net_positive_lower": 0.58,
                "base_p_net_positive": 0.64,
                "base_p_net_positive_lower": 0.55,
                "meta_p_positive_raw": 0.69,
                "meta_p_positive_tree_raw": 0.72,
                "meta_p_positive_linear_raw": 0.63,
                "meta_p_positive": 0.67,
                "meta_p_positive_lower": 0.56,
                "meta_probability_calibration_margin": 0.03,
                "expected_utility_pct": 0.8,
                "conditional_expected_net_return_pct": 1.0,
                "downside_q10_pct": -2.0,
                "model_version": "v3",
                "model_fingerprint": fingerprint,
                "policy_fingerprint": "policy-a",
            }
        ]
    )


def _dual_cohort_predictions() -> pd.DataFrame:
    frames = [_prediction(10.0)]
    for index in range(5):
        frames.append(
            _prediction(
                9.0 + index,
                code=f"6001{index:02d}.SH",
                passes_policy=False,
            ).assign(
                p_net_positive_lower=0.53 - index * 0.01,
                expected_utility_lower_pct=0.10 - index * 0.02,
                selection_score=1.0 - index * 0.05,
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_ledger_locks_fixed_1400_signal_and_separates_both_cohorts():
    ledger = empty_shadow_ledger()
    config = V3Config()
    record_shadow_slot(
        ledger,
        _dual_cohort_predictions(),
        trade_date="20260723",
        signal_slot="14:00",
        config=config,
    )
    session = ledger["sessions"][0]
    candidate = session["candidates"][0]
    assert candidate["first_signal_time"] == "14:00"
    assert candidate["first_signal_price"] == 10.0
    assert candidate["last_signal_price"] == 10.0
    assert candidate["appearance_count"] == 1
    assert candidate["policy_fingerprint"] == "policy-a"
    assert candidate["baseline_all_in_cost_bps"] == 35.0
    assert candidate["entry_benchmark_slot"] == "14:05"
    assert candidate["entry_benchmark_status"] == "PENDING"
    assert candidate["candidate_cohort"] == "QUALIFIED"
    assert candidate["is_user_trade"] is False
    assert candidate["meta_p_positive"] == 0.67
    assert candidate["meta_p_positive_lower"] == 0.56
    assert candidate["base_p_net_positive_lower"] == 0.55
    assert (
        candidate["qualification_evidence"][
            "meta_probability_calibration_margin"
        ]
        == 0.03
    )
    assert "first_signal_features" in candidate
    assert "qualification_evidence" in candidate
    assert len(session["observations"]) == 5
    assert {
        row["ts_code"] for row in session["candidates"]
    }.isdisjoint({row["ts_code"] for row in session["observations"]})
    assert all(
        row["candidate_cohort"] == "OBSERVATION"
        and row["is_user_trade"] is False
        and row["entry_benchmark_slot"] == "14:05"
        for row in session["observations"]
    )
    assert session["policy_fingerprint"] == "policy-a"
    assert_ledger_invariants(ledger, config)


def test_entry_benchmark_is_settled_once_from_the_exact_next_slot():
    ledger = empty_shadow_ledger()
    config = V3Config()
    record_shadow_slot(
        ledger,
        _prediction(10.0),
        trade_date="20260723",
        signal_slot="14:00",
        config=config,
    )
    settlement = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "entry_benchmark_slot": "14:05",
                "entry_benchmark_price": 10.2,
                "entry_benchmark_amount": 20_000_000,
                "entry_benchmark_bar_time": "2026-07-23 14:05:00",
                "data_age_seconds": 0,
                "up_limit": 11.0,
            }
        ]
    )
    settle_entry_benchmarks(
        ledger,
        settlement,
        trade_date="20260723",
        settlement_slot="14:05",
        config=config,
    )
    candidate = ledger["sessions"][0]["candidates"][0]
    assert candidate["entry_benchmark_status"] == "SETTLED"
    assert candidate["entry_benchmark_price"] == 10.2
    assert candidate["entry_price"] == pytest.approx(10.2102)
    assert candidate["entry_fillable"] is True

    changed = settlement.copy()
    changed["entry_benchmark_price"] = 10.3
    with pytest.raises(ValueError, match="cannot rewrite immutable"):
        settle_entry_benchmarks(
            ledger,
            changed,
            trade_date="20260723",
            settlement_slot="14:05",
            config=config,
        )


def test_freeze_fails_integrity_when_dual_cohort_entries_are_pending():
    ledger = empty_shadow_ledger()
    config = V3Config()
    record_shadow_slot(
        ledger,
        _dual_cohort_predictions(),
        trade_date="20260723",
        signal_slot="14:00",
        config=config,
    )
    freeze_shadow_session(
        ledger,
        trade_date="20260723",
        config=config,
    )
    assert ledger["sessions"][0]["integrity_status"] == "INCOMPLETE_ENTRY"
    assert ledger["sessions"][0]["pending_entry_benchmark_count"] == 6


def test_freeze_reports_observation_shortfall_without_fabrication():
    ledger = empty_shadow_ledger()
    config = V3Config()
    record_shadow_slot(
        ledger,
        _prediction(10.0),
        trade_date="20260723",
        signal_slot="14:00",
        config=config,
    )
    settlement = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "entry_benchmark_slot": "14:05",
                "entry_benchmark_price": 10.2,
                "entry_benchmark_amount": 20_000_000,
                "entry_benchmark_bar_time": "2026-07-23 14:05:00",
                "data_age_seconds": 0,
                "up_limit": 11.0,
            }
        ]
    )
    settle_entry_benchmarks(
        ledger,
        settlement,
        trade_date="20260723",
        settlement_slot="14:05",
        config=config,
    )
    freeze_shadow_session(
        ledger,
        trade_date="20260723",
        config=config,
    )
    session = ledger["sessions"][0]
    assert session["observations"] == []
    assert session["integrity_status"] == "INCOMPLETE_OBSERVATION"


def test_frozen_session_rejects_new_candidates():
    ledger = empty_shadow_ledger()
    config = V3Config()
    freeze_shadow_session(ledger, trade_date="20260723", config=config)
    with pytest.raises(ValueError, match="frozen"):
        record_shadow_slot(
            ledger,
            _prediction(10.0),
            trade_date="20260723",
            signal_slot="14:00",
            config=config,
        )


def test_repeating_the_same_slot_is_idempotent():
    ledger = empty_shadow_ledger()
    config = V3Config()
    for _ in range(2):
        record_shadow_slot(
            ledger,
            _prediction(10.0),
            trade_date="20260723",
            signal_slot="14:00",
            config=config,
        )
    candidate = ledger["sessions"][0]["candidates"][0]
    assert candidate["appearance_count"] == 1
    assert ledger["sessions"][0]["covered_slots"] == ["14:00"]


def test_same_day_recovery_is_immutable_and_non_prospective():
    ledger = empty_shadow_ledger()
    config = V3Config()
    record_shadow_slot(
        ledger,
        _dual_cohort_predictions(),
        trade_date="20260723",
        signal_slot="14:00",
        config=config,
        evidence_tier="RECOVERED_SAME_DAY",
        prospective_eligible=False,
        recovery_reason="missed_scheduler",
    )
    session = ledger["sessions"][0]

    assert session["evidence_tier"] == "RECOVERED_SAME_DAY"
    assert session["prospective_eligible"] is False
    assert session["recovery_reason"] == "missed_scheduler"
    assert all(
        record["evidence_tier"] == "RECOVERED_SAME_DAY"
        and record["prospective_eligible"] is False
        for record in [
            *session["candidates"],
            *session["observations"],
        ]
    )
    assert_ledger_invariants(ledger, config)


def test_repeating_freeze_preserves_the_original_freeze_time():
    ledger = empty_shadow_ledger()
    config = V3Config()
    freeze_shadow_session(
        ledger,
        trade_date="20260723",
        config=config,
        frozen_at="2026-07-23T14:55:00+08:00",
        model_fingerprint="abc",
    )
    freeze_shadow_session(
        ledger,
        trade_date="20260723",
        config=config,
        frozen_at="2026-07-23T15:00:00+08:00",
        model_fingerprint="abc",
    )
    assert ledger["sessions"][0]["frozen_at"] == "2026-07-23T14:55:00+08:00"
    assert ledger["sessions"][0]["integrity_status"] == "INCOMPLETE"


def test_frozen_session_rejects_a_different_model_fingerprint():
    ledger = empty_shadow_ledger()
    config = V3Config()
    freeze_shadow_session(
        ledger,
        trade_date="20260723",
        config=config,
        model_fingerprint="abc",
    )
    with pytest.raises(ValueError, match="cannot change frozen session model"):
        freeze_shadow_session(
            ledger,
            trade_date="20260723",
            config=config,
            model_fingerprint="different",
        )
