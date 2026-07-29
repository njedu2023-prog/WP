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


def _prediction(price: float, fingerprint: str = "abc") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "name": "测试",
                "target_trade_date": "20260724",
                "signal_price": price,
                "passes_policy": True,
                "p_entry_fill": 0.99,
                "p_exit_fill_given_entry": 0.995,
                "p_round_trip_fill": 0.985,
                "p_conditional_net_positive": 0.68,
                "p_net_positive": 0.67,
                "p_net_positive_lower": 0.58,
                "expected_utility_pct": 0.8,
                "conditional_expected_net_return_pct": 1.0,
                "downside_q10_pct": -2.0,
                "model_version": "v3",
                "model_fingerprint": fingerprint,
                "policy_fingerprint": "policy-a",
            }
        ]
    )


def test_ledger_locks_first_signal_and_only_updates_last_observation():
    ledger = empty_shadow_ledger()
    config = V3Config()
    record_shadow_slot(
        ledger,
        _prediction(10.0),
        trade_date="20260723",
        signal_slot="14:20",
        config=config,
    )
    record_shadow_slot(
        ledger,
        _prediction(10.4),
        trade_date="20260723",
        signal_slot="14:30",
        config=config,
    )
    candidate = ledger["sessions"][0]["candidates"][0]
    assert candidate["first_signal_time"] == "14:20"
    assert candidate["first_signal_price"] == 10.0
    assert candidate["last_signal_price"] == 10.4
    assert candidate["appearance_count"] == 2
    assert candidate["policy_fingerprint"] == "policy-a"
    assert candidate["baseline_all_in_cost_bps"] == 35.0
    assert candidate["entry_benchmark_slot"] == "14:25"
    assert candidate["entry_benchmark_status"] == "PENDING"
    assert "first_signal_features" in candidate
    assert "qualification_evidence" in candidate
    assert ledger["sessions"][0]["policy_fingerprint"] == "policy-a"
    assert_ledger_invariants(ledger, config)


def test_entry_benchmark_is_settled_once_from_the_exact_next_slot():
    ledger = empty_shadow_ledger()
    config = V3Config()
    record_shadow_slot(
        ledger,
        _prediction(10.0),
        trade_date="20260723",
        signal_slot="14:20",
        config=config,
    )
    settlement = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "entry_benchmark_slot": "14:25",
                "entry_benchmark_price": 10.2,
                "entry_benchmark_amount": 20_000_000,
                "entry_benchmark_bar_time": "2026-07-23 14:25:00",
                "data_age_seconds": 0,
                "up_limit": 11.0,
            }
        ]
    )
    settle_entry_benchmarks(
        ledger,
        settlement,
        trade_date="20260723",
        settlement_slot="14:25",
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
            settlement_slot="14:25",
            config=config,
        )


def test_freeze_fails_integrity_when_a_v7_entry_is_still_pending():
    ledger = empty_shadow_ledger()
    config = V3Config()
    for slot in config.strategy.signal_slots:
        predictions = (
            _prediction(10.0)
            if slot == "14:50"
            else _prediction(10.0).assign(passes_policy=False)
        )
        record_shadow_slot(
            ledger,
            predictions,
            trade_date="20260723",
            signal_slot=slot,
            config=config,
        )
    freeze_shadow_session(
        ledger,
        trade_date="20260723",
        config=config,
    )
    assert ledger["sessions"][0]["integrity_status"] == "INCOMPLETE_ENTRY"
    assert ledger["sessions"][0]["pending_entry_benchmark_count"] == 1


def test_frozen_session_rejects_new_candidates():
    ledger = empty_shadow_ledger()
    config = V3Config()
    freeze_shadow_session(ledger, trade_date="20260723", config=config)
    with pytest.raises(ValueError, match="frozen"):
        record_shadow_slot(
            ledger,
            _prediction(10.0),
            trade_date="20260723",
            signal_slot="14:20",
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
            signal_slot="14:20",
            config=config,
        )
    candidate = ledger["sessions"][0]["candidates"][0]
    assert candidate["appearance_count"] == 1
    assert ledger["sessions"][0]["covered_slots"] == ["14:20"]


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
