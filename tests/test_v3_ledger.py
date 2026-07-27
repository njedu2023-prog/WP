from __future__ import annotations

import pandas as pd
import pytest

from wp.v3.contracts import V3Config
from wp.v3.ledger import (
    assert_ledger_invariants,
    empty_shadow_ledger,
    freeze_shadow_session,
    record_shadow_slot,
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
                "p_net_positive": 0.67,
                "p_net_positive_lower": 0.58,
                "expected_net_return_pct": 0.8,
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
    assert ledger["sessions"][0]["policy_fingerprint"] == "policy-a"
    assert_ledger_invariants(ledger, config)


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
