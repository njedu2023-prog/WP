from __future__ import annotations

import pandas as pd

from wp.v3.v29_peer_ranker import (
    MODEL_FEATURES,
    FrozenPeerPolicy,
    PeerPolicySpec,
    apply_peer_policy,
    policy_eligible_rows,
    validate_feature_contract,
)


def scored_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [
                "20250102",
                "20250102",
                "20250102",
                "20250102",
            ],
            "signal_slot": ["14:20", "14:20", "14:25", "14:30"],
            "ts_code": [
                "000001.SZ",
                "000002.SZ",
                "000001.SZ",
                "000003.SZ",
            ],
            "v23_point_in_time_complete": [True, True, True, True],
            "v29_peer_features_complete": [True, True, True, True],
            "p_round_trip_fill_lower": [0.99, 0.99, 0.99, 0.99],
            "p_severe_loss": [0.10, 0.10, 0.10, 0.10],
            "data_age_seconds": [10.0, 10.0, 10.0, 10.0],
            "v29_peer_score": [0.8, 0.7, 0.9, 0.6],
        }
    )


def test_feature_contract_uses_only_v29_features() -> None:
    assert PeerPolicySpec().policy_id == "v29-peer-rate0.25-k3"
    assert validate_feature_contract(MODEL_FEATURES)


def test_policy_has_no_model_probability_hard_gate() -> None:
    eligible = policy_eligible_rows(scored_frame())

    assert len(eligible) == 4


def test_policy_preserves_first_signal_and_daily_cap() -> None:
    policy = FrozenPeerPolicy(
        spec=PeerPolicySpec(
            target_candidate_day_rate=0.25,
            max_candidates_per_day=3,
        ),
        score_threshold=0.0,
        calibration_start="20240101",
        calibration_end="20241231",
        calibration_days=42,
        eligible_days=10,
    )

    selected = apply_peer_policy(scored_frame(), policy)

    assert list(selected["ts_code"]) == ["000001.SZ", "000003.SZ"]
    assert list(selected["signal_slot"]) == ["14:20", "14:30"]
    assert not selected.duplicated(["trade_date", "ts_code"]).any()
