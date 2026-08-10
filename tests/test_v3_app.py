from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from wp.v3.app import (
    _decision_time,
    _evidence_feature_universe,
    _missing_input_state,
    _refresh_data_age,
    _resolve_model_path,
    _source_recovery_authorized,
    _source_signal_authorized,
)
from wp.v3.contracts import V3Config


def test_data_age_is_computed_per_symbol_bar_not_global_latest_bar():
    frame = pd.DataFrame(
        {
            "ts_code": ["600001.SH", "600002.SH"],
            "slot_bar_time": [
                "2026-07-27 14:20:00",
                "2026-07-27 14:10:00",
            ],
        }
    )
    result = _refresh_data_age(
        frame,
        current=datetime(2026, 7, 27, 14, 22, tzinfo=ZoneInfo("Asia/Shanghai")),
        market_time=pd.Timestamp("2026-07-27 14:20:00"),
    )
    assert result["data_age_seconds"].tolist() == [120.0, 720.0]


def test_completed_1355_cutoff_is_authorized_before_1400_publish():
    manifest = {
        "trade_date": "20260727",
        "signal_slot": "14:00",
        "market_data_cutoff_slot": "13:55",
        "market_data_time": "2026-07-27 13:55:00",
        "latest_bar_slot": "13:55",
        "capture_started_at": "2026-07-27T13:55:30+08:00",
        "capture_completed_at": "2026-07-27T13:56:18+08:00",
        "capture_contract": "anchored_prepublication_cutoff_snapshot",
        "evidence_tier": "PROSPECTIVE_LIVE",
    }

    assert _source_signal_authorized(manifest, V3Config()) is True


def test_capture_started_after_1400_cannot_masquerade_as_on_time():
    manifest = {
        "trade_date": "20260727",
        "signal_slot": "14:00",
        "market_data_cutoff_slot": "13:55",
        "market_data_time": "2026-07-27 13:55:00",
        "latest_bar_slot": "13:55",
        "capture_started_at": "2026-07-27T14:07:01+08:00",
        "capture_completed_at": "2026-07-27T14:07:30+08:00",
        "capture_contract": "anchored_prepublication_cutoff_snapshot",
        "evidence_tier": "PROSPECTIVE_LIVE",
    }

    assert _source_signal_authorized(manifest, V3Config()) is False


def test_bar_after_the_1355_cutoff_cannot_enter_the_decision():
    manifest = {
        "trade_date": "20260727",
        "signal_slot": "14:00",
        "market_data_cutoff_slot": "13:55",
        "market_data_time": "2026-07-27 13:56:00",
        "latest_bar_slot": "13:56",
        "capture_started_at": "2026-07-27T13:56:00+08:00",
        "capture_completed_at": "2026-07-27T13:56:30+08:00",
        "capture_contract": "anchored_prepublication_cutoff_snapshot",
        "evidence_tier": "PROSPECTIVE_LIVE",
    }

    assert _source_signal_authorized(manifest, V3Config()) is False


def test_same_day_recovery_is_authorized_only_as_non_prospective_evidence():
    manifest = {
        "trade_date": "20260727",
        "signal_slot": "14:00",
        "market_data_cutoff_slot": "13:55",
        "market_data_time": "2026-07-27 13:55:00",
        "latest_bar_slot": "13:55",
        "capture_started_at": "2026-07-27T14:45:00+08:00",
        "capture_completed_at": "2026-07-27T15:03:00+08:00",
        "capture_contract": "retrospective_same_day_rt_min_daily",
        "evidence_tier": "RECOVERED_SAME_DAY",
        "prospective_eligible": False,
        "source_api": "rt_min_daily",
        "recovered_at": "2026-07-27T15:03:00+08:00",
        "recovery_reason": "missed_scheduler",
        "decision_reference_time": "2026-07-27T14:00:30+08:00",
        "row_count": 3000,
        "fresh_row_count": 3000,
        "open_universe_coverage": 0.95,
        "tail_universe_coverage": 0.95,
    }

    assert _source_signal_authorized(manifest, V3Config()) is False
    assert _source_recovery_authorized(manifest, V3Config()) is True
    assert _decision_time(
        manifest,
        fallback=datetime(
            2026,
            7,
            27,
            15,
            3,
            tzinfo=ZoneInfo("Asia/Shanghai"),
        ),
    ).strftime("%H:%M") == "14:00"


def test_recovery_cannot_be_marked_as_prospective():
    manifest = {
        "trade_date": "20260727",
        "signal_slot": "14:00",
        "market_data_cutoff_slot": "13:55",
        "market_data_time": "2026-07-27 13:55:00",
        "latest_bar_slot": "13:55",
        "capture_contract": "retrospective_same_day_rt_min_daily",
        "evidence_tier": "RECOVERED_SAME_DAY",
        "prospective_eligible": True,
        "source_api": "rt_min_daily",
        "recovered_at": "2026-07-27T15:03:00+08:00",
        "decision_reference_time": "2026-07-27T14:00:30+08:00",
        "row_count": 3000,
        "fresh_row_count": 3000,
        "open_universe_coverage": 0.95,
        "tail_universe_coverage": 0.95,
    }

    assert _source_recovery_authorized(manifest, V3Config()) is False


def test_evidence_features_follow_the_pruned_model_universe():
    features = pd.DataFrame(
        {
            "ts_code": ["600001.SH", "600002.SH", "600003.SH"],
            "feature": [1.0, 2.0, 3.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "ts_code": ["600003.SH", "600001.SH"],
            "passes_policy": [False, True],
        }
    )

    result = _evidence_feature_universe(features, predictions)

    assert result["ts_code"].tolist() == ["600001.SH", "600003.SH"]


def test_evidence_features_reject_predictions_outside_the_source_universe():
    features = pd.DataFrame({"ts_code": ["600001.SH"]})
    predictions = pd.DataFrame({"ts_code": ["600002.SH"]})

    with pytest.raises(ValueError, match="absent from source features"):
        _evidence_feature_universe(features, predictions)


def test_live_model_resolution_uses_designated_fingerprint_not_contract_fingerprint():
    registry = {
        "active_model_fingerprint": None,
        "shadow_model_fingerprint": "model-123",
        "shadow_policy_fingerprint": "learned-policy-456",
        "models": [
            {
                "fingerprint": "model-123",
                "policy_fingerprint": "learned-policy-456",
                "artifact_path": "artifacts/wp_v3_research/wp_v3_model.joblib",
            }
        ],
    }

    assert _resolve_model_path(registry).as_posix().endswith(
        "artifacts/wp_v3_research/wp_v3_model.joblib"
    )


def test_missing_input_is_normal_after_market_close_when_model_exists():
    state, health, message = _missing_input_state(
        phase="CLOSED",
        model_status="SHADOW",
        error="missing source",
    )

    assert state == "SHADOW"
    assert health == "ok"
    assert "盘后补造" in message


def test_missing_input_remains_a_fault_during_signal_window():
    state, health, message = _missing_input_state(
        phase="SIGNAL",
        model_status="SHADOW",
        error="missing source",
    )

    assert state == "SHADOW"
    assert health == "v3_input_not_ready"
    assert message == "missing source"
