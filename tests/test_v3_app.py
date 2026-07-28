from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from wp.v3.app import (
    _missing_input_state,
    _refresh_data_age,
    _resolve_model_path,
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


def test_legal_1450_capture_remains_authorized_when_computation_crosses_1451():
    manifest = {
        "trade_date": "20260727",
        "signal_slot": "14:50",
        "market_data_time": "2026-07-27 14:50:00",
        "capture_started_at": "2026-07-27T14:50:42+08:00",
        "capture_completed_at": "2026-07-27T14:51:18+08:00",
    }

    assert _source_signal_authorized(manifest, V3Config()) is True


def test_post_deadline_capture_cannot_masquerade_as_the_1450_signal():
    manifest = {
        "trade_date": "20260727",
        "signal_slot": "14:50",
        "market_data_time": "2026-07-27 14:50:00",
        "capture_started_at": "2026-07-27T14:51:01+08:00",
        "capture_completed_at": "2026-07-27T14:51:30+08:00",
    }

    assert _source_signal_authorized(manifest, V3Config()) is False


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
