from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wp.utils import write_json
from wp.v3.contracts import V3Config
from wp.v3.evidence import archive_signal_evidence, verify_signal_evidence
from wp.v3.io import atomic_write_json


@pytest.mark.parametrize("writer", [write_json, atomic_write_json])
def test_json_writers_emit_strict_json_for_numpy_and_missing_values(
    tmp_path: Path,
    writer,
):
    target = tmp_path / "payload.json"
    writer(
        target,
        {
            "nan": np.nan,
            "infinity": np.float64(np.inf),
            "missing": pd.NA,
            "integer": np.int64(7),
            "timestamp": pd.Timestamp("2026-07-28 14:20:00"),
            "path": Path("outputs/audit"),
        },
    )

    raw = target.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert json.loads(raw) == {
        "infinity": None,
        "integer": 7,
        "missing": None,
        "nan": None,
        "path": "outputs/audit",
        "timestamp": "2026-07-28T14:20:00",
    }


def test_slot_evidence_is_idempotent_but_rejects_changed_decisions(
    tmp_path: Path,
):
    config = V3Config()
    features = pd.DataFrame(
        [{"trade_date": "20260728", "signal_slot": "14:20", "ts_code": "600001.SH"}]
    )
    predictions = features.assign(
        p_net_positive=0.61,
        execution_eligible=True,
        passes_policy=True,
        candidate_state="QUALIFIED",
        rejection_reasons="",
    )
    source_manifest = {
        "schema_version": "wp_live_input_v7",
        "trade_date": "20260728",
        "target_trade_date": "20260729",
        "signal_slot": "14:20",
        "market_data_time": "2026-07-28T14:20:00+08:00",
        "latest_bar_slot": "14:20",
        "row_count": 1,
        "fresh_row_count": 1,
        "eligible_count": 1,
        "feature_version": config.model.feature_version,
        "capture_completed_at": "2026-07-28T14:20:03+08:00",
    }
    inference_manifest = {
        "v3_model_fingerprint": "model-a",
        "v3_policy_fingerprint": "policy-a",
        "v3_state": "shadow",
        "v3_formal_authorization": False,
    }

    first = archive_signal_evidence(
        tmp_path,
        features=features,
        predictions=predictions,
        source_manifest=source_manifest,
        inference_manifest=inference_manifest,
        config=config,
    )
    repeated = archive_signal_evidence(
        tmp_path,
        features=features,
        predictions=predictions,
        source_manifest={
            **source_manifest,
            "capture_completed_at": "2026-07-28T14:20:10+08:00",
        },
        inference_manifest=inference_manifest,
        config=config,
    )

    evidence_dir = tmp_path / "audit" / "2026" / "20260728" / "1420"
    assert repeated["evidence_digest"] == first["evidence_digest"]
    assert verify_signal_evidence(evidence_dir)["evidence_digest"] == first[
        "evidence_digest"
    ]

    with pytest.raises(RuntimeError, match="immutable signal evidence conflict"):
        archive_signal_evidence(
            tmp_path,
            features=features,
            predictions=predictions.assign(p_net_positive=0.62),
            source_manifest=source_manifest,
            inference_manifest=inference_manifest,
            config=config,
        )
