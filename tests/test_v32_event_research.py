from __future__ import annotations

import json

import pandas as pd
import pytest

from research_wp_v32_public_event import (
    event_rank_diagnostics,
    join_event_features,
    load_v32_event_features,
)
from wp.v3.io import atomic_write_parquet, file_sha256
from wp.v3.v32_event_features import (
    ADMITTED_SOURCES,
    DETAIL_FEATURES_BY_SOURCE,
    V32_EVENT_FEATURE_COLUMNS,
)


DATA_SCHEMA_VERSION = "wp_v32_public_event_features_1"


def event_features() -> pd.DataFrame:
    rows = []
    for index, code in enumerate(("000001.SZ", "000002.SZ")):
        row = {
            "trade_date": "20260724",
            "signal_slot": "14:20",
            "ts_code": code,
            "fold": 3,
        }
        for source in ADMITTED_SOURCES:
            row[f"v32_{source}_event_count_5d"] = float(index + 1)
            row[f"v32_{source}_active_5d"] = 1.0
            row[f"v32_{source}_latest_age_td"] = 1.0
            for offset, column in enumerate(
                DETAIL_FEATURES_BY_SOURCE[source]
            ):
                row[column] = float(index + 1 + offset / 100.0)
        rows.append(row)
    return pd.DataFrame(rows)


def write_data_artifact(tmp_path) -> None:
    features = event_features()
    candidate_index = features.loc[
        :, ["trade_date", "signal_slot", "ts_code", "fold"]
    ]
    feature_path = atomic_write_parquet(
        features,
        tmp_path / "wp_v32_public_event_features.parquet",
    )
    index_path = atomic_write_parquet(
        candidate_index,
        tmp_path / "wp_v32_outcome_blind_candidate_index.parquet",
    )
    manifest = {
        "schema_version": DATA_SCHEMA_VERSION,
        "profit_outcomes_read": False,
        "future_information_allowed": False,
        "v32_model_research_authorized": True,
        "coverage_audit": {
            "query_contract_passed": True,
            "event_coverage_gate_passed": True,
            "probe_parity": {"passed": True},
            "candidate_features": {
                "candidate_rows": len(features),
                "coverage_passed": True,
            },
        },
        "artifacts": {
            "event_features": {"sha256": file_sha256(feature_path)},
            "candidate_index": {"sha256": file_sha256(index_path)},
        },
    }
    (
        tmp_path / "wp_v32_public_event_data_manifest.json"
    ).write_text(json.dumps(manifest), encoding="utf-8")


def test_loader_enforces_outcome_blind_digest_bound_artifact(
    tmp_path,
) -> None:
    write_data_artifact(tmp_path)

    features, manifest, integrity = load_v32_event_features(tmp_path)

    assert len(features) == 2
    assert manifest["profit_outcomes_read"] is False
    assert integrity
    assert set(V32_EVENT_FEATURE_COLUMNS).issubset(features.columns)

    manifest_path = (
        tmp_path / "wp_v32_public_event_data_manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["profit_outcomes_read"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not outcome blind"):
        load_v32_event_features(tmp_path)


def test_join_preserves_identity_fold_and_event_flags() -> None:
    features = event_features()
    source = features.loc[
        :, ["trade_date", "signal_slot", "ts_code", "fold"]
    ].copy()
    source["net_return_pct"] = [1.0, -1.0]

    joined = join_event_features(source, features)

    assert len(joined) == 2
    assert joined["v32_event_features_complete"].all()
    assert joined["v32_event_any"].all()
    assert joined["net_return_pct"].tolist() == [1.0, -1.0]


def test_rank_diagnostics_use_only_event_active_rows() -> None:
    scored = event_features().loc[
        :, ["trade_date", "signal_slot", "ts_code"]
    ].copy()
    scored["v32_within_slot_rank_score"] = [0.9, 0.1]
    scored["net_return_pct"] = [2.0, -1.0]
    scored["v32_event_any"] = [True, True]

    diagnostics = event_rank_diagnostics(
        scored,
        seed=32,
        bootstrap_samples=100,
    )

    assert diagnostics["groups"] == 1
    assert diagnostics["mean_within_slot_ic"] == pytest.approx(1.0)
    assert diagnostics["mean_top_minus_bottom_return_pct"] == 3.0
