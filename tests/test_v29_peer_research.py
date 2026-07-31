from __future__ import annotations

import json

import pandas as pd
import pytest

from research_wp_v29_peer_shrinkage import (
    join_peer_features,
    load_v29_peer_features,
    peer_rank_diagnostics,
)
from wp.v3.io import atomic_write_parquet, file_sha256
from wp.v3.v29_peer_shrinkage import (
    SCHEMA_VERSION,
    V29_FEATURE_COLUMNS,
)


def peer_features() -> pd.DataFrame:
    rows = []
    for index, code in enumerate(("000001.SZ", "000002.SZ")):
        row = {
            "trade_date": "20260724",
            "signal_slot": "14:20",
            "ts_code": code,
            "fold": 3,
        }
        for offset, column in enumerate(V29_FEATURE_COLUMNS):
            row[column] = float(index + offset / 100.0)
        rows.append(row)
    return pd.DataFrame(rows)


def write_data_artifact(tmp_path) -> None:
    features = peer_features()
    candidate_index = features.loc[
        :, ["trade_date", "signal_slot", "ts_code", "fold"]
    ]
    feature_path = atomic_write_parquet(
        features,
        tmp_path / "wp_v29_hierarchical_peer_features.parquet",
    )
    index_path = atomic_write_parquet(
        candidate_index,
        tmp_path / "wp_v29_outcome_blind_candidate_index.parquet",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "profit_outcomes_read": False,
        "future_information_allowed": False,
        "v29_model_research_authorized": True,
        "requirements": {"candidate_rows": len(features)},
        "coverage_audit": {
            "coverage_passed": True,
            "finite_feature_coverage": 1.0,
        },
        "artifacts": {
            "features": {"sha256": file_sha256(feature_path)},
            "candidate_index": {"sha256": file_sha256(index_path)},
        },
    }
    (
        tmp_path / "wp_v29_peer_shrinkage_data_manifest.json"
    ).write_text(json.dumps(manifest), encoding="utf-8")


def test_loader_enforces_outcome_blind_digest_bound_artifact(
    tmp_path,
) -> None:
    write_data_artifact(tmp_path)

    features, manifest, integrity = load_v29_peer_features(tmp_path)

    assert len(features) == 2
    assert manifest["profit_outcomes_read"] is False
    assert integrity

    manifest_path = (
        tmp_path / "wp_v29_peer_shrinkage_data_manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["profit_outcomes_read"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not outcome blind"):
        load_v29_peer_features(tmp_path)


def test_join_preserves_identity_fold_and_finite_features() -> None:
    features = peer_features()
    source = features.loc[
        :, ["trade_date", "signal_slot", "ts_code", "fold"]
    ].copy()
    source["net_return_pct"] = [1.0, -1.0]

    joined = join_peer_features(source, features)

    assert len(joined) == 2
    assert joined["v29_peer_features_complete"].all()
    assert joined["net_return_pct"].tolist() == [1.0, -1.0]


def test_rank_diagnostics_use_v29_same_slot_score() -> None:
    scored = peer_features().loc[
        :, ["trade_date", "signal_slot", "ts_code"]
    ].copy()
    scored["v29_within_slot_rank_score"] = [0.9, 0.1]
    scored["net_return_pct"] = [2.0, -1.0]

    diagnostics = peer_rank_diagnostics(
        scored,
        seed=29,
        bootstrap_samples=100,
    )

    assert diagnostics["groups"] == 1
    assert diagnostics["mean_within_slot_ic"] == pytest.approx(1.0)
    assert diagnostics["mean_top_minus_bottom_return_pct"] == 3.0
