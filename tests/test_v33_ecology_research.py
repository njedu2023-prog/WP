from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_wp_v33_limit_industry_ecology import (
    DATA_SCHEMA_VERSION,
    join_ecology_features,
    load_v33_ecology_features,
)
from wp.v3.io import file_sha256
from wp.v3.v33_limit_ecology import V33_LIMIT_ECOLOGY_FEATURE_COLUMNS


def candidate_rows() -> pd.DataFrame:
    rows = []
    for index in range(3):
        row = {
            "trade_date": "20260723",
            "signal_slot": f"14:{20 + index * 5:02d}",
            "ts_code": f"{index:06d}.SZ",
            "fold": 1,
            "l2_code": "L2",
            "l3_code": f"L3{index}",
            "v33_membership_available": 1.0,
            "v33_ecology_active_before_signal": 1.0,
        }
        row.update(
            {
                column: float(index + offset)
                for offset, column in enumerate(
                    V33_LIMIT_ECOLOGY_FEATURE_COLUMNS
                )
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_data_artifact(root: Path) -> None:
    features = candidate_rows()
    index = features[
        ["trade_date", "signal_slot", "ts_code", "fold"]
    ].copy()
    feature_path = (
        root / "wp_v33_limit_industry_ecology_features.parquet"
    )
    index_path = root / "wp_v33_outcome_blind_candidate_index.parquet"
    features.to_parquet(feature_path, index=False)
    index.to_parquet(index_path, index=False)
    manifest = {
        "schema_version": DATA_SCHEMA_VERSION,
        "profit_outcomes_read": False,
        "future_information_allowed": False,
        "v33_model_research_authorized": True,
        "source_contract": {
            "v24": {"manifest_sha256": "placeholder"}
        },
        "coverage_audit": {
            "query_contract_passed": True,
            "date_contract_passed": True,
            "candidate_features": {
                "coverage_passed": True,
                "candidate_rows": 3,
            },
            "full_three_year_coverage": {"passed": True},
            "probe_feature_parity": {"passed": True},
        },
        "artifacts": {
            "ecology_features": {
                "sha256": file_sha256(feature_path)
            },
            "candidate_index": {
                "sha256": file_sha256(index_path)
            },
        },
    }
    (
        root / "wp_v33_limit_industry_ecology_data_manifest.json"
    ).write_text(json.dumps(manifest), encoding="utf-8")


def test_load_v33_ecology_features_verifies_artifacts(
    tmp_path: Path,
) -> None:
    write_data_artifact(tmp_path)

    features, manifest, integrity = load_v33_ecology_features(tmp_path)

    assert len(features) == 3
    assert manifest["schema_version"] == DATA_SCHEMA_VERSION
    assert integrity is True


def test_join_ecology_features_requires_exact_identities() -> None:
    ecology = candidate_rows()
    source = ecology[
        ["trade_date", "signal_slot", "ts_code", "fold"]
    ].copy()
    source["label_available"] = True

    joined = join_ecology_features(source, ecology)

    assert len(joined) == 3
    assert joined["v33_ecology_features_complete"].all()
    numeric = joined[
        list(V33_LIMIT_ECOLOGY_FEATURE_COLUMNS)
    ].to_numpy(dtype=float)
    assert np.isfinite(numeric).all()


def test_join_ecology_features_rejects_missing_identity() -> None:
    ecology = candidate_rows().iloc[:-1].copy()
    source = candidate_rows()[
        ["trade_date", "signal_slot", "ts_code", "fold"]
    ]

    try:
        join_ecology_features(source, ecology)
    except RuntimeError as error:
        assert "missed source identities" in str(error)
    else:
        raise AssertionError("V33 missing identity was accepted")
