from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from build_wp_v33_limit_industry_ecology import (
    DATA_SCHEMA_VERSION,
    audit_full_coverage,
    audit_probe_feature_parity,
    load_v33_probe,
)
from wp.v3.io import file_sha256
from wp.v3.meta_alpha import IDENTITY_COLUMNS
from wp.v3.v33_limit_ecology import V33_LIMIT_ECOLOGY_FEATURE_COLUMNS


def make_features(rows: int = 10) -> pd.DataFrame:
    records = []
    for index in range(rows):
        record = {
            "trade_date": f"202401{index + 1:02d}",
            "signal_slot": "14:20",
            "ts_code": f"{index:06d}.SZ",
        }
        record.update(
            {
                column: 1.0
                for column in V33_LIMIT_ECOLOGY_FEATURE_COLUMNS
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def test_audit_full_coverage_passes_dense_features() -> None:
    result = audit_full_coverage(make_features())

    assert result["passed"] is True
    assert result["l2_active_row_rate"] == 1.0
    assert result["l3_active_date_rate"] == 1.0


def test_audit_full_coverage_rejects_sparse_l3() -> None:
    features = make_features()
    features.loc[:, "v33_l3_limit_hit_count"] = 0.0

    result = audit_full_coverage(features)

    assert result["passed"] is False
    assert result["l3_active_rows"] == 0


def test_probe_feature_parity_is_exact() -> None:
    probe = make_features(3)
    full = make_features(5)

    result = audit_probe_feature_parity(full, probe)

    assert result["passed"] is True
    assert result["mismatch_rows"] == 0


def test_probe_feature_parity_rejects_numeric_drift() -> None:
    probe = make_features(3)
    full = make_features(5)
    full.loc[
        full["trade_date"].eq("20240101"),
        "v33_l2_limit_hit_count",
    ] = 2.0

    result = audit_probe_feature_parity(full, probe)

    assert result["passed"] is False
    assert result["mismatch_rows"] == 1


def test_load_v33_probe_verifies_contract_and_digest(
    tmp_path: Path,
) -> None:
    features = make_features(2)
    feature_path = (
        tmp_path / "wp_v33_probe_candidate_industry_ecology.parquet"
    )
    features.to_parquet(feature_path, index=False)
    manifest = {
        "schema_version": "wp_v33_limit_industry_ecology_probe_1",
        "profit_outcomes_read": False,
        "future_information_allowed": False,
        "full_backfill_authorized": True,
        "artifacts": {
            "candidate_industry_ecology": {
                "sha256": file_sha256(feature_path)
            }
        },
    }
    (
        tmp_path / "wp_v33_limit_industry_ecology_probe.json"
    ).write_text(json.dumps(manifest), encoding="utf-8")

    loaded, source = load_v33_probe(tmp_path)

    assert len(loaded) == 2
    assert source["source_integrity"] is True


def test_builder_is_outcome_blind_by_static_contract() -> None:
    source = Path(
        "scripts/build_wp_v33_limit_industry_ecology.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert DATA_SCHEMA_VERSION in string_literals
    assert "net_return" not in source
    assert "gross_return" not in source
    assert "exit_close" not in source
    assert "target_positive" not in source
    assert "profit_outcomes_read" in source
    assert set(IDENTITY_COLUMNS).issubset(
        {"trade_date", "signal_slot", "ts_code"}
    )
