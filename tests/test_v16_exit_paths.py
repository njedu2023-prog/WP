from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.verify_wp_v16_exit_paths import verify_exit_paths
from wp.v3.io import atomic_write_parquet, canonical_digest, file_sha256
from wp.v3.v16_data import SCHEMA_VERSION


def write_dataset(root: Path) -> dict[str, object]:
    minute = root / "minute"
    minute.mkdir(parents=True)
    partition = atomic_write_parquet(
        pd.DataFrame(
            {
                "target_trade_date": ["20260721"],
                "ts_code": ["600001.SH"],
                "close": [10.0],
            }
        ),
        minute / "wp_v16_t1_minutes_202607.parquet",
    )
    quality = atomic_write_parquet(
        pd.DataFrame(
            {
                "target_trade_date": ["20260721"],
                "ts_code": ["600001.SH"],
                "covered": [True],
            }
        ),
        root / "wp_v16_t1_path_quality.parquet",
    )
    source = {"v11_frontier_sha256": "abc"}
    contract = {
        "bar_frequency": "5min",
        "minimum_pair_coverage": 0.98,
    }
    partitions = [
        {
            "month": "202607",
            "path": partition.name,
            "rows": 1,
            "pairs": 1,
            "sha256": file_sha256(partition),
        }
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "source_digest": canonical_digest(source),
        "contract": contract,
        "required_pairs": 1,
        "covered_pairs": 1,
        "pair_coverage": 1.0,
        "rows": 1,
        "quality_sha256": file_sha256(quality),
        "partitions": partitions,
    }
    manifest["dataset_fingerprint"] = canonical_digest(
        {
            "schema_version": manifest["schema_version"],
            "source_digest": manifest["source_digest"],
            "contract": contract,
            "required_pairs": 1,
            "covered_pairs": 1,
            "partitions": [
                {
                    "month": item["month"],
                    "rows": item["rows"],
                    "pairs": item["pairs"],
                    "sha256": item["sha256"],
                }
                for item in partitions
            ],
        }
    )
    (root / "wp_v16_t1_path_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return manifest


def test_exit_path_verifier_accepts_complete_fingerprinted_data(
    tmp_path: Path,
) -> None:
    write_dataset(tmp_path)

    result = verify_exit_paths(
        tmp_path,
        minimum_pair_coverage=0.98,
    )

    assert result["verified"] is True
    assert result["required_pairs"] == 1
    assert result["rows"] == 1


def test_exit_path_verifier_rejects_partition_tampering(
    tmp_path: Path,
) -> None:
    write_dataset(tmp_path)
    partition = tmp_path / "minute/wp_v16_t1_minutes_202607.parquet"
    partition.write_bytes(partition.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="partition digest mismatch"):
        verify_exit_paths(
            tmp_path,
            minimum_pair_coverage=0.98,
        )
