from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from wp.v3.io import canonical_digest, file_sha256
from wp.v3.v16_data import SCHEMA_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a complete cached V16 T+1 five-minute dataset."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--minimum-pair-coverage",
        type=float,
        default=0.98,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify_exit_paths(
        args.dataset_dir,
        minimum_pair_coverage=args.minimum_pair_coverage,
    )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


def verify_exit_paths(
    dataset_dir: str | Path,
    *,
    minimum_pair_coverage: float,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest_path = root / "wp_v16_t1_path_manifest.json"
    quality_path = root / "wp_v16_t1_path_quality.parquet"
    if not manifest_path.is_file() or not quality_path.is_file():
        raise FileNotFoundError("V16 exit-path manifest or quality file missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported V16 exit-path schema")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("V16 exit-path source contract missing")
    if canonical_digest(source) != manifest.get("source_digest"):
        raise RuntimeError("V16 exit-path source digest mismatch")
    coverage = float(manifest.get("pair_coverage") or 0.0)
    required = int(manifest.get("required_pairs") or 0)
    covered = int(manifest.get("covered_pairs") or 0)
    if required <= 0 or covered != required:
        raise RuntimeError(
            f"V16 exit-path pair count mismatch: {covered}/{required}"
        )
    if coverage < minimum_pair_coverage:
        raise RuntimeError(
            f"V16 exit-path coverage {coverage:.2%} below "
            f"{minimum_pair_coverage:.2%}"
        )
    if file_sha256(quality_path) != manifest.get("quality_sha256"):
        raise RuntimeError("V16 exit-path quality digest mismatch")
    expected_fingerprint = canonical_digest(
        {
            "schema_version": manifest["schema_version"],
            "source_digest": manifest["source_digest"],
            "contract": manifest.get("contract"),
            "required_pairs": required,
            "covered_pairs": covered,
            "partitions": [
                {
                    "month": partition.get("month"),
                    "rows": partition.get("rows"),
                    "pairs": partition.get("pairs"),
                    "sha256": partition.get("sha256"),
                }
                for partition in manifest.get("partitions") or []
            ],
        }
    )
    if expected_fingerprint != manifest.get("dataset_fingerprint"):
        raise RuntimeError("V16 exit-path dataset fingerprint mismatch")
    rows = 0
    pairs = 0
    for partition in manifest.get("partitions") or []:
        path = root / "minute" / str(partition.get("path") or "")
        if not path.is_file():
            raise FileNotFoundError(
                f"V16 exit-path partition missing: {path.name}"
            )
        if file_sha256(path) != partition.get("sha256"):
            raise RuntimeError(
                f"V16 exit-path partition digest mismatch: {path.name}"
            )
        rows += int(partition.get("rows") or 0)
        pairs += int(partition.get("pairs") or 0)
    if rows != int(manifest.get("rows") or -1):
        raise RuntimeError(
            f"V16 exit-path row count mismatch: {rows} != "
            f"{manifest.get('rows')}"
        )
    if pairs != covered:
        raise RuntimeError(
            f"V16 exit-path partition pair count mismatch: {pairs} != "
            f"{covered}"
        )
    if not manifest.get("partitions"):
        raise RuntimeError("V16 exit-path dataset has no partitions")
    return {
        "verified": True,
        "schema_version": SCHEMA_VERSION,
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "required_pairs": required,
        "covered_pairs": covered,
        "pair_coverage": coverage,
        "rows": rows,
        "partition_pair_counts": pairs,
        "partitions": len(manifest["partitions"]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
