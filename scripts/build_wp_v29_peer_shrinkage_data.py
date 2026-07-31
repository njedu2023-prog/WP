from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from wp.v3.io import atomic_write_json, atomic_write_parquet, file_sha256
from wp.v3.v24_data import SCHEMA_VERSION as V24_DATA_SCHEMA_VERSION
from wp.v3.v28_industry_peer import V28_PEER_FEATURE_COLUMNS
from wp.v3.v29_peer_shrinkage import (
    IDENTITY_COLUMNS,
    L3_SHRINKAGE_PSEUDO_PEERS,
    V29_FEATURE_COLUMNS,
    audit_hierarchical_feature_coverage,
    build_hierarchical_peer_features,
)


V28_DATA_SCHEMA_VERSION = "wp_v28_fine_industry_peer_features_1"
V28_DIAGNOSIS_SCHEMA_VERSION = "wp_v28_feature_coverage_diagnosis_1"
V29_DATA_SCHEMA_VERSION = "wp_v29_hierarchical_peer_features_1"
SOURCE_V24_DATA_RUN_ID = 30_635_569_735
SOURCE_V28_DATA_RUN_ID = 30_656_696_310
SOURCE_V28_DIAGNOSIS_RUN_ID = 30_659_154_353


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build outcome-blind V29 hierarchical peer shrinkage features."
        )
    )
    parser.add_argument("--v24-data-dir", required=True)
    parser.add_argument("--v28-data-dir", required=True)
    parser.add_argument("--v28-diagnosis-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    v24_manifest, v24_manifest_path = load_single_manifest(
        args.v24_data_dir,
        "wp_v24_data_manifest.json",
        expected_schema=V24_DATA_SCHEMA_VERSION,
    )
    v28_manifest, v28_manifest_path = load_single_manifest(
        args.v28_data_dir,
        "wp_v28_fine_industry_peer_data_manifest.json",
        expected_schema=V28_DATA_SCHEMA_VERSION,
    )
    diagnosis, diagnosis_path = load_single_manifest(
        args.v28_diagnosis_dir,
        "wp_v28_feature_coverage_diagnosis.json",
        expected_schema=V28_DIAGNOSIS_SCHEMA_VERSION,
    )
    validate_source_contracts(v24_manifest, v28_manifest, diagnosis)

    candidates, candidate_path = load_verified_parquet(
        args.v28_data_dir,
        "wp_v28_outcome_blind_candidate_index.parquet",
        v28_manifest,
        artifact_key="candidate_index",
    )
    v28_features, v28_feature_path = load_verified_parquet(
        args.v28_data_dir,
        "wp_v28_fine_industry_peer_features.parquet",
        v28_manifest,
        artifact_key="peer_features",
    )
    validate_v28_identity(candidates, v28_features)
    validate_v24_identity_contract(
        args.v24_data_dir,
        v24_manifest,
        v28_manifest,
        candidates,
    )

    features = build_hierarchical_peer_features(v28_features)
    audit = audit_hierarchical_feature_coverage(features, candidates)
    authorized = bool(
        audit["coverage_passed"]
        and diagnosis["diagnosis"][
            "incomplete_with_sufficient_peer_depth"
        ]
        == 0
    )

    candidate_output = atomic_write_parquet(
        candidates,
        output / "wp_v29_outcome_blind_candidate_index.parquet",
    )
    feature_output = atomic_write_parquet(
        features,
        output / "wp_v29_hierarchical_peer_features.parquet",
    )
    manifest = {
        "schema_version": V29_DATA_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "profit_outcomes_read": False,
        "future_information_allowed": False,
        "source_runs": {
            "v24_data_run_id": SOURCE_V24_DATA_RUN_ID,
            "v28_data_run_id": SOURCE_V28_DATA_RUN_ID,
            "v28_diagnosis_run_id": SOURCE_V28_DIAGNOSIS_RUN_ID,
        },
        "source": v24_manifest.get("source"),
        "trade_calendar": v24_manifest.get("trade_calendar"),
        "requirements": {
            "candidate_rows": int(len(candidates)),
            "feature_rows": int(len(features)),
            "minimum_finite_feature_coverage": 1.0,
        },
        "source_contract": {
            "v24_manifest_sha256": file_sha256(v24_manifest_path),
            "v28_manifest_sha256": file_sha256(v28_manifest_path),
            "v28_diagnosis_sha256": file_sha256(diagnosis_path),
            "v28_candidate_sha256": file_sha256(candidate_path),
            "v28_peer_feature_sha256": file_sha256(v28_feature_path),
            "v28_closed_without_outcomes": True,
            "v28_incomplete_rows": int(
                diagnosis["diagnosis"]["incomplete_rows"]
            ),
            "v28_incomplete_with_sufficient_peer_depth": int(
                diagnosis["diagnosis"][
                    "incomplete_with_sufficient_peer_depth"
                ]
            ),
        },
        "transformation_contract": {
            "source_features": list(V28_PEER_FEATURE_COLUMNS),
            "output_features": list(V29_FEATURE_COLUMNS),
            "candidate_excluded_from_source_peer_aggregates": True,
            "l3_shrinkage_target": "leave_one_out_l2_peer_metric",
            "l3_shrinkage_pseudo_peers": L3_SHRINKAGE_PSEUDO_PEERS,
            "l3_weight_formula": "n_l3/(n_l3+6)",
            "l3_missing_fallback": "leave_one_out_l2_peer_metric",
            "both_levels_missing_value": 0.0,
            "both_levels_missing_indicator": (
                "v29_peer_no_peer_context"
            ),
            "candidate_rows_dropped": 0,
            "outcomes_used": False,
        },
        "coverage_audit": audit,
        "artifacts": {
            "candidate_index": file_artifact(candidate_output),
            "features": file_artifact(feature_output),
        },
        "v29_model_research_authorized": authorized,
        "next_gate": (
            "freeze_v29_nested_oos_model_protocol"
            if authorized
            else "close_v29_data_contract"
        ),
    }
    atomic_write_json(
        output / "wp_v29_peer_shrinkage_data_manifest.json",
        manifest,
    )
    print(
        "WP_V29_DATA_RESULT="
        + json.dumps(
            {
                "candidate_rows": int(len(candidates)),
                "feature_rows": int(len(features)),
                "coverage_audit": audit,
                "v29_model_research_authorized": authorized,
                "next_gate": manifest["next_gate"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    if not authorized:
        raise RuntimeError("V29 outcome-blind data contract failed")
    return 0


def load_single_manifest(
    root: str | Path,
    filename: str,
    *,
    expected_schema: str,
) -> tuple[dict[str, Any], Path]:
    paths = list(Path(root).rglob(filename))
    if len(paths) != 1:
        raise RuntimeError(
            f"expected one {filename}; found {len(paths)} under {root}"
        )
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    if payload.get("schema_version") != expected_schema:
        raise RuntimeError(f"{filename} schema mismatch")
    return payload, paths[0]


def load_verified_parquet(
    root: str | Path,
    filename: str,
    manifest: dict[str, Any],
    *,
    artifact_key: str,
) -> tuple[pd.DataFrame, Path]:
    paths = list(Path(root).rglob(filename))
    if len(paths) != 1:
        raise RuntimeError(
            f"expected one {filename}; found {len(paths)} under {root}"
        )
    expected_sha = str(
        (manifest.get("artifacts") or {})
        .get(artifact_key, {})
        .get("sha256")
        or ""
    )
    actual_sha = file_sha256(paths[0])
    if not expected_sha or actual_sha != expected_sha:
        raise RuntimeError(f"{filename} digest mismatch")
    return pd.read_parquet(paths[0]), paths[0]


def validate_source_contracts(
    v24_manifest: dict[str, Any],
    v28_manifest: dict[str, Any],
    diagnosis: dict[str, Any],
) -> None:
    if v24_manifest.get("profit_outcomes_read") is not False:
        raise RuntimeError("V24 source is not outcome blind")
    if (v24_manifest.get("source") or {}).get("source_integrity") is not True:
        raise RuntimeError("V24 source integrity is not established")
    if v28_manifest.get("profit_outcomes_read") is not False:
        raise RuntimeError("V28 source is not outcome blind")
    if v28_manifest.get("future_information_allowed") is not False:
        raise RuntimeError("V28 source allowed future information")
    if v28_manifest.get("v28_model_research_authorized") is not False:
        raise RuntimeError("V28 source was not the frozen failed-gate build")
    if diagnosis.get("profit_outcomes_read") is not False:
        raise RuntimeError("V28 diagnosis is not outcome blind")
    if diagnosis.get("next_gate") != (
        "close_v28_data_contract_and_preregister_new_direction"
    ):
        raise RuntimeError("V28 diagnosis did not close the data contract")
    expected_feature_sha = str(
        (v28_manifest.get("artifacts") or {})
        .get("peer_features", {})
        .get("sha256")
        or ""
    )
    if diagnosis.get("source_feature_sha256") != expected_feature_sha:
        raise RuntimeError("V28 diagnosis does not match the source features")


def validate_v28_identity(
    candidates: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    required_candidate = {*IDENTITY_COLUMNS, "fold"}
    missing_candidate = sorted(required_candidate - set(candidates.columns))
    missing_feature = sorted(
        required_candidate.union(V28_PEER_FEATURE_COLUMNS)
        - set(features.columns)
    )
    if missing_candidate or missing_feature:
        raise RuntimeError(
            "V28 identity contract missing columns: "
            f"candidate={missing_candidate} feature={missing_feature}"
        )
    if candidates.duplicated(list(IDENTITY_COLUMNS)).any():
        raise RuntimeError("V28 candidate identities are duplicated")
    if features.duplicated(list(IDENTITY_COLUMNS)).any():
        raise RuntimeError("V28 feature identities are duplicated")
    merged = candidates.loc[
        :, [*IDENTITY_COLUMNS, "fold"]
    ].merge(
        features.loc[:, [*IDENTITY_COLUMNS, "fold"]],
        on=[*IDENTITY_COLUMNS, "fold"],
        how="outer",
        indicator=True,
    )
    if len(merged) != len(candidates) or not merged["_merge"].eq("both").all():
        raise RuntimeError("V28 candidate and feature identities differ")


def validate_v24_identity_contract(
    root: str | Path,
    v24_manifest: dict[str, Any],
    v28_manifest: dict[str, Any],
    v28_candidates: pd.DataFrame,
) -> None:
    paths = list(
        Path(root).rglob("wp_v24_outcome_blind_candidate_index.parquet")
    )
    if len(paths) != 1:
        raise RuntimeError("V29 requires one immutable V24 candidate index")
    expected_sha = str(
        (v24_manifest.get("artifacts") or {})
        .get("candidate_index", {})
        .get("sha256")
        or ""
    )
    if not expected_sha or file_sha256(paths[0]) != expected_sha:
        raise RuntimeError("V24 candidate index digest mismatch")
    v28_source_sha = str(
        (v28_manifest.get("source_contract") or {})
        .get("v24_candidate_source", {})
        .get("candidate_index_sha256")
        or ""
    )
    if expected_sha != v28_source_sha:
        raise RuntimeError("V28 did not derive from the immutable V24 index")
    v24 = pd.read_parquet(
        paths[0],
        columns=[*IDENTITY_COLUMNS, "fold"],
    )
    for frame in (v24, v28_candidates):
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame["signal_slot"] = frame["signal_slot"].astype(str)
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(
            int
        )
    comparison = v24.merge(
        v28_candidates.loc[:, [*IDENTITY_COLUMNS, "fold"]],
        on=[*IDENTITY_COLUMNS, "fold"],
        how="outer",
        indicator=True,
    )
    if (
        len(comparison) != len(v28_candidates)
        or not comparison["_merge"].eq("both").all()
    ):
        raise RuntimeError("V29 V24 and V28 candidate identities differ")


def file_artifact(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    return {
        "path": file_path.as_posix(),
        "sha256": file_sha256(file_path),
        "bytes": int(file_path.stat().st_size),
    }


if __name__ == "__main__":
    raise SystemExit(main())
