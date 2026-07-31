from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from wp.v3.io import atomic_write_json, file_sha256
from wp.v3.v28_industry_peer import (
    MINIMUM_L2_PEERS,
    MINIMUM_L3_PEERS,
    V28_PEER_FEATURE_COLUMNS,
)


DATA_SCHEMA_VERSION = "wp_v28_fine_industry_peer_features_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the outcome-blind V28 peer-feature coverage shortfall."
        )
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.data_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifests = list(
        root.rglob("wp_v28_fine_industry_peer_data_manifest.json")
    )
    feature_paths = list(
        root.rglob("wp_v28_fine_industry_peer_features.parquet")
    )
    if len(manifests) != 1 or len(feature_paths) != 1:
        raise RuntimeError(
            "expected exactly one V28 data manifest and feature parquet"
        )
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATA_SCHEMA_VERSION:
        raise RuntimeError("V28 diagnostic source schema mismatch")
    if manifest.get("profit_outcomes_read") is not False:
        raise RuntimeError("V28 diagnostic source is not outcome blind")
    expected_sha = str(
        (manifest.get("artifacts") or {})
        .get("peer_features", {})
        .get("sha256")
        or ""
    )
    actual_sha = file_sha256(feature_paths[0])
    if not expected_sha or actual_sha != expected_sha:
        raise RuntimeError("V28 diagnostic feature digest mismatch")
    features = pd.read_parquet(feature_paths[0])
    summary = coverage_diagnosis(features)
    payload = {
        "schema_version": "wp_v28_feature_coverage_diagnosis_1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "profit_outcomes_read": False,
        "source_feature_sha256": actual_sha,
        "source_run_id": 30_656_696_310,
        "source_job_id": 91_242_705_039,
        "diagnosis": summary,
        "threshold_changed": False,
        "next_gate": (
            "engineering_fix_only"
            if summary["incomplete_with_sufficient_peer_depth"] > 0
            else "close_v28_data_contract_and_preregister_new_direction"
        ),
    }
    atomic_write_json(
        output / "wp_v28_feature_coverage_diagnosis.json",
        payload,
    )
    print(
        "WP_V28_COVERAGE_DIAGNOSIS="
        + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def coverage_diagnosis(features: pd.DataFrame) -> dict[str, Any]:
    missing = sorted(set(V28_PEER_FEATURE_COLUMNS) - set(features.columns))
    if missing:
        raise RuntimeError(f"V28 feature columns missing: {missing}")
    numeric = features.loc[:, V28_PEER_FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    l2_columns = [
        column
        for column in V28_PEER_FEATURE_COLUMNS
        if column.startswith("v28_l2_")
    ]
    l3_columns = [
        column
        for column in V28_PEER_FEATURE_COLUMNS
        if column.startswith("v28_l3_")
    ]
    l2_count = numeric["v28_l2_peer_count"]
    l3_count = numeric["v28_l3_peer_count"]
    sufficient_depth = l2_count.ge(MINIMUM_L2_PEERS) & l3_count.ge(
        MINIMUM_L3_PEERS
    )
    all_non_null = numeric.notna().all(axis=1)
    complete = sufficient_depth & all_non_null
    incomplete = ~complete
    missing_l2 = numeric[l2_columns].isna().any(axis=1)
    missing_l3 = numeric[l3_columns].isna().any(axis=1)
    amount_columns = [
        "v28_l2_peer_own_log_amount_excess",
        "v28_l3_peer_own_log_amount_excess",
    ]
    non_amount_columns = [
        column
        for column in V28_PEER_FEATURE_COLUMNS
        if column not in amount_columns
    ]
    amount_only_missing = (
        numeric[amount_columns].isna().any(axis=1)
        & numeric[non_amount_columns].notna().all(axis=1)
        & sufficient_depth
    )
    incomplete_dates = (
        features.loc[incomplete, "trade_date"].astype(str).nunique()
        if "trade_date" in features
        else 0
    )
    incomplete_by_year = {}
    if "trade_date" in features:
        years = features.loc[incomplete, "trade_date"].astype(str).str[:4]
        incomplete_by_year = {
            str(year): int(count)
            for year, count in years.value_counts().sort_index().items()
        }
    missing_by_feature = {
        column: int(numeric[column].isna().sum())
        for column in V28_PEER_FEATURE_COLUMNS
        if numeric[column].isna().any()
    }
    return {
        "rows": int(len(features)),
        "complete_rows": int(complete.sum()),
        "complete_coverage": (
            float(complete.mean()) if len(features) else 0.0
        ),
        "incomplete_rows": int(incomplete.sum()),
        "incomplete_trade_dates": int(incomplete_dates),
        "l2_peer_depth_below_4": int(l2_count.lt(MINIMUM_L2_PEERS).sum()),
        "l3_peer_depth_below_2": int(l3_count.lt(MINIMUM_L3_PEERS).sum()),
        "either_peer_depth_too_shallow": int((~sufficient_depth).sum()),
        "missing_any_l2_feature": int(missing_l2.sum()),
        "missing_any_l3_feature": int(missing_l3.sum()),
        "incomplete_with_sufficient_peer_depth": int(
            (incomplete & sufficient_depth).sum()
        ),
        "amount_only_missing_with_sufficient_depth": int(
            amount_only_missing.sum()
        ),
        "zero_l2_peers": int(l2_count.eq(0).sum()),
        "zero_l3_peers": int(l3_count.eq(0).sum()),
        "l2_peer_count_quantiles": finite_quantiles(l2_count),
        "l3_peer_count_quantiles": finite_quantiles(l3_count),
        "missing_by_feature": missing_by_feature,
        "incomplete_by_year": incomplete_by_year,
    }


def finite_quantiles(values: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {
            "p00": None,
            "p01": None,
            "p05": None,
            "p10": None,
            "p50": None,
        }
    return {
        "p00": float(numeric.min()),
        "p01": float(numeric.quantile(0.01)),
        "p05": float(numeric.quantile(0.05)),
        "p10": float(numeric.quantile(0.10)),
        "p50": float(numeric.quantile(0.50)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
