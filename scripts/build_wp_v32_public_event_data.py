from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import tushare as ts

from wp.v3.history import TushareHistoryClient
from wp.v3.io import atomic_write_json, atomic_write_parquet, file_sha256
from wp.v3.meta_alpha import IDENTITY_COLUMNS
from wp.v3.v24_data import SCHEMA_VERSION as V24_DATA_SCHEMA_VERSION
from wp.v3.v31_public_event import SOURCE_SPECS
from wp.v3.v32_event_features import (
    ADMITTED_SOURCES,
    V32_EVENT_FEATURE_COLUMNS,
    audit_candidate_event_features,
    build_candidate_event_features,
)
from wp.v3.v32_public_event import (
    PROBE_DATES,
    audit_a_share_event_frame,
    build_lookback_map,
    normalize_a_share_event_frame,
)


DATA_SCHEMA_VERSION = "wp_v32_public_event_features_1"
SOURCE_V24_DATA_RUN_ID = 30_635_569_735
SOURCE_V32_PROBE_RUN_ID = 30_664_863_503
V32_PROBE_SCHEMA_VERSION = "wp_v32_a_share_public_event_data_probe_1"
MIN_EVENT_ROW_RATE = 0.05
MIN_EVENT_DATE_RATE = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build outcome-blind V32 public-event features."
    )
    parser.add_argument("--v24-data-dir", required=True)
    parser.add_argument("--v32-probe-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for V32 data build")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates, candidate_source = load_v24_candidate_index(
        args.v24_data_dir
    )
    probe_manifest, probe_presence = load_v32_probe(args.v32_probe_dir)
    target_dates = tuple(
        sorted(candidates["trade_date"].astype(str).unique())
    )
    lookback_map = build_lookback_map(
        candidate_source["open_dates"],
        target_dates,
    )
    required_dates = sorted(
        {
            event_date
            for dates in lookback_map.values()
            for event_date in dates
        }
    )
    candidate_codes = set(candidates["ts_code"].astype(str))
    client = TushareHistoryClient(
        ts.pro_api(token),
        args.cache_dir,
        page_size=8_000,
        requests_per_minute=180,
        attempts=4,
    )

    query_records: list[dict[str, Any]] = []
    query_failures: list[dict[str, str]] = []
    events_by_source: dict[str, pd.DataFrame] = {}
    completed_queries = 0
    expected_queries = len(required_dates) * len(ADMITTED_SOURCES)
    for source in ADMITTED_SOURCES:
        frames: list[pd.DataFrame] = []
        spec = SOURCE_SPECS[source]
        for requested_date in required_dates:
            try:
                frame = client.query(
                    source,
                    cache_key=f"{requested_date}_{source}_v32_full",
                    paged=True,
                    fields=spec["fields"],
                    **{spec["date_arg"]: requested_date},
                )
                record = audit_a_share_event_frame(
                    frame,
                    source=source,
                    requested_date=requested_date,
                )
                query_records.append(record)
                normalized = normalize_a_share_event_frame(
                    frame,
                    source=source,
                )
                normalized = normalized.loc[
                    normalized["ts_code"].isin(candidate_codes)
                ].copy()
                if not normalized.empty:
                    frames.append(normalized)
            except Exception as error:
                failure = {
                    "source": source,
                    "requested_date": requested_date,
                    "error": str(error)[:500],
                }
                query_failures.append(failure)
                query_records.append(
                    {
                        "status": "error",
                        "source": source,
                        "requested_date": requested_date,
                        "raw_rows": 0,
                        "rows": 0,
                        "coverage_pass": False,
                        "error": failure["error"],
                    }
                )
            completed_queries += 1
            if completed_queries % 50 == 0:
                print(
                    "[wp-v32-data] "
                    f"queries={completed_queries}/{expected_queries} "
                    f"failures={len(query_failures)}",
                    flush=True,
                )
        events_by_source[source] = (
            pd.concat(frames, ignore_index=True).drop_duplicates()
            if frames
            else pd.DataFrame(
                columns=[
                    *spec["fields"].split(","),
                    "event_date",
                    "event_source",
                ]
            )
        )

    features = build_candidate_event_features(
        candidates,
        events_by_source,
        lookback_map,
    )
    feature_audit = audit_candidate_event_features(features, candidates)
    probe_parity = audit_probe_parity(features, probe_presence)
    query_contract = bool(
        len(query_records) == expected_queries
        and not query_failures
        and all(record.get("coverage_pass") for record in query_records)
    )
    coverage_gate = bool(
        feature_audit["event_union_row_rate"] >= MIN_EVENT_ROW_RATE
        and feature_audit["event_union_trade_date_rate"]
        >= MIN_EVENT_DATE_RATE
    )
    authorized = bool(
        candidate_source["source_integrity"]
        and query_contract
        and feature_audit["coverage_passed"]
        and coverage_gate
        and probe_parity["passed"]
    )

    candidate_path = atomic_write_parquet(
        candidates,
        output / "wp_v32_outcome_blind_candidate_index.parquet",
    )
    feature_path = atomic_write_parquet(
        features,
        output / "wp_v32_public_event_features.parquet",
    )
    audit_path = output / "wp_v32_source_date_audit.json"
    atomic_write_json(
        audit_path,
        {
            "schema_version": "wp_v32_source_date_audit_1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "required_dates": required_dates,
            "query_records": query_records,
            "query_failures": query_failures,
        },
    )
    manifest = {
        "schema_version": DATA_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "profit_outcomes_read": False,
        "future_information_allowed": False,
        "source_runs": {
            "v24_data_run_id": SOURCE_V24_DATA_RUN_ID,
            "v32_probe_run_id": SOURCE_V32_PROBE_RUN_ID,
        },
        "source_contract": {
            "v24_candidate_source": {
                key: value
                for key, value in candidate_source.items()
                if key != "open_dates"
            },
            "v32_probe_manifest_sha256": probe_manifest[
                "manifest_sha256"
            ],
            "admitted_sources": list(ADMITTED_SOURCES),
            "lookback_trade_days": 5,
            "required_source_dates": int(len(required_dates)),
            "expected_queries": int(expected_queries),
        },
        "feature_contract": {
            "feature_columns": list(V32_EVENT_FEATURE_COLUMNS),
            "signal_price_used_only_for_causal_relative_price_features": True,
            "candidate_profit_fields_used": False,
        },
        "coverage_audit": {
            "query_contract_passed": query_contract,
            "query_failures": int(len(query_failures)),
            "candidate_features": feature_audit,
            "minimum_event_union_row_rate": MIN_EVENT_ROW_RATE,
            "minimum_event_union_trade_date_rate": MIN_EVENT_DATE_RATE,
            "event_coverage_gate_passed": coverage_gate,
            "probe_parity": probe_parity,
        },
        "artifacts": {
            "candidate_index": file_artifact(candidate_path),
            "event_features": file_artifact(feature_path),
            "source_date_audit": file_artifact(audit_path),
        },
        "v32_model_research_authorized": authorized,
        "next_gate": (
            "freeze_v32_nested_oos_model_protocol"
            if authorized
            else "stop_and_diagnose_v32_data_contract"
        ),
    }
    atomic_write_json(
        output / "wp_v32_public_event_data_manifest.json",
        manifest,
    )
    print(
        "WP_V32_FULL_DATA_RESULT="
        + json.dumps(
            {
                "candidate_rows": int(len(candidates)),
                "feature_rows": int(len(features)),
                "trade_dates": int(len(target_dates)),
                "required_source_dates": int(len(required_dates)),
                "queries": int(len(query_records)),
                "query_failures": int(len(query_failures)),
                "feature_audit": feature_audit,
                "probe_parity": probe_parity,
                "v32_model_research_authorized": authorized,
                "next_gate": manifest["next_gate"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    if not authorized:
        raise RuntimeError(
            "V32 full outcome-blind event dataset failed its frozen contract"
        )
    return 0


def load_v24_candidate_index(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(data_dir)
    manifests = list(root.rglob("wp_v24_data_manifest.json"))
    paths = list(root.rglob("wp_v24_outcome_blind_candidate_index.parquet"))
    if len(manifests) != 1 or len(paths) != 1:
        raise RuntimeError("V32 expected one immutable V24 data source")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != V24_DATA_SCHEMA_VERSION:
        raise RuntimeError("V32 V24 candidate manifest schema mismatch")
    if manifest.get("profit_outcomes_read") is not False:
        raise RuntimeError("V32 V24 source did not preserve outcome blindness")
    source = manifest.get("source") or {}
    if source.get("source_integrity") is not True:
        raise RuntimeError("V32 V24 source integrity is not established")
    expected_sha = str(
        (manifest.get("artifacts") or {})
        .get("candidate_index", {})
        .get("sha256")
        or ""
    )
    actual_sha = file_sha256(paths[0])
    if not expected_sha or actual_sha != expected_sha:
        raise RuntimeError("V32 V24 candidate index digest mismatch")
    projected = [*IDENTITY_COLUMNS, "fold", "signal_price"]
    available = set(pq.read_schema(paths[0]).names)
    missing = sorted(set(projected) - available)
    if missing:
        raise RuntimeError(f"V32 V24 candidate columns missing: {missing}")
    candidates = pq.read_table(paths[0], columns=projected).to_pandas()
    candidates["trade_date"] = candidates["trade_date"].astype(str)
    candidates["signal_slot"] = candidates["signal_slot"].astype(str)
    candidates["ts_code"] = candidates["ts_code"].astype(str)
    candidates["fold"] = pd.to_numeric(
        candidates["fold"],
        errors="raise",
    ).astype(int)
    candidates["signal_price"] = pd.to_numeric(
        candidates["signal_price"],
        errors="raise",
    )
    if candidates.duplicated(list(IDENTITY_COLUMNS)).any():
        raise RuntimeError("V32 V24 candidate identities are duplicated")
    expected_rows = int(source.get("candidate_rows", -1))
    if len(candidates) != expected_rows:
        raise RuntimeError("V32 V24 candidate row count mismatch")
    open_dates = [
        str(value)
        for value in manifest["trade_calendar"]["open_dates"]
    ]
    candidates.sort_values(
        ["fold", *IDENTITY_COLUMNS],
        kind="stable",
        inplace=True,
    )
    candidates.reset_index(drop=True, inplace=True)
    return candidates, {
        "source_integrity": True,
        "candidate_rows": int(len(candidates)),
        "trade_dates": int(candidates["trade_date"].nunique()),
        "candidate_index_sha256": actual_sha,
        "manifest_sha256": file_sha256(manifests[0]),
        "open_dates": open_dates,
    }


def load_v32_probe(
    data_dir: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    root = Path(data_dir)
    manifests = list(root.rglob("wp_v32_public_event_probe.json"))
    paths = list(root.rglob("wp_v32_probe_candidate_event_presence.csv"))
    if len(manifests) != 1 or len(paths) != 1:
        raise RuntimeError("V32 expected one immutable probe source")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != V32_PROBE_SCHEMA_VERSION:
        raise RuntimeError("V32 probe manifest schema mismatch")
    if manifest.get("profit_outcomes_read") is not False:
        raise RuntimeError("V32 probe source is not outcome blind")
    if manifest.get("full_backfill_authorized") is not True:
        raise RuntimeError("V32 probe did not authorize the full build")
    expected_sha = str(
        (manifest.get("artifacts") or {})
        .get("candidate_event_presence", {})
        .get("sha256")
        or ""
    )
    if not expected_sha or file_sha256(paths[0]) != expected_sha:
        raise RuntimeError("V32 probe presence digest mismatch")
    return {
        **manifest,
        "manifest_sha256": file_sha256(manifests[0]),
    }, pd.read_csv(paths[0], dtype={"trade_date": str, "ts_code": str})


def audit_probe_parity(
    features: pd.DataFrame,
    probe_presence: pd.DataFrame,
) -> dict[str, Any]:
    probe = probe_presence.loc[
        probe_presence["trade_date"].astype(str).isin(PROBE_DATES),
        ["trade_date", "ts_code"]
        + [f"event_{source}" for source in ADMITTED_SOURCES],
    ].copy()
    probe["trade_date"] = probe["trade_date"].astype(str)
    probe["ts_code"] = probe["ts_code"].astype(str)
    for source in ADMITTED_SOURCES:
        column = f"event_{source}"
        probe[column] = normalize_bool(probe[column])
    sample = features.loc[
        features["trade_date"].astype(str).isin(PROBE_DATES)
    ].copy()
    records = sample[["trade_date", "ts_code"]].drop_duplicates()
    for source in ADMITTED_SOURCES:
        active = (
            sample.groupby(["trade_date", "ts_code"])[
                f"v32_{source}_active_5d"
            ]
            .max()
            .gt(0)
            .rename(f"full_{source}")
            .reset_index()
        )
        records = records.merge(
            active,
            on=["trade_date", "ts_code"],
            how="left",
            validate="one_to_one",
        )
    merged = probe.merge(
        records,
        on=["trade_date", "ts_code"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    mismatches = pd.Series(False, index=merged.index)
    for source in ADMITTED_SOURCES:
        mismatches |= (
            normalize_bool(merged[f"event_{source}"])
            != normalize_bool(merged[f"full_{source}"])
        )
    identity_match = bool(merged["_merge"].eq("both").all())
    mismatch_rows = int(mismatches.sum())
    return {
        "expected_identities": int(len(probe)),
        "full_identities": int(len(records)),
        "identity_match": identity_match,
        "presence_mismatch_rows": mismatch_rows,
        "passed": bool(identity_match and mismatch_rows == 0),
    }


def normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def file_artifact(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    return {
        "path": str(file_path.as_posix()),
        "sha256": file_sha256(file_path),
        "bytes": int(file_path.stat().st_size),
    }


if __name__ == "__main__":
    raise SystemExit(main())
