from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tushare as ts

from build_wp_v32_public_event_data import load_v24_candidate_index
from probe_wp_v33_limit_industry_ecology import (
    TAG_CACHE_KEYS,
    load_v28_membership,
    membership_coverage,
    previous_trade_dates,
)
from wp.v3.history import TushareHistoryClient
from wp.v3.io import atomic_write_json, atomic_write_parquet, file_sha256
from wp.v3.meta_alpha import IDENTITY_COLUMNS
from wp.v3.v30_limit_event import (
    KPL_FIELDS,
    KPL_TAGS,
    SIGNAL_SLOTS,
    normalize_kpl_frame,
)
from wp.v3.v33_limit_ecology import (
    V33_LIMIT_ECOLOGY_FEATURE_COLUMNS,
    audit_decision_tape_frame,
    audit_ecology_feature_coverage,
    build_date_candidate_ecology,
    build_projection,
)


DATA_SCHEMA_VERSION = "wp_v33_limit_industry_ecology_features_1"
PROBE_SCHEMA_VERSION = "wp_v33_limit_industry_ecology_probe_1"
SOURCE_V24_DATA_RUN_ID = 30_635_569_735
SOURCE_V28_DATA_RUN_ID = 30_656_696_310
SOURCE_V33_PROBE_RUN_ID = 30_671_024_383
MIN_L2_ACTIVE_ROW_RATE = 0.20
MIN_L3_ACTIVE_ROW_RATE = 0.10
MIN_L2_ACTIVE_DATE_RATE = 0.70
MIN_L3_ACTIVE_DATE_RATE = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build three-year outcome-blind V33 ecology features."
    )
    parser.add_argument("--v24-data-dir", required=True)
    parser.add_argument("--v28-data-dir", required=True)
    parser.add_argument("--v33-probe-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for V33 data build")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    candidates, candidate_source = load_v24_candidate_index(
        args.v24_data_dir
    )
    membership, membership_source = load_v28_membership(args.v28_data_dir)
    probe_features, probe_source = load_v33_probe(args.v33_probe_dir)
    target_dates = tuple(
        sorted(candidates["trade_date"].astype(str).unique())
    )
    previous_dates = previous_trade_dates(
        candidate_source["open_dates"],
        target_dates,
    )
    required_dates = sorted(
        set(target_dates).union(previous_dates.values())
    )
    expected_queries = len(required_dates) * len(KPL_TAGS)
    client = TushareHistoryClient(
        ts.pro_api(token),
        args.cache_dir,
        page_size=8_000,
        requests_per_minute=180,
        attempts=4,
    )

    query_records: list[dict[str, Any]] = []
    query_failures: list[dict[str, str]] = []
    projections: dict[str, pd.DataFrame] = {}
    stock_events: dict[str, pd.DataFrame] = {}
    date_records: list[dict[str, Any]] = []
    completed_queries = 0
    for trade_date in required_dates:
        normalized_frames: list[pd.DataFrame] = []
        for tag in KPL_TAGS:
            try:
                frame = client.query(
                    "kpl_list",
                    cache_key=(
                        f"{trade_date}_{TAG_CACHE_KEYS[tag]}_v33_full"
                    ),
                    paged=True,
                    trade_date=trade_date,
                    tag=tag,
                    fields=KPL_FIELDS,
                )
                record = audit_decision_tape_frame(
                    frame,
                    trade_date=trade_date,
                    requested_tag=tag,
                )
                query_records.append(record)
                normalized_frames.append(
                    normalize_kpl_frame(
                        frame,
                        trade_date=trade_date,
                        requested_tag=tag,
                    )
                )
            except Exception as error:
                failure = {
                    "trade_date": trade_date,
                    "tag": tag,
                    "error": str(error)[:500],
                }
                query_failures.append(failure)
                query_records.append(
                    {
                        "trade_date": trade_date,
                        "requested_tag": tag,
                        "rows": 0,
                        "schema_ok": False,
                        "coverage_pass": False,
                        "error": failure["error"],
                    }
                )
            completed_queries += 1
            if completed_queries % 50 == 0:
                print(
                    "[wp-v33-data] "
                    f"queries={completed_queries}/{expected_queries} "
                    f"failures={len(query_failures)}",
                    flush=True,
                )
        if len(normalized_frames) != len(KPL_TAGS):
            date_records.append(
                {
                    "trade_date": trade_date,
                    "projection_rows": 0,
                    "stock_event_rows": 0,
                    "coverage_pass": False,
                    "error": "one or more category queries failed",
                }
            )
            continue
        try:
            projection, stocks = build_projection(
                normalized_frames,
                trade_date=trade_date,
            )
            projections[trade_date] = projection
            stock_events[trade_date] = stocks
            date_records.append(
                {
                    "trade_date": trade_date,
                    "projection_rows": int(len(projection)),
                    "stock_event_rows": int(len(stocks)),
                    "coverage_pass": bool(
                        len(projection) == len(SIGNAL_SLOTS)
                        and not projection.duplicated(
                            ["trade_date", "signal_slot"]
                        ).any()
                    ),
                }
            )
        except Exception as error:
            date_records.append(
                {
                    "trade_date": trade_date,
                    "projection_rows": 0,
                    "stock_event_rows": 0,
                    "coverage_pass": False,
                    "error": str(error)[:500],
                }
            )

    current_coverage = membership_coverage(
        {
            trade_date: stock_events.get(trade_date, pd.DataFrame())
            for trade_date in target_dates
        },
        membership,
    )
    previous_coverage = membership_coverage(
        {
            previous_date: stock_events.get(
                previous_date,
                pd.DataFrame(),
            )
            for previous_date in sorted(set(previous_dates.values()))
        },
        membership,
    )

    feature_parts: list[pd.DataFrame] = []
    for completed_dates, trade_date in enumerate(target_dates, start=1):
        previous_date = previous_dates[trade_date]
        if (
            trade_date not in projections
            or trade_date not in stock_events
            or previous_date not in stock_events
        ):
            continue
        feature_parts.append(
            build_date_candidate_ecology(
                candidates,
                trade_date=trade_date,
                previous_trade_date=previous_date,
                current_stock_events=stock_events[trade_date],
                previous_stock_events=stock_events[previous_date],
                market_projection=projections[trade_date],
                membership=membership,
            )
        )
        if completed_dates % 50 == 0:
            print(
                "[wp-v33-data] "
                f"feature_dates={completed_dates}/{len(target_dates)}",
                flush=True,
            )
    features = (
        pd.concat(feature_parts, ignore_index=True)
        if feature_parts
        else pd.DataFrame()
    )

    base_audit = audit_ecology_feature_coverage(
        features,
        candidates,
        current_event_membership_coverage=current_coverage["coverage"],
        previous_event_membership_coverage=previous_coverage["coverage"],
    )
    full_coverage = audit_full_coverage(features)
    probe_parity = audit_probe_feature_parity(
        features,
        probe_features,
    )
    query_contract = bool(
        len(query_records) == expected_queries
        and not query_failures
        and all(record.get("coverage_pass") for record in query_records)
    )
    date_contract = bool(
        len(date_records) == len(required_dates)
        and all(record.get("coverage_pass") for record in date_records)
    )
    source_integrity = bool(
        candidate_source["source_integrity"]
        and membership_source["source_integrity"]
        and probe_source["source_integrity"]
    )
    authorized = bool(
        source_integrity
        and query_contract
        and date_contract
        and base_audit["coverage_passed"]
        and full_coverage["passed"]
        and probe_parity["passed"]
    )

    candidate_path = atomic_write_parquet(
        candidates,
        output / "wp_v33_outcome_blind_candidate_index.parquet",
    )
    feature_path = atomic_write_parquet(
        features,
        output / "wp_v33_limit_industry_ecology_features.parquet",
    )
    projection_frame = pd.concat(
        [projections[date] for date in target_dates if date in projections],
        ignore_index=True,
    )
    projection_path = atomic_write_parquet(
        projection_frame,
        output / "wp_v33_market_projection.parquet",
    )
    audit_path = output / "wp_v33_source_date_audit.json"
    atomic_write_json(
        audit_path,
        {
            "schema_version": "wp_v33_source_date_audit_1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "required_dates": required_dates,
            "query_records": query_records,
            "query_failures": query_failures,
            "date_records": date_records,
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
            "v28_data_run_id": SOURCE_V28_DATA_RUN_ID,
            "v33_probe_run_id": SOURCE_V33_PROBE_RUN_ID,
        },
        "source_contract": {
            "v24": {
                key: value
                for key, value in candidate_source.items()
                if key != "open_dates"
            },
            "v28_membership": membership_source,
            "v33_probe": probe_source,
            "api_name": "kpl_list",
            "categories": list(KPL_TAGS),
            "target_dates": int(len(target_dates)),
            "required_dates": int(len(required_dates)),
            "expected_queries": int(expected_queries),
            "post_1450_events_used_for_current_features": False,
        },
        "feature_contract": {
            "feature_columns": list(
                V33_LIMIT_ECOLOGY_FEATURE_COLUMNS
            ),
            "raw_industry_codes_are_identity_audit_only": True,
            "candidate_profit_fields_used": False,
            "candidate_excluded_from_peer_aggregates": True,
        },
        "coverage_audit": {
            "query_contract_passed": query_contract,
            "date_contract_passed": date_contract,
            "query_failures": int(len(query_failures)),
            "current_event_membership": current_coverage,
            "previous_event_membership": previous_coverage,
            "candidate_features": base_audit,
            "full_three_year_coverage": full_coverage,
            "probe_feature_parity": probe_parity,
        },
        "artifacts": {
            "candidate_index": file_artifact(candidate_path),
            "ecology_features": file_artifact(feature_path),
            "market_projection": file_artifact(projection_path),
            "source_date_audit": file_artifact(audit_path),
        },
        "v33_model_research_authorized": authorized,
        "next_gate": (
            "freeze_v33_nested_oos_model_protocol"
            if authorized
            else "stop_and_diagnose_v33_data_contract"
        ),
    }
    atomic_write_json(
        output / "wp_v33_limit_industry_ecology_data_manifest.json",
        manifest,
    )
    print(
        "WP_V33_FULL_DATA_RESULT="
        + json.dumps(
            {
                "candidate_rows": int(len(candidates)),
                "feature_rows": int(len(features)),
                "trade_dates": int(len(target_dates)),
                "required_dates": int(len(required_dates)),
                "queries": int(len(query_records)),
                "query_failures": int(len(query_failures)),
                "query_contract_passed": query_contract,
                "date_contract_passed": date_contract,
                "current_event_membership": current_coverage,
                "previous_event_membership": previous_coverage,
                "candidate_features": base_audit,
                "full_three_year_coverage": full_coverage,
                "probe_feature_parity": probe_parity,
                "v33_model_research_authorized": authorized,
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
            "V33 full outcome-blind dataset failed its frozen contract"
        )
    return 0


def load_v33_probe(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(data_dir)
    manifests = list(
        root.rglob("wp_v33_limit_industry_ecology_probe.json")
    )
    paths = list(
        root.rglob("wp_v33_probe_candidate_industry_ecology.parquet")
    )
    if len(manifests) != 1 or len(paths) != 1:
        raise RuntimeError("V33 expected one immutable probe source")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PROBE_SCHEMA_VERSION:
        raise RuntimeError("V33 probe manifest schema mismatch")
    if manifest.get("profit_outcomes_read") is not False:
        raise RuntimeError("V33 probe source is not outcome blind")
    if manifest.get("future_information_allowed") is not False:
        raise RuntimeError("V33 probe source allowed future information")
    if manifest.get("full_backfill_authorized") is not True:
        raise RuntimeError("V33 probe did not authorize the full build")
    expected = str(
        (manifest.get("artifacts") or {})
        .get("candidate_industry_ecology", {})
        .get("sha256")
        or ""
    )
    actual = file_sha256(paths[0])
    if not expected or expected != actual:
        raise RuntimeError("V33 probe feature digest mismatch")
    features = pd.read_parquet(paths[0])
    if features.empty:
        raise RuntimeError("V33 probe feature source is empty")
    return features, {
        "source_integrity": True,
        "manifest_sha256": file_sha256(manifests[0]),
        "feature_sha256": actual,
        "rows": int(len(features)),
    }


def audit_full_coverage(features: pd.DataFrame) -> dict[str, Any]:
    if features.empty:
        return {
            "target_dates": 0,
            "l2_active_row_rate": 0.0,
            "l3_active_row_rate": 0.0,
            "l2_active_date_rate": 0.0,
            "l3_active_date_rate": 0.0,
            "passed": False,
        }
    dates = features["trade_date"].astype(str)
    target_dates = int(dates.nunique())
    l2_active = features["v33_l2_limit_hit_count"].gt(0)
    l3_active = features["v33_l3_limit_hit_count"].gt(0)
    l2_dates = int(dates.loc[l2_active].nunique())
    l3_dates = int(dates.loc[l3_active].nunique())
    l2_row_rate = float(l2_active.mean())
    l3_row_rate = float(l3_active.mean())
    l2_date_rate = float(l2_dates / target_dates)
    l3_date_rate = float(l3_dates / target_dates)
    passed = bool(
        l2_row_rate >= MIN_L2_ACTIVE_ROW_RATE
        and l3_row_rate >= MIN_L3_ACTIVE_ROW_RATE
        and l2_date_rate >= MIN_L2_ACTIVE_DATE_RATE
        and l3_date_rate >= MIN_L3_ACTIVE_DATE_RATE
    )
    return {
        "target_dates": target_dates,
        "l2_active_rows": int(l2_active.sum()),
        "l2_active_row_rate": l2_row_rate,
        "minimum_l2_active_row_rate": MIN_L2_ACTIVE_ROW_RATE,
        "l3_active_rows": int(l3_active.sum()),
        "l3_active_row_rate": l3_row_rate,
        "minimum_l3_active_row_rate": MIN_L3_ACTIVE_ROW_RATE,
        "l2_active_dates": l2_dates,
        "l2_active_date_rate": l2_date_rate,
        "minimum_l2_active_date_rate": MIN_L2_ACTIVE_DATE_RATE,
        "l3_active_dates": l3_dates,
        "l3_active_date_rate": l3_date_rate,
        "minimum_l3_active_date_rate": MIN_L3_ACTIVE_DATE_RATE,
        "passed": passed,
    }


def audit_probe_feature_parity(
    full_features: pd.DataFrame,
    probe_features: pd.DataFrame,
) -> dict[str, Any]:
    keys = list(IDENTITY_COLUMNS)
    columns = list(V33_LIMIT_ECOLOGY_FEATURE_COLUMNS)
    probe_dates = set(probe_features["trade_date"].astype(str))
    actual = full_features.loc[
        full_features["trade_date"].astype(str).isin(probe_dates),
        [*keys, *columns],
    ].copy()
    expected = probe_features[[*keys, *columns]].copy()
    for frame in (actual, expected):
        for key in keys:
            frame[key] = frame[key].astype(str)
        frame.sort_values(keys, kind="stable", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    identity_match = bool(
        len(actual) == len(expected)
        and actual[keys].equals(expected[keys])
    )
    mismatch_rows = 0
    max_absolute_difference = 0.0
    if identity_match:
        left = expected[columns].apply(pd.to_numeric, errors="coerce")
        right = actual[columns].apply(pd.to_numeric, errors="coerce")
        differences = np.abs(
            left.to_numpy(dtype=float) - right.to_numpy(dtype=float)
        )
        close = np.isclose(
            left.to_numpy(dtype=float),
            right.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
        mismatch_rows = int((~close.all(axis=1)).sum())
        if differences.size:
            finite = differences[np.isfinite(differences)]
            max_absolute_difference = (
                float(finite.max()) if finite.size else 0.0
            )
    return {
        "expected_rows": int(len(expected)),
        "actual_rows": int(len(actual)),
        "identity_match": identity_match,
        "mismatch_rows": mismatch_rows,
        "max_absolute_difference": max_absolute_difference,
        "passed": bool(identity_match and mismatch_rows == 0),
    }


def file_artifact(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    return {
        "path": file_path.name,
        "bytes": int(file_path.stat().st_size),
        "sha256": file_sha256(file_path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
