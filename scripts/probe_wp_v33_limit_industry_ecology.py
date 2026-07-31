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
from wp.v3.history import TushareHistoryClient
from wp.v3.io import atomic_write_json, atomic_write_parquet, file_sha256
from wp.v3.v28_industry_peer import normalize_membership
from wp.v3.v30_limit_event import (
    KPL_FIELDS,
    KPL_TAGS,
    SIGNAL_SLOTS,
    normalize_kpl_frame,
)
from wp.v3.v33_limit_ecology import (
    SCHEMA_VERSION,
    audit_decision_tape_frame,
    audit_ecology_feature_coverage,
    build_date_candidate_ecology,
    build_projection,
)


SOURCE_V24_DATA_RUN_ID = 30_635_569_735
SOURCE_V28_DATA_RUN_ID = 30_656_696_310
SOURCE_V30_PROBE_RUN_ID = 30_662_958_173
V28_DATA_SCHEMA_VERSION = "wp_v28_fine_industry_peer_features_1"
V30_PROBE_SCHEMA_VERSION = "wp_v30_limit_event_data_probe_1"
PROBE_DATES = (
    "20230825",
    "20231229",
    "20240315",
    "20240927",
    "20250115",
    "20250723",
    "20260115",
    "20260723",
)
TAG_CACHE_KEYS = {"涨停": "up", "炸板": "failed", "跌停": "down"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe outcome-blind V33 limit-industry ecology data."
    )
    parser.add_argument("--v24-data-dir", required=True)
    parser.add_argument("--v28-data-dir", required=True)
    parser.add_argument("--v30-probe-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for the V33 probe")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    candidates, candidate_source = load_v24_candidate_index(
        args.v24_data_dir
    )
    membership, membership_source = load_v28_membership(args.v28_data_dir)
    v30_projection, v30_source = load_v30_projection(args.v30_probe_dir)
    previous_dates = previous_trade_dates(
        candidate_source["open_dates"],
        PROBE_DATES,
    )
    sample_candidates = candidates.loc[
        candidates["trade_date"].astype(str).isin(PROBE_DATES)
    ].copy()
    if sample_candidates.empty:
        raise RuntimeError("V33 probe dates have no immutable candidates")

    required_dates = sorted(
        set(PROBE_DATES).union(previous_dates.values())
    )
    client = TushareHistoryClient(
        ts.pro_api(token),
        output / "cache",
        page_size=8_000,
        requests_per_minute=120,
        attempts=3,
    )
    query_records: list[dict[str, Any]] = []
    normalized_by_date: dict[str, list[pd.DataFrame]] = {}
    for trade_date in required_dates:
        frames: list[pd.DataFrame] = []
        for tag in KPL_TAGS:
            frame = client.query(
                "kpl_list",
                cache_key=(
                    f"{trade_date}_{TAG_CACHE_KEYS[tag]}_v33_probe"
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
            normalized = normalize_kpl_frame(
                frame,
                trade_date=trade_date,
                requested_tag=tag,
            )
            frames.append(normalized)
        normalized_by_date[trade_date] = frames

    projections: dict[str, pd.DataFrame] = {}
    stock_events: dict[str, pd.DataFrame] = {}
    date_records: list[dict[str, Any]] = []
    for trade_date in required_dates:
        try:
            projection, stocks = build_projection(
                normalized_by_date[trade_date],
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
        {date: stock_events.get(date, pd.DataFrame()) for date in PROBE_DATES},
        membership,
    )
    previous_coverage = membership_coverage(
        {
            previous_dates[date]: stock_events.get(
                previous_dates[date],
                pd.DataFrame(),
            )
            for date in PROBE_DATES
        },
        membership,
    )
    feature_parts: list[pd.DataFrame] = []
    for trade_date in PROBE_DATES:
        previous = previous_dates[trade_date]
        if trade_date not in projections or previous not in stock_events:
            continue
        feature_parts.append(
            build_date_candidate_ecology(
                sample_candidates,
                trade_date=trade_date,
                previous_trade_date=previous,
                current_stock_events=stock_events[trade_date],
                previous_stock_events=stock_events[previous],
                market_projection=projections[trade_date],
                membership=membership,
            )
        )
    features = (
        pd.concat(feature_parts, ignore_index=True)
        if feature_parts
        else pd.DataFrame()
    )
    feature_audit = audit_ecology_feature_coverage(
        features,
        sample_candidates,
        current_event_membership_coverage=current_coverage["coverage"],
        previous_event_membership_coverage=previous_coverage["coverage"],
    )
    parity = audit_v30_projection_parity(
        projections,
        v30_projection,
    )
    expected_queries = len(required_dates) * len(KPL_TAGS)
    query_contract = bool(
        len(query_records) == expected_queries
        and all(record.get("coverage_pass") for record in query_records)
    )
    date_contract = bool(
        len(date_records) == len(required_dates)
        and all(record.get("coverage_pass") for record in date_records)
    )
    authorized = bool(
        candidate_source["source_integrity"]
        and membership_source["source_integrity"]
        and v30_source["source_integrity"]
        and query_contract
        and date_contract
        and parity["passed"]
        and feature_audit["coverage_passed"]
    )

    feature_path = atomic_write_parquet(
        features,
        output / "wp_v33_probe_candidate_industry_ecology.parquet",
    )
    projection_frame = pd.concat(
        [projections[date] for date in PROBE_DATES if date in projections],
        ignore_index=True,
    )
    projection_path = atomic_write_parquet(
        projection_frame,
        output / "wp_v33_probe_market_projection.parquet",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "profit_outcomes_read": False,
        "future_information_allowed": False,
        "source_runs": {
            "v24_data_run_id": SOURCE_V24_DATA_RUN_ID,
            "v28_data_run_id": SOURCE_V28_DATA_RUN_ID,
            "v30_probe_run_id": SOURCE_V30_PROBE_RUN_ID,
        },
        "source_contract": {
            "v24": {
                key: value
                for key, value in candidate_source.items()
                if key != "open_dates"
            },
            "v28_membership": membership_source,
            "v30_projection": v30_source,
            "api_name": "kpl_list",
            "categories": list(KPL_TAGS),
            "requested_dates": required_dates,
            "expected_queries": expected_queries,
            "post_1450_events_used_for_current_features": False,
        },
        "probe_dates": list(PROBE_DATES),
        "previous_trade_dates": previous_dates,
        "query_records": query_records,
        "date_records": date_records,
        "coverage_audit": {
            "query_contract_passed": query_contract,
            "date_contract_passed": date_contract,
            "current_event_membership": current_coverage,
            "previous_event_membership": previous_coverage,
            "v30_projection_parity": parity,
            "candidate_features": feature_audit,
        },
        "artifacts": {
            "candidate_industry_ecology": file_artifact(feature_path),
            "market_projection": file_artifact(projection_path),
        },
        "full_backfill_authorized": authorized,
        "model_research_authorized": False,
        "next_gate": (
            "full_three_year_outcome_blind_v33_data_build"
            if authorized
            else "close_v33_data_direction"
        ),
    }
    atomic_write_json(
        output / "wp_v33_limit_industry_ecology_probe.json",
        payload,
    )
    print(
        "WP_V33_PROBE_RESULT="
        + json.dumps(
            {
                "probe_dates": len(PROBE_DATES),
                "required_dates": len(required_dates),
                "queries": len(query_records),
                "query_contract_passed": query_contract,
                "date_contract_passed": date_contract,
                "candidate_rows": int(len(sample_candidates)),
                "feature_rows": int(len(features)),
                "current_event_membership": current_coverage,
                "previous_event_membership": previous_coverage,
                "v30_projection_parity": parity,
                "feature_audit": feature_audit,
                "full_backfill_authorized": authorized,
                "next_gate": payload["next_gate"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    if not authorized:
        raise RuntimeError("V33 data probe failed its frozen contract")
    return 0


def load_v28_membership(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(data_dir)
    manifests = list(
        root.rglob("wp_v28_fine_industry_peer_data_manifest.json")
    )
    paths = list(root.rglob("wp_v28_industry_membership_snapshot.parquet"))
    if len(manifests) != 1 or len(paths) != 1:
        raise RuntimeError("V33 expected one immutable V28 data source")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != V28_DATA_SCHEMA_VERSION:
        raise RuntimeError("V33 V28 membership schema mismatch")
    if manifest.get("profit_outcomes_read") is not False:
        raise RuntimeError("V33 V28 membership source is not outcome blind")
    expected = str(
        (manifest.get("artifacts") or {})
        .get("membership_snapshot", {})
        .get("sha256")
        or ""
    )
    actual = file_sha256(paths[0])
    if not expected or actual != expected:
        raise RuntimeError("V33 V28 membership digest mismatch")
    membership = normalize_membership(pd.read_parquet(paths[0]))
    if membership["ts_code"].nunique() < 5_000:
        raise RuntimeError("V33 V28 membership snapshot is incomplete")
    return membership, {
        "source_integrity": True,
        "manifest_sha256": file_sha256(manifests[0]),
        "membership_sha256": actual,
        "rows": int(len(membership)),
        "unique_codes": int(membership["ts_code"].nunique()),
    }


def load_v30_projection(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(data_dir)
    manifests = list(root.rglob("wp_v30_limit_event_probe.json"))
    paths = list(root.rglob("wp_v30_probe_market_projection.csv"))
    if len(manifests) != 1 or len(paths) != 1:
        raise RuntimeError("V33 expected one immutable V30 probe source")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != V30_PROBE_SCHEMA_VERSION:
        raise RuntimeError("V33 V30 projection schema mismatch")
    if manifest.get("profit_outcomes_read") is not False:
        raise RuntimeError("V33 V30 projection source is not outcome blind")
    expected = str(
        (manifest.get("artifacts") or {})
        .get("market_projection", {})
        .get("sha256")
        or ""
    )
    actual = file_sha256(paths[0])
    if not expected or actual != expected:
        raise RuntimeError("V33 V30 projection digest mismatch")
    frame = pd.read_csv(paths[0], dtype={"trade_date": str})
    return frame, {
        "source_integrity": True,
        "manifest_sha256": file_sha256(manifests[0]),
        "projection_sha256": actual,
        "rows": int(len(frame)),
    }


def previous_trade_dates(
    open_dates: list[str],
    target_dates: tuple[str, ...],
) -> dict[str, str]:
    ordered = sorted(str(value) for value in open_dates)
    mapping: dict[str, str] = {}
    for date in target_dates:
        earlier = [value for value in ordered if value < date]
        if not earlier:
            raise RuntimeError(f"V33 has no previous trade date for {date}")
        mapping[date] = earlier[-1]
    return mapping


def membership_coverage(
    events_by_date: dict[str, pd.DataFrame],
    membership: pd.DataFrame,
) -> dict[str, Any]:
    rows = 0
    covered = 0
    records: list[dict[str, Any]] = []
    from wp.v3.v28_industry_peer import active_membership

    for trade_date, events in sorted(events_by_date.items()):
        if events.empty:
            records.append(
                {
                    "trade_date": trade_date,
                    "event_codes": 0,
                    "covered_codes": 0,
                    "coverage": 1.0,
                }
            )
            continue
        codes = events.loc[
            events[
                [
                    "first_limit_touch",
                    "first_limit_open",
                    "first_limit_down",
                ]
            ]
            .notna()
            .any(axis=1),
            ["ts_code"],
        ].drop_duplicates()
        members = active_membership(
            membership,
            trade_date=trade_date,
        )[["ts_code", "l2_code", "l3_code"]]
        joined = codes.merge(
            members,
            on="ts_code",
            how="left",
            validate="one_to_one",
        )
        available = (
            joined["l2_code"].fillna("").astype(str).ne("")
            & joined["l3_code"].fillna("").astype(str).ne("")
        )
        date_rows = int(len(joined))
        date_covered = int(available.sum())
        rows += date_rows
        covered += date_covered
        records.append(
            {
                "trade_date": trade_date,
                "event_codes": date_rows,
                "covered_codes": date_covered,
                "coverage": (
                    float(date_covered / date_rows) if date_rows else 1.0
                ),
            }
        )
    return {
        "event_codes": rows,
        "covered_codes": covered,
        "coverage": float(covered / rows) if rows else 1.0,
        "date_records": records,
    }


def audit_v30_projection_parity(
    projections: dict[str, pd.DataFrame],
    expected: pd.DataFrame,
) -> dict[str, Any]:
    actual = pd.concat(
        [projections[date] for date in PROBE_DATES if date in projections],
        ignore_index=True,
    )
    keys = ["trade_date", "signal_slot"]
    columns = [
        column
        for column in expected.columns
        if column not in keys
    ]
    left = expected[[*keys, *columns]].copy()
    right = actual[[*keys, *columns]].copy()
    left["trade_date"] = left["trade_date"].astype(str)
    right["trade_date"] = right["trade_date"].astype(str)
    left.sort_values(keys, inplace=True)
    right.sort_values(keys, inplace=True)
    left.reset_index(drop=True, inplace=True)
    right.reset_index(drop=True, inplace=True)
    identity_match = bool(
        len(left) == len(right)
        and left[keys].equals(right[keys])
    )
    mismatch_rows = 0
    if identity_match:
        left_values = left[columns].apply(pd.to_numeric, errors="coerce")
        right_values = right[columns].apply(pd.to_numeric, errors="coerce")
        close = np.isclose(
            left_values.to_numpy(dtype=float),
            right_values.to_numpy(dtype=float),
            equal_nan=True,
        )
        mismatch_rows = int((~close.all(axis=1)).sum())
    return {
        "expected_rows": int(len(left)),
        "actual_rows": int(len(right)),
        "identity_match": identity_match,
        "mismatch_rows": mismatch_rows,
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
