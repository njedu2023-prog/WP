from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tushare as ts

from wp.v3.history import TushareHistoryClient
from wp.v3.io import (
    atomic_write_json,
    atomic_write_parquet,
    file_sha256,
)
from wp.v3.meta_alpha import IDENTITY_COLUMNS
from wp.v3.v24_data import SCHEMA_VERSION as V24_DATA_SCHEMA_VERSION
from wp.v3.v28_industry_peer import (
    MEMBER_FIELDS,
    PEER_LEVELS,
    SIGNAL_SLOTS,
    V28_PEER_FEATURE_COLUMNS,
    audit_peer_feature_coverage,
    audit_stock_slot_frame,
    build_stock_slot_frame,
    leave_one_out_peer_features,
    normalize_membership,
    normalize_trade_date,
)


DATA_SCHEMA_VERSION = "wp_v28_fine_industry_peer_features_1"
SOURCE_PANEL_RUN_ID = 30_600_193_544
SOURCE_V24_DATA_RUN_ID = 30_635_569_735
MINIMUM_DATE_COVERAGE = 0.98
MINIMUM_SLOT_COVERAGE = 0.98
MINUTE_COLUMNS = (
    "ts_code",
    "trade_date",
    "trade_time",
    "close",
    "amount",
)
FORBIDDEN_SOURCE_TOKENS = (
    "gross_return",
    "net_return",
    "target_",
    "label_",
    "t1_",
    "exit_",
    "truth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the outcome-blind V28 fine-industry peer feature dataset."
        )
    )
    parser.add_argument("--history-root", required=True)
    parser.add_argument("--v24-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--source-panel-run-id",
        type=int,
        default=SOURCE_PANEL_RUN_ID,
    )
    parser.add_argument(
        "--source-v24-data-run-id",
        type=int,
        default=SOURCE_V24_DATA_RUN_ID,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.source_panel_run_id != SOURCE_PANEL_RUN_ID:
        raise RuntimeError("V28 source panel run is frozen")
    if args.source_v24_data_run_id != SOURCE_V24_DATA_RUN_ID:
        raise RuntimeError("V28 V24 candidate run is frozen")
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for V28 data build")

    history_root = Path(args.history_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    client = TushareHistoryClient(
        ts.pro_api(token),
        args.cache_dir,
        page_size=8_000,
        requests_per_minute=120,
        attempts=6,
    )

    candidates, candidate_source = load_v24_candidate_index(
        args.v24_data_dir
    )
    history_manifest, minute_manifest = load_history_contract(history_root)
    membership = load_membership(client)
    membership_path = atomic_write_parquet(
        membership,
        output / "wp_v28_industry_membership_snapshot.parquet",
    )

    feature_frames: list[pd.DataFrame] = []
    date_audits: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    verified_partitions: dict[str, str] = {}
    grouped_candidates = {
        str(trade_date): group.copy()
        for trade_date, group in candidates.groupby("trade_date", sort=True)
    }
    candidate_months = sorted(
        candidates["trade_date"].astype(str).str[:6].unique()
    )
    processed_dates = 0
    for month in candidate_months:
        minute_path = (
            history_root / "minute" / f"wp_v3_minutes_{month}.parquet"
        )
        expected_sha = str(
            (minute_manifest.get("partitions") or {})
            .get(month, {})
            .get("sha256")
            or ""
        )
        actual_sha = file_sha256(minute_path)
        if not expected_sha or actual_sha != expected_sha:
            raise RuntimeError(
                f"immutable V9 minute partition digest mismatch for {month}"
            )
        verified_partitions[month] = actual_sha
        minutes = pd.read_parquet(
            minute_path,
            columns=list(MINUTE_COLUMNS),
        )
        minutes["trade_date"] = minutes["trade_date"].map(
            normalize_trade_date
        )
        minutes_by_date = {
            str(trade_date): group.copy()
            for trade_date, group in minutes.groupby(
                "trade_date",
                sort=False,
            )
        }
        month_dates = sorted(
            date
            for date in grouped_candidates
            if date.startswith(month)
        )
        for trade_date in month_dates:
            date_candidates = grouped_candidates[trade_date]
            try:
                daily = client.query(
                    "daily",
                    cache_key=f"{trade_date}_v28_full_preclose",
                    trade_date=trade_date,
                    fields="ts_code,trade_date,pre_close",
                )
                stock_slots = build_stock_slot_frame(
                    minutes_by_date.get(trade_date, pd.DataFrame()),
                    daily,
                    membership,
                    trade_date=trade_date,
                )
                date_audits.append(
                    audit_stock_slot_frame(
                        stock_slots,
                        trade_date=trade_date,
                    )
                )
                identity = date_candidates.loc[
                    :,
                    list(IDENTITY_COLUMNS),
                ]
                levels = [
                    leave_one_out_peer_features(
                        stock_slots,
                        identity,
                        level=level,
                    )
                    for level in PEER_LEVELS
                ]
                feature = levels[0].merge(
                    levels[1],
                    on=list(IDENTITY_COLUMNS),
                    how="outer",
                    validate="one_to_one",
                )
                feature = date_candidates[
                    [*IDENTITY_COLUMNS, "fold"]
                ].merge(
                    feature,
                    on=list(IDENTITY_COLUMNS),
                    how="left",
                    validate="one_to_one",
                )
                feature_frames.append(feature)
            except Exception as error:
                failures.append(
                    {
                        "trade_date": trade_date,
                        "error": str(error)[:500],
                    }
                )
                date_audits.append(
                    failed_date_audit(
                        trade_date,
                        error=str(error)[:500],
                    )
                )
                feature_frames.append(
                    empty_candidate_features(date_candidates)
                )
            processed_dates += 1
            if processed_dates % 25 == 0:
                print(
                    "[wp-v28-data] "
                    f"dates={processed_dates}/{len(grouped_candidates)} "
                    f"failures={len(failures)}",
                    flush=True,
                )

    features = pd.concat(feature_frames, ignore_index=True)
    features.sort_values(
        ["fold", *IDENTITY_COLUMNS],
        kind="stable",
        inplace=True,
    )
    features.reset_index(drop=True, inplace=True)
    feature_audit = audit_peer_feature_coverage(features, candidates)
    date_coverage = (
        sum(bool(record["coverage_pass"]) for record in date_audits)
        / len(grouped_candidates)
        if grouped_candidates
        else 0.0
    )
    slot_records = [
        slot
        for record in date_audits
        for slot in record.get("slot_records", [])
    ]
    expected_slots = len(grouped_candidates) * len(SIGNAL_SLOTS)
    slot_coverage = (
        sum(bool(record["coverage_pass"]) for record in slot_records)
        / expected_slots
        if expected_slots
        else 0.0
    )
    causal_timestamp_pass = bool(
        len(date_audits) == len(grouped_candidates)
        and all(
            bool(record.get("date_consistent"))
            for record in date_audits
        )
    )
    coverage_audit = {
        "requested_trade_dates": int(len(grouped_candidates)),
        "audited_trade_dates": int(len(date_audits)),
        "passed_trade_dates": int(
            sum(bool(record["coverage_pass"]) for record in date_audits)
        ),
        "date_coverage": float(date_coverage),
        "minimum_date_coverage": MINIMUM_DATE_COVERAGE,
        "requested_signal_slots": int(expected_slots),
        "audited_signal_slots": int(len(slot_records)),
        "passed_signal_slots": int(
            sum(bool(record["coverage_pass"]) for record in slot_records)
        ),
        "slot_coverage": float(slot_coverage),
        "minimum_slot_coverage": MINIMUM_SLOT_COVERAGE,
        "causal_timestamp_pass": causal_timestamp_pass,
        "candidate_features": feature_audit,
        "unresolved_query_failures": int(len(failures)),
    }
    authorized = bool(
        candidate_source["source_integrity"]
        and date_coverage >= MINIMUM_DATE_COVERAGE
        and slot_coverage >= MINIMUM_SLOT_COVERAGE
        and causal_timestamp_pass
        and feature_audit["coverage_passed"]
        and not failures
    )

    candidate_path = atomic_write_parquet(
        candidates,
        output / "wp_v28_outcome_blind_candidate_index.parquet",
    )
    feature_path = atomic_write_parquet(
        features,
        output / "wp_v28_fine_industry_peer_features.parquet",
    )
    audit_path = output / "wp_v28_daily_coverage_audit.json"
    atomic_write_json(
        audit_path,
        {
            "schema_version": "wp_v28_daily_coverage_audit_1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "date_audits": date_audits,
            "failures": failures,
        },
    )
    manifest = {
        "schema_version": DATA_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "profit_outcomes_read": False,
        "future_information_allowed": False,
        "source_runs": {
            "v9_panel_run_id": SOURCE_PANEL_RUN_ID,
            "v24_data_run_id": SOURCE_V24_DATA_RUN_ID,
        },
        "source_contract": {
            "v24_candidate_source": candidate_source,
            "v9_history_manifest_sha256": file_sha256(
                history_root / "wp_v3_dataset_manifest.json"
            ),
            "v9_minute_manifest_sha256": file_sha256(
                history_root / "minute" / "manifest.json"
            ),
            "v9_requested_start": history_manifest.get("requested_start"),
            "v9_requested_end": history_manifest.get("requested_end"),
            "verified_minute_partitions": verified_partitions,
            "membership_snapshot_sha256": file_sha256(membership_path),
        },
        "feature_contract": {
            "levels": list(PEER_LEVELS),
            "signal_slots": list(SIGNAL_SLOTS),
            "feature_columns": list(V28_PEER_FEATURE_COLUMNS),
            "candidate_excluded_from_every_peer_aggregate": True,
            "first_signal_or_outcome_used": False,
            "daily_preclose_fields": [
                "ts_code",
                "trade_date",
                "pre_close",
            ],
            "minute_fields": list(MINUTE_COLUMNS),
        },
        "coverage_audit": coverage_audit,
        "query_failures": failures,
        "artifacts": {
            "candidate_index": file_artifact(candidate_path),
            "peer_features": file_artifact(feature_path),
            "membership_snapshot": file_artifact(membership_path),
            "daily_coverage_audit": file_artifact(audit_path),
        },
        "v28_model_research_authorized": authorized,
        "next_gate": (
            "freeze_v28_nested_oos_model_protocol"
            if authorized
            else "stop_and_diagnose_v28_data_contract"
        ),
    }
    atomic_write_json(
        output / "wp_v28_fine_industry_peer_data_manifest.json",
        manifest,
    )
    print(
        "WP_V28_FULL_DATA_RESULT="
        + json.dumps(
            {
                "candidate_rows": int(len(candidates)),
                "feature_rows": int(len(features)),
                "trade_dates": int(len(grouped_candidates)),
                "coverage_audit": coverage_audit,
                "v28_model_research_authorized": authorized,
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
            "V28 full outcome-blind peer dataset failed its frozen contract"
        )
    return 0


def load_v24_candidate_index(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(data_dir)
    manifests = list(root.rglob("wp_v24_data_manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(
            f"expected one V24 data manifest; found {len(manifests)}"
        )
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != V24_DATA_SCHEMA_VERSION:
        raise RuntimeError("V24 candidate manifest schema mismatch")
    if manifest.get("profit_outcomes_read") is not False:
        raise RuntimeError("V24 source did not preserve outcome blindness")
    source = manifest.get("source") or {}
    if source.get("source_integrity") is not True:
        raise RuntimeError("V24 source integrity is not established")
    paths = list(root.rglob("wp_v24_outcome_blind_candidate_index.parquet"))
    if len(paths) != 1:
        raise RuntimeError(
            f"expected one V24 candidate index; found {len(paths)}"
        )
    expected_sha = str(
        (manifest.get("artifacts") or {})
        .get("candidate_index", {})
        .get("sha256")
        or ""
    )
    actual_sha = file_sha256(paths[0])
    if not expected_sha or actual_sha != expected_sha:
        raise RuntimeError("V24 candidate index digest mismatch")
    available = set(pq.read_schema(paths[0]).names)
    projected = [*IDENTITY_COLUMNS, "fold"]
    contaminated = sorted(
        column
        for column in projected
        if any(token in column.lower() for token in FORBIDDEN_SOURCE_TOKENS)
    )
    if contaminated:
        raise RuntimeError(
            f"V28 source projection contains forbidden outcomes: {contaminated}"
        )
    required = set(projected)
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"V24 candidate index missing columns: {missing}")
    candidates = pq.read_table(
        paths[0],
        columns=projected,
    ).to_pandas()
    candidates["trade_date"] = candidates["trade_date"].map(
        normalize_trade_date
    )
    candidates["signal_slot"] = (
        candidates["signal_slot"].fillna("").astype(str).str.strip()
    )
    candidates["ts_code"] = (
        candidates["ts_code"].fillna("").astype(str).str.strip()
    )
    candidates["fold"] = pd.to_numeric(
        candidates["fold"],
        errors="raise",
    ).astype(int)
    if candidates.duplicated(list(IDENTITY_COLUMNS)).any():
        raise RuntimeError("V24 candidate index contains duplicate identities")
    if not candidates["signal_slot"].isin(SIGNAL_SLOTS).all():
        raise RuntimeError("V24 candidate index contains illegal signal slots")
    expected_rows = int(source.get("candidate_rows", -1))
    if len(candidates) != expected_rows:
        raise RuntimeError(
            f"V24 candidate row mismatch: {len(candidates)} != {expected_rows}"
        )
    candidates.sort_values(
        ["fold", *IDENTITY_COLUMNS],
        kind="stable",
        inplace=True,
    )
    candidates.reset_index(drop=True, inplace=True)
    return candidates, {
        "schema_version": source.get("schema_version"),
        "source_integrity": True,
        "profit_outcomes_read": False,
        "candidate_rows": int(len(candidates)),
        "trade_dates": int(candidates["trade_date"].nunique()),
        "candidate_index_sha256": actual_sha,
        "manifest_sha256": file_sha256(manifests[0]),
    }


def load_history_contract(
    history_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    history_path = history_root / "wp_v3_dataset_manifest.json"
    minute_path = history_root / "minute" / "manifest.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    minute = json.loads(minute_path.read_text(encoding="utf-8"))
    if history.get("schema_version") != "wp_point_in_time_panel_5":
        raise RuntimeError("V28 history source is not the frozen V9 panel")
    if minute.get("schema_version") != "wp_historical_minutes_3":
        raise RuntimeError("V28 minute source is not the frozen V9 substrate")
    return history, minute


def load_membership(client: TushareHistoryClient) -> pd.DataFrame:
    frames = [
        client.query(
            "index_member_all",
            cache_key=f"{state}_v28_full_membership",
            paged=True,
            is_new=state,
            fields=MEMBER_FIELDS,
        )
        for state in ("Y", "N")
    ]
    membership = normalize_membership(pd.concat(frames, ignore_index=True))
    if membership["ts_code"].nunique() < 5_000:
        raise RuntimeError("V28 historical industry membership is incomplete")
    return membership


def empty_candidate_features(candidates: pd.DataFrame) -> pd.DataFrame:
    result = candidates[[*IDENTITY_COLUMNS, "fold"]].copy()
    for column in V28_PEER_FEATURE_COLUMNS:
        result[column] = np.nan
    return result


def failed_date_audit(
    trade_date: str,
    *,
    error: str,
) -> dict[str, Any]:
    return {
        "trade_date": str(trade_date),
        "rows": 0,
        "date_consistent": False,
        "duplicate_identity": False,
        "slot_count": 0,
        "slot_records": [],
        "coverage_pass": False,
        "error": error,
    }


def file_artifact(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    return {
        "path": str(file_path.as_posix()),
        "sha256": file_sha256(file_path),
        "bytes": int(file_path.stat().st_size),
    }


if __name__ == "__main__":
    raise SystemExit(main())
