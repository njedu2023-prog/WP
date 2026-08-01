from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tushare as ts

from build_wp_v23_point_in_time_data import (
    month_end,
    month_start,
    read_partitions,
)
from build_wp_v32_public_event_data import load_v24_candidate_index
from wp.v3.history import MINUTE_FIELDS, TushareHistoryClient
from wp.v3.io import atomic_write_json, atomic_write_parquet, file_sha256
from wp.v3.meta_alpha import IDENTITY_COLUMNS
from wp.v3.v23_data import required_stock_months
from wp.v3.v34_intraday_path import (
    SCHEMA_VERSION as PROBE_SCHEMA_VERSION,
    V34_INTRADAY_PATH_FEATURE_COLUMNS,
    V34_QUALITY_COLUMNS,
    audit_intraday_path_coverage,
    build_intraday_path_features,
    normalize_historical_minutes,
)


DATA_SCHEMA_VERSION = "wp_v34_full_session_path_features_1"
SOURCE_V24_DATA_RUN_ID = 30_635_569_735
SOURCE_V34_PROBE_RUN_ID = 30_676_761_165
T = TypeVar("T")
R = TypeVar("R")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build three-year outcome-blind V34 intraday paths."
    )
    parser.add_argument("--v24-data-dir", required=True)
    parser.add_argument("--v34-probe-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for V34 data build")
    output = Path(args.output_dir)
    raw_root = output / "raw" / "one_minute"
    output.mkdir(parents=True, exist_ok=True)

    candidates, candidate_source = load_v24_candidate_index(
        args.v24_data_dir
    )
    probe_features, probe_source = load_v34_probe(args.v34_probe_dir)
    client = TushareHistoryClient(
        ts.pro_api(token),
        args.cache_dir,
        page_size=8_000,
        requests_per_minute=180,
        attempts=6,
    )
    minute_paths, query_failures = fetch_full_session_minute_partitions(
        client,
        candidates,
        output_dir=raw_root,
        workers=args.workers,
    )
    minutes = read_partitions(minute_paths)
    if not minutes.empty:
        minutes.sort_values(
            ["trade_date", "ts_code", "trade_time"],
            kind="stable",
            inplace=True,
        )
        minutes.reset_index(drop=True, inplace=True)
    features = build_intraday_path_features(candidates, minutes)
    coverage = audit_intraday_path_coverage(
        features,
        candidates,
        query_failures=len(query_failures),
    )
    probe_parity = audit_probe_feature_parity(
        features,
        probe_features,
    )
    source_integrity = bool(
        candidate_source["source_integrity"]
        and probe_source["source_integrity"]
    )
    authorized = bool(
        source_integrity
        and coverage["coverage_passed"]
        and probe_parity["passed"]
        and not query_failures
    )

    candidate_path = atomic_write_parquet(
        candidates,
        output / "wp_v34_outcome_blind_candidate_index.parquet",
    )
    feature_path = atomic_write_parquet(
        features,
        output / "wp_v34_intraday_path_features.parquet",
    )
    failure_path = output / "wp_v34_minute_query_failures.json"
    atomic_write_json(
        failure_path,
        {
            "schema_version": "wp_v34_minute_query_failures_1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "failures": query_failures,
        },
    )
    requirements = required_stock_months(candidates)
    manifest = {
        "schema_version": DATA_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "profit_outcomes_read": False,
        "future_information_allowed": False,
        "source_runs": {
            "v24_data_run_id": SOURCE_V24_DATA_RUN_ID,
            "v34_probe_run_id": SOURCE_V34_PROBE_RUN_ID,
        },
        "source_contract": {
            "v24_candidate_source": {
                key: value
                for key, value in candidate_source.items()
                if key != "open_dates"
            },
            "v34_probe_source": probe_source,
            "historical_api": "stk_mins",
            "historical_frequency": "1min",
            "historical_start": "09:25",
            "historical_end": "14:50",
            "feature_cutoff": "at_or_before_each_signal_slot",
            "post_signal_bars_used": False,
            "candidate_outcomes_read": False,
        },
        "requirements": {
            "candidate_rows": int(len(candidates)),
            "trade_dates": int(candidates["trade_date"].nunique()),
            "stock_month_queries": int(
                sum(len(codes) for codes in requirements.values())
            ),
            "month_partitions": int(len(requirements)),
        },
        "feature_contract": {
            "feature_columns": list(V34_INTRADAY_PATH_FEATURE_COLUMNS),
            "quality_columns": list(V34_QUALITY_COLUMNS),
            "signal_price_use": "parity_audit_only",
            "outcome_driven_row_dropping": False,
        },
        "query_failures": query_failures,
        "coverage_audit": coverage,
        "probe_feature_parity": probe_parity,
        "artifacts": {
            "candidate_index": file_artifact(candidate_path),
            "features": file_artifact(feature_path),
            "query_failures": json_artifact(failure_path),
            "one_minute_partitions": [
                file_artifact(path) for path in minute_paths
            ],
        },
        "v34_model_research_authorized": authorized,
        "next_gate": (
            "freeze_v34_nested_oos_model_protocol"
            if authorized
            else "stop_and_diagnose_v34_data_contract"
        ),
    }
    manifest_path = output / "wp_v34_intraday_path_data_manifest.json"
    atomic_write_json(manifest_path, manifest)
    print(
        "WP_V34_FULL_DATA_RESULT="
        + json.dumps(
            {
                "candidate_rows": int(len(candidates)),
                "feature_rows": int(len(features)),
                "minute_rows": int(len(minutes)),
                "requirements": manifest["requirements"],
                "query_failures": int(len(query_failures)),
                "coverage_audit": coverage,
                "probe_feature_parity": probe_parity,
                "v34_model_research_authorized": authorized,
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
            "V34 full outcome-blind dataset failed its frozen contract"
        )
    return 0


def fetch_full_session_minute_partitions(
    client: TushareHistoryClient,
    candidates: pd.DataFrame,
    *,
    output_dir: Path,
    workers: int,
) -> tuple[list[Path], list[dict[str, str]]]:
    requirements = required_stock_months(candidates)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    failures: list[dict[str, str]] = []
    completed = 0
    total = sum(len(codes) for codes in requirements.values())
    for month, codes in requirements.items():
        tasks = [
            (month, ts_code, dates)
            for ts_code, dates in sorted(codes.items())
        ]

        def fetch(
            task: tuple[str, str, tuple[str, ...]],
        ) -> tuple[pd.DataFrame, dict[str, str] | None]:
            task_month, ts_code, dates = task
            try:
                raw = client.query(
                    "stk_mins",
                    cache_key=(
                        f"{ts_code.replace('.', '_')}_{task_month}_"
                        "0925_1450_1min_v34_full"
                    ),
                    paged=True,
                    ts_code=ts_code,
                    start_date=f"{month_start(task_month)} 09:25:00",
                    end_date=f"{month_end(task_month)} 14:50:00",
                    freq="1min",
                    fields=MINUTE_FIELDS,
                )
                normalized = normalize_historical_minutes(raw)
                selected = normalized.loc[
                    normalized["trade_date"].astype(str).isin(dates)
                ].copy()
                returned = set(selected["trade_date"].astype(str).unique())
                missing_dates = sorted(set(dates) - returned)
                if missing_dates:
                    raise RuntimeError(
                        "missing candidate dates: " + ",".join(missing_dates)
                    )
                return selected, None
            except Exception as error:
                return pd.DataFrame(), {
                    "month": task_month,
                    "ts_code": ts_code,
                    "error": str(error)[:500],
                }

        rows = parallel_map(fetch, tasks, workers=workers)
        month_frames: list[pd.DataFrame] = []
        for frame, failure in rows:
            completed += 1
            if failure:
                failures.append(failure)
            elif not frame.empty:
                month_frames.append(frame)
            if completed % 100 == 0:
                print(
                    f"[wp-v34-data] stock-months={completed}/{total} "
                    f"failures={len(failures)}",
                    flush=True,
                )
        month_frame = (
            pd.concat(month_frames, ignore_index=True)
            if month_frames
            else pd.DataFrame(columns=[*MINUTE_FIELDS.split(","), "trade_date"])
        )
        if not month_frame.empty:
            month_frame.sort_values(
                ["trade_date", "ts_code", "trade_time"],
                kind="stable",
                inplace=True,
            )
            month_frame.drop_duplicates(
                ["ts_code", "trade_time"],
                keep="last",
                inplace=True,
            )
            month_frame.reset_index(drop=True, inplace=True)
        path = atomic_write_parquet(
            month_frame,
            output_dir / f"wp_v34_full_session_minutes_{month}.parquet",
        )
        paths.append(path)
        print(
            f"[wp-v34-data] month={month} rows={len(month_frame):,}",
            flush=True,
        )
    return paths, failures


def load_v34_probe(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(data_dir)
    manifests = list(root.rglob("wp_v34_intraday_path_probe.json"))
    paths = list(root.rglob("wp_v34_probe_intraday_path_features.parquet"))
    if len(manifests) != 1 or len(paths) != 1:
        raise RuntimeError("V34 expected one immutable probe source")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PROBE_SCHEMA_VERSION:
        raise RuntimeError("V34 probe manifest schema mismatch")
    if manifest.get("profit_outcomes_read") is not False:
        raise RuntimeError("V34 probe source is not outcome blind")
    if manifest.get("full_backfill_authorized") is not True:
        raise RuntimeError("V34 probe did not authorize the full build")
    expected_sha = str(
        (manifest.get("artifacts") or {})
        .get("features", {})
        .get("sha256")
        or ""
    )
    actual_sha = file_sha256(paths[0])
    if not expected_sha or actual_sha != expected_sha:
        raise RuntimeError("V34 probe feature digest mismatch")
    projected = [
        *IDENTITY_COLUMNS,
        "fold",
        "signal_price",
        *V34_QUALITY_COLUMNS,
        *V34_INTRADAY_PATH_FEATURE_COLUMNS,
    ]
    available = set(pq.read_schema(paths[0]).names)
    missing = sorted(set(projected) - available)
    if missing:
        raise RuntimeError(f"V34 probe feature columns missing: {missing}")
    frame = pq.read_table(paths[0], columns=projected).to_pandas()
    for column in IDENTITY_COLUMNS:
        frame[column] = frame[column].astype(str)
    frame.sort_values(list(IDENTITY_COLUMNS), kind="stable", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame, {
        "source_integrity": True,
        "manifest_sha256": file_sha256(manifests[0]),
        "feature_sha256": actual_sha,
        "probe_dates": [str(value) for value in manifest["probe_dates"]],
        "candidate_rows": int(len(frame)),
    }


def audit_probe_feature_parity(
    full_features: pd.DataFrame,
    probe_features: pd.DataFrame,
) -> dict[str, Any]:
    identity = list(IDENTITY_COLUMNS)
    probe_dates = set(probe_features["trade_date"].astype(str))
    full = full_features.loc[
        full_features["trade_date"].astype(str).isin(probe_dates)
    ].copy()
    columns = [
        *identity,
        "fold",
        "signal_price",
        *V34_QUALITY_COLUMNS,
        *V34_INTRADAY_PATH_FEATURE_COLUMNS,
    ]
    full = full.reindex(columns=columns).sort_values(
        identity,
        kind="stable",
    ).reset_index(drop=True)
    probe = probe_features.reindex(columns=columns).sort_values(
        identity,
        kind="stable",
    ).reset_index(drop=True)
    identities_match = bool(
        len(full) == len(probe)
        and full[identity].astype(str).equals(probe[identity].astype(str))
    )
    numeric_columns = [
        "fold",
        "signal_price",
        "v34_observed_rows",
        "v34_expected_rows",
        "v34_coverage_ratio",
        "v34_signal_price_error_bps",
        *V34_INTRADAY_PATH_FEATURE_COLUMNS,
    ]
    numeric_match = bool(
        identities_match
        and np.allclose(
            full[numeric_columns].apply(
                pd.to_numeric,
                errors="coerce",
            ).to_numpy(dtype=float),
            probe[numeric_columns].apply(
                pd.to_numeric,
                errors="coerce",
            ).to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
    )
    exact_columns = [
        "v34_latest_time",
        "v34_causal_ok",
        "v34_signal_price_parity_ok",
        "v34_path_complete",
    ]
    exact_match = bool(
        identities_match
        and full[exact_columns].fillna("<NA>").astype(str).equals(
            probe[exact_columns].fillna("<NA>").astype(str)
        )
    )
    return {
        "probe_rows": int(len(probe)),
        "full_probe_rows": int(len(full)),
        "probe_dates": int(len(probe_dates)),
        "identities_match": identities_match,
        "numeric_features_match": numeric_match,
        "quality_fields_match": exact_match,
        "passed": bool(identities_match and numeric_match and exact_match),
    }


def parallel_map(
    function: Callable[[T], R],
    values: list[T],
    *,
    workers: int,
) -> list[R]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        return list(executor.map(function, values))


def file_artifact(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    return {
        "path": str(resolved.as_posix()),
        "sha256": file_sha256(resolved),
        "bytes": int(resolved.stat().st_size),
        "rows": int(pq.ParquetFile(resolved).metadata.num_rows),
    }


def json_artifact(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    return {
        "path": str(resolved.as_posix()),
        "sha256": file_sha256(resolved),
        "bytes": int(resolved.stat().st_size),
    }


if __name__ == "__main__":
    raise SystemExit(main())
