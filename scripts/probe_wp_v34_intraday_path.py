from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import tushare as ts

from build_wp_v32_public_event_data import load_v24_candidate_index
from wp.v3.history import MINUTE_FIELDS, TushareHistoryClient
from wp.v3.io import atomic_write_json, atomic_write_parquet, file_sha256
from wp.v3.v34_intraday_path import (
    SCHEMA_VERSION,
    V34_INTRADAY_PATH_FEATURE_COLUMNS,
    audit_intraday_path_coverage,
    build_intraday_path_features,
    normalize_historical_minutes,
)


SOURCE_V24_DATA_RUN_ID = 30_635_569_735
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe outcome-blind V34 full-session intraday paths."
    )
    parser.add_argument("--v24-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for the V34 probe")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates, source = load_v24_candidate_index(args.v24_data_dir)
    sample = candidates.loc[
        candidates["trade_date"].astype(str).isin(PROBE_DATES)
    ].copy()
    if sample.empty or sample["trade_date"].nunique() != len(PROBE_DATES):
        raise RuntimeError("V34 immutable probe dates are incomplete")
    sample = sample[
        ["trade_date", "signal_slot", "ts_code", "fold", "signal_price"]
    ].copy()
    client = TushareHistoryClient(
        ts.pro_api(token),
        output / "cache",
        page_size=8_000,
        requests_per_minute=120,
        attempts=4,
    )
    requirements = sorted(
        {
            (str(row.trade_date), str(row.ts_code))
            for row in sample[["trade_date", "ts_code"]]
            .drop_duplicates()
            .itertuples(index=False)
        }
    )
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    def fetch(requirement: tuple[str, str]) -> pd.DataFrame:
        trade_date, ts_code = requirement
        raw = client.query(
            "stk_mins",
            cache_key=(
                f"{ts_code.replace('.', '_')}_{trade_date}_"
                "0930_1450_1min_v34_probe"
            ),
            paged=True,
            ts_code=ts_code,
            start_date=f"{_iso_date(trade_date)} 09:25:00",
            end_date=f"{_iso_date(trade_date)} 14:50:00",
            freq="1min",
            fields=MINUTE_FIELDS,
        )
        normalized = normalize_historical_minutes(raw)
        return normalized.loc[
            normalized["trade_date"].astype(str).eq(trade_date)
            & normalized["ts_code"].astype(str).eq(ts_code)
        ].copy()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(fetch, requirement): requirement
            for requirement in requirements
        }
        for future in as_completed(futures):
            trade_date, ts_code = futures[future]
            try:
                frame = future.result()
                if frame.empty:
                    raise RuntimeError("empty historical minute response")
                frames.append(frame)
            except Exception as error:
                failures.append(
                    {
                        "trade_date": trade_date,
                        "ts_code": ts_code,
                        "error": str(error)[:500],
                    }
                )
    minutes = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=MINUTE_FIELDS.split(","))
    )
    if not minutes.empty:
        minutes.sort_values(
            ["trade_date", "ts_code", "trade_time"],
            kind="stable",
            inplace=True,
        )
        minutes.reset_index(drop=True, inplace=True)
    features = build_intraday_path_features(sample, minutes)
    audit = audit_intraday_path_coverage(
        features,
        sample,
        query_failures=len(failures),
    )
    authorized = bool(
        source["source_integrity"]
        and audit["coverage_passed"]
        and not failures
    )
    minute_path = atomic_write_parquet(
        minutes,
        output / "wp_v34_probe_full_session_minutes.parquet",
    )
    feature_path = atomic_write_parquet(
        features,
        output / "wp_v34_probe_intraday_path_features.parquet",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "profit_outcomes_read": False,
        "future_information_allowed": False,
        "source_run": {
            "v24_data_run_id": SOURCE_V24_DATA_RUN_ID,
            **{key: value for key, value in source.items() if key != "open_dates"},
        },
        "probe_dates": list(PROBE_DATES),
        "source_contract": {
            "historical_api": "stk_mins",
            "historical_frequency": "1min",
            "historical_start": "09:25",
            "feature_cutoff": "at_or_before_each_signal_slot",
            "live_api": "rt_min_daily",
            "live_frequency": "1MIN",
            "normalized_schema": MINUTE_FIELDS.split(","),
            "post_signal_bars_used": False,
            "candidate_outcomes_read": False,
        },
        "requirements": {
            "candidate_rows": int(len(sample)),
            "trade_dates": int(sample["trade_date"].nunique()),
            "stock_date_queries": int(len(requirements)),
        },
        "query_failures": failures,
        "coverage_audit": audit,
        "feature_columns": list(V34_INTRADAY_PATH_FEATURE_COLUMNS),
        "artifacts": {
            "minutes": _file_artifact(minute_path),
            "features": _file_artifact(feature_path),
        },
        "full_backfill_authorized": authorized,
        "model_research_authorized": False,
        "next_gate": (
            "full_three_year_outcome_blind_v34_data_build"
            if authorized
            else "close_v34_data_direction"
        ),
    }
    atomic_write_json(
        output / "wp_v34_intraday_path_probe.json",
        payload,
    )
    print(
        "WP_V34_PROBE_RESULT="
        + json.dumps(
            {
                "probe_dates": len(PROBE_DATES),
                "candidate_rows": int(len(sample)),
                "stock_date_queries": int(len(requirements)),
                "query_failures": len(failures),
                "coverage_audit": audit,
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
        raise RuntimeError("V34 data probe failed its frozen contract")
    return 0


def _iso_date(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def _file_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
