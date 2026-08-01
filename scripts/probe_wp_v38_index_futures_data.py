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

from wp.v3.history import TushareHistoryClient
from wp.v3.io import atomic_write_json, atomic_write_parquet, file_sha256
from wp.v3.v38_index_futures import (
    ETF_FIELDS,
    FUTURE_FIELDS,
    MAPPING_FIELDS,
    PAIR_SPECS,
    SCHEMA_VERSION,
    V38_FEATURE_COLUMNS,
    audit_probe_contract,
    build_regime_features,
    normalize_historical_etf_minutes,
    normalize_historical_future_minutes,
    normalize_mapping,
)


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
        description="Probe outcome-blind V38 index-futures regime data."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for the V38 probe")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    client = TushareHistoryClient(
        ts.pro_api(token),
        output / "cache",
        page_size=8_000,
        requests_per_minute=120,
        attempts=3,
    )

    mapping_frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for spec in PAIR_SPECS:
        for trade_date in PROBE_DATES:
            try:
                frame = client.query(
                    "fut_mapping",
                    cache_key=(
                        f"{spec['continuous_code'].replace('.', '_')}_"
                        f"{trade_date}_v38_probe"
                    ),
                    ts_code=spec["continuous_code"],
                    trade_date=trade_date,
                    fields=MAPPING_FIELDS,
                )
                normalized = normalize_mapping(
                    frame,
                    required_dates=(trade_date,),
                )
                normalized = normalized.loc[
                    normalized["ts_code"].eq(spec["continuous_code"])
                ].copy()
                if len(normalized) != 1:
                    raise RuntimeError(
                        "expected exactly one continuous-contract mapping"
                    )
                mapping_frames.append(normalized)
            except Exception as error:
                failures.append(
                    {
                        "stage": "fut_mapping",
                        "trade_date": trade_date,
                        "ts_code": spec["continuous_code"],
                        "error": _clean_error(error, token),
                    }
                )
    mappings = (
        pd.concat(mapping_frames, ignore_index=True)
        if mapping_frames
        else pd.DataFrame(columns=MAPPING_FIELDS.split(","))
    )

    requirements: list[dict[str, str]] = []
    mapping_lookup = {
        (str(row.trade_date), str(row.ts_code)): str(row.mapping_ts_code)
        for row in mappings.itertuples(index=False)
    }
    for spec in PAIR_SPECS:
        for trade_date in PROBE_DATES:
            mapped = mapping_lookup.get(
                (trade_date, spec["continuous_code"]),
                "",
            )
            if not mapped:
                continue
            requirements.append(
                {
                    "kind": "etf",
                    "trade_date": trade_date,
                    "ts_code": spec["etf_code"],
                }
            )
            requirements.append(
                {
                    "kind": "future",
                    "trade_date": trade_date,
                    "ts_code": mapped,
                }
            )

    def fetch(requirement: dict[str, str]) -> pd.DataFrame:
        trade_date = requirement["trade_date"]
        ts_code = requirement["ts_code"]
        kind = requirement["kind"]
        api_name = "etf_mins" if kind == "etf" else "ft_mins"
        fields = ETF_FIELDS if kind == "etf" else FUTURE_FIELDS
        raw = client.query(
            api_name,
            cache_key=(
                f"{ts_code.replace('.', '_')}_{trade_date}_"
                f"0930_1450_1min_v38_probe"
            ),
            paged=True,
            ts_code=ts_code,
            start_date=f"{_iso_date(trade_date)} 09:25:00",
            end_date=f"{_iso_date(trade_date)} 14:50:00",
            freq="1min",
            fields=fields,
        )
        normalized = (
            normalize_historical_etf_minutes(raw)
            if kind == "etf"
            else normalize_historical_future_minutes(raw)
        )
        normalized = normalized.loc[
            normalized["trade_date"].eq(trade_date)
            & normalized["ts_code"].eq(ts_code)
        ].copy()
        if normalized.empty:
            raise RuntimeError("empty historical minute response")
        return normalized

    etf_frames: list[pd.DataFrame] = []
    future_frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_jobs = {
            executor.submit(fetch, requirement): requirement
            for requirement in requirements
        }
        for job in as_completed(future_jobs):
            requirement = future_jobs[job]
            try:
                frame = job.result()
                if requirement["kind"] == "etf":
                    etf_frames.append(frame)
                else:
                    future_frames.append(frame)
            except Exception as error:
                failures.append(
                    {
                        "stage": (
                            "etf_mins"
                            if requirement["kind"] == "etf"
                            else "ft_mins"
                        ),
                        "trade_date": requirement["trade_date"],
                        "ts_code": requirement["ts_code"],
                        "error": _clean_error(error, token),
                    }
                )
    etf_minutes = (
        pd.concat(etf_frames, ignore_index=True)
        if etf_frames
        else pd.DataFrame(columns=ETF_FIELDS.split(","))
    )
    future_minutes = (
        pd.concat(future_frames, ignore_index=True)
        if future_frames
        else pd.DataFrame(columns=FUTURE_FIELDS.split(","))
    )
    features = build_regime_features(
        PROBE_DATES,
        mappings,
        etf_minutes,
        future_minutes,
    )
    audit = audit_probe_contract(
        features,
        mappings,
        probe_dates=PROBE_DATES,
        query_failures=len(failures),
    )
    authorized = bool(audit["coverage_passed"])

    mapping_path = atomic_write_parquet(
        mappings,
        output / "wp_v38_probe_futures_mapping.parquet",
    )
    etf_path = atomic_write_parquet(
        etf_minutes,
        output / "wp_v38_probe_etf_minutes.parquet",
    )
    future_path = atomic_write_parquet(
        future_minutes,
        output / "wp_v38_probe_future_minutes.parquet",
    )
    feature_path = atomic_write_parquet(
        features,
        output / "wp_v38_probe_regime_features.parquet",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "profit_outcomes_read": False,
        "future_information_allowed": False,
        "probe_dates": list(PROBE_DATES),
        "pair_specs": list(PAIR_SPECS),
        "source_contract": {
            "mapping_api": "fut_mapping",
            "historical_etf_api": "etf_mins",
            "historical_future_api": "ft_mins",
            "live_etf_api": "rt_etf_min_daily",
            "live_future_api": "rt_fut_min_daily",
            "historical_frequency": "1min",
            "live_frequency": "1MIN",
            "feature_cutoff": "at_or_before_each_signal_slot",
            "post_signal_bars_used": False,
            "candidate_outcomes_read": False,
            "actual_cash_futures_basis_claimed": False,
            "hedge_spread_definition": (
                "futures_return_minus_tracking_etf_return"
            ),
        },
        "requirements": {
            "mapping_queries": len(PROBE_DATES) * len(PAIR_SPECS),
            "minute_queries": len(PROBE_DATES) * len(PAIR_SPECS) * 2,
            "date_slot_rows": len(PROBE_DATES) * 7,
        },
        "query_failures": failures,
        "coverage_audit": audit,
        "feature_columns": list(V38_FEATURE_COLUMNS),
        "artifacts": {
            "mappings": _file_artifact(mapping_path),
            "etf_minutes": _file_artifact(etf_path),
            "future_minutes": _file_artifact(future_path),
            "regime_features": _file_artifact(feature_path),
        },
        "full_backfill_authorized": authorized,
        "model_research_authorized": False,
        "next_gate": (
            "full_three_year_outcome_blind_v38_data_build"
            if authorized
            else "close_v38_data_direction"
        ),
    }
    atomic_write_json(
        output / "wp_v38_index_futures_data_probe.json",
        payload,
    )
    print(
        "WP_V38_PROBE_RESULT="
        + json.dumps(
            {
                "probe_dates": len(PROBE_DATES),
                "mapping_rows": len(mappings),
                "requirements": len(requirements),
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
    if failures:
        print(
            "WP_V38_QUERY_FAILURES="
            + json.dumps(
                _summarize_failures(failures),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
            flush=True,
        )
    if not authorized:
        raise RuntimeError("V38 data probe failed its frozen contract")
    return 0


def _iso_date(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def _file_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _clean_error(error: Exception, token: str) -> str:
    message = str(error).replace(token, "***") if token else str(error)
    return " ".join(message.split())[:500]


def _summarize_failures(
    failures: list[dict[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for failure in failures:
        key = (failure["stage"], failure["error"])
        summary = grouped.setdefault(
            key,
            {
                "stage": failure["stage"],
                "error": failure["error"],
                "count": 0,
                "examples": [],
            },
        )
        summary["count"] += 1
        if len(summary["examples"]) < 2:
            summary["examples"].append(
                {
                    "trade_date": failure["trade_date"],
                    "ts_code": failure["ts_code"],
                }
            )
    return sorted(
        grouped.values(),
        key=lambda item: (-item["count"], item["stage"], item["error"]),
    )[:12]


if __name__ == "__main__":
    raise SystemExit(main())
