from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import tushare as ts

from wp.v3.history import TushareHistoryClient
from wp.v3.io import atomic_write_json, atomic_write_parquet, file_sha256
from wp.v3.v39_derivatives_daily import (
    FUTURE_SPECS,
    OPTION_SPECS,
    SCHEMA_VERSION,
    V39_FUTURE_FEATURE_COLUMNS,
    V39_OPTION_FEATURE_COLUMNS,
    audit_probe_contract,
    build_derivative_features,
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
MAPPING_FIELDS = "ts_code,trade_date,mapping_ts_code"
FUTURE_FIELDS = (
    "ts_code,trade_date,pre_close,pre_settle,open,high,low,close,"
    "settle,vol,amount,oi,oi_chg"
)
INDEX_FIELDS = (
    "ts_code,trade_date,pre_close,open,high,low,close,pct_chg,"
    "vol,amount"
)
OPTION_BASIC_FIELDS = (
    "ts_code,symbol,exchange,name,opt_code,opt_type,call_put,"
    "exercise_price,maturity_date,list_date,delist_date"
)
OPTION_DAILY_FIELDS = (
    "ts_code,trade_date,exchange,pre_settle,pre_close,open,high,"
    "low,close,settle,vol,amount,oi"
)
FUND_FIELDS = INDEX_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe outcome-blind V39 T-1 derivatives daily data."
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for the V39 probe")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    client = TushareHistoryClient(
        ts.pro_api(token),
        output / "cache",
        page_size=15_000,
        requests_per_minute=100,
        attempts=3,
    )
    failures: list[dict[str, str]] = []

    previous_dates = _load_previous_dates(client, failures, token)
    source_dates = sorted(set(previous_dates.values()))

    mapping_frames: list[pd.DataFrame] = []
    futures_frames: list[pd.DataFrame] = []
    index_frames: list[pd.DataFrame] = []
    option_basic_frames: list[pd.DataFrame] = []
    option_frames: list[pd.DataFrame] = []
    fund_frames: list[pd.DataFrame] = []

    for target_date, source_date in previous_dates.items():
        for spec in FUTURE_SPECS:
            mapping_frames.extend(
                _query_one(
                    client,
                    failures,
                    token,
                    family="futures",
                    stage="fut_mapping",
                    target_date=target_date,
                    source_date=source_date,
                    evidence_code=spec["continuous_code"],
                    api_name="fut_mapping",
                    cache_key=(
                        f"{spec['continuous_code'].replace('.', '_')}_"
                        f"{source_date}_v39_probe"
                    ),
                    fields=MAPPING_FIELDS,
                    ts_code=spec["continuous_code"],
                    trade_date=source_date,
                )
            )
            index_frames.extend(
                _query_one(
                    client,
                    failures,
                    token,
                    family="futures",
                    stage="index_daily",
                    target_date=target_date,
                    source_date=source_date,
                    evidence_code=spec["index_code"],
                    api_name="index_daily",
                    cache_key=(
                        f"{spec['index_code'].replace('.', '_')}_"
                        f"{source_date}_v39_probe"
                    ),
                    fields=INDEX_FIELDS,
                    ts_code=spec["index_code"],
                    trade_date=source_date,
                )
            )
        for spec in OPTION_SPECS:
            fund_frames.extend(
                _query_one(
                    client,
                    failures,
                    token,
                    family="options",
                    stage="fund_daily",
                    target_date=target_date,
                    source_date=source_date,
                    evidence_code=spec["underlying_code"],
                    api_name="fund_daily",
                    cache_key=(
                        f"{spec['underlying_code'].replace('.', '_')}_"
                        f"{source_date}_v39_probe"
                    ),
                    fields=FUND_FIELDS,
                    ts_code=spec["underlying_code"],
                    trade_date=source_date,
                )
            )

    for source_date in source_dates:
        futures_frames.extend(
            _query_one(
                client,
                failures,
                token,
                family="futures",
                stage="fut_daily",
                target_date=_target_for_source(
                    source_date,
                    previous_dates,
                ),
                source_date=source_date,
                evidence_code="CFFEX",
                api_name="fut_daily",
                cache_key=f"cffex_{source_date}_v39_probe",
                fields=FUTURE_FIELDS,
                trade_date=source_date,
                exchange="CFFEX",
            )
        )
        option_frames.extend(
            _query_one(
                client,
                failures,
                token,
                family="options",
                stage="opt_daily",
                target_date=_target_for_source(
                    source_date,
                    previous_dates,
                ),
                source_date=source_date,
                evidence_code="SSE",
                api_name="opt_daily",
                cache_key=f"sse_{source_date}_v39_probe",
                fields=OPTION_DAILY_FIELDS,
                trade_date=source_date,
                exchange="SSE",
            )
        )

    option_basic_frames.extend(
        _query_one(
            client,
            failures,
            token,
            family="options",
            stage="opt_basic",
            target_date="all",
            source_date="all",
            evidence_code="SSE",
            api_name="opt_basic",
            cache_key="sse_all_v39_probe",
            fields=OPTION_BASIC_FIELDS,
            paged=True,
            exchange="SSE",
        )
    )

    mappings = _concat(mapping_frames, MAPPING_FIELDS)
    futures_daily = _concat(futures_frames, FUTURE_FIELDS)
    index_daily = _concat(index_frames, INDEX_FIELDS)
    option_basic = _concat(option_basic_frames, OPTION_BASIC_FIELDS)
    options_daily = _concat(option_frames, OPTION_DAILY_FIELDS)
    fund_daily = _concat(fund_frames, FUND_FIELDS)
    features = build_derivative_features(
        PROBE_DATES,
        previous_dates,
        mappings,
        futures_daily,
        index_daily,
        option_basic,
        options_daily,
        fund_daily,
    )
    family_failures = {
        family: sum(
            failure["family"] == family for failure in failures
        )
        for family in ("futures", "options")
    }
    audit = audit_probe_contract(
        features,
        mappings,
        target_dates=PROBE_DATES,
        family_query_failures=family_failures,
        expected_previous_dates=previous_dates,
    )

    artifact_paths = {
        "mappings": atomic_write_parquet(
            mappings,
            output / "wp_v39_probe_futures_mapping.parquet",
        ),
        "futures_daily": atomic_write_parquet(
            futures_daily,
            output / "wp_v39_probe_futures_daily.parquet",
        ),
        "index_daily": atomic_write_parquet(
            index_daily,
            output / "wp_v39_probe_index_daily.parquet",
        ),
        "option_basic": atomic_write_parquet(
            option_basic,
            output / "wp_v39_probe_option_basic.parquet",
        ),
        "options_daily": atomic_write_parquet(
            options_daily,
            output / "wp_v39_probe_options_daily.parquet",
        ),
        "fund_daily": atomic_write_parquet(
            fund_daily,
            output / "wp_v39_probe_fund_daily.parquet",
        ),
        "features": atomic_write_parquet(
            features,
            output / "wp_v39_probe_derivative_features.parquet",
        ),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "profit_outcomes_read": False,
        "future_information_allowed": False,
        "probe_dates": list(PROBE_DATES),
        "previous_trade_dates": previous_dates,
        "source_contract": {
            "futures": {
                "mapping_api": "fut_mapping",
                "daily_api": "fut_daily",
                "index_api": "index_daily",
                "feature_availability": "T-1 close",
            },
            "options": {
                "contract_api": "opt_basic",
                "daily_api": "opt_daily",
                "underlying_api": "fund_daily",
                "feature_availability": "T-1 close",
            },
            "candidate_outcomes_read": False,
            "target_day_data_read": False,
        },
        "query_failures": failures,
        "query_failure_summary": _summarize_failures(failures),
        "coverage_audit": audit,
        "feature_columns": {
            "futures": list(V39_FUTURE_FEATURE_COLUMNS),
            "options": list(V39_OPTION_FEATURE_COLUMNS),
        },
        "artifacts": {
            name: _file_artifact(path)
            for name, path in artifact_paths.items()
        },
        "full_backfill_authorized": bool(
            audit["full_backfill_authorized"]
        ),
        "authorized_source_families": audit[
            "selected_source_families"
        ],
        "model_research_authorized": False,
        "next_gate": (
            "full_three_year_outcome_blind_v39_data_build"
            if audit["full_backfill_authorized"]
            else "close_v39_data_direction"
        ),
    }
    atomic_write_json(
        output / "wp_v39_derivatives_daily_probe.json",
        payload,
    )
    print(
        "WP_V39_PROBE_RESULT="
        + json.dumps(
            {
                "probe_dates": len(PROBE_DATES),
                "previous_trade_dates": previous_dates,
                "mapping_rows": len(mappings),
                "futures_daily_rows": len(futures_daily),
                "index_daily_rows": len(index_daily),
                "option_basic_rows": len(option_basic),
                "options_daily_rows": len(options_daily),
                "fund_daily_rows": len(fund_daily),
                "family_query_failures": family_failures,
                "coverage_audit": audit,
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
            "WP_V39_QUERY_FAILURES="
            + json.dumps(
                _summarize_failures(failures),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
            flush=True,
        )
    if not audit["full_backfill_authorized"]:
        raise RuntimeError("V39 data probe failed its frozen contract")
    return 0


def _load_previous_dates(
    client: TushareHistoryClient,
    failures: list[dict[str, str]],
    token: str,
) -> dict[str, str]:
    try:
        calendar = client.query(
            "trade_cal",
            cache_key="20230701_20260723_sse_v39_probe",
            exchange="SSE",
            start_date="20230701",
            end_date=max(PROBE_DATES),
            is_open="1",
            fields="exchange,cal_date,is_open,pretrade_date",
        )
        open_dates = sorted(
            calendar.loc[
                calendar["is_open"].astype(str).eq("1"),
                "cal_date",
            ].astype(str)
        )
        previous = {
            target_date: max(
                date for date in open_dates if date < target_date
            )
            for target_date in PROBE_DATES
        }
        if len(previous) != len(PROBE_DATES):
            raise RuntimeError("incomplete SSE T-1 calendar mapping")
        return previous
    except Exception as error:
        for family in ("futures", "options"):
            failures.append(
                {
                    "family": family,
                    "stage": "trade_cal",
                    "target_date": "all",
                    "source_date": "all",
                    "ts_code": "SSE",
                    "error": _clean_error(error, token),
                }
            )
        return {}


def _query_one(
    client: TushareHistoryClient,
    failures: list[dict[str, str]],
    token: str,
    *,
    family: str,
    stage: str,
    target_date: str,
    source_date: str,
    evidence_code: str,
    api_name: str,
    cache_key: str,
    fields: str,
    paged: bool = False,
    **params: Any,
) -> list[pd.DataFrame]:
    try:
        frame = client.query(
            api_name,
            cache_key=cache_key,
            paged=paged,
            fields=fields,
            **params,
        )
        if frame.empty:
            raise RuntimeError("empty response")
        return [frame]
    except Exception as error:
        failures.append(
            {
                "family": family,
                "stage": stage,
                "target_date": target_date,
                "source_date": source_date,
                "ts_code": evidence_code,
                "error": _clean_error(error, token),
            }
        )
        return []


def _concat(
    frames: list[pd.DataFrame],
    fields: str,
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=fields.split(","))
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def _target_for_source(
    source_date: str,
    previous_dates: dict[str, str],
) -> str:
    return next(
        (
            target
            for target, previous in previous_dates.items()
            if previous == source_date
        ),
        "",
    )


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
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for failure in failures:
        key = (
            failure["family"],
            failure["stage"],
            failure["error"],
        )
        summary = grouped.setdefault(
            key,
            {
                "family": failure["family"],
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
                    "target_date": failure["target_date"],
                    "source_date": failure["source_date"],
                    "ts_code": failure["ts_code"],
                }
            )
    return sorted(
        grouped.values(),
        key=lambda item: (
            item["family"],
            -item["count"],
            item["stage"],
        ),
    )[:16]


if __name__ == "__main__":
    raise SystemExit(main())
