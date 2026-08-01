from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import tushare as ts

from probe_wp_v39_derivatives_daily import (
    FUND_FIELDS,
    FUTURE_FIELDS,
    INDEX_FIELDS,
    MAPPING_FIELDS,
    OPTION_BASIC_FIELDS,
    OPTION_DAILY_FIELDS,
)
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


BUILD_SCHEMA_VERSION = "wp_v39_tminus1_derivatives_data_1"
DEFAULT_START_DATE = "20230727"
DEFAULT_END_DATE = "20260724"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen three-year outcome-blind V39 derivatives data."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for the V39 build")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    client = TushareHistoryClient(
        ts.pro_api(token),
        args.cache_dir,
        page_size=8_000,
        requests_per_minute=100,
        attempts=6,
    )
    failures: list[dict[str, str]] = []
    target_dates, previous_dates = _load_calendar(
        client,
        args.start_date,
        args.end_date,
    )
    if len(target_dates) < 700:
        raise RuntimeError(
            f"V39 requires at least 700 target days; found {len(target_dates)}"
        )
    source_dates = sorted(set(previous_dates.values()))
    source_start = source_dates[0]
    source_end = source_dates[-1]
    month_ranges = _month_ranges(source_dates)

    mappings = _load_mappings(
        client,
        source_start,
        source_end,
        failures,
        token,
    )
    futures_daily = _load_monthly(
        client,
        month_ranges,
        failures,
        token,
        family="futures",
        stage="fut_daily",
        api_name="fut_daily",
        fields=FUTURE_FIELDS,
        cache_prefix="cffex",
        exchange="CFFEX",
    )
    index_daily = _load_codes(
        client,
        (
            str(spec["index_code"])
            for spec in FUTURE_SPECS
        ),
        source_start,
        source_end,
        failures,
        token,
        family="futures",
        stage="index_daily",
        api_name="index_daily",
        fields=INDEX_FIELDS,
    )
    option_basic = _load_option_basic(client, failures, token)
    options_daily = _load_monthly(
        client,
        month_ranges,
        failures,
        token,
        family="options",
        stage="opt_daily",
        api_name="opt_daily",
        fields=OPTION_DAILY_FIELDS,
        cache_prefix="sse",
        exchange="SSE",
    )
    fund_daily = _load_codes(
        client,
        (
            str(spec["underlying_code"])
            for spec in OPTION_SPECS
        ),
        source_start,
        source_end,
        failures,
        token,
        family="options",
        stage="fund_daily",
        api_name="fund_daily",
        fields=FUND_FIELDS,
    )
    features = build_derivative_features(
        target_dates,
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
        target_dates=target_dates,
        family_query_failures=family_failures,
        expected_previous_dates=previous_dates,
    )

    paths = {
        "mappings": atomic_write_parquet(
            mappings,
            output / "wp_v39_futures_mapping.parquet",
        ),
        "futures_daily": atomic_write_parquet(
            futures_daily,
            output / "wp_v39_futures_daily.parquet",
        ),
        "index_daily": atomic_write_parquet(
            index_daily,
            output / "wp_v39_index_daily.parquet",
        ),
        "option_basic": atomic_write_parquet(
            option_basic,
            output / "wp_v39_option_basic.parquet",
        ),
        "options_daily": atomic_write_parquet(
            options_daily,
            output / "wp_v39_options_daily.parquet",
        ),
        "fund_daily": atomic_write_parquet(
            fund_daily,
            output / "wp_v39_fund_daily.parquet",
        ),
        "features": atomic_write_parquet(
            features,
            output / "wp_v39_derivative_features.parquet",
        ),
    }
    manifest = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "feature_schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "profit_outcomes_read": False,
        "future_information_allowed": False,
        "target_window": {
            "start_date": target_dates[0],
            "end_date": target_dates[-1],
            "trade_days": len(target_dates),
        },
        "source_window": {
            "start_date": source_start,
            "end_date": source_end,
            "trade_days": len(source_dates),
            "availability": "fully known after T-1 close",
        },
        "source_contract": {
            "futures": [
                dict(spec) for spec in FUTURE_SPECS
            ],
            "options": [
                dict(spec) for spec in OPTION_SPECS
            ],
            "target_day_data_read": False,
            "candidate_outcomes_read": False,
        },
        "query_failures": failures,
        "query_failure_counts": family_failures,
        "coverage_audit": audit,
        "feature_columns": {
            "futures": list(V39_FUTURE_FEATURE_COLUMNS),
            "options": list(V39_OPTION_FEATURE_COLUMNS),
        },
        "artifacts": {
            name: _file_artifact(path)
            for name, path in paths.items()
        },
        "authorized_source_families": audit[
            "selected_source_families"
        ],
        "v39_model_research_authorized": bool(
            audit["full_backfill_authorized"]
        ),
    }
    atomic_write_json(
        output / "wp_v39_derivatives_data_manifest.json",
        manifest,
    )
    print(
        "WP_V39_DATA_RESULT="
        + json.dumps(
            {
                "target_days": len(target_dates),
                "source_days": len(source_dates),
                "mapping_rows": len(mappings),
                "futures_daily_rows": len(futures_daily),
                "index_daily_rows": len(index_daily),
                "option_basic_rows": len(option_basic),
                "options_daily_rows": len(options_daily),
                "fund_daily_rows": len(fund_daily),
                "feature_rows": len(features),
                "family_query_failures": family_failures,
                "coverage_audit": audit,
                "v39_model_research_authorized": manifest[
                    "v39_model_research_authorized"
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    if not manifest["v39_model_research_authorized"]:
        raise RuntimeError(
            "V39 three-year data failed its frozen coverage contract"
        )
    return 0


def _load_calendar(
    client: TushareHistoryClient,
    start_date: str,
    end_date: str,
) -> tuple[list[str], dict[str, str]]:
    calendar_start = (
        datetime.strptime(start_date, "%Y%m%d") - timedelta(days=40)
    ).strftime("%Y%m%d")
    calendar = client.query(
        "trade_cal",
        cache_key=f"{calendar_start}_{end_date}_sse_v39_build",
        exchange="SSE",
        start_date=calendar_start,
        end_date=end_date,
        is_open="1",
        fields="exchange,cal_date,is_open,pretrade_date",
    )
    open_dates = sorted(
        calendar.loc[
            calendar["is_open"].astype(str).eq("1"),
            "cal_date",
        ].astype(str)
    )
    target_dates = [
        date for date in open_dates if start_date <= date <= end_date
    ]
    position = {date: index for index, date in enumerate(open_dates)}
    previous = {
        date: open_dates[position[date] - 1]
        for date in target_dates
        if position[date] > 0
    }
    if len(previous) != len(target_dates):
        raise RuntimeError("incomplete V39 previous-trading-day calendar")
    return target_dates, previous


def _load_mappings(
    client: TushareHistoryClient,
    start_date: str,
    end_date: str,
    failures: list[dict[str, str]],
    token: str,
) -> pd.DataFrame:
    frames = []
    for spec in FUTURE_SPECS:
        code = str(spec["continuous_code"])
        frames.extend(
            _query(
                client,
                failures,
                token,
                family="futures",
                stage="fut_mapping",
                identity=code,
                api_name="fut_mapping",
                cache_key=(
                    f"{code.replace('.', '_')}_{start_date}_"
                    f"{end_date}_v39_build"
                ),
                fields=MAPPING_FIELDS,
                paged=True,
                ts_code=code,
                start_date=start_date,
                end_date=end_date,
            )
        )
    return _concat(frames, MAPPING_FIELDS)


def _load_monthly(
    client: TushareHistoryClient,
    month_ranges: list[tuple[str, str, str]],
    failures: list[dict[str, str]],
    token: str,
    *,
    family: str,
    stage: str,
    api_name: str,
    fields: str,
    cache_prefix: str,
    exchange: str,
) -> pd.DataFrame:
    frames = []
    for month, start_date, end_date in month_ranges:
        frames.extend(
            _query(
                client,
                failures,
                token,
                family=family,
                stage=stage,
                identity=f"{exchange}:{month}",
                api_name=api_name,
                cache_key=f"{cache_prefix}_{month}_v39_build",
                fields=fields,
                paged=True,
                exchange=exchange,
                start_date=start_date,
                end_date=end_date,
            )
        )
        print(
            f"[wp-v39-data] {stage} month={month} frames={len(frames)}",
            flush=True,
        )
    return _concat(frames, fields)


def _load_codes(
    client: TushareHistoryClient,
    codes: Any,
    start_date: str,
    end_date: str,
    failures: list[dict[str, str]],
    token: str,
    *,
    family: str,
    stage: str,
    api_name: str,
    fields: str,
) -> pd.DataFrame:
    frames = []
    for code in codes:
        frames.extend(
            _query(
                client,
                failures,
                token,
                family=family,
                stage=stage,
                identity=code,
                api_name=api_name,
                cache_key=(
                    f"{code.replace('.', '_')}_{start_date}_"
                    f"{end_date}_v39_build"
                ),
                fields=fields,
                paged=True,
                ts_code=code,
                start_date=start_date,
                end_date=end_date,
            )
        )
    return _concat(frames, fields)


def _load_option_basic(
    client: TushareHistoryClient,
    failures: list[dict[str, str]],
    token: str,
) -> pd.DataFrame:
    frames = _query(
        client,
        failures,
        token,
        family="options",
        stage="opt_basic",
        identity="SSE",
        api_name="opt_basic",
        cache_key="sse_all_v39_build",
        fields=OPTION_BASIC_FIELDS,
        paged=True,
        exchange="SSE",
    )
    return _concat(frames, OPTION_BASIC_FIELDS)


def _query(
    client: TushareHistoryClient,
    failures: list[dict[str, str]],
    token: str,
    *,
    family: str,
    stage: str,
    identity: str,
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
        message = str(error).replace(token, "***")
        failures.append(
            {
                "family": family,
                "stage": stage,
                "identity": identity,
                "error": " ".join(message.split())[:500],
            }
        )
        return []


def _month_ranges(
    dates: list[str],
) -> list[tuple[str, str, str]]:
    grouped: dict[str, list[str]] = {}
    for date in dates:
        grouped.setdefault(date[:6], []).append(date)
    return [
        (month, min(values), max(values))
        for month, values in sorted(grouped.items())
    ]


def _concat(
    frames: list[pd.DataFrame],
    fields: str,
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=fields.split(","))
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def _file_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
