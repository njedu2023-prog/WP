from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd
import pyarrow.parquet as pq
import tushare as ts

from wp.v3.contracts import load_v3_config
from wp.v3.history import MINUTE_FIELDS, TushareHistoryClient
from wp.v3.io import (
    atomic_write_json,
    atomic_write_parquet,
    file_sha256,
)
from wp.v3.v19_recall import (
    DEFAULT_EXPLORATION_PER_SLOT,
    DEFAULT_TOP_PER_SOURCE,
)
from wp.v3.v23_data import (
    AUCTION_FIELDS,
    MINIMUM_DATASET_COVERAGE,
    MONEYFLOW_FIELDS,
    SCHEMA_VERSION,
    assemble_v23_feature_frame,
    attach_previous_trade_dates,
    build_auction_features,
    build_minute_features,
    build_moneyflow_features,
    feature_coverage_audit,
    load_v23_source_leaders,
    normalize_one_minute,
    required_codes_by_date,
    required_stock_months,
)


ROOT = Path(__file__).resolve().parents[1]
T = TypeVar("T")
R = TypeVar("R")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the outcome-blind V23 point-in-time microstructure dataset "
            "for immutable V9 leaders."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--top-per-source",
        type=int,
        default=DEFAULT_TOP_PER_SOURCE,
    )
    parser.add_argument(
        "--exploration-per-slot",
        type=int,
        default=DEFAULT_EXPLORATION_PER_SLOT,
    )
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for V23 backfill")
    config = load_v3_config(args.config)
    output = Path(args.output_dir)
    raw_root = output / "raw"
    output.mkdir(parents=True, exist_ok=True)
    client = TushareHistoryClient(
        ts.pro_api(token),
        args.cache_dir,
        page_size=config.history.tushare_page_size,
        requests_per_minute=config.history.tushare_requests_per_minute,
        attempts=6,
    )

    leaders, source = load_v23_source_leaders(
        args.shard_dir,
        evaluation_end=config.history.evaluation_end_date,
        top_per_source=args.top_per_source,
        exploration_per_slot=args.exploration_per_slot,
    )
    print(
        f"[wp-v23-data] immutable outcome-blind leaders={len(leaders):,}",
        flush=True,
    )
    open_dates = load_trade_calendar(
        client,
        start_date=config.history.start_date,
        end_date=config.history.evaluation_end_date,
    )
    leaders = attach_previous_trade_dates(leaders, open_dates)

    minute_paths, minute_failures = fetch_minute_partitions(
        client,
        leaders,
        output_dir=raw_root / "one_minute",
        workers=args.workers,
    )
    auction_paths, auction_failures = fetch_cross_section_partitions(
        client,
        api_name="stk_auction_o",
        fields=AUCTION_FIELDS,
        required=required_codes_by_date(leaders),
        output_dir=raw_root / "opening_auction",
        file_prefix="wp_v23_auction",
        workers=args.workers,
    )
    previous_requirements = {
        str(previous_date): tuple(
            sorted(group["ts_code"].astype(str).unique())
        )
        for previous_date, group in leaders.groupby(
            "v23_prev_trade_date",
            sort=True,
        )
    }
    moneyflow_paths, moneyflow_failures = fetch_cross_section_partitions(
        client,
        api_name="moneyflow",
        fields=MONEYFLOW_FIELDS,
        required=previous_requirements,
        output_dir=raw_root / "previous_moneyflow",
        file_prefix="wp_v23_moneyflow",
        workers=args.workers,
    )

    minutes = read_partitions(minute_paths)
    auctions = read_partitions(auction_paths)
    moneyflow = read_partitions(moneyflow_paths)
    minute_features = build_minute_features(leaders, minutes)
    auction_features = build_auction_features(leaders, auctions)
    moneyflow_features = build_moneyflow_features(leaders, moneyflow)
    features = assemble_v23_feature_frame(
        leaders,
        minute_features,
        auction_features,
        moneyflow_features,
    )
    audit = feature_coverage_audit(features)

    leader_path = atomic_write_parquet(
        leaders,
        output / "wp_v23_outcome_blind_leader_index.parquet",
    )
    feature_path = atomic_write_parquet(
        features,
        output / "wp_v23_point_in_time_features.parquet",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "profit_outcomes_read": False,
        "future_information_allowed": False,
        "source": source,
        "protocol": {
            "selection": (
                "immutable V9 broad-recall frontier and one source-eligible "
                "leader per signal slot"
            ),
            "minute_frequency": "1min",
            "minute_observation_start": "13:55",
            "minute_feature_cutoff": "at_or_before_signal_slot",
            "auction_use": "same_trade_day_completed_opening_auction",
            "moneyflow_use": "immediately_previous_A_share_trade_day_only",
            "minimum_row_minute_coverage": 0.90,
            "minimum_dataset_coverage": MINIMUM_DATASET_COVERAGE,
            "outcome_driven_row_dropping": False,
        },
        "requirements": {
            "leader_rows": int(len(leaders)),
            "trade_dates": int(leaders["trade_date"].nunique()),
            "stock_month_pairs": int(
                sum(
                    len(codes)
                    for codes in required_stock_months(leaders).values()
                )
            ),
            "auction_dates": int(leaders["trade_date"].nunique()),
            "moneyflow_previous_dates": int(
                leaders["v23_prev_trade_date"].nunique()
            ),
        },
        "trade_calendar": {
            "exchange": "SSE",
            "open_dates": open_dates,
            "open_date_count": int(len(open_dates)),
            "start_date": open_dates[0],
            "end_date": open_dates[-1],
        },
        "query_failures": {
            "one_minute": minute_failures,
            "opening_auction": auction_failures,
            "previous_moneyflow": moneyflow_failures,
        },
        "coverage_audit": audit,
        "artifacts": {
            "leader_index": file_artifact(leader_path),
            "features": file_artifact(feature_path),
            "one_minute_partitions": [
                file_artifact(path) for path in minute_paths
            ],
            "opening_auction_partitions": [
                file_artifact(path) for path in auction_paths
            ],
            "previous_moneyflow_partitions": [
                file_artifact(path) for path in moneyflow_paths
            ],
        },
        "v23_model_research_authorized": bool(
            audit["coverage_passed"]
            and not minute_failures
            and not auction_failures
            and not moneyflow_failures
        ),
    }
    atomic_write_json(output / "wp_v23_data_manifest.json", manifest)
    print(
        "WP_V23_DATA_RESULT="
        + json.dumps(
            {
                "leader_rows": int(len(leaders)),
                "requirements": manifest["requirements"],
                "coverage_audit": audit,
                "failure_counts": {
                    key: len(value)
                    for key, value in manifest["query_failures"].items()
                },
                "v23_model_research_authorized": (
                    manifest["v23_model_research_authorized"]
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    if not manifest["v23_model_research_authorized"]:
        raise RuntimeError(
            "V23 point-in-time dataset failed the preregistered coverage gate"
        )
    return 0


def load_trade_calendar(
    client: TushareHistoryClient,
    *,
    start_date: str,
    end_date: str,
) -> list[str]:
    padded_start = (
        pd.Timestamp(start_date) - pd.Timedelta(days=45)
    ).strftime("%Y%m%d")
    frame = client.query(
        "trade_cal",
        cache_key=f"{padded_start}_{end_date}_sse_v23",
        exchange="SSE",
        start_date=padded_start,
        end_date=end_date,
        is_open="1",
        fields="exchange,cal_date,is_open,pretrade_date",
    )
    return sorted(
        frame.loc[
            frame["is_open"].astype(str).eq("1"),
            "cal_date",
        ]
        .astype(str)
        .unique()
    )


def fetch_minute_partitions(
    client: TushareHistoryClient,
    leaders: pd.DataFrame,
    *,
    output_dir: Path,
    workers: int,
) -> tuple[list[Path], list[dict[str, str]]]:
    requirements = required_stock_months(leaders)
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
                        f"{ts_code.replace('.', '_')}_{task_month}_1min_v23"
                    ),
                    paged=True,
                    ts_code=ts_code,
                    start_date=f"{month_start(task_month)} 13:55:00",
                    end_date=f"{month_end(task_month)} 15:00:00",
                    freq="1min",
                    fields=MINUTE_FIELDS,
                )
                normalized = normalize_one_minute(raw)
                selected = normalized.loc[
                    normalized["trade_date"].isin(dates)
                ].copy()
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
                    f"[wp-v23-data] minute pairs={completed}/{total} "
                    f"failures={len(failures)}",
                    flush=True,
                )
        month_frame = (
            pd.concat(month_frames, ignore_index=True)
            if month_frames
            else pd.DataFrame(columns=MINUTE_FIELDS.split(","))
        )
        path = atomic_write_parquet(
            month_frame,
            output_dir / f"wp_v23_minutes_{month}.parquet",
        )
        paths.append(path)
        print(
            f"[wp-v23-data] minute month={month} "
            f"rows={len(month_frame):,}",
            flush=True,
        )
    return paths, failures


def fetch_cross_section_partitions(
    client: TushareHistoryClient,
    *,
    api_name: str,
    fields: str,
    required: dict[str, tuple[str, ...]],
    output_dir: Path,
    file_prefix: str,
    workers: int,
) -> tuple[list[Path], list[dict[str, str]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = sorted(required.items())

    def fetch(
        task: tuple[str, tuple[str, ...]],
    ) -> tuple[pd.DataFrame, dict[str, str] | None]:
        trade_date, codes = task
        try:
            frame = client.query(
                api_name,
                cache_key=f"{trade_date}_v23",
                paged=True,
                trade_date=trade_date,
                fields=fields,
            )
            frame["trade_date"] = frame["trade_date"].astype(str)
            frame["ts_code"] = frame["ts_code"].astype(str)
            return frame.loc[frame["ts_code"].isin(codes)].copy(), None
        except Exception as error:
            return pd.DataFrame(), {
                "trade_date": trade_date,
                "error": str(error)[:500],
            }

    rows = parallel_map(fetch, tasks, workers=workers)
    frames_by_year: dict[str, list[pd.DataFrame]] = {}
    failures: list[dict[str, str]] = []
    for index, (frame, failure) in enumerate(rows, start=1):
        if failure:
            failures.append(failure)
        elif not frame.empty:
            year = str(frame["trade_date"].iloc[0])[:4]
            frames_by_year.setdefault(year, []).append(frame)
        if index % 100 == 0:
            print(
                f"[wp-v23-data] {api_name} dates={index}/{len(tasks)} "
                f"failures={len(failures)}",
                flush=True,
            )
    paths: list[Path] = []
    for year in sorted({date[:4] for date in required}):
        frame = (
            pd.concat(frames_by_year.get(year, []), ignore_index=True)
            if frames_by_year.get(year)
            else pd.DataFrame(columns=fields.split(","))
        )
        path = atomic_write_parquet(
            frame,
            output_dir / f"{file_prefix}_{year}.parquet",
        )
        paths.append(path)
        print(
            f"[wp-v23-data] {api_name} year={year} rows={len(frame):,}",
            flush=True,
        )
    return paths, failures


def parallel_map(
    function: Callable[[T], R],
    values: list[T],
    *,
    workers: int,
) -> list[R]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        return list(executor.map(function, values))


def month_start(month: str) -> str:
    timestamp = pd.Period(month, freq="M").start_time
    return timestamp.strftime("%Y-%m-%d")


def month_end(month: str) -> str:
    timestamp = pd.Period(month, freq="M").end_time
    return timestamp.strftime("%Y-%m-%d")


def read_partitions(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def file_artifact(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    return {
        "path": str(resolved.as_posix()),
        "sha256": file_sha256(resolved),
        "bytes": int(resolved.stat().st_size),
        "rows": int(pq.ParquetFile(resolved).metadata.num_rows),
    }


if __name__ == "__main__":
    raise SystemExit(main())
