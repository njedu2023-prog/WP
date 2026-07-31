from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import tushare as ts

from build_wp_v23_point_in_time_data import (
    fetch_cross_section_partitions,
    fetch_minute_partitions,
    file_artifact,
    load_trade_calendar,
    read_partitions,
)
from wp.v3.contracts import load_v3_config
from wp.v3.history import TushareHistoryClient
from wp.v3.io import atomic_write_json, atomic_write_parquet
from wp.v3.v19_recall import (
    DEFAULT_EXPLORATION_PER_SLOT,
    DEFAULT_TOP_PER_SOURCE,
)
from wp.v3.v23_data import (
    AUCTION_FIELDS,
    MINIMUM_DATASET_COVERAGE,
    MONEYFLOW_FIELDS,
    attach_previous_trade_dates,
    build_auction_features,
    build_minute_features,
    build_moneyflow_features,
    feature_coverage_audit,
    required_codes_by_date,
    required_stock_months,
)
from wp.v3.v24_data import (
    SCHEMA_VERSION,
    SOURCE_CANDIDATES_PER_SLOT,
    V24_DERIVED_SOURCE_FEATURE_COLUMNS,
    assemble_v24_feature_frame,
    load_v24_source_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the outcome-blind V24 top-5 point-in-time opportunity set."
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
    parser.add_argument(
        "--candidates-per-slot",
        type=int,
        default=SOURCE_CANDIDATES_PER_SLOT,
    )
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for V24 backfill")
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

    candidates, source = load_v24_source_candidates(
        args.shard_dir,
        evaluation_end=config.history.evaluation_end_date,
        top_per_source=args.top_per_source,
        exploration_per_slot=args.exploration_per_slot,
        candidates_per_slot=args.candidates_per_slot,
    )
    print(
        f"[wp-v24-data] immutable outcome-blind candidates={len(candidates):,}",
        flush=True,
    )
    open_dates = load_trade_calendar(
        client,
        start_date=config.history.start_date,
        end_date=config.history.evaluation_end_date,
    )
    candidates = attach_previous_trade_dates(candidates, open_dates)

    minute_paths, minute_failures = fetch_minute_partitions(
        client,
        candidates,
        output_dir=raw_root / "one_minute",
        workers=args.workers,
    )
    auction_paths, auction_failures = fetch_cross_section_partitions(
        client,
        api_name="stk_auction_o",
        fields=AUCTION_FIELDS,
        required=required_codes_by_date(candidates),
        output_dir=raw_root / "opening_auction",
        file_prefix="wp_v24_auction",
        workers=args.workers,
    )
    previous_requirements = {
        str(previous_date): tuple(
            sorted(group["ts_code"].astype(str).unique())
        )
        for previous_date, group in candidates.groupby(
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
        file_prefix="wp_v24_moneyflow",
        workers=args.workers,
    )

    minutes = read_partitions(minute_paths)
    auctions = read_partitions(auction_paths)
    moneyflow = read_partitions(moneyflow_paths)
    minute_features = build_minute_features(candidates, minutes)
    auction_features = build_auction_features(candidates, auctions)
    moneyflow_features = build_moneyflow_features(candidates, moneyflow)
    features = assemble_v24_feature_frame(
        candidates,
        minute_features,
        auction_features,
        moneyflow_features,
    )
    audit = feature_coverage_audit(features)
    source_feature_coverage = {
        column: float(
            pd.to_numeric(features[column], errors="coerce").notna().mean()
        )
        for column in V24_DERIVED_SOURCE_FEATURE_COLUMNS
        if column in features
    }

    candidate_path = atomic_write_parquet(
        candidates,
        output / "wp_v24_outcome_blind_candidate_index.parquet",
    )
    feature_path = atomic_write_parquet(
        features,
        output / "wp_v24_point_in_time_features.parquet",
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
                "immutable V9 broad-recall frontier and fixed top five "
                "selection-score candidates per signal slot"
            ),
            "candidates_per_slot": SOURCE_CANDIDATES_PER_SLOT,
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
            "candidate_rows": int(len(candidates)),
            "trade_dates": int(candidates["trade_date"].nunique()),
            "stock_month_pairs": int(
                sum(
                    len(codes)
                    for codes in required_stock_months(candidates).values()
                )
            ),
            "auction_dates": int(candidates["trade_date"].nunique()),
            "moneyflow_previous_dates": int(
                candidates["v23_prev_trade_date"].nunique()
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
        "coverage_audit": {
            **audit,
            "derived_source_feature_non_null_coverage": (
                source_feature_coverage
            ),
        },
        "artifacts": {
            "candidate_index": file_artifact(candidate_path),
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
        "v24_model_research_authorized": bool(
            audit["coverage_passed"]
            and not minute_failures
            and not auction_failures
            and not moneyflow_failures
        ),
    }
    atomic_write_json(output / "wp_v24_data_manifest.json", manifest)
    print(
        "WP_V24_DATA_RESULT="
        + json.dumps(
            {
                "candidate_rows": int(len(candidates)),
                "requirements": manifest["requirements"],
                "coverage_audit": audit,
                "failure_counts": {
                    key: len(value)
                    for key, value in manifest["query_failures"].items()
                },
                "v24_model_research_authorized": (
                    manifest["v24_model_research_authorized"]
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    if not manifest["v24_model_research_authorized"]:
        raise RuntimeError(
            "V24 point-in-time dataset failed the preregistered coverage gate"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
