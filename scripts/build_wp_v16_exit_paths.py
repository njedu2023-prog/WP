from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import tushare as ts

from wp.v3.contracts import load_v3_config
from wp.v3.history import (
    TushareHistoryClient,
    load_historical_minutes,
    load_panel_partitions,
)
from wp.v3.v16_data import (
    IDENTITY_COLUMNS,
    candidate_exit_pairs,
    normalize_full_day_minutes,
    write_exit_path_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a versioned full-day T+1 five-minute path dataset for the "
            "immutable V11 causal candidate frontier."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--v11-source-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-pair-coverage", type=float, default=0.98)
    parser.add_argument("--minimum-bars", type=int, default=44)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for V16 exit-path build")
    config = load_v3_config(args.config)
    frontier, source = load_v11_frontier(args.v11_source_dir)
    panel = load_panel_partitions(
        args.panel_dir,
        columns=[*IDENTITY_COLUMNS, "target_trade_date"],
        start_date=config.history.evaluation_start_date,
        end_date=config.history.evaluation_end_date,
    )
    pairs = candidate_exit_pairs(frontier, panel)
    print(
        f"[wp-v16-data] immutable_frontier={len(frontier):,} "
        f"unique_t1_pairs={len(pairs):,}",
        flush=True,
    )

    client = TushareHistoryClient(
        ts.pro_api(token),
        args.cache_dir,
        page_size=config.history.tushare_page_size,
        requests_per_minute=config.history.tushare_requests_per_minute,
    )
    raw_frames: list[pd.DataFrame] = []
    grouped = list(pairs.groupby("ts_code", sort=True))
    for index, (ts_code, symbol_pairs) in enumerate(grouped, start=1):
        target_dates = set(symbol_pairs["target_trade_date"].astype(str))
        raw = load_historical_minutes(
            client,
            ts_code=str(ts_code),
            start_date=min(target_dates),
            end_date=max(target_dates),
        )
        if not raw.empty:
            raw_dates = pd.to_datetime(
                raw["trade_time"],
                errors="coerce",
            ).dt.strftime("%Y%m%d")
            selected = raw.loc[raw_dates.isin(target_dates)].copy()
            if not selected.empty:
                raw_frames.append(selected)
        if index % 50 == 0 or index == len(grouped):
            rows = sum(len(frame) for frame in raw_frames)
            print(
                f"[wp-v16-data] symbols={index}/{len(grouped)} "
                f"selected_bars={rows:,}",
                flush=True,
            )
    raw_minutes = (
        pd.concat(raw_frames, ignore_index=True)
        if raw_frames
        else pd.DataFrame()
    )
    minutes, quality = normalize_full_day_minutes(
        raw_minutes,
        pairs,
        minimum_bars=args.minimum_bars,
    )
    source.update(
        {
            "panel_contract": {
                "evaluation_start": config.history.evaluation_start_date,
                "evaluation_end": config.history.evaluation_end_date,
                "entry_contract": config.execution.entry_price_contract,
                "exit_contract": config.strategy.exit_contract,
            },
            "required_pair_digest": stable_pair_digest(pairs),
        }
    )
    manifest = write_exit_path_dataset(
        minutes,
        quality,
        args.output_dir,
        source=source,
        minimum_pair_coverage=args.minimum_pair_coverage,
        minimum_bars=args.minimum_bars,
    )
    print(
        "WP_V16_EXIT_PATH_RESULT="
        + json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "dataset_fingerprint": manifest["dataset_fingerprint"],
                "required_pairs": manifest["required_pairs"],
                "covered_pairs": manifest["covered_pairs"],
                "pair_coverage": manifest["pair_coverage"],
                "rows": manifest["rows"],
                "trade_dates": manifest["trade_dates"],
                "symbols": manifest["symbols"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def load_v11_frontier(
    path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(path)
    frontier_paths = sorted(root.rglob("wp_v11_exit_frontier.parquet"))
    summary_paths = sorted(root.rglob("wp_v11_exit_summary.json"))
    if len(frontier_paths) != 1 or len(summary_paths) != 1:
        raise FileNotFoundError(
            "V11 source must contain exactly one frontier and one summary"
        )
    frontier_path = frontier_paths[0]
    summary_path = summary_paths[0]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    actual = file_digest(frontier_path)
    expected = str(summary.get("candidate_frontier_sha256") or "")
    if not expected or expected != actual:
        raise RuntimeError(
            f"V11 frontier digest mismatch: {actual} != {expected}"
        )
    frame = pd.read_parquet(
        frontier_path,
        columns=list(IDENTITY_COLUMNS),
    )
    return frame, {
        "v11_schema_version": summary.get("schema_version"),
        "v11_frontier_sha256": actual,
        "v11_summary_sha256": file_digest(summary_path),
        "v11_frontier_rows": int(len(frame)),
    }


def stable_pair_digest(frame: pd.DataFrame) -> str:
    content = (
        frame.sort_values(["target_trade_date", "ts_code"], kind="stable")
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    )
    return hashlib.sha256(content).hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
