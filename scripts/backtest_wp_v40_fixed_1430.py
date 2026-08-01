from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wp.v3.contracts import load_v3_config
from wp.v3.io import (
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
)
from wp.v3.v40 import evaluate_v40_fixed_1430


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the immutable V40 fixed-14:30 dual-cohort contract."
        )
    )
    parser.add_argument("--config", default="config/wp_v3.yml")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--start-date", default="20260501")
    parser.add_argument("--end-date", default="20260731")
    parser.add_argument("--source-run-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = Path(args.source)
    if source_path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source_path)
    else:
        frame = pd.read_csv(source_path, low_memory=False)
    config = load_v3_config(args.config)
    result = evaluate_v40_fixed_1430(
        frame,
        config,
        start_date=args.start_date,
        end_date=args.end_date,
        source_run_id=args.source_run_id,
    )
    output = Path(args.output_dir)
    json_path = output / "json" / "wp_v40_backtest_202605_202607.json"
    qualified_path = (
        output / "csv" / "wp_v40_backtest_qualified_202605_202607.csv"
    )
    observations_path = (
        output / "csv" / "wp_v40_backtest_observations_202605_202607.csv"
    )
    atomic_write_csv(result.qualified, qualified_path)
    atomic_write_csv(result.observations, observations_path)
    summary = {
        **result.summary,
        "artifacts": {
            "source_sha256": file_sha256(source_path),
            "qualified_csv": qualified_path.as_posix(),
            "observations_csv": observations_path.as_posix(),
        },
    }
    atomic_write_json(json_path, summary)
    print(
        "WP_V40_FIXED_1430_BACKTEST_RESULT="
        + json.dumps(
            {
                "status": summary["status"],
                "source": summary["source"],
                "integrity": {
                    key: value
                    for key, value in summary["integrity"].items()
                    if key != "day_audit"
                },
                "qualified": summary["qualified"],
                "observations": summary["observations"],
                "interpretation": summary["interpretation"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
