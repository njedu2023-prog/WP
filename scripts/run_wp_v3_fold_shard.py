from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from wp.v3.backtest import (
    walk_forward_backtest,
    walk_forward_fold_count,
    walk_forward_fold_dates,
)
from wp.v3.contracts import load_v3_config
from wp.v3.history import load_panel_partitions
from wp.v3.sharding import shard_fold_numbers, write_walk_forward_shard


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one deterministic shard of the WP V7 nested walk-forward backtest."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "wp_v3.yml"))
    parser.add_argument(
        "--panel-dir",
        default=str(ROOT / "artifacts" / "wp_v3_history" / "panel"),
    )
    parser.add_argument(
        "--dataset-manifest",
        default=str(
            ROOT
            / "artifacts"
            / "wp_v3_history"
            / "wp_v3_dataset_manifest.json"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_v3_config(args.config)
    calendar = load_panel_partitions(
        args.panel_dir,
        columns=["trade_date"],
    )
    _assert_research_calendar(calendar, config)
    calendar_dates = np.array(
        sorted(calendar["trade_date"].astype(str).unique())
    )
    total_folds = walk_forward_fold_count(calendar, config)
    fold_numbers = shard_fold_numbers(
        total_folds,
        args.shard_index,
        args.shard_count,
    )
    if not fold_numbers:
        raise RuntimeError(
            f"shard {args.shard_index}/{args.shard_count} has no assigned folds"
        )
    print(
        f"[wp-v7] shard={args.shard_index}/{args.shard_count} "
        f"folds={list(fold_numbers)} total_folds={total_folds}",
        flush=True,
    )
    windows = [
        walk_forward_fold_dates(calendar_dates, config, fold_number)
        for fold_number in fold_numbers
    ]
    panel_start = min(str(train_dates[0]) for train_dates, _ in windows)
    panel_end = max(str(test_dates[-1]) for _, test_dates in windows)
    del calendar
    gc.collect()
    panel = load_panel_partitions(
        args.panel_dir,
        start_date=panel_start,
        end_date=panel_end,
    )
    print(
        f"[wp-v7] selective panel={panel_start}..{panel_end} "
        f"rows={len(panel):,}",
        flush=True,
    )
    result = walk_forward_backtest(
        panel,
        config,
        fold_numbers=fold_numbers,
        calendar_dates=calendar_dates,
        evaluate=False,
    )
    manifest = write_walk_forward_shard(
        result,
        args.output_dir,
        config=config,
        dataset_manifest_path=args.dataset_manifest,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        total_folds=total_folds,
    )
    print(
        f"[wp-v7] shard complete index={args.shard_index} "
        f"rows={manifest['prediction_rows']:,} "
        f"sha256={manifest['prediction_sha256']}",
        flush=True,
    )
    return 0


def _assert_research_calendar(calendar: pd.DataFrame, config) -> None:
    trade_dates = calendar["trade_date"].astype(str)
    unique_dates = sorted(trade_dates.unique())
    evaluation_dates = [
        date
        for date in unique_dates
        if (
            config.history.evaluation_start_date
            <= date
            <= config.history.evaluation_end_date
        )
    ]
    if len(evaluation_dates) < 700:
        raise RuntimeError(
            "full three-year OOS evaluation requires at least 700 covered "
            f"trade days; received {len(evaluation_dates)}"
        )
    if (
        evaluation_dates[0] != config.history.evaluation_start_date
        or evaluation_dates[-1] != config.history.evaluation_end_date
    ):
        raise RuntimeError(
            "panel does not cover the exact declared OOS evaluation boundary"
        )
    pre_evaluation_days = sum(
        date < config.history.evaluation_start_date for date in unique_dates
    )
    required_warmup_days = (
        config.model.minimum_train_days
        + config.model.calibration_days
        + 2 * config.model.purge_days
        + config.model.policy_design_days
        + config.model.policy_confirmation_days
    )
    if pre_evaluation_days < required_warmup_days:
        raise RuntimeError(
            f"causal panel has {pre_evaluation_days} pre-evaluation days; "
            f"{required_warmup_days} are required for model and policy warmup"
        )


if __name__ == "__main__":
    raise SystemExit(main())
