from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wp.v3.backtest import walk_forward_backtest, walk_forward_fold_count
from wp.v3.contracts import load_v3_config
from wp.v3.dataset import audit_panel
from wp.v3.history import load_panel_partitions
from wp.v3.sharding import shard_fold_numbers, write_walk_forward_shard


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one deterministic shard of the WP V5 nested walk-forward backtest."
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
    panel = load_panel_partitions(args.panel_dir)
    _assert_three_year_panel(panel)
    total_folds = walk_forward_fold_count(panel, config)
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
        f"[wp-v5] shard={args.shard_index}/{args.shard_count} "
        f"folds={list(fold_numbers)} total_folds={total_folds}",
        flush=True,
    )
    result = walk_forward_backtest(
        panel,
        config,
        fold_numbers=fold_numbers,
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
        f"[wp-v5] shard complete index={args.shard_index} "
        f"rows={manifest['prediction_rows']:,} "
        f"sha256={manifest['prediction_sha256']}",
        flush=True,
    )
    return 0


def _assert_three_year_panel(panel: pd.DataFrame) -> None:
    audit = audit_panel(panel)
    if audit.trade_days < 700:
        raise RuntimeError(
            f"three-year research contract requires at least 700 covered trade days; "
            f"received {audit.trade_days}"
        )
    start = pd.Timestamp(str(panel["trade_date"].min()))
    end = pd.Timestamp(str(panel["trade_date"].max()))
    if (end - start).days < 1_000:
        raise RuntimeError(
            f"dataset covers only {(end - start).days} calendar days; "
            "three years are required"
        )


if __name__ == "__main__":
    raise SystemExit(main())
