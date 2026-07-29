from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from wp.v3.backtest import (
    BacktestResult,
    WalkForwardFold,
    walk_forward_fold_count,
    walk_forward_fold_dates,
)
from wp.v3.contracts import V3Config
from wp.v3.sharding import (
    load_walk_forward_shards,
    shard_fold_numbers,
    write_walk_forward_shard,
)


def test_shard_fold_numbers_partition_every_fold_exactly_once():
    partitions = [
        shard_fold_numbers(10, shard_index, 5)
        for shard_index in range(5)
    ]
    assert partitions == [
        (1, 6),
        (2, 7),
        (3, 8),
        (4, 9),
        (5, 10),
    ]
    assert sorted(number for part in partitions for number in part) == list(
        range(1, 11)
    )


def test_fold_date_window_retains_only_model_history_and_outer_purge():
    base = V3Config()
    config = replace(
        base,
        model=replace(
            base.model,
            minimum_train_days=10,
            calibration_days=4,
            purge_days=2,
            test_days=3,
            ensemble_windows_days=(8, 12),
        ),
    )
    dates = [
        date.strftime("%Y%m%d")
        for date in pd.bdate_range("2026-01-01", periods=30)
    ]

    train_dates, test_dates = walk_forward_fold_dates(dates, config, 1)

    assert len(train_dates) == 16
    assert len(test_dates) == 3
    assert dates.index(str(test_dates[0])) - dates.index(str(train_dates[-1])) == 3


def test_shard_aggregation_rejects_missing_or_tampered_evidence(
    tmp_path,
    monkeypatch,
):
    base = V3Config()
    config = replace(
        base,
        model=replace(
            base.model,
            minimum_train_days=4,
            calibration_days=2,
            purge_days=1,
            test_days=3,
        ),
        history=replace(
            base.history,
            start_date="20251201",
            end_date="20260121",
            evaluation_start_date="20260113",
            evaluation_end_date="20260121",
        ),
    )
    dates = [date.strftime("%Y%m%d") for date in pd.bdate_range("2026-01-01", periods=15)]
    panel = pd.DataFrame(
        {
            "trade_date": dates,
            "signal_slot": ["14:20"] * len(dates),
            "ts_code": ["600001.SH"] * len(dates),
        }
    )
    assert walk_forward_fold_count(panel, config) == 3
    dataset_manifest = tmp_path / "dataset.json"
    dataset_manifest.write_text('{"version":"test"}\n', encoding="utf-8")

    starts = (8, 11, 14)
    shard_root = tmp_path / "shards"
    for shard_index in range(2):
        assigned = shard_fold_numbers(3, shard_index, 2)
        folds = []
        frames = []
        for fold_number in assigned:
            start = starts[fold_number - 1]
            test_dates = dates[start : start + 3]
            folds.append(
                WalkForwardFold(
                    fold=fold_number,
                    train_start=dates[0],
                    train_end=dates[start - 1],
                    test_start=test_dates[0],
                    test_end=test_dates[-1],
                    train_days=start,
                    test_days=len(test_dates),
                )
            )
            frames.append(
                pd.DataFrame(
                    {
                        "trade_date": test_dates,
                        "target_trade_date": test_dates,
                        "signal_slot": ["14:20"] * len(test_dates),
                        "ts_code": ["600001.SH"] * len(test_dates),
                        "signal_price": [10.0] * len(test_dates),
                        "passes_policy": [True] * len(test_dates),
                        "fold": [fold_number] * len(test_dates),
                        "unneeded_training_feature": [999.0] * len(test_dates),
                    }
                )
            )
        write_walk_forward_shard(
            BacktestResult(
                folds=folds,
                metrics={},
                predictions=pd.concat(frames, ignore_index=True),
                candidates=pd.DataFrame(),
            ),
            shard_root / f"shard-{shard_index}",
            config=config,
            dataset_manifest_path=dataset_manifest,
            shard_index=shard_index,
            shard_count=2,
            total_folds=3,
        )

    monkeypatch.setattr(
        "wp.v3.sharding.evaluate_predictions",
        lambda predictions, candidates, config: {
            "rows": len(predictions),
            "candidates": len(candidates),
        },
    )
    combined = load_walk_forward_shards(
        shard_root,
        panel=panel,
        config=config,
        dataset_manifest_path=dataset_manifest,
    )
    assert [fold.fold for fold in combined.folds] == [1, 2, 3]
    assert combined.metrics["rows"] == 7
    assert combined.metrics["candidates"] == 0
    assert combined.metrics["nested_policy"]["final"]["policy"]["authorized"] is False
    assert "unneeded_training_feature" not in combined.predictions.columns

    (shard_root / "shard-1" / "wp_v9_fold_shard_manifest.json").unlink()
    with pytest.raises(RuntimeError, match="incomplete walk-forward shards"):
        load_walk_forward_shards(
            shard_root,
            panel=panel,
            config=config,
            dataset_manifest_path=dataset_manifest,
        )
