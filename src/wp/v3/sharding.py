from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .backtest import (
    BacktestResult,
    WalkForwardFold,
    evaluation_contract_summary,
    evaluation_window,
    evaluate_predictions,
    walk_forward_fold_count,
)
from .contracts import V3Config, policy_fingerprint
from .dataset import IDENTITY_COLUMNS, first_crossing_candidates
from .io import atomic_write_json, atomic_write_parquet
from .policy import apply_nested_oos_policies


SHARD_SCHEMA_VERSION = "wp_v9_walk_forward_shard_1"
SHARD_MANIFEST_NAME = "wp_v9_fold_shard_manifest.json"
SHARD_PREDICTIONS_NAME = "wp_v9_fold_predictions.parquet"

# Fold workers need the full feature panel, but aggregation only needs immutable
# identity, execution truth, model outputs, and audit metadata. Keeping this
# boundary explicit prevents multi-gigabyte feature matrices from being loaded
# again after every fold has already finished.
AGGREGATE_PREDICTION_COLUMNS = (
    *IDENTITY_COLUMNS,
    "target_trade_date",
    "name",
    "board",
    "signal_price",
    "entry_price",
    "t1_close",
    "gross_return_pct",
    "net_return_pct",
    "ret_from_prev_close_pct",
    "execution_eligible",
    "entry_fillable",
    "exit_fillable",
    "execution_success",
    "label_available",
    "target_net_positive",
    "target_entry_fillable",
    "target_exit_fillable",
    "target_conditional_net_positive",
    "target_cross_section_top",
    "target_severe_loss",
    "target_conditional_severe_loss",
    "conditional_gross_return_pct",
    "conditional_net_return_pct",
    "data_age_seconds",
    "p_net_positive_raw",
    "p_conditional_net_positive_raw",
    "p_net_positive_direct",
    "p_net_positive_component_product",
    "p_net_positive_model_gap",
    "p_entry_fill",
    "p_exit_fill_given_entry",
    "p_round_trip_fill",
    "p_round_trip_fill_lower",
    "p_conditional_net_positive",
    "p_net_positive",
    "p_net_positive_lower",
    "p_cross_section_top",
    "p_conditional_severe_loss",
    "p_severe_loss_direct",
    "p_severe_loss_component_product",
    "p_severe_loss",
    "probability_model_spread",
    "fill_probability_model_spread",
    "conditional_expected_net_return_pct",
    "direct_all_in_expected_return_pct",
    "expected_utility_pct",
    "expected_utility_lower_pct",
    "expected_utility_residual_q10_adjustment_pct",
    "expected_return_model_spread",
    "conditional_downside_q10_pct",
    "direct_all_in_downside_q10_pct",
    "exit_failure_probability",
    "downside_q10_pct",
    "ranking_score",
    "selection_score",
    "selection_rank_pct",
    "selection_rank_spread",
    "model_version",
    "model_fingerprint",
    "policy_fingerprint",
    "fold",
    "test_start",
    "test_end",
)
SHARD_REQUIRED_COLUMNS = (*IDENTITY_COLUMNS, "fold")


def shard_fold_numbers(
    total_folds: int,
    shard_index: int,
    shard_count: int,
) -> tuple[int, ...]:
    if total_folds < 1:
        raise ValueError("total_folds must be positive")
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"shard_index must be in 0..{shard_count - 1}; received {shard_index}"
        )
    return tuple(
        fold_number
        for fold_number in range(1, total_folds + 1)
        if (fold_number - 1) % shard_count == shard_index
    )


def write_walk_forward_shard(
    result: BacktestResult,
    output_dir: str | Path,
    *,
    config: V3Config,
    dataset_manifest_path: str | Path,
    shard_index: int,
    shard_count: int,
    total_folds: int,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    expected_folds = shard_fold_numbers(total_folds, shard_index, shard_count)
    produced_folds = tuple(sorted(int(fold.fold) for fold in result.folds))
    if produced_folds != expected_folds:
        raise RuntimeError(
            f"shard {shard_index} produced folds {produced_folds}; "
            f"expected {expected_folds}"
        )
    prediction_folds = tuple(
        sorted(
            pd.to_numeric(
                result.predictions.get("fold"),
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
        )
    )
    if prediction_folds != expected_folds:
        raise RuntimeError(
            f"shard {shard_index} prediction folds {prediction_folds}; "
            f"expected {expected_folds}"
        )

    prediction_path = output / SHARD_PREDICTIONS_NAME
    missing_required = sorted(
        set(SHARD_REQUIRED_COLUMNS) - set(result.predictions.columns)
    )
    if missing_required:
        raise RuntimeError(
            f"shard {shard_index} predictions missing required columns "
            f"{missing_required}"
        )
    retained_columns = [
        column
        for column in AGGREGATE_PREDICTION_COLUMNS
        if column in result.predictions.columns
    ]
    ordered = result.predictions.loc[:, retained_columns].sort_values(
        ["fold", "trade_date", "signal_slot", "ts_code"],
        kind="stable",
    ).reset_index(drop=True)
    atomic_write_parquet(ordered, prediction_path)
    manifest = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "policy_fingerprint": policy_fingerprint(config),
        "dataset_manifest_sha256": _sha256_file(Path(dataset_manifest_path)),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "total_folds": int(total_folds),
        "expected_folds": list(expected_folds),
        "produced_folds": list(produced_folds),
        "prediction_rows": int(len(ordered)),
        "prediction_sha256": _sha256_file(prediction_path),
        "folds": [asdict(fold) for fold in result.folds],
    }
    atomic_write_json(output / SHARD_MANIFEST_NAME, manifest)
    return manifest


def load_walk_forward_shards(
    shard_root: str | Path,
    *,
    panel: pd.DataFrame,
    config: V3Config,
    dataset_manifest_path: str | Path,
) -> BacktestResult:
    root = Path(shard_root)
    manifest_paths = sorted(root.rglob(SHARD_MANIFEST_NAME))
    if not manifest_paths:
        raise FileNotFoundError(f"no walk-forward shard manifests under {root}")

    expected_total_folds = walk_forward_fold_count(panel, config)
    expected_policy = policy_fingerprint(config)
    expected_dataset = _sha256_file(Path(dataset_manifest_path))
    manifests: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    fold_records: dict[int, WalkForwardFold] = {}
    shard_indices: set[int] = set()
    shard_count: int | None = None

    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SHARD_SCHEMA_VERSION:
            raise RuntimeError(f"invalid shard schema in {manifest_path}")
        if manifest.get("policy_fingerprint") != expected_policy:
            raise RuntimeError(f"policy fingerprint mismatch in {manifest_path}")
        if manifest.get("dataset_manifest_sha256") != expected_dataset:
            raise RuntimeError(f"dataset fingerprint mismatch in {manifest_path}")
        if int(manifest.get("total_folds", -1)) != expected_total_folds:
            raise RuntimeError(f"walk-forward fold count mismatch in {manifest_path}")

        current_shard_count = int(manifest.get("shard_count", 0))
        if shard_count is None:
            shard_count = current_shard_count
        if current_shard_count != shard_count:
            raise RuntimeError("walk-forward shards use inconsistent shard counts")
        shard_index = int(manifest.get("shard_index", -1))
        if shard_index in shard_indices:
            raise RuntimeError(f"duplicate walk-forward shard index {shard_index}")
        shard_indices.add(shard_index)

        expected_folds = shard_fold_numbers(
            expected_total_folds,
            shard_index,
            current_shard_count,
        )
        produced_folds = tuple(int(value) for value in manifest.get("produced_folds", []))
        if produced_folds != expected_folds:
            raise RuntimeError(
                f"shard {shard_index} contains folds {produced_folds}; "
                f"expected {expected_folds}"
            )

        prediction_path = manifest_path.parent / SHARD_PREDICTIONS_NAME
        if _sha256_file(prediction_path) != manifest.get("prediction_sha256"):
            raise RuntimeError(f"prediction digest mismatch for shard {shard_index}")
        available_columns = set(
            pq.ParquetFile(prediction_path).schema_arrow.names
        )
        missing_required = sorted(
            set(SHARD_REQUIRED_COLUMNS) - available_columns
        )
        if missing_required:
            raise RuntimeError(
                f"prediction shard {shard_index} missing required columns "
                f"{missing_required}"
            )
        retained_columns = [
            column
            for column in AGGREGATE_PREDICTION_COLUMNS
            if column in available_columns
        ]
        frame = pd.read_parquet(
            prediction_path,
            columns=retained_columns,
        )
        if int(manifest.get("prediction_rows", -1)) != len(frame):
            raise RuntimeError(f"prediction row count mismatch for shard {shard_index}")
        frame_folds = tuple(
            sorted(
                pd.to_numeric(frame.get("fold"), errors="coerce")
                .dropna()
                .astype(int)
                .unique()
            )
        )
        if frame_folds != expected_folds:
            raise RuntimeError(
                f"prediction fold mismatch for shard {shard_index}: {frame_folds}"
            )

        for record in manifest.get("folds", []):
            fold = WalkForwardFold(**record)
            if fold.fold in fold_records:
                raise RuntimeError(f"duplicate walk-forward fold {fold.fold}")
            fold_frame = frame.loc[
                pd.to_numeric(frame["fold"], errors="coerce").eq(fold.fold)
            ]
            observed_dates = fold_frame["trade_date"].astype(str)
            if (
                observed_dates.empty
                or observed_dates.min() != fold.test_start
                or observed_dates.max() != fold.test_end
            ):
                raise RuntimeError(
                    f"test date boundary mismatch for fold {fold.fold}"
                )
            fold_records[fold.fold] = fold

        manifests.append(manifest)
        predictions.append(frame)

    if shard_count is None:
        raise RuntimeError("walk-forward shard count is missing")
    expected_indices = set(range(shard_count))
    if shard_indices != expected_indices:
        raise RuntimeError(
            f"incomplete walk-forward shards: found {sorted(shard_indices)}, "
            f"expected {sorted(expected_indices)}"
        )
    expected_folds = set(range(1, expected_total_folds + 1))
    if set(fold_records) != expected_folds:
        raise RuntimeError(
            f"incomplete walk-forward folds: found {sorted(fold_records)}, "
            f"expected {sorted(expected_folds)}"
        )

    combined = pd.concat(predictions, ignore_index=True)
    missing_identity = sorted(set(IDENTITY_COLUMNS) - set(combined.columns))
    if missing_identity:
        raise RuntimeError(f"shard predictions missing identity columns {missing_identity}")
    if combined.duplicated(list(IDENTITY_COLUMNS), keep=False).any():
        raise RuntimeError("walk-forward shard predictions contain duplicate identities")
    combined = combined.sort_values(
        ["fold", "trade_date", "signal_slot", "ts_code"],
        kind="stable",
    ).reset_index(drop=True)
    combined, policy_audit, final_selection = apply_nested_oos_policies(
        combined,
        config,
    )
    all_oos_predictions = combined
    combined = evaluation_window(all_oos_predictions, config)
    candidates = first_crossing_candidates(combined, config)
    metrics = evaluate_predictions(combined, candidates, config)
    metrics["evaluation_contract"] = evaluation_contract_summary(
        all_oos_predictions,
        combined,
        config,
    )
    metrics["nested_policy"] = {
        "folds": policy_audit,
        "final": final_selection.as_dict(),
    }
    return BacktestResult(
        folds=[fold_records[number] for number in sorted(fold_records)],
        metrics=metrics,
        predictions=combined,
        candidates=candidates,
        policy_audit=policy_audit,
        final_policy=final_selection.as_dict(),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
