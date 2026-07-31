from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .io import file_sha256
from .meta_alpha import IDENTITY_COLUMNS
from .sharding import (
    SHARD_MANIFEST_NAME,
    SHARD_PREDICTIONS_NAME,
    SHARD_SCHEMA_VERSION,
)
from .v19_recall import build_recall_frontier
from .v20_opportunity import V20_FEATURES, build_opportunity_leaders
from .v23_data import (
    OPTIONAL_SOURCE_SELECTION_COLUMNS,
    REQUIRED_SOURCE_SELECTION_COLUMNS,
    SOURCE_SELECTION_COLUMNS,
    assemble_v23_feature_frame,
)


SCHEMA_VERSION = "wp_v24_point_in_time_features_1"
SOURCE_SCHEMA_VERSION = "wp_v24_v9_top5_source_1"
SOURCE_CANDIDATES_PER_SLOT = 5

FORBIDDEN_SOURCE_TOKENS = (
    "target",
    "label",
    "truth",
    "future",
    "gross_return",
    "net_return",
    "t1_",
    "exit_",
)

V24_DERIVED_SOURCE_FEATURE_COLUMNS = tuple(
    column
    for column in V20_FEATURES
    if column not in SOURCE_SELECTION_COLUMNS
)


def load_v24_source_candidates(
    shard_dir: str | Path,
    *,
    evaluation_end: str,
    top_per_source: int,
    exploration_per_slot: int,
    candidates_per_slot: int = SOURCE_CANDIDATES_PER_SLOT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select a fixed top-K opportunity set without reading return outcomes."""
    if candidates_per_slot != SOURCE_CANDIDATES_PER_SLOT:
        raise ValueError(
            "V24 candidates_per_slot is preregistered at "
            f"{SOURCE_CANDIDATES_PER_SLOT}"
        )
    root = Path(shard_dir)
    manifests = sorted(root.rglob(SHARD_MANIFEST_NAME))
    if not manifests:
        raise FileNotFoundError(f"no V9 shard manifests under {root}")
    contaminated = [
        column
        for column in SOURCE_SELECTION_COLUMNS
        if any(token in column.lower() for token in FORBIDDEN_SOURCE_TOKENS)
    ]
    if contaminated:
        raise RuntimeError(
            f"V24 source projection contains outcomes: {contaminated}"
        )

    frames: list[pd.DataFrame] = []
    source_rows = 0
    expected_folds: set[int] = set()
    produced_folds: set[int] = set()
    source_shards: list[dict[str, Any]] = []
    dataset_digests: set[str] = set()
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SHARD_SCHEMA_VERSION:
            raise RuntimeError(f"invalid V9 shard schema: {manifest_path}")
        expected = {int(value) for value in manifest["expected_folds"]}
        produced = {int(value) for value in manifest["produced_folds"]}
        if not expected or produced != expected:
            raise RuntimeError(
                f"V9 shard fold mismatch: expected={expected} produced={produced}"
            )
        expected_folds.update(expected)
        dataset_digests.add(
            str(manifest.get("dataset_manifest_sha256") or "")
        )

        prediction_path = manifest_path.with_name(SHARD_PREDICTIONS_NAME)
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        actual_sha = file_sha256(prediction_path)
        if actual_sha != str(manifest.get("prediction_sha256") or ""):
            raise RuntimeError(
                f"V9 prediction digest mismatch: {prediction_path}"
            )
        available = set(pq.read_schema(prediction_path).names)
        missing_required = sorted(
            set(REQUIRED_SOURCE_SELECTION_COLUMNS) - available
        )
        if missing_required:
            raise RuntimeError(
                "V9 source projection missing required columns "
                f"{missing_required}: {prediction_path}"
            )
        projected = [
            column
            for column in SOURCE_SELECTION_COLUMNS
            if column in available
        ]
        frame = pq.read_table(
            prediction_path,
            columns=projected,
        ).to_pandas()
        missing_optional = sorted(
            set(OPTIONAL_SOURCE_SELECTION_COLUMNS) - available
        )
        for column in missing_optional:
            frame[column] = np.nan
        if len(frame) != int(manifest.get("prediction_rows", -1)):
            raise RuntimeError(
                f"V9 prediction row count mismatch: {prediction_path}"
            )
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame = frame.loc[
            frame["trade_date"].le(str(evaluation_end))
        ].copy()
        folds = {
            int(value)
            for value in pd.to_numeric(frame["fold"], errors="coerce")
            .dropna()
            .astype(int)
        }
        overlap = produced_folds.intersection(folds)
        if overlap:
            raise RuntimeError(f"duplicate V9 folds: {sorted(overlap)}")
        produced_folds.update(folds)
        source_rows += len(frame)

        frontier = build_recall_frontier(
            frame,
            top_per_source=top_per_source,
            exploration_per_slot=exploration_per_slot,
            require_label=False,
        )
        candidates = build_opportunity_leaders(
            frontier,
            leaders_per_slot=candidates_per_slot,
        )
        frames.append(candidates)
        source_shards.append(
            {
                "manifest": str(manifest_path.relative_to(root)),
                "prediction_sha256": actual_sha,
                "folds": sorted(folds),
                "source_rows": int(len(frame)),
                "frontier_rows": int(len(frontier)),
                "candidate_rows": int(len(candidates)),
                "missing_optional_columns": missing_optional,
            }
        )

    if not frames:
        raise RuntimeError("V24 source contains no candidate rows")
    if produced_folds != expected_folds:
        raise RuntimeError(
            "V24 source folds incomplete: "
            f"produced={sorted(produced_folds)} "
            f"expected={sorted(expected_folds)}"
        )
    if len(dataset_digests - {""}) != 1 or "" in dataset_digests:
        raise RuntimeError("V24 source dataset digest is inconsistent")

    result = pd.concat(frames, ignore_index=True)
    result.sort_values(["fold", *IDENTITY_COLUMNS], kind="stable", inplace=True)
    result.reset_index(drop=True, inplace=True)
    if result.duplicated(list(IDENTITY_COLUMNS), keep=False).any():
        raise RuntimeError("V24 source candidates contain duplicate identities")
    rank = pd.to_numeric(
        result["v20_stock_rank_in_slot"],
        errors="coerce",
    )
    if rank.isna().any() or not rank.between(
        1,
        SOURCE_CANDIDATES_PER_SLOT,
    ).all():
        raise RuntimeError("V24 source candidate ranks violate top-K contract")
    maximum_slot_rows = int(
        result.groupby(["trade_date", "signal_slot"]).size().max()
    )
    if maximum_slot_rows > SOURCE_CANDIDATES_PER_SLOT:
        raise RuntimeError("V24 source exceeds the fixed per-slot maximum")

    retained = (
        *IDENTITY_COLUMNS,
        "fold",
        "signal_price",
        "ret_from_prev_close_pct",
        "selection_score",
        "model_version",
        "model_fingerprint",
        "policy_fingerprint",
        *V24_DERIVED_SOURCE_FEATURE_COLUMNS,
    )
    result = result.reindex(columns=tuple(dict.fromkeys(retained)))
    return result, {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "profit_outcomes_read": False,
        "source_rows": int(source_rows),
        "candidate_rows": int(len(result)),
        "candidates_per_slot": SOURCE_CANDIDATES_PER_SLOT,
        "folds": sorted(produced_folds),
        "dataset_manifest_sha256": next(
            iter(dataset_digests - {""})
        ),
        "shards": source_shards,
        "source_integrity": True,
    }


def assemble_v24_feature_frame(
    candidates: pd.DataFrame,
    minute_features: pd.DataFrame,
    auction_features: pd.DataFrame,
    moneyflow_features: pd.DataFrame,
) -> pd.DataFrame:
    microstructure = assemble_v23_feature_frame(
        candidates,
        minute_features,
        auction_features,
        moneyflow_features,
    )
    derived = candidates.reindex(
        columns=(
            *IDENTITY_COLUMNS,
            *V24_DERIVED_SOURCE_FEATURE_COLUMNS,
        )
    )
    result = microstructure.merge(
        derived,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    result.sort_values(["fold", *IDENTITY_COLUMNS], kind="stable", inplace=True)
    result.reset_index(drop=True, inplace=True)
    if result.duplicated(list(IDENTITY_COLUMNS), keep=False).any():
        raise RuntimeError("V24 feature frame contains duplicate identities")
    return result
