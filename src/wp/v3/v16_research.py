from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .exit_research import materialize_contract
from .features import FEATURE_COLUMNS
from .io import file_sha256
from .meta_alpha import IDENTITY_COLUMNS
from .v16_policy import (
    ExpertPolicy,
    apply_expert_policy,
    benjamini_hochberg,
    expert_policy_grid,
    policy_metrics,
)


SCHEMA_VERSION = "wp_v16_specialist_nested_oos_1"
EXIT_CONTRACT_ID = "t1_close_auction"
MODEL_MAX_TRAIN_DAYS = 504
MODEL_MIN_TRAIN_DAYS = 252
MODEL_CALIBRATION_DAYS = 42
POLICY_DESIGN_DAYS = 84
POLICY_CONFIRMATION_DAYS = 42
PURGE_DAYS = 2


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
    expected = str(summary.get("candidate_frontier_sha256") or "")
    actual = file_sha256(frontier_path)
    if not expected or actual != expected:
        raise RuntimeError(
            f"V11 frontier digest mismatch: {actual} != {expected}"
        )
    frame = pd.read_parquet(frontier_path)
    required = {
        *IDENTITY_COLUMNS,
        "fold",
        "entry_fillable",
        "net_t1_close_auction_pct",
        "exit_t1_close_auction_fillable",
        "p_round_trip_fill_lower",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V11 frontier missing columns: {missing}")
    for column in IDENTITY_COLUMNS:
        frame[column] = frame[column].astype(str)
    duplicates = frame.duplicated(list(IDENTITY_COLUMNS), keep=False)
    if duplicates.any():
        raise RuntimeError(
            f"V11 frontier has {int(duplicates.sum())} duplicate identities"
        )
    return frame, {
        "v11_schema_version": summary.get("schema_version"),
        "v11_frontier_sha256": actual,
        "v11_summary_sha256": file_sha256(summary_path),
        "v11_frontier_rows": int(len(frame)),
    }


def attach_original_features(
    frontier: pd.DataFrame,
    panel_dir: str | Path,
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    result = frontier.copy()
    for column in IDENTITY_COLUMNS:
        result[column] = result[column].astype(str)
    missing_features = [
        column for column in FEATURE_COLUMNS if column not in result
    ]
    if not missing_features:
        return result
    panel_columns = list(
        dict.fromkeys([*IDENTITY_COLUMNS, *missing_features])
    )
    requested = result.loc[:, IDENTITY_COLUMNS].drop_duplicates().copy()
    requested["month"] = (
        requested["trade_date"]
        .astype(str)
        .str.replace("-", "", regex=False)
        .str[:6]
    )
    paths = {
        path.stem.rsplit("_", 1)[-1]: path
        for path in Path(panel_dir).glob("wp_v3_panel_*.parquet")
        if start_date.replace("-", "")[:6]
        <= path.stem.rsplit("_", 1)[-1]
        <= end_date.replace("-", "")[:6]
    }
    feature_frames: list[pd.DataFrame] = []
    for month, month_requested in requested.groupby("month", sort=True):
        path = paths.get(str(month))
        if path is None:
            raise FileNotFoundError(
                f"missing panel feature partition for {month}"
            )
        panel = pd.read_parquet(path, columns=panel_columns)
        for column in IDENTITY_COLUMNS:
            panel[column] = panel[column].astype(str)
        month_identities = month_requested.loc[:, IDENTITY_COLUMNS]
        matched = panel.merge(
            month_identities,
            on=list(IDENTITY_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        if len(matched) != len(month_identities):
            raise RuntimeError(
                f"matched {len(matched)} of {len(month_identities)} "
                f"frontier identities for {month}"
            )
        feature_frames.append(matched)
    if not feature_frames:
        raise RuntimeError("no panel feature partitions matched V11 frontier")
    panel_features = pd.concat(feature_frames, ignore_index=True)
    result = result.merge(
        panel_features,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    matched_features = sum(
        column in result and result[column].notna().any()
        for column in FEATURE_COLUMNS
    )
    if matched_features < 12:
        raise RuntimeError(
            f"only {matched_features} causal features matched V11 frontier"
        )
    return result


def materialize_close_labels(
    frame: pd.DataFrame,
    *,
    severe_loss_threshold_pct: float,
) -> pd.DataFrame:
    result = materialize_contract(frame, EXIT_CONTRACT_ID)
    result["net_return_pct"] = pd.to_numeric(
        result["net_return_pct"],
        errors="coerce",
    )
    result["label_available"] = result["net_return_pct"].notna()
    result["target_net_positive"] = result["net_return_pct"].gt(0).where(
        result["label_available"]
    )
    result["target_severe_loss"] = (
        result["net_return_pct"].le(severe_loss_threshold_pct)
    ).where(result["label_available"])
    return result


def eligible_labeled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "trade_date",
        "net_return_pct",
        "target_net_positive",
        "target_severe_loss",
        "label_available",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"labeled research frame missing columns: {missing}")
    mask = (
        _boolean(frame["label_available"])
        & pd.to_numeric(frame["net_return_pct"], errors="coerce").notna()
        & pd.to_numeric(
            frame["target_net_positive"],
            errors="coerce",
        ).notna()
        & pd.to_numeric(
            frame["target_severe_loss"],
            errors="coerce",
        ).notna()
    )
    return frame.loc[mask].copy()


def rolling_model_segments(
    prior_dates: Iterable[str],
    *,
    reserve_final_purge: bool = True,
) -> tuple[list[str], list[str]] | None:
    dates = sorted(set(map(str, prior_dates)))
    final_purge = PURGE_DAYS if reserve_final_purge else 0
    minimum = (
        MODEL_MIN_TRAIN_DAYS
        + PURGE_DAYS
        + MODEL_CALIBRATION_DAYS
        + final_purge
    )
    if len(dates) < minimum:
        return None
    calibration_end = len(dates) - final_purge
    calibration_start = calibration_end - MODEL_CALIBRATION_DAYS
    train_end = calibration_start - PURGE_DAYS
    train_start = max(0, train_end - MODEL_MAX_TRAIN_DAYS)
    train = dates[train_start:train_end]
    calibration = dates[calibration_start:calibration_end]
    if len(train) < MODEL_MIN_TRAIN_DAYS:
        return None
    return train, calibration


def rolling_policy_segments(
    prior_dates: Iterable[str],
    *,
    reserve_final_purge: bool = True,
) -> tuple[list[str], list[str]] | None:
    dates = sorted(set(map(str, prior_dates)))
    final_purge = PURGE_DAYS if reserve_final_purge else 0
    needed = (
        POLICY_DESIGN_DAYS
        + PURGE_DAYS
        + POLICY_CONFIRMATION_DAYS
        + final_purge
    )
    if len(dates) < needed:
        return None
    selected = dates[-needed:]
    design = selected[:POLICY_DESIGN_DAYS]
    confirmation_start = POLICY_DESIGN_DAYS + PURGE_DAYS
    confirmation = selected[
        confirmation_start : confirmation_start + POLICY_CONFIRMATION_DAYS
    ]
    return design, confirmation


def descriptive_policy_frontier(
    scored: pd.DataFrame,
    *,
    total_days: int,
    policies: Iterable[ExpertPolicy] | None = None,
    seed: int,
    bootstrap_samples: int = 1_000,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grid = tuple(policies or expert_policy_grid())
    for offset, policy in enumerate(grid):
        selected = apply_expert_policy(scored, policy)
        metrics = policy_metrics(
            selected,
            total_days=total_days,
            seed=seed + offset,
            bootstrap_samples=bootstrap_samples,
        )
        rows.append({**policy.as_dict(), **metrics})
    q_values = benjamini_hochberg(
        float(row["mean_return_p_value"]) for row in rows
    )
    for row, q_value in zip(rows, q_values, strict=True):
        row["mean_return_q_value"] = q_value
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["pareto_efficient"] = pareto_mask(result)
    return result.sort_values(
        [
            "pareto_efficient",
            "clustered_mean_lower_pct",
            "candidate_day_rate",
        ],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def pareto_mask(frame: pd.DataFrame) -> pd.Series:
    frequency = pd.to_numeric(
        frame["candidate_day_rate"],
        errors="coerce",
    ).fillna(0.0)
    mean_return = pd.to_numeric(
        frame["mean_net_return_pct"],
        errors="coerce",
    ).fillna(float("-inf"))
    win_rate = pd.to_numeric(
        frame["win_rate"],
        errors="coerce",
    ).fillna(0.0)
    efficient = []
    for index in frame.index:
        dominates = (
            frequency.ge(frequency.loc[index])
            & mean_return.ge(mean_return.loc[index])
            & win_rate.ge(win_rate.loc[index])
            & (
                frequency.gt(frequency.loc[index])
                | mean_return.gt(mean_return.loc[index])
                | win_rate.gt(win_rate.loc[index])
            )
        )
        efficient.append(not bool(dominates.any()))
    return pd.Series(efficient, index=frame.index, dtype=bool)


def research_readiness(
    metrics: dict[str, Any],
    *,
    temporal_integrity: bool,
) -> dict[str, Any]:
    gates = {
        "minimum_nested_oos_candidates": int(metrics.get("events", 0)) >= 250,
        "minimum_nested_oos_candidate_days": (
            int(metrics.get("candidate_days", 0)) >= 50
        ),
        "minimum_win_rate": float(metrics.get("win_rate", 0.0)) >= 0.55,
        "minimum_wilson_lower": (
            float(metrics.get("win_rate_wilson_lower", 0.0)) >= 0.52
        ),
        "minimum_clustered_win_rate_lower": (
            float(metrics.get("clustered_win_rate_lower", 0.0)) >= 0.52
        ),
        "minimum_mean_net_return_pct": (
            float(metrics.get("mean_net_return_pct") or -999.0) >= 0.20
        ),
        "clustered_mean_lower_positive": (
            float(metrics.get("clustered_mean_lower_pct") or -999.0) > 0.0
        ),
        "minimum_profit_factor": (
            float(metrics.get("profit_factor") or 0.0) >= 1.20
        ),
        "real_50bps_stress_nonnegative": (
            float(
                metrics.get("stress_50bps_mean_net_return_pct") or -999.0
            )
            >= 0.0
        ),
        "return_p10_above_minus_3pct": (
            float(metrics.get("return_p10_pct") or -999.0) >= -3.0
        ),
        "temporal_integrity": bool(temporal_integrity),
    }
    return {
        "all_historical_gates_passed": all(gates.values()),
        "gates": gates,
        "failed_gates": [
            name for name, passed in gates.items() if not passed
        ],
        "production_authorized": False,
        "future_shadow_days_required": 150,
        "reason": (
            "historical_gates_passed_future_shadow_still_required"
            if all(gates.values())
            else "historical_evidence_insufficient"
        ),
    }


def _boolean(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y", "t"})
