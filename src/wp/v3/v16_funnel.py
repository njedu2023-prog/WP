from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .forward_risk import SAFE_RISK_RANK_MAX, frozen_meta_policy


@dataclass(frozen=True)
class FunnelStage:
    stage_id: str
    label: str
    predicate: Callable[[pd.DataFrame], pd.Series]


def v15_funnel_stages() -> tuple[FunnelStage, ...]:
    policy = frozen_meta_policy()
    return (
        FunnelStage(
            "early_slot",
            "14:20-14:35",
            lambda frame: frame["signal_slot"].astype(str).isin(
                {"14:20", "14:25", "14:30", "14:35"}
            ),
        ),
        FunnelStage(
            "positive_probability",
            f"meta_p_positive >= {policy.probability_min:.2f}",
            lambda frame: _numeric(frame, "meta_p_positive").ge(
                policy.probability_min
            ),
        ),
        FunnelStage(
            "positive_expectancy",
            (
                "meta_expected_net_return_pct >= "
                f"{policy.expected_return_min_pct:.2f}%"
            ),
            lambda frame: _numeric(
                frame,
                "meta_expected_net_return_pct",
            ).ge(policy.expected_return_min_pct),
        ),
        FunnelStage(
            "severe_loss_risk",
            f"meta_p_severe_loss <= {policy.severe_loss_max:.2f}",
            lambda frame: _numeric(frame, "meta_p_severe_loss").le(
                policy.severe_loss_max
            ),
        ),
        FunnelStage(
            "round_trip_fill",
            f"p_round_trip_fill_lower >= {policy.round_trip_fill_min:.2f}",
            lambda frame: _numeric(
                frame,
                "p_round_trip_fill_lower",
            ).ge(policy.round_trip_fill_min),
        ),
        FunnelStage(
            "meta_rank",
            f"meta_rank_pct >= {policy.meta_rank_min:.2f}",
            lambda frame: _numeric(frame, "meta_rank_pct").ge(
                policy.meta_rank_min
            ),
        ),
        FunnelStage(
            "exit_failure_safe_half",
            f"risk_failure_rank_pct <= {SAFE_RISK_RANK_MAX:.2f}",
            lambda frame: _numeric(
                frame,
                "risk_failure_rank_pct",
            ).le(SAFE_RISK_RANK_MAX),
        ),
    )


def build_funnel(
    frame: pd.DataFrame,
    *,
    stages: tuple[FunnelStage, ...] | None = None,
    max_candidates_per_day: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return stage-level counts and row-level first-rejection attribution."""
    required = {"trade_date", "signal_slot", "ts_code"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"funnel frame missing identity columns: {missing}")
    ordered_stages = stages or v15_funnel_stages()
    result = frame.copy()
    result["trade_date"] = result["trade_date"].astype(str)
    result["signal_slot"] = result["signal_slot"].astype(str)
    result["ts_code"] = result["ts_code"].astype(str)
    result["_funnel_pass"] = True
    result["first_rejection_stage"] = pd.Series(
        "passed_thresholds",
        index=result.index,
        dtype="string",
    )
    rows = [
        _stage_summary(
            result,
            stage_id="source",
            label="immutable scored frontier",
            pass_mask=result["_funnel_pass"],
            rejected_at_stage=0,
        )
    ]
    for stage in ordered_stages:
        current = result["_funnel_pass"].fillna(False).astype(bool)
        stage_pass = stage.predicate(result).fillna(False).astype(bool)
        rejected = current & ~stage_pass
        result.loc[rejected, "first_rejection_stage"] = stage.stage_id
        result["_funnel_pass"] = current & stage_pass
        rows.append(
            _stage_summary(
                result,
                stage_id=stage.stage_id,
                label=stage.label,
                pass_mask=result["_funnel_pass"],
                rejected_at_stage=int(rejected.sum()),
            )
        )

    if max_candidates_per_day is not None:
        if max_candidates_per_day < 1:
            raise ValueError("max_candidates_per_day must be positive")
        passed = result.loc[result["_funnel_pass"]].copy()
        passed["_slot_minute"] = _slot_minute(passed["signal_slot"])
        passed["_score"] = _numeric(passed, "meta_score")
        passed.sort_values(
            ["trade_date", "_slot_minute", "_score", "ts_code"],
            ascending=[True, True, False, True],
            kind="stable",
            inplace=True,
        )
        kept: list[int] = []
        for _, day in passed.groupby("trade_date", sort=False):
            seen: set[str] = set()
            for index, row in day.iterrows():
                code = str(row["ts_code"])
                if code in seen:
                    continue
                seen.add(code)
                kept.append(index)
                if len(seen) >= max_candidates_per_day:
                    break
        keep_mask = result.index.isin(kept)
        rejected = result["_funnel_pass"] & ~keep_mask
        result.loc[rejected, "first_rejection_stage"] = "daily_dedup_top_k"
        result["_funnel_pass"] = result["_funnel_pass"] & keep_mask
        rows.append(
            _stage_summary(
                result,
                stage_id="daily_dedup_top_k",
                label=f"first occurrence, maximum {max_candidates_per_day}/day",
                pass_mask=result["_funnel_pass"],
                rejected_at_stage=int(rejected.sum()),
            )
        )
    result["funnel_selected"] = result["_funnel_pass"].astype(bool)
    result.drop(columns="_funnel_pass", inplace=True)
    summary = pd.DataFrame(rows)
    source_rows = max(int(summary.iloc[0]["rows_remaining"]), 1)
    source_days = max(int(summary.iloc[0]["days_remaining"]), 1)
    summary["row_retention"] = summary["rows_remaining"] / source_rows
    summary["day_retention"] = summary["days_remaining"] / source_days
    return summary, result


def rejection_summary(attributed: pd.DataFrame) -> pd.DataFrame:
    required = {"first_rejection_stage", "trade_date", "ts_code"}
    missing = sorted(required - set(attributed.columns))
    if missing:
        raise ValueError(f"attributed frame missing columns: {missing}")
    return (
        attributed.groupby("first_rejection_stage", dropna=False, sort=False)
        .agg(
            rows=("ts_code", "size"),
            trade_days=("trade_date", "nunique"),
            symbols=("ts_code", "nunique"),
        )
        .reset_index()
        .sort_values(["rows", "first_rejection_stage"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _stage_summary(
    frame: pd.DataFrame,
    *,
    stage_id: str,
    label: str,
    pass_mask: pd.Series,
    rejected_at_stage: int,
) -> dict[str, object]:
    remaining = frame.loc[pass_mask.fillna(False).astype(bool)]
    return {
        "stage_id": stage_id,
        "label": label,
        "rows_remaining": int(len(remaining)),
        "days_remaining": int(remaining["trade_date"].nunique()),
        "symbols_remaining": int(remaining["ts_code"].nunique()),
        "rejected_at_stage": int(rejected_at_stage),
    }


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(float("nan"), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _slot_minute(values: pd.Series) -> pd.Series:
    parsed = values.astype(str).str.extract(
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2})"
    )
    return (
        pd.to_numeric(parsed["hour"], errors="coerce") * 60
        + pd.to_numeric(parsed["minute"], errors="coerce")
    )
