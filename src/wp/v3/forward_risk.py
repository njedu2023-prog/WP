from __future__ import annotations

import pandas as pd

from .meta_alpha import MetaPolicy, apply_meta_policy


DISCOVERY_FOLD = 15
DISCOVERY_END_DATE = "20250527"
FORWARD_VALIDATION_FOLDS = (16, 17, 18, 19, 20, 21, 22)
SAFE_RISK_RANK_MAX = 0.50


def frozen_meta_policy() -> MetaPolicy:
    return MetaPolicy(
        probability_min=0.54,
        expected_return_min_pct=0.00,
        severe_loss_max=0.35,
        round_trip_fill_min=0.95,
        meta_rank_min=0.95,
        max_candidates_per_day=3,
        slot_group="early",
    )


def select_forward_candidates(
    frame: pd.DataFrame,
    *,
    apply_exit_risk_gate: bool,
) -> pd.DataFrame:
    candidates = frame.copy()
    if apply_exit_risk_gate:
        risk_rank = pd.to_numeric(
            candidates.get("risk_failure_rank_pct"),
            errors="coerce",
        )
        candidates = candidates.loc[
            risk_rank.le(SAFE_RISK_RANK_MAX)
        ].copy()
    return apply_meta_policy(candidates, frozen_meta_policy())


def assert_strictly_forward(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("forward validation frame is empty")
    folds = set(
        pd.to_numeric(frame["fold"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    unexpected = sorted(folds - set(FORWARD_VALIDATION_FOLDS))
    if unexpected:
        raise ValueError(
            f"unexpected folds in forward validation: {unexpected}"
        )
    dates = (
        frame["trade_date"]
        .astype(str)
        .str.replace("-", "", regex=False)
    )
    if dates.le(DISCOVERY_END_DATE).any():
        earliest = str(dates.min())
        raise ValueError(
            "forward validation contains discovery-period data: "
            f"{earliest} <= {DISCOVERY_END_DATE}"
        )
