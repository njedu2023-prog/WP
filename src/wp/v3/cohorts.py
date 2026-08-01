from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .contracts import V3Config
from .v40 import V40Policy, select_v40_cohorts


QUALIFIED = "QUALIFIED"
OBSERVATION = "OBSERVATION"

HARD_GATE_COLUMNS = (
    "passes_execution",
    "passes_freshness",
)

RESEARCH_GATE_COLUMNS = (
    "passes_entry_fill_probability",
    "passes_exit_fill_probability",
    "passes_round_trip_fill_probability",
    "passes_probability",
    "passes_probability_lower",
    "passes_conditional_probability",
    "passes_severe_loss",
    "passes_selection_rank",
    "passes_expected_utility",
    "passes_expected_utility_lower",
    "passes_downside",
    "passes_stability",
    "passes_prior_oos_evidence",
)

V40_GATE_COLUMNS = (
    "passes_execution",
    "passes_freshness",
    "passes_meta_probability",
    "passes_meta_expected_return",
    "passes_meta_severe_loss",
    "passes_round_trip_fill",
    "passes_meta_rank",
    "passes_exit_failure_rank",
)

GATE_LABELS = {
    "passes_execution": "不可成交",
    "passes_freshness": "数据过期",
    "passes_entry_fill_probability": "买入成交率不足",
    "passes_exit_fill_probability": "卖出成交率不足",
    "passes_round_trip_fill_probability": "完整成交率不足",
    "passes_probability": "盈利概率不足",
    "passes_probability_lower": "盈利概率下界不足",
    "passes_conditional_probability": "成交后盈利概率不足",
    "passes_severe_loss": "大亏风险过高",
    "passes_selection_rank": "横截面排名不足",
    "passes_expected_utility": "期望收益不足",
    "passes_expected_utility_lower": "期望收益下界不足",
    "passes_downside": "下行分位过差",
    "passes_stability": "模型稳定性不足",
    "passes_prior_oos_evidence": "历史样本外证据不足",
    "passes_meta_probability": "元模型盈利概率不足",
    "passes_meta_expected_return": "元模型期望收益不足",
    "passes_meta_severe_loss": "元模型大亏风险过高",
    "passes_round_trip_fill": "完整成交率不足",
    "passes_meta_rank": "元模型横截面排名不足",
    "passes_exit_failure_rank": "次日退出风险过高",
}


@dataclass(frozen=True)
class CohortSelection:
    qualified: pd.DataFrame
    observations: pd.DataFrame
    observation_target_count: int
    observation_selection_status: str
    observation_shortfall_reason: str | None


def select_live_cohorts(
    predictions: pd.DataFrame,
    config: V3Config,
) -> CohortSelection:
    frame = predictions.copy()
    if frame.empty:
        return _empty_selection(
            frame,
            config,
            status="NO_PREDICTIONS",
            reason="prediction_frame_empty",
        )

    frame["ts_code"] = frame.get(
        "ts_code",
        pd.Series("", index=frame.index),
    ).astype(str)
    passes = _bool_column(frame, "passes_policy")
    frame["candidate_cohort"] = ""
    v40 = _is_v40_frame(frame)
    gate_columns = V40_GATE_COLUMNS if v40 else (
        *HARD_GATE_COLUMNS,
        *RESEARCH_GATE_COLUMNS,
    )
    frame["failed_gates"] = [
        _failed_gates(row, gate_columns)
        for _, row in frame.iterrows()
    ]
    frame["failed_gate_labels"] = frame["failed_gates"].map(
        lambda value: "、".join(
            GATE_LABELS.get(item, item)
            for item in str(value).split("|")
            if item
        )
    )
    frame["failed_gate_count"] = frame["failed_gates"].map(
        lambda value: len([item for item in str(value).split("|") if item])
    )

    if v40:
        qualified, observations, _ = select_v40_cohorts(
            frame,
            V40Policy(
                observation_count=config.strategy.observation_count
            ),
        )
        observations["candidate_state"] = "RESEARCH_OBSERVATION"
        shortfall = config.strategy.observation_count - len(observations)
        return CohortSelection(
            qualified=qualified,
            observations=observations,
            observation_target_count=config.strategy.observation_count,
            observation_selection_status=(
                "COMPLETE"
                if shortfall == 0
                else "INSUFFICIENT_ELIGIBLE_POOL"
            ),
            observation_shortfall_reason=(
                None
                if shortfall == 0
                else f"eligible_nonqualified_rows={len(observations)}"
            ),
        )

    qualified = frame.loc[passes].copy()
    qualified["candidate_cohort"] = QUALIFIED
    qualified["cohort_rank"] = _rank_series(qualified.index)

    hard_eligible = pd.Series(True, index=frame.index, dtype=bool)
    for gate in HARD_GATE_COLUMNS:
        if gate in frame:
            hard_eligible &= _bool_column(frame, gate)
    signal_price = pd.to_numeric(
        frame.get("signal_price", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    )
    hard_eligible &= frame["ts_code"].ne("") & signal_price.gt(0)
    model_ready = _model_ready(frame)
    pool = frame.loc[~passes & hard_eligible].copy()
    if not model_ready:
        return CohortSelection(
            qualified=qualified,
            observations=pool.iloc[0:0].copy(),
            observation_target_count=config.strategy.observation_count,
            observation_selection_status="MODEL_NOT_READY",
            observation_shortfall_reason="ranking_scores_unavailable",
        )

    sort_columns = [
        "failed_gate_count",
        "_rank_probability_lower",
        "_rank_utility_lower",
        "_rank_selection_score",
        "ts_code",
    ]
    pool["_rank_probability_lower"] = -_numeric(
        pool,
        "p_net_positive_lower",
    )
    pool["_rank_utility_lower"] = -_numeric(
        pool,
        "expected_utility_lower_pct",
    )
    pool["_rank_selection_score"] = -_numeric(pool, "selection_score")
    pool = pool.sort_values(
        sort_columns,
        ascending=[True, True, True, True, True],
        kind="stable",
        na_position="last",
    )
    observations = pool.head(config.strategy.observation_count).copy()
    observations["candidate_cohort"] = OBSERVATION
    observations["candidate_state"] = "RESEARCH_OBSERVATION"
    observations["cohort_rank"] = _rank_series(observations.index)
    observations = observations.drop(
        columns=[
            "_rank_probability_lower",
            "_rank_utility_lower",
            "_rank_selection_score",
        ],
        errors="ignore",
    )

    shortfall = config.strategy.observation_count - len(observations)
    return CohortSelection(
        qualified=qualified,
        observations=observations,
        observation_target_count=config.strategy.observation_count,
        observation_selection_status=(
            "COMPLETE" if shortfall == 0 else "INSUFFICIENT_ELIGIBLE_POOL"
        ),
        observation_shortfall_reason=(
            None
            if shortfall == 0
            else f"eligible_nonqualified_rows={len(pool)}"
        ),
    )


def attach_cohort_labels(
    predictions: pd.DataFrame,
    selection: CohortSelection,
) -> pd.DataFrame:
    result = predictions.copy()
    result["candidate_cohort"] = ""
    result["cohort_rank"] = pd.NA
    result["failed_gates"] = ""
    result["failed_gate_labels"] = ""
    result["failed_gate_count"] = pd.NA
    for cohort in (selection.qualified, selection.observations):
        if cohort.empty:
            continue
        for column in (
            "candidate_cohort",
            "cohort_rank",
            "failed_gates",
            "failed_gate_labels",
            "failed_gate_count",
        ):
            if column in cohort:
                result.loc[cohort.index, column] = cohort[column]
    return result


def _failed_gates(
    row: pd.Series,
    gate_columns: tuple[str, ...],
) -> str:
    failed = []
    for gate in gate_columns:
        if gate in row.index and not _as_bool(row.get(gate)):
            failed.append(gate)
    return "|".join(failed)


def _is_v40_frame(frame: pd.DataFrame) -> bool:
    return {
        "meta_p_positive",
        "meta_expected_net_return_pct",
        "meta_p_severe_loss",
        "p_round_trip_fill_lower",
        "meta_rank_pct",
        "risk_failure_rank_pct",
        "failed_gate_count",
        "failed_gate_distance",
        "meta_score",
    }.issubset(frame.columns)


def _model_ready(frame: pd.DataFrame) -> bool:
    fingerprints = frame.get(
        "model_fingerprint",
        pd.Series("", index=frame.index),
    ).astype(str)
    if fingerprints.str.strip().ne("").any():
        return True
    score_columns = (
        "p_net_positive_lower",
        "expected_utility_lower_pct",
        "selection_score",
    )
    return any(
        pd.to_numeric(frame.get(column), errors="coerce").notna().any()
        for column in score_columns
        if column in frame
    )


def _empty_selection(
    frame: pd.DataFrame,
    config: V3Config,
    *,
    status: str,
    reason: str,
) -> CohortSelection:
    empty = frame.iloc[0:0].copy()
    return CohortSelection(
        qualified=empty,
        observations=empty.copy(),
        observation_target_count=config.strategy.observation_count,
        observation_selection_status=status,
        observation_shortfall_reason=reason,
    )


def _bool_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[name].map(_as_bool).fillna(False).astype(bool)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    try:
        return bool(value) if pd.notna(value) else False
    except (TypeError, ValueError):
        return False


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(
        frame.get(name, pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    )


def _rank_series(index: pd.Index) -> pd.Series:
    return pd.Series(
        range(1, len(index) + 1),
        index=index,
        dtype="Int64",
    )
