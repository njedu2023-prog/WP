from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .meta_alpha import IDENTITY_COLUMNS
from .v17_selector import SelectorBundle, fit_selector
from .v19_recall import FULL_CONTEXT_FEATURES, RETRIEVAL_FEATURES


SCHEMA_VERSION = "wp_v20_hierarchical_opportunity_1"
DEFAULT_LEADERS_PER_SLOT = 3
GATE_TRAIN_DAYS = 126
GATE_CALIBRATION_DAYS = 42
GATE_PURGE_DAYS = 2
FIXED_TARGET_CANDIDATE_DAY_RATE = 0.18
FIXED_MAX_CANDIDATES_PER_DAY = 2

STOCK_FEATURES = (
    "p_net_positive",
    "p_net_positive_lower",
    "p_conditional_net_positive",
    "p_cross_section_top",
    "p_severe_loss",
    "p_round_trip_fill_lower",
    "probability_model_spread",
    "fill_probability_model_spread",
    "expected_utility_pct",
    "expected_utility_lower_pct",
    "expected_return_model_spread",
    "downside_q10_pct",
    "selection_score",
    "selection_rank_pct",
    "selection_rank_spread",
    "ret_from_prev_close_pct",
)

LEADER_FEATURES = (
    "v20_stock_score",
    "v20_stock_rank_in_slot",
    "v20_stock_score_gap_to_top",
    "v20_slot_frontier_size",
    "v20_slot_score_max",
    "v20_slot_score_median",
    "v20_slot_score_std",
    "v20_slot_score_q90",
    "v20_slot_top_score_margin",
    "v20_slot_probability_max",
    "v20_slot_probability_median",
    "v20_slot_expected_utility_max",
    "v20_slot_expected_utility_median",
    "v20_slot_severe_risk_min",
    "v20_leader_appearances_so_far",
    "v20_stock_score_mean_so_far",
    "v20_stock_score_delta_1",
    "v20_stock_best_rank_so_far",
)

V20_FEATURES = tuple(
    dict.fromkeys(
        (
            *STOCK_FEATURES,
            *FULL_CONTEXT_FEATURES,
            *RETRIEVAL_FEATURES,
            *LEADER_FEATURES,
        )
    )
)

GATE_OUTPUT_COLUMNS = {
    "selector_p_positive": "v20_gate_p_positive",
    "selector_p_positive_lower": "v20_gate_p_positive_lower",
    "selector_probability_spread": "v20_gate_probability_spread",
    "selector_expected_net_return_pct": (
        "v20_gate_expected_net_return_pct"
    ),
    "selector_return_q25_pct": "v20_gate_return_q25_pct",
    "selector_score": "v20_gate_score",
    "selector_score_rank_pct": "v20_gate_score_rank_pct",
}


@dataclass
class OpportunityGateBundle:
    selector: SelectorBundle
    source_train_rows: int
    source_calibration_rows: int

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.selector.feature_columns

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        scored = self.selector.predict(frame)
        return scored.rename(columns=GATE_OUTPUT_COLUMNS)


@dataclass(frozen=True)
class OpportunityPolicySpec:
    target_candidate_day_rate: float = FIXED_TARGET_CANDIDATE_DAY_RATE
    max_candidates_per_day: int = FIXED_MAX_CANDIDATES_PER_DAY

    @property
    def policy_id(self) -> str:
        return (
            f"fixed-rate{self.target_candidate_day_rate:.2f}-"
            f"k{self.max_candidates_per_day}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "target_candidate_day_rate": self.target_candidate_day_rate,
            "max_candidates_per_day": self.max_candidates_per_day,
        }


@dataclass(frozen=True)
class FrozenOpportunityPolicy:
    spec: OpportunityPolicySpec
    score_threshold: float
    threshold_calibration_start: str
    threshold_calibration_end: str
    threshold_calibration_days: int
    threshold_eligible_days: int

    @property
    def policy_id(self) -> str:
        return self.spec.policy_id

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.spec.as_dict(),
            "score_threshold": self.score_threshold,
            "threshold_calibration_start": self.threshold_calibration_start,
            "threshold_calibration_end": self.threshold_calibration_end,
            "threshold_calibration_days": self.threshold_calibration_days,
            "threshold_eligible_days": self.threshold_eligible_days,
        }


def build_opportunity_leaders(
    frame: pd.DataFrame,
    *,
    leaders_per_slot: int = DEFAULT_LEADERS_PER_SLOT,
) -> pd.DataFrame:
    if leaders_per_slot < 1:
        raise ValueError("leaders_per_slot must be positive")
    required = {*IDENTITY_COLUMNS, "selection_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V20 leader frame missing columns: {missing}")
    if frame.empty:
        return frame.copy()

    prepared = frame.copy().reset_index(drop=True)
    keys = ["trade_date", "signal_slot"]
    prepared["v20_stock_score"] = _numeric(
        prepared,
        "selection_score",
    )
    prepared.sort_values(
        [*keys, "v20_stock_score", "ts_code"],
        ascending=[True, True, False, True],
        kind="stable",
        inplace=True,
    )
    grouped = prepared.groupby(keys, sort=False)
    prepared["v20_stock_rank_in_slot"] = grouped.cumcount() + 1
    prepared["v20_slot_frontier_size"] = grouped["ts_code"].transform(
        "size"
    )
    score_group = grouped["v20_stock_score"]
    prepared["v20_slot_score_max"] = score_group.transform("max")
    prepared["v20_slot_score_median"] = score_group.transform("median")
    prepared["v20_slot_score_std"] = score_group.transform("std").fillna(0.0)
    prepared["v20_slot_score_q90"] = score_group.transform(
        lambda values: values.quantile(0.90)
    )
    prepared["v20_stock_score_gap_to_top"] = (
        prepared["v20_slot_score_max"] - prepared["v20_stock_score"]
    )

    second_scores = (
        prepared.loc[
            prepared["v20_stock_rank_in_slot"].eq(2),
            [*keys, "v20_stock_score"],
        ]
        .set_index(keys)["v20_stock_score"]
    )
    group_index = pd.MultiIndex.from_frame(prepared.loc[:, keys])
    prepared["v20_slot_top_score_margin"] = (
        prepared["v20_slot_score_max"]
        - pd.Series(group_index.map(second_scores), index=prepared.index)
    ).fillna(0.0)

    prepared["v20_slot_probability_max"] = _group_transform(
        prepared,
        keys,
        "p_net_positive_lower",
        "max",
    )
    prepared["v20_slot_probability_median"] = _group_transform(
        prepared,
        keys,
        "p_net_positive_lower",
        "median",
    )
    prepared["v20_slot_expected_utility_max"] = _group_transform(
        prepared,
        keys,
        "expected_utility_lower_pct",
        "max",
    )
    prepared["v20_slot_expected_utility_median"] = _group_transform(
        prepared,
        keys,
        "expected_utility_lower_pct",
        "median",
    )
    prepared["v20_slot_severe_risk_min"] = _group_transform(
        prepared,
        keys,
        "p_severe_loss",
        "min",
    )

    leaders = prepared.loc[
        prepared["v20_stock_rank_in_slot"].le(leaders_per_slot)
    ].copy()
    leaders["_v20_slot_absolute"] = _slot_absolute(
        leaders["signal_slot"]
    )
    leaders.sort_values(
        [
            "trade_date",
            "_v20_slot_absolute",
            "v20_stock_rank_in_slot",
            "ts_code",
        ],
        kind="stable",
        inplace=True,
    )
    stock_keys = [leaders["trade_date"], leaders["ts_code"]]
    stock_group = leaders.groupby(["trade_date", "ts_code"], sort=False)
    leaders["v20_leader_appearances_so_far"] = stock_group.cumcount() + 1
    leaders["v20_stock_score_mean_so_far"] = stock_group[
        "v20_stock_score"
    ].transform(
        lambda values: values.expanding(min_periods=1).mean()
    )
    leaders["v20_stock_score_delta_1"] = stock_group[
        "v20_stock_score"
    ].diff()
    leaders["v20_stock_best_rank_so_far"] = leaders[
        "v20_stock_rank_in_slot"
    ].groupby(stock_keys, sort=False).cummin()
    return (
        leaders.drop(columns="_v20_slot_absolute")
        .sort_values(list(IDENTITY_COLUMNS), kind="stable")
        .reset_index(drop=True)
    )


def fit_opportunity_gate(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    random_seed: int,
    minimum_train_rows: int = 1_500,
    minimum_calibration_rows: int = 350,
) -> OpportunityGateBundle:
    selector = fit_selector(
        train,
        calibration,
        random_seed=random_seed,
        minimum_train_rows=minimum_train_rows,
        minimum_calibration_rows=minimum_calibration_rows,
        feature_candidates=V20_FEATURES,
    )
    return OpportunityGateBundle(
        selector=selector,
        source_train_rows=int(len(train)),
        source_calibration_rows=int(len(calibration)),
    )


def rolling_opportunity_model_segments(
    prior_dates: Iterable[str],
    *,
    reserve_final_purge: bool = True,
) -> tuple[list[str], list[str]] | None:
    dates = sorted(set(map(str, prior_dates)))
    final_purge = GATE_PURGE_DAYS if reserve_final_purge else 0
    needed = (
        GATE_TRAIN_DAYS
        + GATE_PURGE_DAYS
        + GATE_CALIBRATION_DAYS
        + final_purge
    )
    if len(dates) < needed:
        return None
    selected = dates[-needed:]
    train = selected[:GATE_TRAIN_DAYS]
    calibration_start = GATE_TRAIN_DAYS + GATE_PURGE_DAYS
    calibration = selected[
        calibration_start : calibration_start + GATE_CALIBRATION_DAYS
    ]
    return train, calibration


def calibrate_opportunity_policy(
    scored_calibration: pd.DataFrame,
    *,
    calibration_dates: Iterable[str],
    spec: OpportunityPolicySpec | None = None,
) -> FrozenOpportunityPolicy:
    frozen_spec = spec or OpportunityPolicySpec()
    dates = sorted(set(map(str, calibration_dates)))
    if not dates:
        raise ValueError("V20 policy calibration has no dates")
    calibration = scored_calibration.loc[
        scored_calibration["trade_date"].astype(str).isin(dates)
    ].copy()
    eligible = _base_eligible(calibration)
    daily_max = (
        eligible.groupby("trade_date", sort=False)["v20_gate_score"]
        .max()
        .sort_values(ascending=False)
    )
    target_days = max(
        1,
        int(np.ceil(frozen_spec.target_candidate_day_rate * len(dates))),
    )
    threshold = (
        float("inf")
        if daily_max.empty
        else float(daily_max.iloc[min(target_days, len(daily_max)) - 1])
    )
    return FrozenOpportunityPolicy(
        spec=frozen_spec,
        score_threshold=threshold,
        threshold_calibration_start=dates[0],
        threshold_calibration_end=dates[-1],
        threshold_calibration_days=len(dates),
        threshold_eligible_days=int(len(daily_max)),
    )


def apply_opportunity_policy(
    scored: pd.DataFrame,
    policy: FrozenOpportunityPolicy,
) -> pd.DataFrame:
    eligible = _base_eligible(scored)
    qualified = eligible.loc[
        _numeric(eligible, "v20_gate_score").ge(policy.score_threshold)
    ].copy()
    if qualified.empty:
        qualified["v20_policy_id"] = policy.policy_id
        return qualified
    qualified["_v20_slot_absolute"] = _slot_absolute(
        qualified["signal_slot"]
    )
    qualified.sort_values(
        [
            "trade_date",
            "_v20_slot_absolute",
            "v20_gate_score",
            "v20_stock_score",
            "ts_code",
        ],
        ascending=[True, True, False, False, True],
        kind="stable",
        inplace=True,
    )
    first_signal = qualified.drop_duplicates(
        ["trade_date", "ts_code"],
        keep="first",
    ).copy()
    within_day = first_signal.groupby("trade_date", sort=False).cumcount()
    selected = first_signal.loc[
        within_day.lt(policy.spec.max_candidates_per_day)
    ].drop(columns="_v20_slot_absolute")
    selected["v20_policy_id"] = policy.policy_id
    selected["v20_score_threshold"] = policy.score_threshold
    selected["v20_target_candidate_day_rate"] = (
        policy.spec.target_candidate_day_rate
    )
    return selected.reset_index(drop=True)


def v20_research_readiness(
    metrics: dict[str, Any],
    *,
    yearly: list[dict[str, Any]],
    temporal_integrity: bool,
    source_integrity: bool,
) -> dict[str, Any]:
    active_years = [
        row for row in yearly if int(row.get("events", 0)) > 0
    ]
    positive_years = sum(
        float(row.get("mean_net_return_pct") or -999.0) > 0.0
        for row in active_years
    )
    worst_year = min(
        (
            float(row.get("mean_net_return_pct") or -999.0)
            for row in active_years
        ),
        default=-999.0,
    )
    gates = {
        "minimum_nested_oos_candidates": int(metrics.get("events", 0)) >= 50,
        "minimum_nested_oos_candidate_days": (
            int(metrics.get("candidate_days", 0)) >= 30
        ),
        "practical_candidate_day_rate": (
            0.10 <= float(metrics.get("candidate_day_rate", 0.0)) <= 0.25
        ),
        "minimum_win_rate": float(metrics.get("win_rate", 0.0)) >= 0.55,
        "minimum_wilson_lower": (
            float(metrics.get("win_rate_wilson_lower", 0.0)) >= 0.48
        ),
        "minimum_clustered_win_rate_lower": (
            float(metrics.get("clustered_win_rate_lower", 0.0)) >= 0.48
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
        "minimum_two_positive_calendar_years": positive_years >= 2,
        "worst_calendar_year_above_minus_0_20pct": worst_year >= -0.20,
        "temporal_integrity": bool(temporal_integrity),
        "source_integrity": bool(source_integrity),
    }
    passed = all(gates.values())
    return {
        "all_historical_gates_passed": passed,
        "gates": gates,
        "failed_gates": [
            name for name, gate_passed in gates.items() if not gate_passed
        ],
        "production_authorized": False,
        "future_shadow_days_required": 150,
        "future_shadow_min_candidates": 60,
        "future_shadow_min_candidate_days": 30,
        "reason": (
            "historical_screen_passed_future_shadow_still_required"
            if passed
            else "historical_evidence_insufficient"
        ),
    }


def _base_eligible(scored: pd.DataFrame) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "v20_gate_score",
        "v20_gate_probability_spread",
        "v20_stock_score",
        "p_severe_loss",
        "p_round_trip_fill_lower",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"V20 policy frame missing columns: {missing}")
    age = _numeric(scored, "data_age_seconds")
    fresh = age.isna() | age.le(420.0)
    mask = (
        _numeric(scored, "p_severe_loss").le(0.40)
        & _numeric(scored, "p_round_trip_fill_lower").ge(0.95)
        & _numeric(scored, "v20_gate_probability_spread").le(0.30)
        & fresh
    )
    return scored.loc[mask].copy()


def _group_transform(
    frame: pd.DataFrame,
    keys: list[str],
    column: str,
    operation: str,
) -> pd.Series:
    numeric = _numeric(frame, column)
    return numeric.groupby(
        [frame[key] for key in keys],
        sort=False,
    ).transform(operation)


def _slot_absolute(values: pd.Series) -> pd.Series:
    parsed = values.astype(str).str.extract(
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2})"
    )
    return (
        pd.to_numeric(parsed["hour"], errors="coerce") * 60
        + pd.to_numeric(parsed["minute"], errors="coerce")
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")
