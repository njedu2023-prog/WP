from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .meta_alpha import IDENTITY_COLUMNS
from .v16_policy import benjamini_hochberg, policy_metrics
from .v17_selector import (
    SELECTOR_FEATURES,
    SelectorBundle,
    fit_selector,
    prepare_selector_frame,
)


SCHEMA_VERSION = "wp_v18_causal_frequency_selector_1"
THRESHOLD_CALIBRATION_DAYS = 42
POLICY_DESIGN_DAYS = 42
POLICY_CONFIRMATION_DAYS = 42
POLICY_PURGE_DAYS = 2

PERSISTENCE_FEATURES = (
    "v18_slots_seen",
    "v18_quality_hits_so_far",
    "v18_probability_hits_so_far",
    "v18_p_positive_delta_1",
    "v18_p_positive_change_from_1420",
    "v18_p_positive_mean_so_far",
    "v18_p_positive_min_so_far",
    "v18_utility_delta_1",
    "v18_utility_change_from_1420",
    "v18_utility_mean_so_far",
    "v18_utility_min_so_far",
    "v18_selection_score_delta_1",
    "v18_selection_score_change_from_1420",
    "v18_selection_score_mean_so_far",
    "v18_selection_score_min_so_far",
    "v18_selection_score_stability_so_far",
    "v18_return_delta_1",
    "v18_return_change_from_1420",
    "v18_return_mean_so_far",
    "v18_return_min_so_far",
    "v18_severe_loss_delta_1",
    "v18_severe_loss_max_so_far",
)
V18_FEATURES = tuple(dict.fromkeys((*SELECTOR_FEATURES, *PERSISTENCE_FEATURES)))


@dataclass
class RankedSelectorBundle:
    selector: SelectorBundle

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.selector.feature_columns

    @property
    def train_rows(self) -> int:
        return self.selector.train_rows

    @property
    def calibration_rows(self) -> int:
        return self.selector.calibration_rows

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.selector.predict(prepare_ranked_frame(frame))


@dataclass(frozen=True)
class RankedPolicySpec:
    target_candidate_day_rate: float
    max_candidates_per_day: int
    minimum_quality_hits: int

    @property
    def spec_id(self) -> str:
        return (
            f"rate{self.target_candidate_day_rate:.2f}-"
            f"k{self.max_candidates_per_day}-"
            f"hits{self.minimum_quality_hits}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "target_candidate_day_rate": self.target_candidate_day_rate,
            "max_candidates_per_day": self.max_candidates_per_day,
            "minimum_quality_hits": self.minimum_quality_hits,
        }


@dataclass(frozen=True)
class FrozenRankedPolicy:
    spec: RankedPolicySpec
    score_threshold: float
    threshold_calibration_start: str
    threshold_calibration_end: str
    threshold_calibration_days: int
    threshold_eligible_days: int

    @property
    def policy_id(self) -> str:
        return self.spec.spec_id

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.spec.as_dict(),
            "policy_id": self.policy_id,
            "score_threshold": self.score_threshold,
            "threshold_calibration_start": self.threshold_calibration_start,
            "threshold_calibration_end": self.threshold_calibration_end,
            "threshold_calibration_days": self.threshold_calibration_days,
            "threshold_eligible_days": self.threshold_eligible_days,
        }


@dataclass(frozen=True)
class RankedPolicySelection:
    policy: FrozenRankedPolicy | None
    threshold_calibration: dict[str, Any]
    design: dict[str, Any]
    confirmation: dict[str, Any]
    policies_evaluated: int
    design_passed: int
    confirmation_passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.as_dict() if self.policy else None,
            "threshold_calibration": self.threshold_calibration,
            "design": self.design,
            "confirmation": self.confirmation,
            "search": {
                "policies_evaluated": self.policies_evaluated,
                "design_passed": self.design_passed,
                "confirmation_passed": self.confirmation_passed,
            },
        }


def ranked_policy_grid() -> tuple[RankedPolicySpec, ...]:
    return tuple(
        RankedPolicySpec(*values)
        for values in product(
            (0.08, 0.12, 0.16, 0.20),
            (1, 2),
            (1, 2),
        )
    )


def prepare_ranked_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_selector_frame(frame).reset_index(drop=True)
    prepared["_v18_slot_absolute"] = _slot_absolute(
        prepared["signal_slot"]
    )
    invalid = ~prepared["_v18_slot_absolute"].between(14 * 60 + 20, 14 * 60 + 50)
    if invalid.any():
        slots = sorted(prepared.loc[invalid, "signal_slot"].unique())
        raise ValueError(f"V18 received non-tail signal slots: {slots[:5]}")
    prepared.sort_values(
        ["trade_date", "ts_code", "_v18_slot_absolute"],
        kind="stable",
        inplace=True,
    )
    group_keys = [prepared["trade_date"], prepared["ts_code"]]
    prepared["v18_slots_seen"] = (
        prepared.groupby(["trade_date", "ts_code"], sort=False).cumcount() + 1
    )
    quality_hit = _numeric(
        prepared,
        "selection_rank_recomputed_pct",
    ).ge(0.90)
    probability_hit = _numeric(
        prepared,
        "p_net_positive_lower",
    ).ge(0.50)
    prepared["v18_quality_hits_so_far"] = quality_hit.groupby(
        group_keys,
        sort=False,
    ).cumsum()
    prepared["v18_probability_hits_so_far"] = probability_hit.groupby(
        group_keys,
        sort=False,
    ).cumsum()

    _attach_history(
        prepared,
        source="p_net_positive_lower",
        prefix="v18_p_positive",
        include_stability=False,
    )
    _attach_history(
        prepared,
        source="expected_utility_lower_pct",
        prefix="v18_utility",
        include_stability=False,
    )
    _attach_history(
        prepared,
        source="selection_score",
        prefix="v18_selection_score",
        include_stability=True,
    )
    _attach_history(
        prepared,
        source="ret_from_prev_close_pct",
        prefix="v18_return",
        include_stability=False,
    )
    severe = _numeric(prepared, "p_severe_loss")
    severe_group = severe.groupby(group_keys, sort=False)
    prepared["v18_severe_loss_delta_1"] = severe - severe_group.shift(1)
    prepared["v18_severe_loss_max_so_far"] = severe_group.transform(
        lambda values: values.expanding(min_periods=1).max()
    )
    return prepared.drop(columns="_v18_slot_absolute").reset_index(drop=True)


def fit_ranked_selector(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    random_seed: int,
    minimum_train_rows: int = 2_000,
    minimum_calibration_rows: int = 500,
) -> RankedSelectorBundle:
    selector = fit_selector(
        prepare_ranked_frame(train),
        prepare_ranked_frame(calibration),
        random_seed=random_seed,
        minimum_train_rows=minimum_train_rows,
        minimum_calibration_rows=minimum_calibration_rows,
        feature_candidates=V18_FEATURES,
    )
    return RankedSelectorBundle(selector=selector)


def calibrate_ranked_policy(
    scored: pd.DataFrame,
    spec: RankedPolicySpec,
    *,
    calibration_dates: Iterable[str],
) -> FrozenRankedPolicy:
    dates = sorted(set(map(str, calibration_dates)))
    if not dates:
        raise ValueError("V18 threshold calibration has no dates")
    calibration = scored.loc[
        scored["trade_date"].astype(str).isin(dates)
    ].copy()
    eligible = _base_eligible(calibration, spec)
    daily_max = (
        eligible.groupby("trade_date", sort=False)["selector_score"]
        .max()
        .sort_values(ascending=False)
    )
    target_days = max(
        1,
        int(np.ceil(spec.target_candidate_day_rate * len(dates))),
    )
    if daily_max.empty:
        threshold = float("inf")
    else:
        threshold = float(
            daily_max.iloc[min(target_days, len(daily_max)) - 1]
        )
    return FrozenRankedPolicy(
        spec=spec,
        score_threshold=threshold,
        threshold_calibration_start=dates[0],
        threshold_calibration_end=dates[-1],
        threshold_calibration_days=len(dates),
        threshold_eligible_days=int(len(daily_max)),
    )


def apply_ranked_policy(
    scored: pd.DataFrame,
    policy: FrozenRankedPolicy,
) -> pd.DataFrame:
    eligible = _base_eligible(scored, policy.spec)
    qualified = eligible.loc[
        _numeric(eligible, "selector_score").ge(policy.score_threshold)
    ].copy()
    if qualified.empty:
        qualified["v18_policy_id"] = policy.policy_id
        return qualified
    qualified["_v18_slot_absolute"] = _slot_absolute(
        qualified["signal_slot"]
    )
    qualified.sort_values(
        [
            "trade_date",
            "_v18_slot_absolute",
            "selector_score",
            "ts_code",
        ],
        ascending=[True, True, False, True],
        kind="stable",
        inplace=True,
    )
    first_signal = qualified.drop_duplicates(
        ["trade_date", "ts_code"],
        keep="first",
    ).copy()
    first_signal.sort_values(
        [
            "trade_date",
            "_v18_slot_absolute",
            "selector_score",
            "ts_code",
        ],
        ascending=[True, True, False, True],
        kind="stable",
        inplace=True,
    )
    within_day = first_signal.groupby("trade_date", sort=False).cumcount()
    selected = first_signal.loc[
        within_day.lt(policy.spec.max_candidates_per_day)
    ].drop(columns="_v18_slot_absolute")
    selected["v18_policy_id"] = policy.policy_id
    selected["v18_score_threshold"] = policy.score_threshold
    selected["v18_target_candidate_day_rate"] = (
        policy.spec.target_candidate_day_rate
    )
    return selected.reset_index(drop=True)


def select_ranked_policy(
    threshold_calibration: pd.DataFrame,
    design: pd.DataFrame,
    confirmation: pd.DataFrame,
    *,
    threshold_calibration_dates: Iterable[str],
    design_total_days: int,
    confirmation_total_days: int,
    specs: Iterable[RankedPolicySpec] | None = None,
    seed: int,
    bootstrap_samples: int = 800,
) -> RankedPolicySelection:
    grid = tuple(specs or ranked_policy_grid())
    rows: list[dict[str, Any]] = []
    frozen_by_id: dict[str, FrozenRankedPolicy] = {}
    for offset, spec in enumerate(grid):
        frozen = calibrate_ranked_policy(
            threshold_calibration,
            spec,
            calibration_dates=threshold_calibration_dates,
        )
        frozen_by_id[spec.spec_id] = frozen
        metrics = policy_metrics(
            apply_ranked_policy(design, frozen),
            total_days=design_total_days,
            seed=seed + offset,
            bootstrap_samples=bootstrap_samples,
        )
        rows.append({**frozen.as_dict(), **metrics})
    q_values = benjamini_hochberg(
        float(row["mean_return_p_value"]) for row in rows
    )
    for row, q_value in zip(rows, q_values, strict=True):
        row["mean_return_q_value"] = q_value
        row["design_gate_passed"] = ranked_design_gate(row)
    passed = [row for row in rows if row["design_gate_passed"]]
    calibration_summary = {
        "start": min(map(str, threshold_calibration_dates)),
        "end": max(map(str, threshold_calibration_dates)),
        "days": len(set(map(str, threshold_calibration_dates))),
    }
    if not passed:
        return RankedPolicySelection(
            policy=None,
            threshold_calibration=calibration_summary,
            design={
                "reason": "no_design_policy_passed_predeclared_gates",
                "policies": rows,
            },
            confirmation={"reason": "not_run"},
            policies_evaluated=len(rows),
            design_passed=0,
            confirmation_passed=False,
        )
    champion_row = max(
        passed,
        key=lambda row: (
            float(row["clustered_mean_lower_pct"]),
            float(row["win_rate_wilson_lower"]),
            -abs(
                float(row["candidate_day_rate"])
                - float(row["target_candidate_day_rate"])
            ),
            float(row["mean_net_return_pct"]),
        ),
    )
    champion = frozen_by_id[str(champion_row["spec_id"])]
    confirmation_metrics = policy_metrics(
        apply_ranked_policy(confirmation, champion),
        total_days=confirmation_total_days,
        seed=seed + len(grid) + 1,
        bootstrap_samples=max(1_000, bootstrap_samples),
    )
    confirmed = ranked_confirmation_gate(confirmation_metrics)
    if not confirmed:
        return RankedPolicySelection(
            policy=None,
            threshold_calibration=calibration_summary,
            design=champion_row,
            confirmation={
                **confirmation_metrics,
                "reason": "frozen_design_champion_failed_confirmation",
            },
            policies_evaluated=len(rows),
            design_passed=len(passed),
            confirmation_passed=False,
        )
    return RankedPolicySelection(
        policy=champion,
        threshold_calibration=calibration_summary,
        design=champion_row,
        confirmation={
            **confirmation_metrics,
            "reason": "frozen_design_champion_passed_once",
        },
        policies_evaluated=len(rows),
        design_passed=len(passed),
        confirmation_passed=True,
    )


def ranked_design_gate(metrics: dict[str, Any]) -> bool:
    rate = float(metrics["candidate_day_rate"])
    return bool(
        int(metrics["events"]) >= 10
        and int(metrics["candidate_days"]) >= 6
        and 0.06 <= rate <= 0.30
        and float(metrics["win_rate"]) >= 0.53
        and float(metrics["mean_net_return_pct"]) > 0.0
        and float(metrics["profit_factor"]) >= 1.10
        and float(metrics["stress_50bps_mean_net_return_pct"]) >= 0.0
        and float(metrics["mean_return_q_value"]) <= 0.10
    )


def ranked_confirmation_gate(metrics: dict[str, Any]) -> bool:
    rate = float(metrics["candidate_day_rate"])
    return bool(
        int(metrics["events"]) >= 8
        and int(metrics["candidate_days"]) >= 5
        and 0.05 <= rate <= 0.35
        and float(metrics["win_rate"]) >= 0.50
        and float(metrics["mean_net_return_pct"]) > 0.0
        and float(metrics["profit_factor"]) >= 1.0
        and float(metrics["stress_50bps_mean_net_return_pct"]) >= 0.0
    )


def rolling_ranked_policy_segments(
    prior_dates: Iterable[str],
    *,
    reserve_final_purge: bool = True,
) -> tuple[list[str], list[str], list[str]] | None:
    dates = sorted(set(map(str, prior_dates)))
    final_purge = POLICY_PURGE_DAYS if reserve_final_purge else 0
    needed = (
        THRESHOLD_CALIBRATION_DAYS
        + POLICY_PURGE_DAYS
        + POLICY_DESIGN_DAYS
        + POLICY_PURGE_DAYS
        + POLICY_CONFIRMATION_DAYS
        + final_purge
    )
    if len(dates) < needed:
        return None
    selected = dates[-needed:]
    threshold = selected[:THRESHOLD_CALIBRATION_DAYS]
    design_start = THRESHOLD_CALIBRATION_DAYS + POLICY_PURGE_DAYS
    design = selected[design_start : design_start + POLICY_DESIGN_DAYS]
    confirmation_start = (
        design_start + POLICY_DESIGN_DAYS + POLICY_PURGE_DAYS
    )
    confirmation = selected[
        confirmation_start : confirmation_start + POLICY_CONFIRMATION_DAYS
    ]
    return threshold, design, confirmation


def v18_research_readiness(
    metrics: dict[str, Any],
    *,
    yearly: list[dict[str, Any]],
    temporal_integrity: bool,
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
        "minimum_nested_oos_candidates": int(metrics.get("events", 0)) >= 100,
        "minimum_nested_oos_candidate_days": (
            int(metrics.get("candidate_days", 0)) >= 40
        ),
        "practical_candidate_day_rate": (
            0.08 <= float(metrics.get("candidate_day_rate", 0.0)) <= 0.30
        ),
        "minimum_win_rate": float(metrics.get("win_rate", 0.0)) >= 0.55,
        "minimum_wilson_lower": (
            float(metrics.get("win_rate_wilson_lower", 0.0)) >= 0.50
        ),
        "minimum_clustered_win_rate_lower": (
            float(metrics.get("clustered_win_rate_lower", 0.0)) >= 0.50
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
            "historical_gates_passed_future_shadow_still_required"
            if passed
            else "historical_evidence_insufficient"
        ),
    }


def _base_eligible(
    scored: pd.DataFrame,
    spec: RankedPolicySpec,
) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "selector_p_positive_lower",
        "selector_expected_net_return_pct",
        "selector_return_q25_pct",
        "selector_probability_spread",
        "selector_score",
        "selector_score_rank_pct",
        "p_severe_loss",
        "p_round_trip_fill_lower",
        "v18_quality_hits_so_far",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"V18 policy frame missing columns: {missing}")
    mask = (
        _numeric(scored, "selector_p_positive_lower").ge(0.45)
        & _numeric(scored, "selector_expected_net_return_pct").ge(-0.25)
        & _numeric(scored, "selector_return_q25_pct").ge(-1.50)
        & _numeric(scored, "p_severe_loss").le(0.40)
        & _numeric(scored, "p_round_trip_fill_lower").ge(0.95)
        & _numeric(scored, "selector_probability_spread").le(0.25)
        & _numeric(scored, "selector_score_rank_pct").ge(0.80)
        & _numeric(scored, "v18_quality_hits_so_far").ge(
            spec.minimum_quality_hits
        )
    )
    return scored.loc[mask].copy()


def _attach_history(
    frame: pd.DataFrame,
    *,
    source: str,
    prefix: str,
    include_stability: bool,
) -> None:
    values = _numeric(frame, source)
    keys = [frame["trade_date"], frame["ts_code"]]
    grouped = values.groupby(keys, sort=False)
    first = grouped.transform("first")
    frame[f"{prefix}_delta_1"] = values - grouped.shift(1)
    frame[f"{prefix}_change_from_1420"] = values - first
    frame[f"{prefix}_mean_so_far"] = grouped.transform(
        lambda item: item.expanding(min_periods=1).mean()
    )
    frame[f"{prefix}_min_so_far"] = grouped.transform(
        lambda item: item.expanding(min_periods=1).min()
    )
    if include_stability:
        frame[f"{prefix}_stability_so_far"] = grouped.transform(
            lambda item: item.expanding(min_periods=2).std()
        ).fillna(0.0)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _slot_absolute(values: pd.Series) -> pd.Series:
    parsed = values.astype(str).str.extract(
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2})"
    )
    return (
        pd.to_numeric(parsed["hour"], errors="coerce") * 60
        + pd.to_numeric(parsed["minute"], errors="coerce")
    )
