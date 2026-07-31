from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .meta_alpha import IDENTITY_COLUMNS, attach_meta_context
from .v16_policy import benjamini_hochberg, policy_metrics
from .v18_ranked import (
    V18_FEATURES,
    RankedSelectorBundle,
    fit_ranked_selector,
)


SCHEMA_VERSION = "wp_v19_broad_recall_selector_1"
MODEL_TRAIN_DAYS = 252
MODEL_CALIBRATION_DAYS = 42
MODEL_PURGE_DAYS = 2
THRESHOLD_CALIBRATION_DAYS = 42
POLICY_DESIGN_DAYS = 84
POLICY_CONFIRMATION_DAYS = 84
POLICY_PURGE_DAYS = 2
DEFAULT_TOP_PER_SOURCE = 48
DEFAULT_EXPLORATION_PER_SLOT = 24
DEFAULT_MAX_TRAIN_STOCKS_PER_DAY = 160

RECALL_SOURCES = (
    ("positive", "p_net_positive", False),
    ("conditional_positive", "p_conditional_net_positive", False),
    ("cross_section", "p_cross_section_top", False),
    ("utility", "expected_utility_pct", False),
    ("selection", "selection_score", False),
    ("momentum", "ret_from_prev_close_pct", False),
    ("low_severe_risk", "p_severe_loss", True),
)

FULL_CONTEXT_COLUMNS = (
    "context_universe_size",
    "context_return_mean_pct",
    "context_return_median_pct",
    "context_return_dispersion_pct",
    "context_breadth_positive",
    "context_breadth_above_2pct",
    "context_breadth_above_5pct",
    "context_breadth_above_7pct",
    "context_probability_mean",
    "context_probability_dispersion",
    "context_utility_mean_pct",
    "context_return_change_from_1420_pct",
    "return_context_relative_pct",
    "return_context_zscore",
    "return_rank_pct",
    "probability_rank_pct",
    "utility_rank_pct",
    "selection_rank_recomputed_pct",
    "severe_quality_rank_pct",
)
FULL_CONTEXT_FEATURES = tuple(
    f"v19_full_{column}" for column in FULL_CONTEXT_COLUMNS
)
RETRIEVAL_FEATURES = (
    "v19_retrieval_source_count",
    *(f"v19_retrieval_{name}" for name, _, _ in RECALL_SOURCES),
    "v19_retrieval_exploration",
)
V19_FEATURES = tuple(
    dict.fromkeys((*V18_FEATURES, *FULL_CONTEXT_FEATURES, *RETRIEVAL_FEATURES))
)


@dataclass
class RecallSelectorBundle:
    selector: RankedSelectorBundle
    source_train_rows: int
    sampled_train_rows: int
    source_calibration_rows: int
    sampled_calibration_rows: int

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
        return self.selector.predict(frame)


@dataclass(frozen=True)
class RecallPolicySpec:
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
class FrozenRecallPolicy:
    spec: RecallPolicySpec
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
class RecallPolicySelection:
    policy: FrozenRecallPolicy | None
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


def build_recall_frontier(
    frame: pd.DataFrame,
    *,
    top_per_source: int = DEFAULT_TOP_PER_SOURCE,
    exploration_per_slot: int = DEFAULT_EXPLORATION_PER_SLOT,
    require_label: bool = True,
) -> pd.DataFrame:
    if top_per_source < 1:
        raise ValueError("top_per_source must be positive")
    if exploration_per_slot < 0:
        raise ValueError("exploration_per_slot cannot be negative")
    contextual = attach_meta_context(frame).reset_index(drop=True)
    execution = _boolean(
        contextual.get(
            "execution_eligible",
            pd.Series(False, index=contextual.index),
        )
    )
    eligible = contextual.loc[execution].copy()
    if require_label:
        labelled = _boolean(
            eligible.get(
                "label_available",
                pd.Series(False, index=eligible.index),
            )
        )
        eligible = eligible.loc[labelled].copy()
    eligible.reset_index(drop=True, inplace=True)
    if eligible.empty:
        return eligible

    for column in FULL_CONTEXT_COLUMNS:
        eligible[f"v19_full_{column}"] = _numeric(eligible, column)

    keys = ["trade_date", "signal_slot"]
    sources_by_index: dict[int, set[str]] = {}
    for source_name, column, ascending in RECALL_SOURCES:
        if column not in eligible:
            continue
        ranked = eligible.dropna(subset=[column]).sort_values(
            [*keys, column, "ts_code"],
            ascending=[True, True, ascending, True],
            kind="stable",
        )
        selected = ranked.groupby(keys, sort=False).head(top_per_source)
        for index in selected.index:
            sources_by_index.setdefault(int(index), set()).add(source_name)

    if exploration_per_slot:
        eligible["_v19_hash"] = pd.util.hash_pandas_object(
            eligible.loc[:, list(IDENTITY_COLUMNS)],
            index=False,
        ).astype("uint64")
        explored = (
            eligible.sort_values(
                [*keys, "_v19_hash", "ts_code"],
                kind="stable",
            )
            .groupby(keys, sort=False)
            .head(exploration_per_slot)
        )
        for index in explored.index:
            sources_by_index.setdefault(int(index), set()).add("exploration")

    if not sources_by_index:
        return eligible.head(0).drop(columns="_v19_hash", errors="ignore")
    selected_indices = sorted(sources_by_index)
    frontier = eligible.loc[selected_indices].copy()
    for source_name, _, _ in RECALL_SOURCES:
        frontier[f"v19_retrieval_{source_name}"] = [
            float(source_name in sources_by_index[int(index)])
            for index in frontier.index
        ]
    frontier["v19_retrieval_exploration"] = [
        float("exploration" in sources_by_index[int(index)])
        for index in frontier.index
    ]
    frontier["v19_retrieval_source_count"] = [
        float(len(sources_by_index[int(index)]))
        for index in frontier.index
    ]
    frontier["v19_retrieval_sources"] = [
        "|".join(sorted(sources_by_index[int(index)]))
        for index in frontier.index
    ]
    return (
        frontier.drop(columns="_v19_hash", errors="ignore")
        .sort_values([*IDENTITY_COLUMNS], kind="stable")
        .reset_index(drop=True)
    )


def recall_frontier_audit(
    source: pd.DataFrame,
    frontier: pd.DataFrame,
) -> dict[str, Any]:
    execution = _boolean(
        source.get(
            "execution_eligible",
            pd.Series(False, index=source.index),
        )
    )
    labelled = _boolean(
        source.get(
            "label_available",
            pd.Series(False, index=source.index),
        )
    )
    eligible = source.loc[execution & labelled].copy()
    eligible_positive = _numeric(
        eligible,
        "target_net_positive",
    ).eq(1)
    frontier_positive = _numeric(
        frontier,
        "target_net_positive",
    ).eq(1)
    eligible_severe = _numeric(eligible, "net_return_pct").le(-2.0)
    frontier_severe = _numeric(frontier, "net_return_pct").le(-2.0)
    positive_total = int(eligible_positive.sum())
    severe_total = int(eligible_severe.sum())
    return {
        "source_rows": int(len(source)),
        "eligible_labelled_rows": int(len(eligible)),
        "frontier_rows": int(len(frontier)),
        "frontier_fraction_of_eligible": (
            len(frontier) / len(eligible) if len(eligible) else 0.0
        ),
        "positive_rows_total": positive_total,
        "positive_rows_retained": int(frontier_positive.sum()),
        "positive_row_recall": (
            float(frontier_positive.sum()) / positive_total
            if positive_total
            else 0.0
        ),
        "severe_rows_total": severe_total,
        "severe_rows_retained": int(frontier_severe.sum()),
        "severe_row_recall": (
            float(frontier_severe.sum()) / severe_total
            if severe_total
            else 0.0
        ),
    }


def deterministic_stock_day_sample(
    frame: pd.DataFrame,
    *,
    max_stocks_per_day: int = DEFAULT_MAX_TRAIN_STOCKS_PER_DAY,
) -> pd.DataFrame:
    if max_stocks_per_day < 4:
        raise ValueError("max_stocks_per_day must be at least four")
    if frame.empty:
        return frame.copy()
    prepared = frame.copy().reset_index(drop=True)
    stock_day = (
        prepared.groupby(["trade_date", "ts_code"], sort=False)
        .agg(
            selection_score=("selection_score", "max"),
            p_net_positive=("p_net_positive", "max"),
            p_severe_loss=("p_severe_loss", "min"),
        )
        .reset_index()
    )
    per_source = max(1, max_stocks_per_day // 4)
    selected: set[tuple[str, str]] = set()
    for column, ascending in (
        ("selection_score", False),
        ("p_net_positive", False),
        ("p_severe_loss", True),
    ):
        ranked = stock_day.dropna(subset=[column]).sort_values(
            ["trade_date", column, "ts_code"],
            ascending=[True, ascending, True],
            kind="stable",
        )
        for row in ranked.groupby("trade_date", sort=False).head(per_source).itertuples():
            selected.add((str(row.trade_date), str(row.ts_code)))
    stock_day["_v19_hash"] = pd.util.hash_pandas_object(
        stock_day.loc[:, ["trade_date", "ts_code"]],
        index=False,
    ).astype("uint64")
    explored = (
        stock_day.sort_values(
            ["trade_date", "_v19_hash", "ts_code"],
            kind="stable",
        )
        .groupby("trade_date", sort=False)
        .head(per_source)
    )
    for row in explored.itertuples():
        selected.add((str(row.trade_date), str(row.ts_code)))
    key = list(
        zip(
            prepared["trade_date"].astype(str),
            prepared["ts_code"].astype(str),
            strict=True,
        )
    )
    mask = pd.Series(
        [item in selected for item in key],
        index=prepared.index,
    )
    return prepared.loc[mask].reset_index(drop=True)


def fit_recall_selector(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    random_seed: int,
    max_stocks_per_day: int = DEFAULT_MAX_TRAIN_STOCKS_PER_DAY,
    minimum_train_rows: int = 2_000,
    minimum_calibration_rows: int = 500,
) -> RecallSelectorBundle:
    sampled_train = deterministic_stock_day_sample(
        train,
        max_stocks_per_day=max_stocks_per_day,
    )
    sampled_calibration = deterministic_stock_day_sample(
        calibration,
        max_stocks_per_day=max_stocks_per_day,
    )
    selector = fit_ranked_selector(
        sampled_train,
        sampled_calibration,
        random_seed=random_seed,
        minimum_train_rows=minimum_train_rows,
        minimum_calibration_rows=minimum_calibration_rows,
        feature_candidates=V19_FEATURES,
    )
    return RecallSelectorBundle(
        selector=selector,
        source_train_rows=int(len(train)),
        sampled_train_rows=int(len(sampled_train)),
        source_calibration_rows=int(len(calibration)),
        sampled_calibration_rows=int(len(sampled_calibration)),
    )


def recall_policy_grid() -> tuple[RecallPolicySpec, ...]:
    return tuple(
        RecallPolicySpec(*values)
        for values in product(
            (0.12, 0.18, 0.24, 0.30),
            (1, 2),
            (1, 2),
        )
    )


def calibrate_recall_policy(
    scored: pd.DataFrame,
    spec: RecallPolicySpec,
    *,
    calibration_dates: Iterable[str],
) -> FrozenRecallPolicy:
    dates = sorted(set(map(str, calibration_dates)))
    if not dates:
        raise ValueError("V19 threshold calibration has no dates")
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
    threshold = (
        float("inf")
        if daily_max.empty
        else float(daily_max.iloc[min(target_days, len(daily_max)) - 1])
    )
    return FrozenRecallPolicy(
        spec=spec,
        score_threshold=threshold,
        threshold_calibration_start=dates[0],
        threshold_calibration_end=dates[-1],
        threshold_calibration_days=len(dates),
        threshold_eligible_days=int(len(daily_max)),
    )


def apply_recall_policy(
    scored: pd.DataFrame,
    policy: FrozenRecallPolicy,
) -> pd.DataFrame:
    eligible = _base_eligible(scored, policy.spec)
    qualified = eligible.loc[
        _numeric(eligible, "selector_score").ge(policy.score_threshold)
    ].copy()
    if qualified.empty:
        qualified["v19_policy_id"] = policy.policy_id
        return qualified
    qualified["_v19_slot"] = _slot_absolute(qualified["signal_slot"])
    qualified.sort_values(
        ["trade_date", "_v19_slot", "selector_score", "ts_code"],
        ascending=[True, True, False, True],
        kind="stable",
        inplace=True,
    )
    first_signal = qualified.drop_duplicates(
        ["trade_date", "ts_code"],
        keep="first",
    ).copy()
    first_signal.sort_values(
        ["trade_date", "_v19_slot", "selector_score", "ts_code"],
        ascending=[True, True, False, True],
        kind="stable",
        inplace=True,
    )
    within_day = first_signal.groupby("trade_date", sort=False).cumcount()
    selected = first_signal.loc[
        within_day.lt(policy.spec.max_candidates_per_day)
    ].drop(columns="_v19_slot")
    selected["v19_policy_id"] = policy.policy_id
    selected["v19_score_threshold"] = policy.score_threshold
    selected["v19_target_candidate_day_rate"] = (
        policy.spec.target_candidate_day_rate
    )
    return selected.reset_index(drop=True)


def select_recall_policy(
    threshold_calibration: pd.DataFrame,
    design: pd.DataFrame,
    confirmation: pd.DataFrame,
    *,
    threshold_calibration_dates: Iterable[str],
    design_total_days: int,
    confirmation_total_days: int,
    specs: Iterable[RecallPolicySpec] | None = None,
    seed: int,
    bootstrap_samples: int = 800,
) -> RecallPolicySelection:
    grid = tuple(specs or recall_policy_grid())
    rows: list[dict[str, Any]] = []
    frozen_by_id: dict[str, FrozenRecallPolicy] = {}
    for offset, spec in enumerate(grid):
        frozen = calibrate_recall_policy(
            threshold_calibration,
            spec,
            calibration_dates=threshold_calibration_dates,
        )
        frozen_by_id[spec.spec_id] = frozen
        metrics = policy_metrics(
            apply_recall_policy(design, frozen),
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
        row["design_gate_passed"] = recall_design_gate(row)
    passed = [row for row in rows if row["design_gate_passed"]]
    calibration_summary = {
        "start": min(map(str, threshold_calibration_dates)),
        "end": max(map(str, threshold_calibration_dates)),
        "days": len(set(map(str, threshold_calibration_dates))),
    }
    if not passed:
        return RecallPolicySelection(
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
        apply_recall_policy(confirmation, champion),
        total_days=confirmation_total_days,
        seed=seed + len(grid) + 1,
        bootstrap_samples=max(1_000, bootstrap_samples),
    )
    confirmed = recall_confirmation_gate(confirmation_metrics)
    if not confirmed:
        return RecallPolicySelection(
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
    return RecallPolicySelection(
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


def recall_design_gate(metrics: dict[str, Any]) -> bool:
    rate = float(metrics["candidate_day_rate"])
    return bool(
        int(metrics["events"]) >= 10
        and int(metrics["candidate_days"]) >= 8
        and 0.08 <= rate <= 0.35
        and float(metrics["win_rate"]) >= 0.54
        and float(metrics["mean_net_return_pct"]) >= 0.20
        and float(metrics["profit_factor"]) >= 1.20
        and float(metrics["stress_50bps_mean_net_return_pct"]) >= 0.0
        and float(metrics["return_p10_pct"]) >= -3.0
        and float(metrics["mean_return_q_value"]) <= 0.10
    )


def recall_confirmation_gate(metrics: dict[str, Any]) -> bool:
    rate = float(metrics["candidate_day_rate"])
    return bool(
        int(metrics["events"]) >= 8
        and int(metrics["candidate_days"]) >= 7
        and 0.07 <= rate <= 0.38
        and float(metrics["win_rate"]) >= 0.52
        and float(metrics["mean_net_return_pct"]) > 0.0
        and float(metrics["profit_factor"]) >= 1.0
        and float(metrics["stress_50bps_mean_net_return_pct"]) >= 0.0
        and float(metrics["return_p10_pct"]) >= -3.0
    )


def rolling_recall_policy_segments(
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


def v19_research_readiness(
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
        "minimum_nested_oos_candidates": int(metrics.get("events", 0)) >= 80,
        "minimum_nested_oos_candidate_days": (
            int(metrics.get("candidate_days", 0)) >= 40
        ),
        "practical_candidate_day_rate": (
            0.10 <= float(metrics.get("candidate_day_rate", 0.0)) <= 0.32
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


def _base_eligible(
    scored: pd.DataFrame,
    spec: RecallPolicySpec,
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
        raise ValueError(f"V19 policy frame missing columns: {missing}")
    age = _numeric(scored, "data_age_seconds")
    fresh = age.isna() | age.le(420.0)
    mask = (
        _numeric(scored, "selector_p_positive_lower").ge(0.50)
        & _numeric(scored, "selector_expected_net_return_pct").ge(0.0)
        & _numeric(scored, "selector_return_q25_pct").ge(-1.50)
        & _numeric(scored, "p_severe_loss").le(0.35)
        & _numeric(scored, "p_round_trip_fill_lower").ge(0.95)
        & _numeric(scored, "selector_probability_spread").le(0.20)
        & _numeric(scored, "selector_score_rank_pct").ge(0.90)
        & _numeric(scored, "v18_quality_hits_so_far").ge(
            spec.minimum_quality_hits
        )
        & fresh
    )
    return scored.loc[mask].copy()


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


def _boolean(values: pd.Series | bool) -> pd.Series:
    if isinstance(values, bool):
        return pd.Series(dtype=bool)
    if values.dtype == bool:
        return values.fillna(False)
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y", "qualified", "pass"})
    )
