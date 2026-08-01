from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .contracts import V3Config
from .overlay import performance_summary


IDENTITY_COLUMNS = ("trade_date", "signal_slot", "ts_code")
FIXED_SIGNAL_SLOT = "14:30"


@dataclass(frozen=True)
class V40Policy:
    probability_min: float = 0.54
    expected_return_min_pct: float = 0.00
    severe_loss_max: float = 0.35
    round_trip_fill_min: float = 0.95
    meta_rank_min: float = 0.95
    exit_failure_rank_max: float = 0.50
    signal_slot: str = FIXED_SIGNAL_SLOT
    observation_count: int = 5

    @property
    def policy_id(self) -> str:
        return (
            "v40-1430-"
            f"p{self.probability_min:.2f}-"
            f"e{self.expected_return_min_pct:.2f}-"
            f"s{self.severe_loss_max:.2f}-"
            f"f{self.round_trip_fill_min:.2f}-"
            f"r{self.meta_rank_min:.2f}-"
            f"x{self.exit_failure_rank_max:.2f}-"
            f"obs{self.observation_count}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {"policy_id": self.policy_id, **asdict(self)}


@dataclass(frozen=True)
class V40Backtest:
    qualified: pd.DataFrame
    observations: pd.DataFrame
    summary: dict[str, Any]


def evaluate_v40_fixed_1430(
    frame: pd.DataFrame,
    config: V3Config,
    *,
    start_date: str,
    end_date: str,
    source_run_id: str | None = None,
    policy: V40Policy | None = None,
    expected_trade_dates: Iterable[str] | None = None,
) -> V40Backtest:
    active_policy = policy or V40Policy(
        observation_count=config.strategy.observation_count
    )
    source = _prepare_source(
        frame,
        start_date=start_date,
        end_date=end_date,
        policy=active_policy,
    )
    qualified, observations, day_audit = _select_cohorts(
        source,
        active_policy,
    )
    qualified_metrics = performance_summary(
        qualified,
        config,
        bootstrap_samples=4_000,
        seed=config.model.random_seed + 40_001,
    )
    observation_metrics = performance_summary(
        observations,
        config,
        bootstrap_samples=4_000,
        seed=config.model.random_seed + 40_002,
    )
    source_dates = sorted(source["trade_date"].astype(str).unique())
    expected_dates = (
        sorted(
            {
                _date(value)
                for value in expected_trade_dates
                if _date(start_date) <= _date(value) <= _date(end_date)
            }
        )
        if expected_trade_dates is not None
        else []
    )
    evaluated_records = pd.concat(
        [qualified, observations],
        ignore_index=True,
    )
    truth_dates = sorted(
        evaluated_records.loc[
            _label_available(evaluated_records),
            "trade_date",
        ].astype(str).unique()
    )
    requested_months = _month_sequence(start_date, end_date)
    monthly = [
        _month_summary(
            month,
            source=source,
            qualified=qualified,
            observations=observations,
            config=config,
        )
        for month in requested_months
    ]
    incomplete_days = [
        row["trade_date"]
        for row in day_audit
        if row["observation_status"] != "COMPLETE"
    ]
    source_end = source_dates[-1] if source_dates else None
    truth_end = truth_dates[-1] if truth_dates else None
    missing_source_dates = sorted(set(expected_dates) - set(source_dates))
    unexpected_source_dates = sorted(set(source_dates) - set(expected_dates))
    source_range_complete = (
        bool(expected_dates)
        and not missing_source_dates
        and not unexpected_source_dates
        if expected_trade_dates is not None
        else bool(
            source_dates
            and source_dates[0][:6] == _date(start_date)[:6]
            and source_dates[-1][:6] == _date(end_date)[:6]
        )
    )
    truth_range_complete = bool(
        source_range_complete
        and not evaluated_records.empty
        and _label_available(evaluated_records).all()
    )
    observation_complete = not incomplete_days
    status = (
        "COMPLETE"
        if source_range_complete
        and truth_range_complete
        and observation_complete
        else "INCOMPLETE"
    )
    summary = {
        "schema_version": "wp_v40_fixed_1430_backtest_1",
        "status": status,
        "research_only": True,
        "production_authorized": False,
        "objective": (
            "At 14:30 publish every fixed-gate qualified candidate and five "
            "separate near-gate research observations, then evaluate the "
            "immutable 14:35 entry benchmark against the T+1 close after costs."
        ),
        "evidence_contract": {
            "retrospective_start_date": _date(start_date),
            "retrospective_end_date": _date(end_date),
            "live_shadow_start_date": (
                config.evidence.live_shadow_start_date
            ),
            "retrospective_and_live_statistics_are_separate": True,
            "user_trade_decisions_are_not_collected": True,
        },
        "policy": active_policy.as_dict(),
        "policy_change_disclosure": {
            "seed": "V15 forward-risk research",
            "changes": [
                "fixed signal time changed to 14:30 only",
                "V15 daily top-three cap removed",
                "all fixed-gate passers form the qualified cohort",
                "five non-qualified names form a separate research cohort",
            ],
            "consequence": (
                "V15 performance cannot be reused as V40 performance; this "
                "fixed-14:30 contract requires its own retrospective and "
                "forward shadow evidence."
            ),
        },
        "source": {
            "workflow_run_id": source_run_id,
            "input_rows": int(len(frame)),
            "fixed_slot_rows_in_requested_range": int(len(source)),
            "first_source_trade_date": (
                source_dates[0] if source_dates else None
            ),
            "last_source_trade_date": source_end,
            "last_truth_trade_date": truth_end,
            "source_range_complete": source_range_complete,
            "expected_trade_days": (
                len(expected_dates)
                if expected_trade_dates is not None
                else None
            ),
            "missing_source_trade_dates": missing_source_dates,
            "unexpected_source_trade_dates": unexpected_source_dates,
            "truth_range_complete": truth_range_complete,
            "selection_uses_future_truth": False,
        },
        "integrity": {
            "evaluated_trade_days": int(len(source_dates)),
            "qualified_observation_overlap": int(
                len(
                    set(_identity_keys(qualified))
                    & set(_identity_keys(observations))
                )
            ),
            "observation_target_per_day": active_policy.observation_count,
            "observation_complete_days": int(
                sum(
                    row["observation_status"] == "COMPLETE"
                    for row in day_audit
                )
            ),
            "observation_incomplete_days": incomplete_days,
            "day_audit": day_audit,
        },
        "qualified": {
            "days_with_signal": int(
                qualified["trade_date"].astype(str).nunique()
            ),
            "zero_signal_days": int(
                max(len(source_dates) - qualified["trade_date"].nunique(), 0)
            ),
            "metrics": qualified_metrics,
        },
        "observations": {
            "metrics": observation_metrics,
        },
        "monthly": monthly,
        "interpretation": _interpretation(
            status=status,
            metrics=qualified_metrics,
        ),
    }
    return V40Backtest(
        qualified=qualified.reset_index(drop=True),
        observations=observations.reset_index(drop=True),
        summary=summary,
    )


def attach_v40_policy_gates(
    frame: pd.DataFrame,
    policy: V40Policy,
) -> pd.DataFrame:
    result = frame.copy()
    for column in (
        "meta_p_positive",
        "meta_expected_net_return_pct",
        "meta_p_severe_loss",
        "p_round_trip_fill_lower",
        "meta_rank_pct",
        "risk_failure_rank_pct",
        "meta_score",
        "signal_price",
        "net_return_pct",
    ):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result["passes_execution"] = _causal_execution_mask(result)
    gate_values = {
        "passes_meta_probability": result["meta_p_positive"].ge(
            policy.probability_min
        ),
        "passes_meta_expected_return": result[
            "meta_expected_net_return_pct"
        ].ge(policy.expected_return_min_pct),
        "passes_meta_severe_loss": result["meta_p_severe_loss"].le(
            policy.severe_loss_max
        ),
        "passes_round_trip_fill": result[
            "p_round_trip_fill_lower"
        ].ge(policy.round_trip_fill_min),
        "passes_meta_rank": result["meta_rank_pct"].ge(
            policy.meta_rank_min
        ),
        "passes_exit_failure_rank": result[
            "risk_failure_rank_pct"
        ].le(policy.exit_failure_rank_max),
    }
    passes = result["passes_execution"].copy()
    for name, values in gate_values.items():
        result[name] = values.fillna(False).astype(bool)
        passes &= result[name]
    result["passes_policy"] = passes.fillna(False).astype(bool)
    result["failed_gate_count"] = sum(
        (~result[name]).astype(int)
        for name in gate_values
    )
    result["failed_gate_distance"] = (
        (policy.probability_min - result["meta_p_positive"]).clip(lower=0)
        + (
            policy.expected_return_min_pct
            - result["meta_expected_net_return_pct"]
        ).clip(lower=0)
        + (result["meta_p_severe_loss"] - policy.severe_loss_max).clip(
            lower=0
        )
        + (
            policy.round_trip_fill_min
            - result["p_round_trip_fill_lower"]
        ).clip(lower=0)
        + (policy.meta_rank_min - result["meta_rank_pct"]).clip(lower=0)
        + (
            result["risk_failure_rank_pct"]
            - policy.exit_failure_rank_max
        ).clip(lower=0)
    )
    return result


def select_v40_cohorts(
    scored: pd.DataFrame,
    policy: V40Policy,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    return _select_cohorts(scored, policy)


def refresh_v40_backtest_summary(
    summary: dict[str, Any],
    qualified: pd.DataFrame,
    observations: pd.DataFrame,
    config: V3Config,
) -> dict[str, Any]:
    refreshed = dict(summary)
    qualified_metrics = performance_summary(
        qualified,
        config,
        bootstrap_samples=4_000,
        seed=config.model.random_seed + 40_001,
    )
    observation_metrics = performance_summary(
        observations,
        config,
        bootstrap_samples=4_000,
        seed=config.model.random_seed + 40_002,
    )
    selected = pd.concat(
        [qualified, observations],
        ignore_index=True,
    )
    truth_complete = bool(
        not selected.empty and _label_available(selected).all()
    )
    source_complete = bool(
        refreshed.get("source", {}).get("source_range_complete")
    )
    observation_complete = not bool(
        refreshed.get("integrity", {}).get(
            "observation_incomplete_days",
            [],
        )
    )
    status = (
        "COMPLETE"
        if source_complete and truth_complete and observation_complete
        else "INCOMPLETE"
    )
    refreshed["status"] = status
    refreshed.setdefault("source", {})["truth_range_complete"] = truth_complete
    labelled_dates = sorted(
        selected.loc[
            _label_available(selected),
            "trade_date",
        ].astype(str).unique()
    )
    refreshed["source"]["last_truth_trade_date"] = (
        labelled_dates[-1] if labelled_dates else None
    )
    refreshed.setdefault("qualified", {})["metrics"] = qualified_metrics
    refreshed.setdefault("observations", {})["metrics"] = observation_metrics
    existing_months = {
        str(row.get("month")): dict(row)
        for row in refreshed.get("monthly", [])
    }
    month_ids = _month_sequence(
        refreshed.get("evidence_contract", {}).get(
            "retrospective_start_date",
            config.evidence.retrospective_start_date,
        ),
        refreshed.get("evidence_contract", {}).get(
            "retrospective_end_date",
            config.evidence.retrospective_end_date,
        ),
    )
    monthly: list[dict[str, Any]] = []
    for month in month_ids:
        row = existing_months.get(month, {"month": month})
        row["qualified"] = performance_summary(
            qualified.loc[
                qualified["trade_date"].astype(str).str[:6].eq(month)
            ],
            config,
            bootstrap_samples=1_000,
            seed=config.model.random_seed + int(month),
        )
        row["observations"] = performance_summary(
            observations.loc[
                observations["trade_date"].astype(str).str[:6].eq(month)
            ],
            config,
            bootstrap_samples=1_000,
            seed=config.model.random_seed + int(month) + 1,
        )
        monthly.append(row)
    refreshed["monthly"] = monthly
    refreshed["interpretation"] = _interpretation(
        status=status,
        metrics=qualified_metrics,
    )
    return refreshed


def v40_historical_gate(
    summary: dict[str, Any],
    config: V3Config,
) -> dict[str, Any]:
    metrics = summary["qualified"]["metrics"]
    promotion = config.promotion
    gates = {
        "complete_evidence": summary["status"] == "COMPLETE",
        "minimum_candidates": int(metrics.get("events") or 0)
        >= promotion.minimum_oos_candidates,
        "minimum_win_rate": float(metrics.get("win_rate") or 0.0)
        >= promotion.minimum_oos_win_rate,
        "minimum_wilson_lower": float(
            metrics.get("win_rate_wilson_lower") or 0.0
        )
        >= promotion.minimum_oos_win_rate_lower,
        "minimum_clustered_win_lower": float(
            metrics.get("win_rate_day_clustered_lower") or 0.0
        )
        >= promotion.minimum_clustered_win_rate_lower,
        "minimum_mean_net_return": float(
            metrics.get("mean_net_return_pct") or -999.0
        )
        >= promotion.minimum_mean_net_return_pct,
        "clustered_mean_lower_nonnegative": float(
            metrics.get("mean_net_return_day_clustered_lower_pct")
            or -999.0
        )
        >= promotion.minimum_clustered_mean_return_lower_pct,
        "minimum_profit_factor": float(
            metrics.get("profit_factor") or 0.0
        )
        >= promotion.minimum_profit_factor,
        "50bps_stress_nonnegative": bool(
            metrics.get("stress", {})
            .get("50bps", {})
            .get("positive_total_return", False)
        ),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "failed_gates": [
            name for name, passed in gates.items() if not passed
        ],
        "production_authorized": False,
        "reason": "150_day_forward_shadow_still_required",
    }


def _prepare_source(
    frame: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    policy: V40Policy,
) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "meta_p_positive",
        "meta_expected_net_return_pct",
        "meta_p_severe_loss",
        "p_round_trip_fill_lower",
        "meta_rank_pct",
        "risk_failure_rank_pct",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V40 source is missing columns: {missing}")
    result = frame.copy()
    for column in IDENTITY_COLUMNS:
        result[column] = result[column].astype(str)
    result["trade_date"] = result["trade_date"].map(_date)
    duplicated = result.duplicated(list(IDENTITY_COLUMNS), keep=False)
    if duplicated.any():
        raise ValueError(
            "V40 source contains duplicate identities: "
            f"{int(duplicated.sum())}"
        )
    result = result.loc[
        result["signal_slot"].eq(policy.signal_slot)
        & result["trade_date"].between(_date(start_date), _date(end_date))
    ].copy()
    result = attach_v40_policy_gates(result, policy)
    return result.sort_values(
        ["trade_date", "meta_score", "ts_code"],
        ascending=[True, False, True],
        kind="stable",
        na_position="last",
    )


def _select_cohorts(
    source: pd.DataFrame,
    policy: V40Policy,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    qualified_frames: list[pd.DataFrame] = []
    observation_frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for trade_date, day in source.groupby("trade_date", sort=True):
        qualified = day.loc[day["passes_policy"]].copy()
        qualified = qualified.sort_values(
            ["meta_score", "ts_code"],
            ascending=[False, True],
            kind="stable",
            na_position="last",
        )
        qualified["candidate_cohort"] = "QUALIFIED"
        qualified["cohort_rank"] = range(1, len(qualified) + 1)

        hard_eligible = _boolean(day["passes_execution"])
        if "passes_freshness" in day:
            hard_eligible &= _boolean(day["passes_freshness"])
        observation_pool = day.loc[
            hard_eligible & ~_boolean(day["passes_policy"])
        ].copy()
        observation_pool = observation_pool.sort_values(
            [
                "failed_gate_count",
                "failed_gate_distance",
                "meta_score",
                "ts_code",
            ],
            ascending=[True, True, False, True],
            kind="stable",
            na_position="last",
        )
        observations = observation_pool.head(policy.observation_count).copy()
        observations["candidate_cohort"] = "OBSERVATION"
        observations["cohort_rank"] = range(1, len(observations) + 1)
        qualified_frames.append(qualified)
        observation_frames.append(observations)
        audits.append(
            {
                "trade_date": str(trade_date),
                "source_rows": int(len(day)),
                "qualified_count": int(len(qualified)),
                "observation_pool_count": int(len(observation_pool)),
                "observation_count": int(len(observations)),
                "observation_status": (
                    "COMPLETE"
                    if len(observations) == policy.observation_count
                    else "INSUFFICIENT_CAUSAL_ELIGIBLE_POOL"
                ),
            }
        )
    empty = source.iloc[0:0].copy()
    return (
        _concat(qualified_frames, empty),
        _concat(observation_frames, empty),
        audits,
    )


def _month_summary(
    month: str,
    *,
    source: pd.DataFrame,
    qualified: pd.DataFrame,
    observations: pd.DataFrame,
    config: V3Config,
) -> dict[str, Any]:
    source_month = source.loc[source["trade_date"].str[:6].eq(month)]
    qualified_month = qualified.loc[
        qualified["trade_date"].str[:6].eq(month)
    ]
    observation_month = observations.loc[
        observations["trade_date"].str[:6].eq(month)
    ]
    return {
        "month": month,
        "evaluated_trade_days": int(
            source_month["trade_date"].nunique()
        ),
        "qualified": performance_summary(
            qualified_month,
            config,
            bootstrap_samples=1_000,
            seed=config.model.random_seed + int(month),
        ),
        "observations": performance_summary(
            observation_month,
            config,
            bootstrap_samples=1_000,
            seed=config.model.random_seed + int(month) + 1,
        ),
    }


def _causal_execution_mask(frame: pd.DataFrame) -> pd.Series:
    for column in ("execution_eligible", "passes_execution"):
        if column in frame:
            return _boolean(frame[column])
    signal_price = pd.to_numeric(
        frame.get(
            "signal_price",
            pd.Series(1.0, index=frame.index),
        ),
        errors="coerce",
    )
    return signal_price.gt(0).fillna(False)


def _label_available(frame: pd.DataFrame) -> pd.Series:
    if "label_available" in frame:
        return _boolean(frame["label_available"])
    return pd.to_numeric(
        frame.get(
            "net_return_pct",
            pd.Series(index=frame.index, dtype=float),
        ),
        errors="coerce",
    ).notna()


def _identity_keys(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    return (
        frame.loc[:, IDENTITY_COLUMNS]
        .astype(str)
        .agg("|".join, axis=1)
        .tolist()
    )


def _concat(frames: list[pd.DataFrame], empty: pd.DataFrame) -> pd.DataFrame:
    usable = [frame for frame in frames if not frame.empty]
    if not usable:
        return empty.copy()
    return pd.concat(usable, ignore_index=True)


def _month_sequence(start_date: str, end_date: str) -> list[str]:
    start = pd.Period(_date(start_date)[:6], freq="M")
    end = pd.Period(_date(end_date)[:6], freq="M")
    return [str(period).replace("-", "") for period in pd.period_range(start, end)]


def _date(value: Any) -> str:
    return str(value or "").replace("-", "")[:8]


def _boolean(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y", "qualified", "pass"})
    )


def _interpretation(
    *,
    status: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    events = int(metrics.get("events") or 0)
    mean = metrics.get("mean_net_return_pct")
    positive = bool(mean is not None and float(mean) > 0)
    if status != "COMPLETE":
        conclusion = "INCOMPLETE_EVIDENCE"
        text = (
            "The requested May-July evidence is not complete. No profitability "
            "claim is permitted until source coverage, T+1 truth and the five-"
            "observation integrity rule are all complete."
        )
    elif events == 0:
        conclusion = "NO_QUALIFIED_EVENTS"
        text = (
            "The fixed contract produced no qualified historical event. This is "
            "a valid result, not permission to lower the gate."
        )
    elif positive:
        conclusion = "HISTORICALLY_POSITIVE_NOT_PRODUCTION_PROOF"
        text = (
            "The retrospective qualified cohort is positive after stated costs, "
            "but remains historical evidence only. August forward shadow results "
            "must be accumulated separately."
        )
    else:
        conclusion = "HISTORICALLY_NON_POSITIVE"
        text = (
            "The retrospective qualified cohort is not positive after stated "
            "costs. The contract must not be production-authorized."
        )
    return {
        "conclusion": conclusion,
        "text": text,
        "profitability_guaranteed": False,
    }
