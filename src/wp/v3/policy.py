from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from .contracts import V3Config
from .statistics import day_clustered_intervals, wilson_interval


@dataclass(frozen=True)
class CandidatePolicy:
    policy_id: str
    authorized: bool
    reason: str
    entry_fill_probability_min: float = 1.0
    exit_fill_probability_min: float = 1.0
    round_trip_fill_probability_min: float = 1.0
    probability_min: float = 1.0
    probability_lower_min: float = 1.0
    conditional_probability_min: float = 1.0
    severe_loss_probability_max: float = 0.0
    selection_rank_min: float = 1.0
    expected_utility_min_pct: float = 999.0
    expected_utility_lower_min_pct: float = 999.0
    downside_min_pct: float = 999.0


@dataclass(frozen=True)
class PolicySelection:
    policy: CandidatePolicy
    design: dict[str, Any]
    confirmation: dict[str, Any]
    search: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": asdict(self.policy),
            "design": self.design,
            "confirmation": self.confirmation,
            "search": self.search,
        }


def no_signal_policy(reason: str) -> CandidatePolicy:
    payload = {
        "authorized": False,
        "reason": reason,
    }
    return CandidatePolicy(
        policy_id="no-signal-" + _digest(payload),
        authorized=False,
        reason=reason,
    )


def policy_from_dict(raw: dict[str, Any] | None) -> CandidatePolicy:
    if not raw:
        return no_signal_policy("policy_missing")
    fields = CandidatePolicy.__dataclass_fields__
    return CandidatePolicy(
        **{key: value for key, value in raw.items() if key in fields}
    )


def policy_selection_from_dict(raw: dict[str, Any] | None) -> PolicySelection:
    payload = raw or {}
    return PolicySelection(
        policy=policy_from_dict(payload.get("policy")),
        design=dict(payload.get("design") or {}),
        confirmation=dict(payload.get("confirmation") or {}),
        search=dict(payload.get("search") or {}),
    )


def apply_candidate_policy(
    frame: pd.DataFrame,
    policy: CandidatePolicy,
    config: V3Config,
) -> pd.Series:
    if frame.empty or not policy.authorized:
        return pd.Series(
            False,
            index=frame.index,
            dtype=bool,
            name="passes_policy",
        )

    execution = _boolean(
        frame.get("execution_eligible", pd.Series(True, index=frame.index))
    )
    freshness = _numeric_default(frame, "data_age_seconds", 0.0).le(
        config.execution.max_market_data_age_seconds
    )
    mask = (
        execution
        & freshness
        & _numeric(frame, "p_entry_fill").ge(
            policy.entry_fill_probability_min
        )
        & _numeric(frame, "p_exit_fill_given_entry").ge(
            policy.exit_fill_probability_min
        )
        & _numeric(frame, "p_round_trip_fill_lower").ge(
            policy.round_trip_fill_probability_min
        )
        & _numeric(frame, "p_net_positive").ge(policy.probability_min)
        & _numeric(frame, "p_net_positive_lower").ge(
            policy.probability_lower_min
        )
        & _numeric(frame, "p_conditional_net_positive").ge(
            policy.conditional_probability_min
        )
        & _numeric(frame, "p_severe_loss").le(
            policy.severe_loss_probability_max
        )
        & _numeric(frame, "selection_rank_pct").ge(policy.selection_rank_min)
        & _numeric(frame, "expected_utility_pct").ge(
            policy.expected_utility_min_pct
        )
        & _numeric(frame, "expected_utility_lower_pct").ge(
            policy.expected_utility_lower_min_pct
        )
        & _numeric(frame, "downside_q10_pct").ge(policy.downside_min_pct)
        & _numeric(frame, "probability_model_spread").le(
            config.model.max_probability_model_spread
        )
        & _numeric(frame, "fill_probability_model_spread").le(
            config.model.max_fill_probability_model_spread
        )
        & _numeric(frame, "selection_rank_spread").le(
            config.model.max_selection_rank_spread
        )
        & _numeric(frame, "expected_return_model_spread").le(
            config.model.max_expected_return_model_spread_pct
        )
    )
    return mask.fillna(False).astype(bool).rename("passes_policy")


def candidate_policy_diagnostics(
    frame: pd.DataFrame,
    policy: CandidatePolicy,
    config: V3Config,
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    if frame.empty:
        result["passes_policy"] = pd.Series(False, index=frame.index, dtype=bool)
        result["rejection_reasons"] = pd.Series("", index=frame.index, dtype=str)
        return result

    entry_fill_probability = _numeric(frame, "p_entry_fill")
    exit_fill_probability = _numeric(frame, "p_exit_fill_given_entry")
    round_trip_fill_lower = _numeric(frame, "p_round_trip_fill_lower")
    probability = _numeric(frame, "p_net_positive")
    probability_lower = _numeric(frame, "p_net_positive_lower")
    conditional_probability = _numeric(frame, "p_conditional_net_positive")
    severe_probability = _numeric(frame, "p_severe_loss")
    selection_rank = _numeric(frame, "selection_rank_pct")
    expected_utility = _numeric(frame, "expected_utility_pct")
    expected_utility_lower = _numeric(frame, "expected_utility_lower_pct")
    downside = _numeric(frame, "downside_q10_pct")
    probability_spread = _numeric(frame, "probability_model_spread")
    fill_probability_spread = _numeric(
        frame,
        "fill_probability_model_spread",
    )
    rank_spread = _numeric(frame, "selection_rank_spread")
    expected_return_spread = _numeric(
        frame,
        "expected_return_model_spread",
    )
    execution = _boolean(
        frame.get("execution_eligible", pd.Series(True, index=frame.index))
    )
    freshness = _numeric_default(frame, "data_age_seconds", 0.0).le(
        config.execution.max_market_data_age_seconds
    )
    result["passes_execution"] = execution
    result["passes_freshness"] = freshness
    result["passes_entry_fill_probability"] = entry_fill_probability.ge(
        policy.entry_fill_probability_min
    )
    result["passes_exit_fill_probability"] = exit_fill_probability.ge(
        policy.exit_fill_probability_min
    )
    result["passes_round_trip_fill_probability"] = round_trip_fill_lower.ge(
        policy.round_trip_fill_probability_min
    )
    result["passes_probability"] = probability.ge(policy.probability_min)
    result["passes_probability_lower"] = probability_lower.ge(
        policy.probability_lower_min
    )
    result["passes_conditional_probability"] = conditional_probability.ge(
        policy.conditional_probability_min
    )
    result["passes_severe_loss"] = severe_probability.le(
        policy.severe_loss_probability_max
    )
    result["passes_selection_rank"] = selection_rank.ge(policy.selection_rank_min)
    result["passes_expected_utility"] = expected_utility.ge(
        policy.expected_utility_min_pct
    )
    result["passes_expected_utility_lower"] = expected_utility_lower.ge(
        policy.expected_utility_lower_min_pct
    )
    result["passes_downside"] = downside.ge(policy.downside_min_pct)
    result["passes_stability"] = probability_spread.le(
        config.model.max_probability_model_spread
    ) & fill_probability_spread.le(
        config.model.max_fill_probability_model_spread
    ) & rank_spread.le(config.model.max_selection_rank_spread)
    result["passes_stability"] &= expected_return_spread.le(
        config.model.max_expected_return_model_spread_pct
    )
    result["passes_prior_oos_evidence"] = bool(policy.authorized)

    gate_columns = [
        "passes_execution",
        "passes_freshness",
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
    ]
    for column in gate_columns:
        result[column] = result[column].fillna(False).astype(bool)
    result["passes_policy"] = result[gate_columns].all(axis=1)
    reason_names = {
        "passes_execution": "execution",
        "passes_freshness": "freshness",
        "passes_entry_fill_probability": "entry_fill_probability",
        "passes_exit_fill_probability": "exit_fill_probability",
        "passes_round_trip_fill_probability": "round_trip_fill_probability",
        "passes_probability": "probability",
        "passes_probability_lower": "probability_lower",
        "passes_conditional_probability": "conditional_probability",
        "passes_severe_loss": "severe_loss",
        "passes_selection_rank": "selection_rank",
        "passes_expected_utility": "expected_utility",
        "passes_expected_utility_lower": "expected_utility_lower",
        "passes_downside": "downside",
        "passes_stability": "model_stability",
        "passes_prior_oos_evidence": "prior_oos_evidence",
    }
    result["rejection_reasons"] = [
        "|".join(
            reason_names[column]
            for column in gate_columns
            if not bool(row[column])
        )
        for _, row in result[gate_columns].iterrows()
    ]
    if not policy.authorized:
        suffix = str(policy.reason or "not_authorized")
        result["rejection_reasons"] = result["rejection_reasons"].map(
            lambda value: f"policy_not_authorized:{suffix}"
            + (f"|{value}" if value else "")
        )
    return result


def select_candidate_policy(
    design: pd.DataFrame,
    confirmation: pd.DataFrame,
    config: V3Config,
) -> PolicySelection:
    if design.empty or confirmation.empty:
        reason = "policy_period_empty"
        return PolicySelection(
            policy=no_signal_policy(reason),
            design=_empty_metrics(),
            confirmation=_empty_metrics(),
            search={"tested": 0, "design_eligible": 0, "reason": reason},
        )

    design_frame = _ordered(design)
    confirmation_frame = _ordered(confirmation)
    grid = list(_policy_grid(config))
    quick: list[tuple[CandidatePolicy, dict[str, Any]]] = []
    near_misses: list[dict[str, Any]] = []
    for policy in grid:
        metrics = _candidate_metrics(
            design_frame,
            apply_candidate_policy(design_frame, policy, config),
            clustered=False,
            seed=config.model.random_seed,
        )
        gate_status = _quick_design_gate_status(metrics, config)
        near_misses.append(
            {
                "policy_id": policy.policy_id,
                "policy": asdict(policy),
                "passed_gate_count": int(sum(gate_status.values())),
                "total_gate_count": int(len(gate_status)),
                "failed_gates": [
                    gate for gate, passed in gate_status.items() if not passed
                ],
                "gate_status": gate_status,
                "metrics": metrics,
                "proximity_score": _quick_design_proximity(metrics, config),
            }
        )
        if all(gate_status.values()):
            quick.append((policy, metrics))

    near_misses.sort(
        key=lambda item: (
            int(item["passed_gate_count"]),
            float(item["proximity_score"]),
            float(item["metrics"].get("win_rate_wilson_lower", 0.0)),
            float(item["metrics"].get("mean_net_return_pct") or -999.0),
        ),
        reverse=True,
    )
    near_misses = near_misses[:20]
    quick.sort(
        key=lambda item: (
            float(item[1].get("win_rate_wilson_lower", 0.0)),
            float(item[1].get("win_rate", 0.0)),
            float(item[1].get("mean_net_return_pct", -999.0)),
            float(item[1].get("profit_factor", 0.0)),
        ),
        reverse=True,
    )

    reviewed: list[dict[str, Any]] = []
    design_finalists: list[
        tuple[CandidatePolicy, dict[str, Any], int]
    ] = []
    for rank, (policy, _) in enumerate(quick[:20], start=1):
        design_metrics = _candidate_metrics(
            design_frame,
            apply_candidate_policy(design_frame, policy, config),
            clustered=True,
            seed=config.model.random_seed + rank,
        )
        design_pass = _passes_full(
            design_metrics,
            config,
            design_period=True,
        )
        reviewed.append(
            {
                "rank": rank,
                "policy_id": policy.policy_id,
                "design_pass": design_pass,
                "design": design_metrics,
            }
        )
        if design_pass:
            design_finalists.append((policy, design_metrics, rank))

    if not design_finalists:
        reason = "no_design_policy_passed"
        best = reviewed[0] if reviewed else {}
        return PolicySelection(
            policy=no_signal_policy(reason),
            design=best.get("design", _empty_metrics()),
            confirmation=_empty_metrics(),
            search={
                "tested": len(grid),
                "design_eligible": len(quick),
                "design_finalists": 0,
                "confirmation_policies_evaluated": 0,
                "reviewed": reviewed,
                "near_misses": near_misses,
                "reason": reason,
            },
        )

    # The confirmation period is a one-shot holdout. Pick exactly one policy
    # using design data, then expose that single frozen policy to confirmation.
    # Trying alternatives after seeing confirmation outcomes would turn the
    # holdout into another search set and materially overstate live evidence.
    champion, design_metrics, champion_rank = design_finalists[0]
    confirmation_metrics = _candidate_metrics(
        confirmation_frame,
        apply_candidate_policy(confirmation_frame, champion, config),
        clustered=True,
        seed=config.model.random_seed + 10_000 + champion_rank,
    )
    confirmation_pass = _passes_full(
        confirmation_metrics,
        config,
        design_period=False,
    )
    reviewed[champion_rank - 1]["selected_for_confirmation"] = True
    reviewed[champion_rank - 1]["confirmation_pass"] = confirmation_pass
    reviewed[champion_rank - 1]["confirmation"] = confirmation_metrics
    if confirmation_pass:
        authorized = CandidatePolicy(
            **{
                **asdict(champion),
                "authorized": True,
                "reason": "design_champion_confirmed_once",
            }
        )
        return PolicySelection(
            policy=authorized,
            design=design_metrics,
            confirmation=confirmation_metrics,
            search={
                "tested": len(grid),
                "design_eligible": len(quick),
                "design_finalists": len(design_finalists),
                "confirmation_policies_evaluated": 1,
                "selected_design_rank": champion_rank,
                "reviewed": reviewed,
                "near_misses": near_misses,
                "reason": "authorized",
            },
        )

    reason = "design_champion_failed_confirmation"
    return PolicySelection(
        policy=no_signal_policy(reason),
        design=design_metrics,
        confirmation=confirmation_metrics,
        search={
            "tested": len(grid),
            "design_eligible": len(quick),
            "design_finalists": len(design_finalists),
            "confirmation_policies_evaluated": 1,
            "selected_design_rank": champion_rank,
            "reviewed": reviewed,
            "near_misses": near_misses,
            "reason": reason,
        },
    )


def apply_nested_oos_policies(
    predictions: pd.DataFrame,
    config: V3Config,
) -> tuple[pd.DataFrame, list[dict[str, Any]], PolicySelection]:
    if predictions.empty:
        selection = PolicySelection(
            policy=no_signal_policy("oos_predictions_empty"),
            design=_empty_metrics(),
            confirmation=_empty_metrics(),
            search={"tested": 0, "design_eligible": 0},
        )
        return predictions.copy(), [], selection

    result = _ordered(predictions)
    result["passes_policy"] = False
    result["candidate_policy_id"] = ""
    result["candidate_policy_authorized"] = False
    audits: list[dict[str, Any]] = []
    total_policy_days = (
        config.model.policy_design_days
        + config.model.policy_confirmation_days
    )

    folds = sorted(
        pd.to_numeric(result["fold"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    for fold in folds:
        current = pd.to_numeric(result["fold"], errors="coerce").eq(fold)
        current_start = str(result.loc[current, "trade_date"].astype(str).min())
        history = result.loc[
            pd.to_numeric(result["fold"], errors="coerce").lt(fold)
            & result["trade_date"].astype(str).lt(current_start)
        ].copy()
        dates = np.array(sorted(history["trade_date"].astype(str).unique()))
        if len(dates) < total_policy_days:
            selection = PolicySelection(
                policy=no_signal_policy("insufficient_prior_oos_policy_days"),
                design=_empty_metrics(),
                confirmation=_empty_metrics(),
                search={
                    "tested": 0,
                    "design_eligible": 0,
                    "available_oos_days": int(len(dates)),
                    "required_oos_days": int(total_policy_days),
                },
            )
        else:
            selected_dates = dates[-total_policy_days:]
            split = config.model.policy_design_days
            design = history.loc[
                history["trade_date"].astype(str).isin(selected_dates[:split])
            ]
            confirmation = history.loc[
                history["trade_date"].astype(str).isin(selected_dates[split:])
            ]
            selection = select_candidate_policy(design, confirmation, config)

        mask = apply_candidate_policy(
            result.loc[current],
            selection.policy,
            config,
        )
        result.loc[current, "passes_policy"] = mask.to_numpy(dtype=bool)
        result.loc[current, "candidate_policy_id"] = selection.policy.policy_id
        result.loc[current, "candidate_policy_authorized"] = (
            selection.policy.authorized
        )
        audits.append(
            {
                "fold": int(fold),
                "test_start": current_start,
                **selection.as_dict(),
            }
        )

    all_dates = np.array(sorted(result["trade_date"].astype(str).unique()))
    if len(all_dates) < total_policy_days:
        final = PolicySelection(
            policy=no_signal_policy("insufficient_final_oos_policy_days"),
            design=_empty_metrics(),
            confirmation=_empty_metrics(),
            search={
                "available_oos_days": int(len(all_dates)),
                "required_oos_days": int(total_policy_days),
            },
        )
    else:
        selected_dates = all_dates[-total_policy_days:]
        split = config.model.policy_design_days
        final = select_candidate_policy(
            result.loc[
                result["trade_date"].astype(str).isin(selected_dates[:split])
            ],
            result.loc[
                result["trade_date"].astype(str).isin(selected_dates[split:])
            ],
            config,
        )
    return result.reset_index(drop=True), audits, final


def _policy_grid(config: V3Config):
    model = config.model
    for (
        entry_fill_probability,
        exit_fill_probability,
        probability,
        severe_probability,
        selection_rank,
        expected_utility,
        downside,
    ) in product(
        model.entry_fill_probability_grid,
        model.exit_fill_probability_grid,
        model.probability_grid,
        model.severe_loss_probability_grid,
        model.selection_rank_grid,
        model.expected_utility_grid_pct,
        model.downside_grid_pct,
    ):
        payload = {
            "entry_fill_probability_min": float(entry_fill_probability),
            "exit_fill_probability_min": float(exit_fill_probability),
            "round_trip_fill_probability_min": float(
                entry_fill_probability * exit_fill_probability
            ),
            "probability_min": float(probability),
            "probability_lower_min": float(max(0.30, probability - 0.06)),
            # This value is derived from direct P(net positive) and round-trip
            # fill probability in V8. It remains an audit field, not a second
            # near-duplicate search dimension.
            "conditional_probability_min": 0.0,
            "severe_loss_probability_max": float(severe_probability),
            "selection_rank_min": float(selection_rank),
            "expected_utility_min_pct": float(expected_utility),
            "expected_utility_lower_min_pct": float(expected_utility - 0.10),
            "downside_min_pct": float(downside),
        }
        yield CandidatePolicy(
            policy_id="wpv8-" + _digest(payload),
            authorized=True,
            reason="under_evaluation",
            **payload,
        )


def _candidate_metrics(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    clustered: bool,
    seed: int,
) -> dict[str, Any]:
    selected = frame.loc[mask.fillna(False)].drop_duplicates(
        ["trade_date", "ts_code"],
        keep="first",
    )
    returns = _numeric(selected, "net_return_pct")
    target = _numeric(selected, "target_net_positive")
    valid = returns.notna() & target.notna()
    selected = selected.loc[valid].copy()
    returns = returns.loc[valid]
    target = target.loc[valid]
    total = int(len(returns))
    wins = int(target.eq(1).sum())
    lower, upper = wilson_interval(wins, total)
    profits = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    profit_factor = (
        profits / losses
        if losses > 0
        else (float("inf") if profits > 0 else 0.0)
    )
    entry_fill = _numeric(selected, "target_entry_fillable")
    exit_fill = _numeric(selected, "target_exit_fillable")
    entry_known = entry_fill.notna()
    exit_known = exit_fill.notna()
    entry_fill_rate = (
        float(entry_fill.loc[entry_known].mean()) if entry_known.any() else 0.0
    )
    exit_fill_rate = (
        float(exit_fill.loc[exit_known].mean()) if exit_known.any() else 0.0
    )
    round_trip_fill_rate = (
        float(
            (
                entry_fill.eq(1)
                & exit_fill.eq(1)
            ).sum()
            / total
        )
        if total
        else 0.0
    )
    payload: dict[str, Any] = {
        "period_start": (
            str(frame["trade_date"].astype(str).min()) if not frame.empty else None
        ),
        "period_end": (
            str(frame["trade_date"].astype(str).max()) if not frame.empty else None
        ),
        "events": total,
        "trade_days": int(selected["trade_date"].astype(str).nunique())
        if total
        else 0,
        "wins": wins,
        "win_rate": wins / total if total else 0.0,
        "win_rate_wilson_lower": lower,
        "win_rate_wilson_upper": upper,
        "entry_fill_rate": entry_fill_rate,
        "exit_fill_rate_given_entry": exit_fill_rate,
        "round_trip_fill_rate": round_trip_fill_rate,
        "mean_net_return_pct": _finite(returns.mean()) if total else None,
        "median_net_return_pct": _finite(returns.median()) if total else None,
        "profit_factor": (
            _finite(profit_factor) if np.isfinite(profit_factor) else 999.0
        ),
    }
    if clustered and total:
        interval = day_clustered_intervals(
            selected,
            samples=1_500,
            seed=seed,
        )
        payload.update(
            {
                "win_rate_day_clustered_lower": interval.win_rate_lower,
                "win_rate_day_clustered_upper": interval.win_rate_upper,
                "mean_net_return_day_clustered_lower_pct": (
                    interval.mean_return_lower_pct
                ),
                "mean_net_return_day_clustered_upper_pct": (
                    interval.mean_return_upper_pct
                ),
            }
        )
    else:
        payload.update(
            {
                "win_rate_day_clustered_lower": None,
                "win_rate_day_clustered_upper": None,
                "mean_net_return_day_clustered_lower_pct": None,
                "mean_net_return_day_clustered_upper_pct": None,
            }
        )
    return payload


def _passes_quick_design(metrics: dict[str, Any], config: V3Config) -> bool:
    return all(_quick_design_gate_status(metrics, config).values())


def _quick_design_gate_status(
    metrics: dict[str, Any],
    config: V3Config,
) -> dict[str, bool]:
    model = config.model
    return {
        "minimum_events": (
            int(metrics.get("events", 0)) >= model.policy_min_design_events
        ),
        "minimum_trade_days": (
            int(metrics.get("trade_days", 0)) >= model.policy_min_design_days
        ),
        "minimum_win_rate": (
            float(metrics.get("win_rate", 0.0)) >= model.policy_min_win_rate
        ),
        "minimum_wilson_lower": (
            float(metrics.get("win_rate_wilson_lower", 0.0))
            >= model.policy_min_wilson_lower
        ),
        "minimum_entry_fill_rate": (
            float(metrics.get("entry_fill_rate", 0.0))
            >= config.promotion.minimum_entry_fill_rate
        ),
        "minimum_exit_fill_rate": (
            float(metrics.get("exit_fill_rate_given_entry", 0.0))
            >= config.promotion.minimum_exit_fill_rate
        ),
        "minimum_mean_net_return": (
            float(metrics.get("mean_net_return_pct") or -999.0)
            >= model.policy_min_mean_net_return_pct
        ),
        "minimum_profit_factor": (
            float(metrics.get("profit_factor") or 0.0)
            >= model.policy_min_profit_factor
        ),
    }


def _quick_design_proximity(
    metrics: dict[str, Any],
    config: V3Config,
) -> float:
    model = config.model
    pairs = (
        (float(metrics.get("events", 0)), float(model.policy_min_design_events)),
        (float(metrics.get("trade_days", 0)), float(model.policy_min_design_days)),
        (float(metrics.get("win_rate", 0.0)), float(model.policy_min_win_rate)),
        (
            float(metrics.get("win_rate_wilson_lower", 0.0)),
            float(model.policy_min_wilson_lower),
        ),
        (
            float(metrics.get("entry_fill_rate", 0.0)),
            float(config.promotion.minimum_entry_fill_rate),
        ),
        (
            float(metrics.get("exit_fill_rate_given_entry", 0.0)),
            float(config.promotion.minimum_exit_fill_rate),
        ),
        (
            float(metrics.get("mean_net_return_pct") or -999.0),
            float(model.policy_min_mean_net_return_pct),
        ),
        (
            float(metrics.get("profit_factor") or 0.0),
            float(model.policy_min_profit_factor),
        ),
    )
    scores = [
        max(-5.0, min(value / threshold, 1.0))
        if threshold > 0
        else (1.0 if value >= threshold else -1.0)
        for value, threshold in pairs
    ]
    return float(sum(scores) / len(scores))


def _passes_full(
    metrics: dict[str, Any],
    config: V3Config,
    *,
    design_period: bool,
) -> bool:
    model = config.model
    min_events = (
        model.policy_min_design_events
        if design_period
        else model.policy_min_confirmation_events
    )
    min_days = (
        model.policy_min_design_days
        if design_period
        else model.policy_min_confirmation_days
    )
    return (
        int(metrics.get("events", 0)) >= min_events
        and int(metrics.get("trade_days", 0)) >= min_days
        and float(metrics.get("win_rate", 0.0)) >= model.policy_min_win_rate
        and float(metrics.get("win_rate_wilson_lower", 0.0))
        >= model.policy_min_wilson_lower
        and float(metrics.get("win_rate_day_clustered_lower") or 0.0)
        >= model.policy_min_clustered_lower
        and float(metrics.get("entry_fill_rate", 0.0))
        >= config.promotion.minimum_entry_fill_rate
        and float(metrics.get("exit_fill_rate_given_entry", 0.0))
        >= config.promotion.minimum_exit_fill_rate
        and float(metrics.get("mean_net_return_pct") or -999.0)
        >= model.policy_min_mean_net_return_pct
        and float(metrics.get("profit_factor") or 0.0)
        >= model.policy_min_profit_factor
    )


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["trade_date"] = result["trade_date"].astype(str)
    result["signal_slot"] = result["signal_slot"].astype(str)
    return result.sort_values(
        ["trade_date", "signal_slot", "ts_code"],
        kind="stable",
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(
        frame.get(column, pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )


def _numeric_default(
    frame: pd.DataFrame,
    column: str,
    default: float,
) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _boolean(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y", "qualified", "pass"})
    )


def _empty_metrics() -> dict[str, Any]:
    return {
        "events": 0,
        "trade_days": 0,
        "wins": 0,
        "win_rate": 0.0,
        "win_rate_wilson_lower": 0.0,
        "win_rate_day_clustered_lower": 0.0,
        "entry_fill_rate": 0.0,
        "exit_fill_rate_given_entry": 0.0,
        "round_trip_fill_rate": 0.0,
        "mean_net_return_pct": None,
        "mean_net_return_day_clustered_lower_pct": None,
        "median_net_return_pct": None,
        "profit_factor": None,
    }


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None
