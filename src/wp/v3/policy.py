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
    probability_min: float = 1.0
    probability_lower_min: float = 1.0
    market_probability_min: float = 1.0
    cross_section_probability_min: float = 1.0
    severe_loss_probability_max: float = 0.0
    selection_rank_min: float = 1.0
    expected_return_min_pct: float = 999.0
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
        return pd.Series(False, index=frame.index, dtype=bool)

    probability = _numeric(frame, "p_net_positive")
    probability_lower = _numeric(frame, "p_net_positive_lower")
    market_probability = _numeric(frame, "p_market_positive")
    cross_probability = _numeric(frame, "p_cross_section_top")
    severe_probability = _numeric(frame, "p_severe_loss")
    selection_rank = _numeric(frame, "selection_rank_pct")
    expected_return = _numeric(frame, "expected_net_return_pct")
    downside = _numeric(frame, "downside_q10_pct")
    probability_spread = _numeric(frame, "probability_model_spread")
    rank_spread = _numeric(frame, "selection_rank_spread")
    execution = _boolean(
        frame.get("execution_eligible", pd.Series(True, index=frame.index))
    )
    freshness = _numeric_default(frame, "data_age_seconds", 0.0).le(
        config.execution.max_market_data_age_seconds
    )

    return (
        execution
        & freshness
        & probability.ge(policy.probability_min)
        & probability_lower.ge(policy.probability_lower_min)
        & market_probability.ge(policy.market_probability_min)
        & cross_probability.ge(policy.cross_section_probability_min)
        & severe_probability.le(policy.severe_loss_probability_max)
        & selection_rank.ge(policy.selection_rank_min)
        & expected_return.ge(policy.expected_return_min_pct)
        & downside.ge(policy.downside_min_pct)
        & probability_spread.le(config.model.max_probability_model_spread)
        & rank_spread.le(config.model.max_selection_rank_spread)
    ).fillna(False)


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
    for policy in grid:
        metrics = _candidate_metrics(
            design_frame,
            apply_candidate_policy(design_frame, policy, config),
            clustered=False,
            seed=config.model.random_seed,
        )
        if _passes_quick_design(metrics, config):
            quick.append((policy, metrics))

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
    for rank, (policy, _) in enumerate(quick[:20], start=1):
        design_metrics = _candidate_metrics(
            design_frame,
            apply_candidate_policy(design_frame, policy, config),
            clustered=True,
            seed=config.model.random_seed + rank,
        )
        confirmation_metrics = _candidate_metrics(
            confirmation_frame,
            apply_candidate_policy(confirmation_frame, policy, config),
            clustered=True,
            seed=config.model.random_seed + 10_000 + rank,
        )
        design_pass = _passes_full(
            design_metrics,
            config,
            design_period=True,
        )
        confirmation_pass = _passes_full(
            confirmation_metrics,
            config,
            design_period=False,
        )
        reviewed.append(
            {
                "rank": rank,
                "policy_id": policy.policy_id,
                "design_pass": design_pass,
                "confirmation_pass": confirmation_pass,
                "design": design_metrics,
                "confirmation": confirmation_metrics,
            }
        )
        if design_pass and confirmation_pass:
            authorized = CandidatePolicy(
                **{
                    **asdict(policy),
                    "authorized": True,
                    "reason": "design_and_confirmation_passed",
                }
            )
            return PolicySelection(
                policy=authorized,
                design=design_metrics,
                confirmation=confirmation_metrics,
                search={
                    "tested": len(grid),
                    "design_eligible": len(quick),
                    "reviewed": reviewed,
                    "reason": "authorized",
                },
            )

    reason = (
        "no_design_policy_passed"
        if not quick
        else "no_policy_confirmed"
    )
    best = reviewed[0] if reviewed else {}
    return PolicySelection(
        policy=no_signal_policy(reason),
        design=best.get("design", _empty_metrics()),
        confirmation=best.get("confirmation", _empty_metrics()),
        search={
            "tested": len(grid),
            "design_eligible": len(quick),
            "reviewed": reviewed,
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
        probability,
        market_probability,
        cross_probability,
        severe_probability,
        selection_rank,
        expected_return,
        downside,
    ) in product(
        model.probability_grid,
        model.market_probability_grid,
        model.cross_section_probability_grid,
        model.severe_loss_probability_grid,
        model.selection_rank_grid,
        model.expected_return_grid_pct,
        model.downside_grid_pct,
    ):
        payload = {
            "probability_min": float(probability),
            "probability_lower_min": float(max(0.40, probability - 0.08)),
            "market_probability_min": float(market_probability),
            "cross_section_probability_min": float(cross_probability),
            "severe_loss_probability_max": float(severe_probability),
            "selection_rank_min": float(selection_rank),
            "expected_return_min_pct": float(expected_return),
            "downside_min_pct": float(downside),
        }
        yield CandidatePolicy(
            policy_id="wpv5-" + _digest(payload),
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
    returns = pd.to_numeric(
        selected.get("net_return_pct"),
        errors="coerce",
    ).dropna()
    selected = selected.loc[returns.index]
    total = int(len(returns))
    wins = int(returns.gt(0).sum())
    lower, upper = wilson_interval(wins, total)
    profits = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    profit_factor = (
        profits / losses
        if losses > 0
        else (float("inf") if profits > 0 else 0.0)
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
    model = config.model
    return (
        int(metrics.get("events", 0)) >= model.policy_min_design_events
        and int(metrics.get("trade_days", 0)) >= model.policy_min_design_days
        and float(metrics.get("win_rate", 0.0)) >= model.policy_min_win_rate
        and float(metrics.get("win_rate_wilson_lower", 0.0))
        >= model.policy_min_wilson_lower
        and float(metrics.get("mean_net_return_pct") or -999.0)
        >= model.policy_min_mean_net_return_pct
        and float(metrics.get("profit_factor") or 0.0)
        >= model.policy_min_profit_factor
    )


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
