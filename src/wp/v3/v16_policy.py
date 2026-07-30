from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .meta_alpha import IDENTITY_COLUMNS
from .statistics import wilson_interval


@dataclass(frozen=True)
class ExpertPolicy:
    probability_lower_min: float
    expected_return_lower_min_pct: float
    severe_loss_max: float
    round_trip_fill_min: float
    minimum_experts: int
    probability_spread_max: float
    score_rank_min: float
    max_candidates_per_day: int
    slot_group: str

    @property
    def policy_id(self) -> str:
        return (
            f"p{self.probability_lower_min:.2f}-"
            f"e{self.expected_return_lower_min_pct:.2f}-"
            f"s{self.severe_loss_max:.2f}-"
            f"f{self.round_trip_fill_min:.2f}-"
            f"n{self.minimum_experts}-"
            f"d{self.probability_spread_max:.2f}-"
            f"r{self.score_rank_min:.2f}-"
            f"k{self.max_candidates_per_day}-"
            f"{self.slot_group}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "probability_lower_min": self.probability_lower_min,
            "expected_return_lower_min_pct": (
                self.expected_return_lower_min_pct
            ),
            "severe_loss_max": self.severe_loss_max,
            "round_trip_fill_min": self.round_trip_fill_min,
            "minimum_experts": self.minimum_experts,
            "probability_spread_max": self.probability_spread_max,
            "score_rank_min": self.score_rank_min,
            "max_candidates_per_day": self.max_candidates_per_day,
            "slot_group": self.slot_group,
        }


@dataclass(frozen=True)
class PolicySelection:
    policy: ExpertPolicy | None
    design: dict[str, Any]
    confirmation: dict[str, Any]
    design_evaluated: int
    design_gate_passed: int
    confirmation_passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.as_dict() if self.policy else None,
            "design": self.design,
            "confirmation": self.confirmation,
            "search": {
                "design_evaluated": self.design_evaluated,
                "design_gate_passed": self.design_gate_passed,
                "confirmation_passed": self.confirmation_passed,
            },
        }


def expert_policy_grid() -> tuple[ExpertPolicy, ...]:
    policies = []
    for values in product(
        (0.52, 0.56, 0.60),
        (0.00, 0.10),
        (0.25, 0.35),
        (1, 2),
        (0.10, 0.18),
        (0.90, 0.95),
        (2, 3),
        ("all", "early", "late"),
    ):
        (
            probability,
            expected,
            severe,
            experts,
            spread,
            rank,
            maximum,
            slot_group,
        ) = values
        policies.append(
            ExpertPolicy(
                probability_lower_min=probability,
                expected_return_lower_min_pct=expected,
                severe_loss_max=severe,
                round_trip_fill_min=0.95,
                minimum_experts=experts,
                probability_spread_max=spread,
                score_rank_min=rank,
                max_candidates_per_day=maximum,
                slot_group=slot_group,
            )
        )
    return tuple(policies)


def apply_expert_policy(
    frame: pd.DataFrame,
    policy: ExpertPolicy,
) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "expert_count",
        "expert_p_positive_lower",
        "expert_expected_return_lower_pct",
        "expert_p_severe",
        "expert_probability_spread",
        "expert_score",
        "p_round_trip_fill_lower",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"expert policy frame missing columns: {missing}")
    candidates = frame.copy()
    candidates["expert_score_rank_pct"] = candidates.groupby(
        ["trade_date", "signal_slot"],
        sort=False,
    )["expert_score"].rank(method="average", pct=True)
    slot_mask = _slot_group_mask(candidates["signal_slot"], policy.slot_group)
    mask = (
        slot_mask
        & _numeric(candidates, "expert_count").ge(policy.minimum_experts)
        & _numeric(candidates, "expert_p_positive_lower").ge(
            policy.probability_lower_min
        )
        & _numeric(candidates, "expert_expected_return_lower_pct").ge(
            policy.expected_return_lower_min_pct
        )
        & _numeric(candidates, "expert_p_severe").le(policy.severe_loss_max)
        & _numeric(candidates, "p_round_trip_fill_lower").ge(
            policy.round_trip_fill_min
        )
        & _numeric(candidates, "expert_probability_spread").le(
            policy.probability_spread_max
        )
        & _numeric(candidates, "expert_score_rank_pct").ge(
            policy.score_rank_min
        )
    )
    qualified = candidates.loc[mask].copy()
    if qualified.empty:
        qualified["expert_policy_id"] = policy.policy_id
        return qualified
    qualified["_slot_minute"] = _slot_minute(qualified["signal_slot"])
    qualified["_score"] = _numeric(qualified, "expert_score")
    qualified.sort_values(
        ["trade_date", "_slot_minute", "_score", "ts_code"],
        ascending=[True, True, False, True],
        kind="stable",
        inplace=True,
    )
    selected: list[int] = []
    for _, day in qualified.groupby("trade_date", sort=False):
        seen: set[str] = set()
        for index, row in day.iterrows():
            code = str(row["ts_code"])
            if code in seen:
                continue
            selected.append(index)
            seen.add(code)
            if len(seen) >= policy.max_candidates_per_day:
                break
    result = qualified.loc[selected].drop(
        columns=["_slot_minute", "_score"],
    )
    result["expert_policy_id"] = policy.policy_id
    return result.reset_index(drop=True)


def select_nested_policy(
    design: pd.DataFrame,
    confirmation: pd.DataFrame,
    *,
    design_total_days: int,
    confirmation_total_days: int,
    policies: Iterable[ExpertPolicy] | None = None,
    seed: int = 20_260_727,
    bootstrap_samples: int = 800,
) -> PolicySelection:
    grid = tuple(policies or expert_policy_grid())
    design_rows: list[dict[str, Any]] = []
    for offset, policy in enumerate(grid):
        selected = apply_expert_policy(design, policy)
        metrics = policy_metrics(
            selected,
            total_days=design_total_days,
            seed=seed + offset,
            bootstrap_samples=bootstrap_samples,
        )
        design_rows.append({**policy.as_dict(), **metrics})
    p_values = [float(row["mean_return_p_value"]) for row in design_rows]
    q_values = benjamini_hochberg(p_values)
    for row, q_value in zip(design_rows, q_values, strict=True):
        row["mean_return_q_value"] = q_value
        row["design_gate_passed"] = design_gate(row)
    passed = [row for row in design_rows if row["design_gate_passed"]]
    if not passed:
        return PolicySelection(
            policy=None,
            design={
                "reason": "no_design_policy_passed_predeclared_gates",
                "policies": design_rows,
            },
            confirmation={"reason": "not_run"},
            design_evaluated=len(design_rows),
            design_gate_passed=0,
            confirmation_passed=False,
        )
    champion_row = max(
        passed,
        key=lambda row: (
            float(row["clustered_mean_lower_pct"]),
            float(row["win_rate_wilson_lower"]),
            float(row["mean_net_return_pct"]),
            float(row["candidate_day_rate"]),
            -float(row["mean_return_q_value"]),
        ),
    )
    champion = next(
        policy for policy in grid if policy.policy_id == champion_row["policy_id"]
    )
    confirmation_selected = apply_expert_policy(confirmation, champion)
    confirmation_metrics = policy_metrics(
        confirmation_selected,
        total_days=confirmation_total_days,
        seed=seed + len(grid) + 1,
        bootstrap_samples=max(bootstrap_samples, 1_000),
    )
    confirmation_ok = confirmation_gate(confirmation_metrics)
    if not confirmation_ok:
        return PolicySelection(
            policy=None,
            design=champion_row,
            confirmation={
                **confirmation_metrics,
                "reason": "frozen_design_champion_failed_confirmation",
            },
            design_evaluated=len(design_rows),
            design_gate_passed=len(passed),
            confirmation_passed=False,
        )
    return PolicySelection(
        policy=champion,
        design=champion_row,
        confirmation={
            **confirmation_metrics,
            "reason": "frozen_design_champion_passed_once",
        },
        design_evaluated=len(design_rows),
        design_gate_passed=len(passed),
        confirmation_passed=True,
    )


def policy_metrics(
    frame: pd.DataFrame,
    *,
    total_days: int,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    returns = _numeric(frame, "net_return_pct").dropna()
    clean = frame.loc[returns.index].copy()
    events = int(len(clean))
    candidate_days = int(clean["trade_date"].astype(str).nunique())
    wins = int(returns.gt(0).sum())
    lower, upper = wilson_interval(wins, events)
    profits = float(returns.loc[returns > 0].sum())
    losses = float(-returns.loc[returns < 0].sum())
    profit_factor = (
        profits / losses
        if losses > 0
        else (float("inf") if profits > 0 else 0.0)
    )
    clustered = clustered_mean_bootstrap(
        clean,
        seed=seed,
        samples=bootstrap_samples,
    )
    stress_returns = returns - 0.50
    significance = clustered_mean_significance(clean)
    return {
        "events": events,
        "candidate_days": candidate_days,
        "candidate_day_rate": candidate_days / max(total_days, 1),
        "wins": wins,
        "win_rate": wins / events if events else 0.0,
        "win_rate_wilson_lower": lower,
        "win_rate_wilson_upper": upper,
        "mean_net_return_pct": (
            float(returns.mean()) if events else float("nan")
        ),
        "median_net_return_pct": (
            float(returns.median()) if events else float("nan")
        ),
        "return_p10_pct": (
            float(returns.quantile(0.10)) if events else float("nan")
        ),
        "profit_factor": profit_factor,
        "stress_50bps_mean_net_return_pct": (
            float(stress_returns.mean()) if events else float("nan")
        ),
        **significance,
        **clustered,
    }


def clustered_mean_significance(frame: pd.DataFrame) -> dict[str, float]:
    daily = _daily_mean_returns(frame)
    if len(daily) < 2:
        return {
            "mean_return_p_value": 1.0,
            "mean_return_test_days": float(len(daily)),
            "mean_return_hac_lags": 0.0,
        }
    values = daily.to_numpy(dtype=float)
    centered = values - float(values.mean())
    lags = min(5, len(values) - 1)
    long_run_variance = float(np.mean(centered * centered))
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        covariance = float(
            np.sum(centered[lag:] * centered[:-lag]) / len(centered)
        )
        long_run_variance += 2.0 * weight * covariance
    long_run_variance = max(long_run_variance, 0.0)
    standard_error = np.sqrt(long_run_variance / len(values))
    if np.isclose(float(standard_error), 0.0):
        if len(daily) and float(daily.mean()) > 0.0:
            p_value = max(
                float(np.finfo(float).tiny),
                float(0.5 ** len(daily)),
            )
        else:
            p_value = 1.0
        return {
            "mean_return_p_value": p_value,
            "mean_return_test_days": float(len(daily)),
            "mean_return_hac_lags": float(lags),
        }
    test_statistic = float(daily.mean()) / float(standard_error)
    p_value = float(student_t.sf(test_statistic, df=len(values) - 1))
    return {
        "mean_return_p_value": p_value if np.isfinite(p_value) else 1.0,
        "mean_return_test_days": float(len(daily)),
        "mean_return_hac_lags": float(lags),
    }


def clustered_mean_bootstrap(
    frame: pd.DataFrame,
    *,
    seed: int,
    samples: int = 1_000,
    block_days: int = 5,
) -> dict[str, float]:
    if frame.empty:
        return {
            "clustered_mean_lower_pct": float("nan"),
            "clustered_mean_upper_pct": float("nan"),
            "clustered_win_rate_lower": float("nan"),
            "clustered_win_rate_upper": float("nan"),
        }
    daily = _daily_cluster_statistics(frame)
    if daily.empty:
        return {
            "clustered_mean_lower_pct": float("nan"),
            "clustered_mean_upper_pct": float("nan"),
            "clustered_win_rate_lower": float("nan"),
            "clustered_win_rate_upper": float("nan"),
        }
    values = daily["mean_return"].to_numpy(float)
    wins = daily["wins"].to_numpy(float)
    events = daily["events"].to_numpy(float)
    block = max(1, min(block_days, len(values)))
    blocks = int(np.ceil(len(values) / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), size=(samples, blocks))
    offsets = np.arange(block)
    choices = (
        (starts[:, :, None] + offsets[None, None, :]) % len(values)
    ).reshape(samples, -1)[:, : len(values)]
    means = values[choices].mean(axis=1)
    sampled_events = events[choices].sum(axis=1)
    sampled_win_rates = np.divide(
        wins[choices].sum(axis=1),
        sampled_events,
        out=np.zeros_like(sampled_events, dtype=float),
        where=sampled_events > 0,
    )
    return {
        "clustered_mean_lower_pct": float(np.quantile(means, 0.025)),
        "clustered_mean_upper_pct": float(np.quantile(means, 0.975)),
        "clustered_win_rate_lower": float(
            np.quantile(sampled_win_rates, 0.025)
        ),
        "clustered_win_rate_upper": float(
            np.quantile(sampled_win_rates, 0.975)
        ),
    }


def _daily_mean_returns(frame: pd.DataFrame) -> pd.Series:
    return _daily_cluster_statistics(frame)["mean_return"]


def _daily_cluster_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.assign(
        _trade_date=frame["trade_date"].astype(str),
        _return=_numeric(frame, "net_return_pct"),
    ).dropna(subset=["_return"])
    if prepared.empty:
        return pd.DataFrame(columns=["mean_return", "wins", "events"])
    prepared["_win"] = prepared["_return"].gt(0).astype(int)
    return prepared.groupby("_trade_date", sort=True).agg(
        mean_return=("_return", "mean"),
        wins=("_win", "sum"),
        events=("_return", "size"),
    )


def benjamini_hochberg(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    if len(values) == 0:
        return []
    values = np.where(np.isfinite(values), values, 1.0)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result.tolist()


def design_gate(metrics: dict[str, Any]) -> bool:
    return bool(
        int(metrics["events"]) >= 60
        and int(metrics["candidate_days"]) >= 20
        and float(metrics["win_rate"]) >= 0.52
        and float(metrics["mean_net_return_pct"]) > 0.0
        and float(metrics["profit_factor"]) >= 1.05
        and float(metrics["stress_50bps_mean_net_return_pct"]) >= 0.0
        and float(metrics["mean_return_q_value"]) <= 0.10
    )


def confirmation_gate(metrics: dict[str, Any]) -> bool:
    return bool(
        int(metrics["events"]) >= 25
        and int(metrics["candidate_days"]) >= 10
        and float(metrics["win_rate"]) >= 0.50
        and float(metrics["mean_net_return_pct"]) > 0.0
        and float(metrics["profit_factor"]) >= 1.0
        and float(metrics["stress_50bps_mean_net_return_pct"]) >= 0.0
    )


def _slot_group_mask(values: pd.Series, group: str) -> pd.Series:
    slots = values.astype(str)
    if group == "all":
        return pd.Series(True, index=values.index, dtype=bool)
    if group == "early":
        return slots.isin({"14:20", "14:25", "14:30", "14:35"})
    if group == "late":
        return slots.isin({"14:40", "14:45", "14:50"})
    raise ValueError(f"unknown slot group: {group}")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _slot_minute(values: pd.Series) -> pd.Series:
    parsed = values.astype(str).str.extract(
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2})"
    )
    return (
        pd.to_numeric(parsed["hour"], errors="coerce") * 60
        + pd.to_numeric(parsed["minute"], errors="coerce")
    )
