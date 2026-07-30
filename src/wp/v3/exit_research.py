from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .contracts import V3Config
from .meta_alpha import IDENTITY_COLUMNS
from .statistics import wilson_interval


@dataclass(frozen=True)
class ExitContract:
    contract_id: str
    kind: str
    target_net_bps: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "kind": self.kind,
            "target_net_bps": self.target_net_bps,
        }


@dataclass(frozen=True)
class ExitPolicy:
    contract_id: str
    probability_min: float
    expected_utility_min_pct: float
    severe_loss_max: float
    round_trip_fill_min: float
    selection_rank_min: float
    max_candidates_per_day: int

    @property
    def policy_id(self) -> str:
        return (
            f"{self.contract_id}-"
            f"p{self.probability_min:.2f}-"
            f"u{self.expected_utility_min_pct:.2f}-"
            f"s{self.severe_loss_max:.2f}-"
            f"f{self.round_trip_fill_min:.2f}-"
            f"r{self.selection_rank_min:.3f}-"
            f"k{self.max_candidates_per_day}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "contract_id": self.contract_id,
            "probability_min": self.probability_min,
            "expected_utility_min_pct": self.expected_utility_min_pct,
            "severe_loss_max": self.severe_loss_max,
            "round_trip_fill_min": self.round_trip_fill_min,
            "selection_rank_min": self.selection_rank_min,
            "max_candidates_per_day": self.max_candidates_per_day,
        }


EXIT_CONTRACTS = (
    ExitContract("t1_close_auction", "close"),
    ExitContract("t1_open_auction", "open"),
    ExitContract("tp10_close_fallback", "take_profit_close", 10),
    ExitContract("tp25_close_fallback", "take_profit_close", 25),
    ExitContract("tp50_close_fallback", "take_profit_close", 50),
    ExitContract("tp100_close_fallback", "take_profit_close", 100),
    ExitContract("tp200_close_fallback", "take_profit_close", 200),
)

PANEL_EXIT_COLUMNS = (
    *IDENTITY_COLUMNS,
    "adj_factor",
    "t1_adj_factor",
    "t1_open",
    "t1_high",
    "t1_low",
    "t1_close",
    "t1_vol",
    "t1_down_limit",
    "t1_up_limit",
    "exit_fillable",
)


def exit_contracts() -> tuple[ExitContract, ...]:
    return EXIT_CONTRACTS


def attach_exit_truth(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    config: V3Config,
) -> pd.DataFrame:
    missing_prediction = sorted(
        set((*IDENTITY_COLUMNS, "entry_price", "entry_fillable", "net_return_pct"))
        - set(predictions.columns)
    )
    missing_panel = sorted(set(PANEL_EXIT_COLUMNS) - set(panel.columns))
    if missing_prediction:
        raise ValueError(
            f"predictions missing exit-research columns: {missing_prediction}"
        )
    if missing_panel:
        raise ValueError(f"panel missing exit-research columns: {missing_panel}")

    left = predictions.copy()
    right = panel.loc[:, PANEL_EXIT_COLUMNS].copy()
    for frame in (left, right):
        for column in IDENTITY_COLUMNS:
            frame[column] = frame[column].astype(str)
    duplicate_panel = right.duplicated(list(IDENTITY_COLUMNS), keep=False)
    if duplicate_panel.any():
        raise ValueError(
            "panel contains duplicate exit truth identities: "
            f"{int(duplicate_panel.sum())}"
        )
    right = right.rename(columns={"exit_fillable": "panel_close_fillable"})
    result = left.merge(
        right,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_truth = result["_merge"].ne("both")
    if missing_truth.any():
        examples = result.loc[
            missing_truth,
            list(IDENTITY_COLUMNS),
        ].head(5).to_dict(orient="records")
        raise RuntimeError(
            f"exit truth missing for {int(missing_truth.sum())} predictions: "
            f"{examples}"
        )
    result.drop(columns="_merge", inplace=True)

    numeric_columns = (
        "entry_price",
        "adj_factor",
        "t1_adj_factor",
        "t1_open",
        "t1_high",
        "t1_low",
        "t1_close",
        "t1_vol",
        "t1_down_limit",
        "t1_up_limit",
        "net_return_pct",
    )
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    adjustment_ratio = np.where(
        result["adj_factor"].gt(0) & result["t1_adj_factor"].gt(0),
        result["t1_adj_factor"] / result["adj_factor"],
        np.nan,
    )
    result["t1_adjustment_ratio"] = adjustment_ratio
    for source in ("open", "high", "low", "close"):
        result[f"t1_adjusted_{source}"] = (
            result[f"t1_{source}"] * adjustment_ratio
        )

    entry_fillable = _boolean(result["entry_fillable"])
    result["exit_t1_close_auction_fillable"] = _boolean(
        result["panel_close_fillable"]
    )
    result["net_t1_close_auction_pct"] = pd.to_numeric(
        result["net_return_pct"],
        errors="coerce",
    )

    open_fillable = (
        result["t1_open"].gt(0)
        & result["t1_vol"].gt(0)
        & (
            result["t1_down_limit"].isna()
            | result["t1_open"].gt(result["t1_down_limit"] + 0.005)
        )
    )
    result["exit_t1_open_auction_fillable"] = open_fillable
    open_conditional = (
        result["t1_adjusted_open"] / result["entry_price"] - 1.0
    ) * 100.0 - config.execution.round_trip_cost_bps / 100.0
    result["net_t1_open_auction_pct"] = _all_in_return(
        entry_fillable=entry_fillable,
        exit_fillable=open_fillable,
        conditional_return=open_conditional,
        non_fill_penalty_pct=config.execution.non_fill_penalty_pct,
    )

    for contract in EXIT_CONTRACTS:
        if contract.kind != "take_profit_close":
            continue
        if contract.target_net_bps is None:
            raise ValueError(f"take-profit contract has no target: {contract}")
        gross_target_fraction = (
            config.execution.round_trip_cost_bps + contract.target_net_bps
        ) / 10_000.0
        adjusted_target = result["entry_price"] * (1.0 + gross_target_fraction)
        raw_target = adjusted_target / result["t1_adjustment_ratio"]
        # Requiring the daily high to trade at least one tick through the limit
        # avoids assuming queue priority when the high merely equals the order.
        target_hit = (
            entry_fillable
            & result["t1_high"].ge(raw_target + 0.01 - 1e-9)
            & result["t1_vol"].gt(0)
        )
        close_fillable = result["exit_t1_close_auction_fillable"]
        contract_fillable = target_hit | (~target_hit & close_fillable)
        contract_return = result["net_t1_close_auction_pct"].copy()
        contract_return.loc[target_hit] = contract.target_net_bps / 100.0
        contract_return = _all_in_return(
            entry_fillable=entry_fillable,
            exit_fillable=contract_fillable,
            conditional_return=contract_return,
            non_fill_penalty_pct=config.execution.non_fill_penalty_pct,
        )
        result[f"target_hit_{contract.contract_id}"] = target_hit
        result[f"exit_{contract.contract_id}_fillable"] = contract_fillable
        result[f"net_{contract.contract_id}_pct"] = contract_return

    _validate_contract_outputs(result)
    return result


def materialize_contract(
    frame: pd.DataFrame,
    contract_id: str,
) -> pd.DataFrame:
    contract = contract_by_id(contract_id)
    return_column = f"net_{contract.contract_id}_pct"
    fill_column = f"exit_{contract.contract_id}_fillable"
    missing = sorted(
        {return_column, fill_column, "entry_fillable"} - set(frame.columns)
    )
    if missing:
        raise ValueError(
            f"cannot materialize exit contract {contract_id}; missing {missing}"
        )
    result = frame.copy()
    result["net_return_pct"] = pd.to_numeric(
        result[return_column],
        errors="coerce",
    )
    result["exit_fillable"] = _boolean(result[fill_column])
    result["target_net_positive"] = result["net_return_pct"].gt(0).astype("int8")
    result["target_severe_loss"] = (
        result["net_return_pct"].le(-2.0).astype("int8")
    )
    result["exit_contract_id"] = contract.contract_id
    return result


def contract_by_id(contract_id: str) -> ExitContract:
    for contract in EXIT_CONTRACTS:
        if contract.contract_id == contract_id:
            return contract
    raise KeyError(f"unknown exit contract: {contract_id}")


def exit_policy_grid(
    contract_ids: Iterable[str] | None = None,
) -> tuple[ExitPolicy, ...]:
    allowed = tuple(
        contract_ids
        if contract_ids is not None
        else (contract.contract_id for contract in EXIT_CONTRACTS)
    )
    policies: list[ExitPolicy] = []
    for contract_id in allowed:
        contract_by_id(contract_id)
        for values in product(
            (0.50, 0.54),
            (0.00, 0.10),
            (0.25, 0.35),
            (0.90, 0.95),
            (0.98, 0.99),
            (1, 2, 3),
        ):
            policies.append(ExitPolicy(contract_id, *values))
    return tuple(policies)


def apply_exit_policy(
    frame: pd.DataFrame,
    policy: ExitPolicy,
) -> pd.DataFrame:
    contract_frame = materialize_contract(frame, policy.contract_id)
    mask = (
        _numeric(contract_frame, "p_net_positive").ge(policy.probability_min)
        & _numeric(contract_frame, "expected_utility_pct").ge(
            policy.expected_utility_min_pct
        )
        & _numeric(contract_frame, "p_severe_loss").le(policy.severe_loss_max)
        & _numeric(contract_frame, "p_round_trip_fill_lower").ge(
            policy.round_trip_fill_min
        )
        & _numeric(contract_frame, "selection_rank_pct").ge(
            policy.selection_rank_min
        )
    )
    qualified = contract_frame.loc[mask].copy()
    if qualified.empty:
        return qualified
    qualified["_slot_minute"] = _slot_minute(qualified["signal_slot"])
    qualified["_score"] = _numeric(qualified, "selection_score")
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
    return (
        qualified.loc[selected]
        .drop(columns=["_slot_minute", "_score"])
        .reset_index(drop=True)
    )


def fast_exit_metrics(
    frame: pd.DataFrame,
    config: V3Config,
) -> dict[str, Any]:
    clean = frame.copy()
    returns = _numeric(clean, "net_return_pct").dropna()
    clean = clean.loc[returns.index]
    total = int(len(clean))
    wins = int(returns.gt(0).sum())
    lower, upper = wilson_interval(wins, total)
    profits = float(returns.loc[returns > 0].sum())
    losses = float(-returns.loc[returns < 0].sum())
    entry = _boolean(
        clean.get("entry_fillable", pd.Series(False, index=clean.index))
    )
    exit_fill = _boolean(
        clean.get("exit_fillable", pd.Series(False, index=clean.index))
    )
    entered = int(entry.sum())
    extra_stress_pct = (
        50.0 - config.execution.baseline_all_in_cost_bps
    ) / 100.0
    stressed = returns - extra_stress_pct * entry.astype(float)
    daily = (
        clean.assign(_return=returns)
        .groupby("trade_date", sort=True)["_return"]
        .mean()
    )
    equity = (1.0 + daily / 100.0).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "events": total,
        "trade_days": (
            int(clean["trade_date"].astype(str).nunique()) if total else 0
        ),
        "wins": wins,
        "win_rate": wins / total if total else 0.0,
        "win_rate_wilson_lower": lower,
        "win_rate_wilson_upper": upper,
        "mean_net_return_pct": float(returns.mean()) if total else None,
        "median_net_return_pct": float(returns.median()) if total else None,
        "net_return_q10_pct": (
            float(returns.quantile(0.10)) if total else None
        ),
        "profit_factor": (
            profits / losses
            if losses > 0
            else (float("inf") if profits > 0 else 0.0)
        ),
        "entry_fill_rate": entered / total if total else 0.0,
        "exit_fill_rate_given_entry": (
            int((entry & exit_fill).sum()) / entered if entered else 0.0
        ),
        "stress_50bps_mean_net_return_pct": (
            float(stressed.mean()) if total else None
        ),
        "stress_50bps_positive_total_return": bool(stressed.sum() > 0),
        "maximum_day_equal_weight_drawdown_pct": (
            float(drawdown.min() * 100.0) if len(drawdown) else None
        ),
    }


def passes_design(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["events"] >= 30
        and metrics["trade_days"] >= 15
        and metrics["win_rate"] >= 0.52
        and (metrics["mean_net_return_pct"] or -999.0) >= 0.10
        and metrics["profit_factor"] >= 1.10
        and metrics["entry_fill_rate"] >= 0.90
        and metrics["exit_fill_rate_given_entry"] >= 0.95
        and (metrics["stress_50bps_mean_net_return_pct"] or -999.0) >= 0.0
    )


def passes_confirmation(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["events"] >= 15
        and metrics["trade_days"] >= 8
        and metrics["win_rate"] >= 0.50
        and (metrics["mean_net_return_pct"] or -999.0) > 0.0
        and metrics["profit_factor"] > 1.0
        and metrics["entry_fill_rate"] >= 0.88
        and metrics["exit_fill_rate_given_entry"] >= 0.93
        and (metrics["stress_50bps_mean_net_return_pct"] or -999.0) >= 0.0
    )


def select_exit_policy(
    design: pd.DataFrame,
    confirmation: pd.DataFrame,
    config: V3Config,
    *,
    contract_ids: Iterable[str] | None = None,
) -> tuple[ExitPolicy | None, dict[str, Any]]:
    policies = exit_policy_grid(contract_ids)
    design_passed: list[tuple[ExitPolicy, dict[str, Any]]] = []
    for policy in policies:
        metrics = fast_exit_metrics(apply_exit_policy(design, policy), config)
        if passes_design(metrics):
            design_passed.append((policy, metrics))

    confirmed: list[
        tuple[ExitPolicy, dict[str, Any], dict[str, Any]]
    ] = []
    for policy, design_metrics in design_passed:
        confirmation_metrics = fast_exit_metrics(
            apply_exit_policy(confirmation, policy),
            config,
        )
        if passes_confirmation(confirmation_metrics):
            confirmed.append((policy, design_metrics, confirmation_metrics))

    audit = {
        "tested": len(policies),
        "design_passed": len(design_passed),
        "confirmation_passed": len(confirmed),
        "selected_policy": None,
        "design": {},
        "confirmation": {},
    }
    if not confirmed:
        return None, audit
    confirmed.sort(
        key=lambda item: (
            min(
                item[1]["stress_50bps_mean_net_return_pct"],
                item[2]["stress_50bps_mean_net_return_pct"],
            ),
            min(
                item[1]["mean_net_return_pct"],
                item[2]["mean_net_return_pct"],
            ),
            min(item[1]["profit_factor"], item[2]["profit_factor"]),
            min(item[1]["win_rate"], item[2]["win_rate"]),
            item[2]["events"],
            item[0].policy_id,
        ),
        reverse=True,
    )
    policy, design_metrics, confirmation_metrics = confirmed[0]
    audit.update(
        {
            "selected_policy": policy.as_dict(),
            "design": design_metrics,
            "confirmation": confirmation_metrics,
        }
    )
    return policy, audit


def _all_in_return(
    *,
    entry_fillable: pd.Series,
    exit_fillable: pd.Series,
    conditional_return: pd.Series,
    non_fill_penalty_pct: float,
) -> pd.Series:
    return pd.Series(
        np.where(
            ~entry_fillable,
            0.0,
            np.where(
                exit_fillable,
                conditional_return,
                non_fill_penalty_pct,
            ),
        ),
        index=entry_fillable.index,
        dtype=float,
    )


def _validate_contract_outputs(frame: pd.DataFrame) -> None:
    for contract in EXIT_CONTRACTS:
        return_column = f"net_{contract.contract_id}_pct"
        fill_column = f"exit_{contract.contract_id}_fillable"
        if return_column not in frame or fill_column not in frame:
            raise RuntimeError(
                f"exit contract {contract.contract_id} was not materialized"
            )
        missing = pd.to_numeric(
            frame[return_column],
            errors="coerce",
        ).isna()
        if missing.any():
            raise RuntimeError(
                f"exit contract {contract.contract_id} has "
                f"{int(missing.sum())} missing returns"
            )


def _slot_minute(values: pd.Series) -> pd.Series:
    parsed = values.astype(str).str.extract(
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2})"
    )
    return (
        pd.to_numeric(parsed["hour"], errors="coerce") * 60
        + pd.to_numeric(parsed["minute"], errors="coerce")
        - (14 * 60 + 20)
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
