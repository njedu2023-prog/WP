from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .contracts import V3Config
from .meta_alpha import IDENTITY_COLUMNS


@dataclass(frozen=True)
class TailExitContract:
    contract_id: str
    decision_slot: str
    benchmark_slot: str

    def as_dict(self) -> dict[str, str]:
        return {
            "contract_id": self.contract_id,
            "decision_slot": self.decision_slot,
            "benchmark_slot": self.benchmark_slot,
        }


TAIL_EXIT_CONTRACTS = (
    TailExitContract("t1_1420_next5m", "14:20", "14:25"),
    TailExitContract("t1_1430_next5m", "14:30", "14:35"),
    TailExitContract("t1_1440_next5m", "14:40", "14:45"),
    TailExitContract("t1_1450_next5m", "14:50", "14:55"),
)

TAIL_PANEL_COLUMNS = (
    *IDENTITY_COLUMNS,
    "target_trade_date",
    "adj_factor",
    "entry_benchmark_price",
    "entry_benchmark_amount",
    "entry_benchmark_volume",
    "entry_benchmark_bar_lag_minutes",
    "down_limit",
)


def tail_exit_contracts() -> tuple[TailExitContract, ...]:
    return TAIL_EXIT_CONTRACTS


def tail_exit_contract_by_id(contract_id: str) -> TailExitContract:
    for contract in TAIL_EXIT_CONTRACTS:
        if contract.contract_id == contract_id:
            return contract
    raise KeyError(f"unknown T+1 tail exit contract: {contract_id}")


def attach_t1_tail_exit_truth(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    config: V3Config,
    *,
    exit_slippage_bps: float | None = None,
) -> pd.DataFrame:
    prediction_required = {
        *IDENTITY_COLUMNS,
        "entry_price",
        "entry_fillable",
    }
    panel_required = set(TAIL_PANEL_COLUMNS)
    missing_prediction = sorted(prediction_required - set(predictions.columns))
    missing_panel = sorted(panel_required - set(panel.columns))
    if missing_prediction:
        raise ValueError(
            f"predictions missing T+1 tail-exit columns: {missing_prediction}"
        )
    if missing_panel:
        raise ValueError(f"panel missing T+1 tail-exit columns: {missing_panel}")

    result = predictions.copy()
    source = panel.loc[:, TAIL_PANEL_COLUMNS].copy()
    for frame in (result, source):
        for column in IDENTITY_COLUMNS:
            frame[column] = frame[column].astype(str)
    if source.duplicated(list(IDENTITY_COLUMNS), keep=False).any():
        raise ValueError("panel contains duplicate T+1 tail-exit identities")

    mapping = source.loc[
        :,
        [*IDENTITY_COLUMNS, "target_trade_date", "adj_factor"],
    ].rename(columns={"adj_factor": "entry_day_adj_factor"})
    result = result.merge(
        mapping,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    result["target_trade_date"] = result["target_trade_date"].astype(
        "string"
    )
    result["entry_day_adj_factor"] = pd.to_numeric(
        result["entry_day_adj_factor"],
        errors="coerce",
    )
    result["entry_price"] = pd.to_numeric(
        result["entry_price"],
        errors="coerce",
    )
    entry_fillable = _boolean(result["entry_fillable"])
    covered_dates = set(source["trade_date"].astype(str).unique())
    result["_tail_target_date_covered"] = (
        result["target_trade_date"].astype(str).isin(covered_dates)
    )

    slippage_bps = (
        config.execution.entry_slippage_bps
        if exit_slippage_bps is None
        else float(exit_slippage_bps)
    )
    for contract in TAIL_EXIT_CONTRACTS:
        snapshot = source.loc[
            source["signal_slot"].eq(contract.decision_slot),
            [
                "trade_date",
                "ts_code",
                "adj_factor",
                "entry_benchmark_price",
                "entry_benchmark_amount",
                "entry_benchmark_volume",
                "entry_benchmark_bar_lag_minutes",
                "down_limit",
            ],
        ].copy()
        snapshot = snapshot.rename(
            columns={
                "trade_date": "target_trade_date",
                "adj_factor": f"exit_adj_factor_{contract.contract_id}",
                "entry_benchmark_price": (
                    f"exit_benchmark_price_{contract.contract_id}"
                ),
                "entry_benchmark_amount": (
                    f"exit_benchmark_amount_{contract.contract_id}"
                ),
                "entry_benchmark_volume": (
                    f"exit_benchmark_volume_{contract.contract_id}"
                ),
                "entry_benchmark_bar_lag_minutes": (
                    f"exit_benchmark_lag_{contract.contract_id}"
                ),
                "down_limit": f"exit_down_limit_{contract.contract_id}",
            }
        )
        if snapshot.duplicated(
            ["target_trade_date", "ts_code"],
            keep=False,
        ).any():
            raise ValueError(
                f"duplicate target snapshots for {contract.contract_id}"
            )
        result = result.merge(
            snapshot,
            on=["target_trade_date", "ts_code"],
            how="left",
            validate="many_to_one",
        )
        result = _materialize_tail_columns(
            result,
            contract,
            config,
            entry_fillable=entry_fillable,
            slippage_bps=slippage_bps,
        )

    return result.drop(columns="_tail_target_date_covered")


def materialize_tail_exit_contract(
    frame: pd.DataFrame,
    contract_id: str,
) -> pd.DataFrame:
    contract = tail_exit_contract_by_id(contract_id)
    return_column = f"net_{contract.contract_id}_pct"
    fill_column = f"exit_{contract.contract_id}_fillable"
    label_column = f"label_{contract.contract_id}_available"
    missing = sorted(
        {return_column, fill_column, label_column, "entry_fillable"}
        - set(frame.columns)
    )
    if missing:
        raise ValueError(
            f"cannot materialize T+1 tail contract {contract_id}; "
            f"missing {missing}"
        )
    result = frame.copy()
    available = _boolean(result[label_column])
    result["net_return_pct"] = pd.to_numeric(
        result[return_column],
        errors="coerce",
    )
    result["exit_fillable"] = _boolean(result[fill_column])
    result["label_available"] = available
    result["target_net_positive"] = np.where(
        available,
        result["net_return_pct"].gt(0).astype("int8"),
        np.nan,
    )
    result["target_severe_loss"] = np.where(
        available,
        result["net_return_pct"].le(-2.0).astype("int8"),
        np.nan,
    )
    result["exit_contract_id"] = contract.contract_id
    return result


def _materialize_tail_columns(
    frame: pd.DataFrame,
    contract: TailExitContract,
    config: V3Config,
    *,
    entry_fillable: pd.Series,
    slippage_bps: float,
) -> pd.DataFrame:
    contract_id = contract.contract_id
    price = _numeric(frame, f"exit_benchmark_price_{contract_id}")
    amount = _numeric(frame, f"exit_benchmark_amount_{contract_id}")
    volume = _numeric(frame, f"exit_benchmark_volume_{contract_id}")
    lag = _numeric(frame, f"exit_benchmark_lag_{contract_id}")
    down_limit = _numeric(frame, f"exit_down_limit_{contract_id}")
    exit_adj_factor = _numeric(frame, f"exit_adj_factor_{contract_id}")
    entry_adj_factor = _numeric(frame, "entry_day_adj_factor")
    target_date_covered = _boolean(frame["_tail_target_date_covered"])

    snapshot_present = (
        target_date_covered
        & exit_adj_factor.gt(0)
        & entry_adj_factor.gt(0)
    )
    fillable = (
        snapshot_present
        & price.gt(0)
        & amount.ge(config.execution.min_slot_amount)
        & amount.mul(config.execution.max_entry_pct_of_slot_amount).ge(
            config.execution.reference_order_notional
        )
        & volume.gt(0)
        & lag.between(0, 0, inclusive="both")
        & (down_limit.isna() | price.gt(down_limit * 1.0001))
    )
    adjusted_price = price * exit_adj_factor / entry_adj_factor
    execution_price = adjusted_price * (1.0 - slippage_bps / 10_000.0)
    conditional_net = (
        (execution_price / _numeric(frame, "entry_price") - 1.0) * 100.0
        - config.execution.round_trip_cost_bps / 100.0
    )
    label_available = ~entry_fillable | target_date_covered
    net_return = pd.Series(
        np.where(
            ~entry_fillable,
            0.0,
            np.where(
                ~target_date_covered,
                np.nan,
                np.where(
                    fillable,
                    conditional_net,
                    config.execution.non_fill_penalty_pct,
                ),
            ),
        ),
        index=frame.index,
        dtype=float,
    )
    frame[f"exit_{contract_id}_fillable"] = fillable
    frame[f"label_{contract_id}_available"] = label_available
    frame[f"conditional_net_{contract_id}_pct"] = conditional_net
    frame[f"net_{contract_id}_pct"] = net_return
    return frame


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean(values: pd.Series | Any) -> pd.Series:
    if isinstance(values, pd.Series):
        if values.dtype == bool:
            return values.fillna(False)
        normalized = values.astype(str).str.strip().str.lower()
        return normalized.isin(
            {"1", "true", "yes", "y", "qualified", "pass"}
        )
    return pd.Series(dtype=bool)
