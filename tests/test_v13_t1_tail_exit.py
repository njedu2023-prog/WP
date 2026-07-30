from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wp.v3.tail_exit import (
    attach_t1_tail_exit_truth,
    materialize_tail_exit_contract,
    tail_exit_contract_by_id,
)


def _config():
    return SimpleNamespace(
        execution=SimpleNamespace(
            entry_slippage_bps=10,
            round_trip_cost_bps=25,
            min_slot_amount=3_000_000,
            max_entry_pct_of_slot_amount=0.01,
            reference_order_notional=100_000,
            non_fill_penalty_pct=-10.0,
        )
    )


def _predictions(entry_fillable: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20240102"],
            "signal_slot": ["14:20"],
            "ts_code": ["600001.SH"],
            "entry_price": [10.0],
            "entry_fillable": [entry_fillable],
        }
    )


def _panel(include_target: bool = True) -> pd.DataFrame:
    rows = [
        {
            "trade_date": "20240102",
            "signal_slot": "14:20",
            "ts_code": "600001.SH",
            "target_trade_date": "20240103",
            "adj_factor": 1.0,
            "entry_benchmark_price": 10.0,
            "entry_benchmark_amount": 20_000_000,
            "entry_benchmark_volume": 2_000_000,
            "entry_benchmark_bar_lag_minutes": 0.0,
            "down_limit": 9.0,
        }
    ]
    if include_target:
        for slot, price in (
            ("14:20", 11.0),
            ("14:30", 10.8),
            ("14:40", 10.6),
            ("14:50", 10.4),
        ):
            rows.append(
                {
                    "trade_date": "20240103",
                    "signal_slot": slot,
                    "ts_code": "600001.SH",
                    "target_trade_date": "20240104",
                    "adj_factor": 1.0,
                    "entry_benchmark_price": price,
                    "entry_benchmark_amount": 20_000_000,
                    "entry_benchmark_volume": 2_000_000,
                    "entry_benchmark_bar_lag_minutes": 0.0,
                    "down_limit": 9.0,
                }
            )
    return pd.DataFrame(rows)


def test_tail_exit_uses_next_bar_price_slippage_and_costs() -> None:
    attached = attach_t1_tail_exit_truth(
        _predictions(),
        _panel(),
        _config(),
    )
    materialized = materialize_tail_exit_contract(
        attached,
        "t1_1420_next5m",
    )
    expected = ((11.0 * 0.999) / 10.0 - 1.0) * 100.0 - 0.25
    assert materialized.loc[0, "net_return_pct"] == pytest.approx(expected)
    assert bool(materialized.loc[0, "exit_fillable"])
    assert bool(materialized.loc[0, "label_available"])


def test_covered_target_without_liquidity_gets_non_fill_penalty() -> None:
    panel = _panel()
    panel.loc[
        panel["trade_date"].eq("20240103"),
        "entry_benchmark_amount",
    ] = 1_000
    attached = attach_t1_tail_exit_truth(
        _predictions(),
        panel,
        _config(),
    )
    materialized = materialize_tail_exit_contract(
        attached,
        "t1_1430_next5m",
    )
    assert materialized.loc[0, "net_return_pct"] == -10.0
    assert not bool(materialized.loc[0, "exit_fillable"])


def test_uncovered_future_target_is_not_fabricated_as_loss() -> None:
    attached = attach_t1_tail_exit_truth(
        _predictions(),
        _panel(include_target=False),
        _config(),
    )
    materialized = materialize_tail_exit_contract(
        attached,
        "t1_1450_next5m",
    )
    assert np.isnan(materialized.loc[0, "net_return_pct"])
    assert not bool(materialized.loc[0, "label_available"])


def test_failed_entry_remains_zero_return_cash() -> None:
    attached = attach_t1_tail_exit_truth(
        _predictions(entry_fillable=False),
        _panel(include_target=False),
        _config(),
    )
    materialized = materialize_tail_exit_contract(
        attached,
        "t1_1420_next5m",
    )
    assert materialized.loc[0, "net_return_pct"] == 0.0
    assert bool(materialized.loc[0, "label_available"])


def test_unknown_contract_is_rejected() -> None:
    with pytest.raises(KeyError):
        tail_exit_contract_by_id("t1_1500_impossible")
