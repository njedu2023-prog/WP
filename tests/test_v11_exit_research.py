from __future__ import annotations

import pandas as pd

from wp.v3.contracts import V3Config
from wp.v3.exit_research import (
    ExitPolicy,
    apply_exit_policy,
    attach_exit_truth,
    exit_contracts,
    materialize_contract,
)


def _predictions() -> pd.DataFrame:
    rows = []
    for slot, code, entry, close_net in (
        ("14:20", "000001.SZ", 10.0, -1.0),
        ("14:25", "000002.SZ", 20.0, 1.0),
        ("14:30", "000001.SZ", 10.2, 0.5),
    ):
        rows.append(
            {
                "trade_date": "20260105",
                "signal_slot": slot,
                "ts_code": code,
                "entry_price": entry,
                "entry_fillable": True,
                "net_return_pct": close_net,
                "p_net_positive": 0.60,
                "expected_utility_pct": 0.20,
                "p_severe_loss": 0.10,
                "p_round_trip_fill_lower": 0.99,
                "selection_rank_pct": 0.995,
                "selection_score": 1.0 if code == "000001.SZ" else 0.9,
            }
        )
    return pd.DataFrame(rows)


def _panel() -> pd.DataFrame:
    rows = []
    for slot, code, open_price, high_price, close_price in (
        ("14:20", "000001.SZ", 10.10, 10.50, 9.90),
        ("14:25", "000002.SZ", 19.80, 20.02, 20.25),
        ("14:30", "000001.SZ", 10.00, 10.40, 10.28),
    ):
        rows.append(
            {
                "trade_date": "20260105",
                "signal_slot": slot,
                "ts_code": code,
                "adj_factor": 1.0,
                "t1_adj_factor": 1.0,
                "t1_open": open_price,
                "t1_high": high_price,
                "t1_low": min(open_price, close_price) - 0.1,
                "t1_close": close_price,
                "t1_vol": 1_000_000,
                "t1_down_limit": 9.0 if code == "000001.SZ" else 18.0,
                "t1_up_limit": 11.0 if code == "000001.SZ" else 22.0,
                "exit_fillable": True,
            }
        )
    return pd.DataFrame(rows)


def test_attach_exit_truth_builds_open_and_conservative_take_profit_contracts():
    result = attach_exit_truth(_predictions(), _panel(), V3Config())
    first = result.iloc[0]
    assert round(float(first["net_t1_open_auction_pct"]), 4) == 0.75
    assert bool(first["target_hit_tp25_close_fallback"]) is True
    assert round(float(first["net_tp25_close_fallback_pct"]), 4) == 0.25

    second = result.iloc[1]
    assert bool(second["target_hit_tp25_close_fallback"]) is False
    assert round(float(second["net_tp25_close_fallback_pct"]), 4) == 1.0
    assert {
        contract.contract_id for contract in exit_contracts()
    }.issubset(
        {
            column.removeprefix("net_").removesuffix("_pct")
            for column in result.columns
            if column.startswith("net_")
        }
    )


def test_open_at_down_limit_receives_failed_exit_penalty():
    panel = _panel()
    panel.loc[0, "t1_open"] = panel.loc[0, "t1_down_limit"]
    result = attach_exit_truth(_predictions(), panel, V3Config())
    assert result.loc[0, "net_t1_open_auction_pct"] == -10.0
    assert bool(result.loc[0, "exit_t1_open_auction_fillable"]) is False


def test_policy_is_chronological_and_does_not_rewrite_first_signal():
    truth = attach_exit_truth(_predictions(), _panel(), V3Config())
    policy = ExitPolicy(
        contract_id="tp25_close_fallback",
        probability_min=0.50,
        expected_utility_min_pct=0.0,
        severe_loss_max=0.25,
        round_trip_fill_min=0.95,
        selection_rank_min=0.99,
        max_candidates_per_day=2,
    )
    selected = apply_exit_policy(truth, policy)
    assert selected["ts_code"].tolist() == ["000001.SZ", "000002.SZ"]
    assert selected["signal_slot"].tolist() == ["14:20", "14:25"]
    assert selected["exit_contract_id"].eq("tp25_close_fallback").all()


def test_materialized_contract_relabels_profit_and_severe_loss():
    truth = attach_exit_truth(_predictions(), _panel(), V3Config())
    open_contract = materialize_contract(truth, "t1_open_auction")
    assert open_contract["target_net_positive"].tolist() == [1, 0, 0]
    assert open_contract["target_severe_loss"].tolist() == [0, 0, 1]
