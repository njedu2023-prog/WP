from __future__ import annotations

import pandas as pd
import pytest

from wp.v3.contracts import V3Config
from wp.v3.truth import _verify_candidate


def test_truth_uses_immutable_first_signal_price():
    candidate = {
        "ts_code": "600001.SH",
        "first_signal_price": 10.0,
        "entry_adj_factor": 1.0,
        "truth_status": "pending",
    }
    truth = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "close": 10.10,
                "vol": 1000,
                "down_limit": 9.0,
                "adj_factor": 1.0,
            }
        ]
    ).set_index("ts_code")
    _verify_candidate(candidate, truth, V3Config())
    assert candidate["first_signal_price"] == 10.0
    assert candidate["entry_price"] == pytest.approx(10.01)
    assert candidate["net_return_pct"] == pytest.approx(0.6491008991)
    assert candidate["truth_status"] == "verified"


def test_v6_truth_uses_settled_entry_and_never_falls_back_to_signal_price():
    config = V3Config()
    candidate = {
        "ts_code": "600001.SH",
        "first_signal_price": 10.0,
        "entry_contract": config.execution.entry_price_contract,
        "entry_benchmark_status": "SETTLED",
        "entry_benchmark_price": 10.5,
        "entry_price": 10.5105,
        "entry_fillable": True,
        "entry_adj_factor": 1.0,
        "truth_status": "pending",
    }
    truth = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "close": 10.6,
                "vol": 1000,
                "down_limit": 9.0,
                "adj_factor": 1.0,
            }
        ]
    ).set_index("ts_code")

    _verify_candidate(candidate, truth, config)

    assert candidate["entry_price"] == 10.5105
    assert candidate["gross_return_pct"] == pytest.approx(
        (10.6 / 10.5105 - 1.0) * 100.0
    )
    assert candidate["truth_contract"].startswith("immutable_next_5m")


def test_v6_truth_remains_pending_without_entry_benchmark():
    config = V3Config()
    candidate = {
        "ts_code": "600001.SH",
        "first_signal_price": 10.0,
        "entry_contract": config.execution.entry_price_contract,
        "entry_benchmark_status": "PENDING",
        "entry_adj_factor": 1.0,
        "truth_status": "pending",
    }
    truth = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "close": 10.6,
                "vol": 1000,
                "down_limit": 9.0,
                "adj_factor": 1.0,
            }
        ]
    ).set_index("ts_code")

    _verify_candidate(candidate, truth, config)

    assert candidate["truth_status"] == "pending"
    assert candidate["truth_error"] == "missing_entry_benchmark"
    assert candidate.get("entry_price") is None


def test_down_limit_close_is_counted_as_execution_failure():
    candidate = {
        "ts_code": "600001.SH",
        "first_signal_price": 10.0,
        "entry_adj_factor": 1.0,
        "truth_status": "pending",
    }
    truth = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "close": 9.0,
                "vol": 1000,
                "down_limit": 9.0,
                "adj_factor": 1.0,
            }
        ]
    ).set_index("ts_code")
    _verify_candidate(candidate, truth, V3Config())
    assert candidate["exit_fillable"] is False
    assert candidate["net_positive"] is False
    assert candidate["net_return_pct"] <= -10


def test_truth_uses_adjustment_factor_total_return():
    candidate = {
        "ts_code": "600001.SH",
        "first_signal_price": 10.0,
        "entry_adj_factor": 1.0,
        "truth_status": "pending",
    }
    truth = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "close": 9.0,
                "vol": 1000,
                "down_limit": 8.1,
                "adj_factor": 1.1333333333333333,
            }
        ]
    ).set_index("ts_code")
    _verify_candidate(candidate, truth, V3Config())
    assert candidate["t1_total_return_close"] == pytest.approx(10.2)
    assert candidate["net_positive"] is True
    assert candidate["corporate_action_adjustment"] == "adj_factor_total_return"


def test_missing_adjustment_factor_does_not_fabricate_raw_price_truth():
    candidate = {
        "ts_code": "600001.SH",
        "first_signal_price": 10.0,
        "entry_adj_factor": None,
        "truth_status": "pending",
    }
    truth = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "close": 10.2,
                "vol": 1000,
                "down_limit": 9.0,
                "adj_factor": 1.0,
            }
        ]
    ).set_index("ts_code")
    _verify_candidate(candidate, truth, V3Config())
    assert candidate["truth_status"] == "pending"
    assert candidate["truth_error"] == "missing_adjustment_factor"


def test_t1_suspension_is_verified_as_a_contract_failure():
    candidate = {
        "ts_code": "600001.SH",
        "first_signal_price": 10.0,
        "entry_adj_factor": 1.0,
        "truth_status": "pending",
    }
    truth = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "close": float("nan"),
                "vol": float("nan"),
                "down_limit": 9.0,
                "adj_factor": 1.0,
            }
        ]
    ).set_index("ts_code")
    _verify_candidate(candidate, truth, V3Config())
    assert candidate["truth_status"] == "verified"
    assert candidate["exit_fillable"] is False
    assert candidate["net_return_pct"] == -10.0


def test_truth_uses_authoritative_entry_day_adjustment_factor():
    candidate = {
        "trade_date": "20260721",
        "ts_code": "600001.SH",
        "first_signal_price": 10.0,
        "entry_adj_factor": 1.0,
        "truth_status": "pending",
    }
    entry_truth = pd.DataFrame(
        [{"ts_code": "600001.SH", "adj_factor": 1.1}]
    ).set_index("ts_code")
    target_truth = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "close": 10.0,
                "vol": 1000,
                "down_limit": 9.0,
                "adj_factor": 1.1,
            }
        ]
    ).set_index("ts_code")

    _verify_candidate(
        candidate,
        target_truth,
        V3Config(),
        entry_truth=entry_truth,
    )

    assert candidate["entry_adj_factor_truth"] == 1.1
    assert candidate["t1_total_return_close"] == pytest.approx(10.0)
