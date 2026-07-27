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
