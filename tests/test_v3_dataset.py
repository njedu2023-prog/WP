from __future__ import annotations

import pandas as pd
import pytest

from wp.v3.contracts import V3Config
from wp.v3.dataset import build_supervised_panel, first_crossing_candidates
from wp.v3.features import assert_feature_contract


def test_label_uses_delayed_entry_benchmark_and_t1_close_after_costs():
    raw = pd.DataFrame(
        [
            {
                "trade_date": "20260723",
                "target_trade_date": "20260724",
                "signal_slot": "14:20",
                "ts_code": "600001.SH",
                "board": "main_board",
                "signal_price": 10.0,
                "entry_benchmark_price": 10.0,
                "t1_close": 10.10,
                "t1_total_return_close": 10.10,
                "adj_factor": 1.0,
                "listing_days": 500,
                "prev_20d_amount": 300_000_000,
                "slot_amount": 20_000_000,
                "distance_to_up_limit_pct": 3.0,
                "distance_to_down_limit_pct": 12.0,
                "entry_fillable": True,
                "exit_fillable": True,
            }
        ]
    )
    panel = build_supervised_panel(raw, V3Config())
    assert panel.loc[0, "entry_price"] == pytest.approx(10.01)
    assert 0.64 < panel.loc[0, "net_return_pct"] < 0.66
    assert panel.loc[0, "target_net_positive"] == 1


def test_non_executable_exit_is_always_a_failure():
    raw = pd.DataFrame(
        [
            {
                "trade_date": "20260723",
                "target_trade_date": "20260724",
                "signal_slot": "14:20",
                "ts_code": "600001.SH",
                "board": "main_board",
                "signal_price": 10.0,
                "entry_benchmark_price": 10.0,
                "t1_close": 11.0,
                "t1_total_return_close": 11.0,
                "adj_factor": 1.0,
                "listing_days": 500,
                "prev_20d_amount": 300_000_000,
                "slot_amount": 20_000_000,
                "distance_to_up_limit_pct": 3.0,
                "distance_to_down_limit_pct": 12.0,
                "entry_fillable": True,
                "exit_fillable": False,
            }
        ]
    )
    panel = build_supervised_panel(raw, V3Config())
    assert panel.loc[0, "target_net_positive"] == 0
    assert panel.loc[0, "net_return_pct"] == pytest.approx(-10.0)
    assert panel.loc[0, "target_entry_fillable"] == 1
    assert panel.loc[0, "target_exit_fillable"] == 0


def test_missing_t1_close_on_suspension_is_an_observed_failure():
    raw = pd.DataFrame(
        [
            {
                "trade_date": "20260723",
                "target_trade_date": "20260724",
                "signal_slot": "14:20",
                "ts_code": "600001.SH",
                "board": "main_board",
                "signal_price": 10.0,
                "entry_benchmark_price": 10.0,
                "t1_close": float("nan"),
                "t1_total_return_close": float("nan"),
                "adj_factor": 1.0,
                "listing_days": 500,
                "prev_20d_amount": 300_000_000,
                "slot_amount": 20_000_000,
                "distance_to_up_limit_pct": 3.0,
                "distance_to_down_limit_pct": 12.0,
                "entry_fillable": True,
                "exit_fillable": False,
            }
        ]
    )
    panel = build_supervised_panel(raw, V3Config())
    assert bool(panel.loc[0, "label_available"]) is True
    assert panel.loc[0, "target_net_positive"] == 0
    assert panel.loc[0, "net_return_pct"] == pytest.approx(-10.0)
    assert panel.loc[0, "target_entry_fillable"] == 1
    assert panel.loc[0, "target_exit_fillable"] == 0


def test_future_target_truth_is_pending_not_an_exit_failure():
    raw = pd.DataFrame(
        [
            {
                "trade_date": "20260731",
                "target_trade_date": "20260803",
                "signal_slot": "14:00",
                "ts_code": "600001.SH",
                "board": "main_board",
                "signal_price": 10.0,
                "entry_benchmark_price": 10.0,
                "t1_close": float("nan"),
                "t1_total_return_close": float("nan"),
                "adj_factor": 1.0,
                "listing_days": 500,
                "prev_20d_amount": 300_000_000,
                "slot_amount": 20_000_000,
                "distance_to_up_limit_pct": 3.0,
                "distance_to_down_limit_pct": 12.0,
                "entry_fillable": True,
                "exit_fillable": False,
                "target_market_truth_available": False,
            }
        ]
    )

    panel = build_supervised_panel(raw, V3Config())

    assert bool(panel.loc[0, "label_available"]) is False
    assert pd.isna(panel.loc[0, "target_net_positive"])
    assert pd.isna(panel.loc[0, "net_return_pct"])


def test_missing_next_bar_is_an_observed_entry_failure():
    raw = pd.DataFrame(
        [
            {
                "trade_date": "20260723",
                "target_trade_date": "20260724",
                "signal_slot": "14:20",
                "ts_code": "600001.SH",
                "board": "main_board",
                "signal_price": 10.0,
                "entry_benchmark_price": float("nan"),
                "adj_factor": 1.0,
                "t1_close": 10.5,
                "t1_total_return_close": 10.5,
                "listing_days": 500,
                "prev_20d_amount": 300_000_000,
                "slot_amount": 20_000_000,
                "distance_to_up_limit_pct": 3.0,
                "distance_to_down_limit_pct": 12.0,
                "entry_fillable": False,
                "exit_fillable": True,
            }
        ]
    )

    panel = build_supervised_panel(raw, V3Config())

    assert bool(panel.loc[0, "label_available"]) is True
    assert panel.loc[0, "target_net_positive"] == 0
    assert panel.loc[0, "net_return_pct"] == pytest.approx(0.0)
    assert panel.loc[0, "target_entry_fillable"] == 0
    assert pd.isna(panel.loc[0, "target_exit_fillable"])


def test_all_in_rank_target_includes_entry_and_exit_failures():
    base = {
        "trade_date": "20260723",
        "target_trade_date": "20260724",
        "signal_slot": "14:20",
        "board": "main_board",
        "signal_price": 10.0,
        "entry_benchmark_price": 10.0,
        "adj_factor": 1.0,
        "t1_close": 10.2,
        "t1_total_return_close": 10.2,
        "listing_days": 500,
        "prev_20d_amount": 300_000_000,
        "slot_amount": 20_000_000,
        "distance_to_up_limit_pct": 3.0,
        "distance_to_down_limit_pct": 12.0,
        "entry_fillable": True,
        "exit_fillable": True,
    }
    rows = [
        {**base, "ts_code": "600001.SH"},
        {
            **base,
            "ts_code": "600002.SH",
            "entry_benchmark_price": float("nan"),
            "entry_fillable": False,
        },
        {
            **base,
            "ts_code": "600003.SH",
            "exit_fillable": False,
        },
    ]

    panel = build_supervised_panel(pd.DataFrame(rows), V3Config()).set_index(
        "ts_code"
    )

    assert panel.loc["600001.SH", "_target_net_return_rank"] == pytest.approx(1.0)
    assert panel.loc["600002.SH", "_target_net_return_rank"] == pytest.approx(2 / 3)
    assert panel.loc["600003.SH", "_target_net_return_rank"] == pytest.approx(1 / 3)
    assert int(panel["target_cross_section_top"].sum()) == 1


def test_unadjusted_t1_truth_is_rejected_instead_of_used_as_fallback():
    raw = pd.DataFrame(
        [
            {
                "trade_date": "20260723",
                "signal_slot": "14:20",
                "ts_code": "600001.SH",
                "signal_price": 10.0,
                "entry_benchmark_price": 10.0,
                "entry_fillable": True,
                "exit_fillable": True,
                "adj_factor": 1.0,
                "t1_close": 10.2,
            }
        ]
    )

    with pytest.raises(ValueError, match="t1_total_return_close"):
        build_supervised_panel(raw, V3Config())


def test_label_uses_adjustment_factor_total_return_across_ex_dividend_day():
    raw = pd.DataFrame(
        [
            {
                "trade_date": "20260723",
                "target_trade_date": "20260724",
                "signal_slot": "14:20",
                "ts_code": "600001.SH",
                "board": "main_board",
                "signal_price": 10.0,
                "entry_benchmark_price": 10.0,
                "adj_factor": 1.0,
                "t1_close": 9.0,
                "t1_total_return_close": 10.2,
                "listing_days": 500,
                "prev_20d_amount": 300_000_000,
                "slot_amount": 20_000_000,
                "distance_to_up_limit_pct": 3.0,
                "distance_to_down_limit_pct": 12.0,
                "entry_fillable": True,
                "exit_fillable": True,
            }
        ]
    )
    panel = build_supervised_panel(raw, V3Config())
    assert panel.loc[0, "gross_return_pct"] == pytest.approx(
        (10.2 / 10.01 - 1.0) * 100.0
    )
    assert panel.loc[0, "target_net_positive"] == 1


def test_missing_current_adjustment_factor_is_not_execution_eligible():
    raw = pd.DataFrame(
        [
            {
                "trade_date": "20260723",
                "target_trade_date": "20260724",
                "signal_slot": "14:20",
                "ts_code": "600001.SH",
                "board": "main_board",
                "signal_price": 10.0,
                "entry_benchmark_price": 10.0,
                "adj_factor": float("nan"),
                "t1_close": 10.2,
                "t1_total_return_close": 10.2,
                "listing_days": 500,
                "prev_20d_amount": 300_000_000,
                "slot_amount": 20_000_000,
                "distance_to_up_limit_pct": 3.0,
                "distance_to_down_limit_pct": 12.0,
                "entry_fillable": True,
                "exit_fillable": True,
            }
        ]
    )
    panel = build_supervised_panel(raw, V3Config())
    assert bool(panel.loc[0, "execution_eligible"]) is False


def test_stale_symbol_bar_is_not_execution_eligible():
    raw = pd.DataFrame(
        [
            {
                "trade_date": "20260723",
                "target_trade_date": "20260724",
                "signal_slot": "14:20",
                "ts_code": "600001.SH",
                "board": "main_board",
                "signal_price": 10.0,
                "entry_benchmark_price": 10.0,
                "adj_factor": 1.0,
                "t1_close": 10.2,
                "t1_total_return_close": 10.2,
                "listing_days": 500,
                "prev_20d_amount": 300_000_000,
                "slot_amount": 20_000_000,
                "slot_bar_lag_minutes": 10,
                "distance_to_up_limit_pct": 3.0,
                "distance_to_down_limit_pct": 12.0,
                "entry_fillable": True,
                "exit_fillable": True,
            }
        ]
    )
    panel = build_supervised_panel(raw, V3Config())
    assert bool(panel.loc[0, "execution_eligible"]) is False


def test_cold_session_without_five_observations_is_not_execution_eligible():
    raw = pd.DataFrame(
        [
            {
                "trade_date": "20260723",
                "target_trade_date": "20260724",
                "signal_slot": "14:20",
                "ts_code": "600001.SH",
                "board": "main_board",
                "signal_price": 10.0,
                "entry_benchmark_price": 10.0,
                "adj_factor": 1.0,
                "t1_close": 10.2,
                "t1_total_return_close": 10.2,
                "listing_days": 500,
                "prev_20d_amount": 300_000_000,
                "slot_amount": 20_000_000,
                "slot_bar_lag_minutes": 0,
                "intraday_snapshot_count": 1,
                "distance_to_up_limit_pct": 3.0,
                "distance_to_down_limit_pct": 12.0,
                "entry_fillable": True,
                "exit_fillable": True,
            }
        ]
    )
    panel = build_supervised_panel(raw, V3Config())
    assert bool(panel.loc[0, "execution_eligible"]) is False


def test_fixed_1400_contract_ignores_legacy_early_slots():
    predictions = pd.DataFrame(
        [
            {"trade_date": "20260723", "signal_slot": "14:00", "ts_code": "600001.SH", "passes_policy": True, "signal_price": 10.2},
            {"trade_date": "20260723", "signal_slot": "14:20", "ts_code": "600001.SH", "passes_policy": True, "signal_price": 10.0},
            {"trade_date": "20260723", "signal_slot": "14:25", "ts_code": "600002.SH", "passes_policy": False, "signal_price": 8.0},
        ]
    )
    selected = first_crossing_candidates(predictions, V3Config())
    assert selected[["ts_code", "signal_slot", "signal_price"]].to_dict("records") == [
        {"ts_code": "600001.SH", "signal_slot": "14:00", "signal_price": 10.2}
    ]


def test_future_columns_cannot_enter_feature_contract():
    try:
        assert_feature_contract(["next_close"])
    except ValueError as error:
        assert "unregistered" in str(error) or "future-aware" in str(error)
    else:
        raise AssertionError("future feature was accepted")
