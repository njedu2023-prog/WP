from datetime import datetime

import pandas as pd

import wp.buy_validation as buy_validation
from wp.buy_validation import VALIDATION_COLUMNS, _fill_truth, _in_tail_window, _summary
from wp.calendar import CN_TZ


def test_summary_compounds_complete_daily_portfolios_only():
    table = pd.DataFrame(
        [
            {"plan_trade_date": "20260701", "truth_status": "verified", "actual_pct_chg": 10.0, "is_limit_up_t1": True},
            {"plan_trade_date": "20260701", "truth_status": "verified", "actual_pct_chg": -2.0, "is_limit_up_t1": False},
            {"plan_trade_date": "20260702", "truth_status": "verified", "actual_pct_chg": 2.0, "is_limit_up_t1": False},
            {"plan_trade_date": "20260702", "truth_status": "verified", "actual_pct_chg": 2.0, "is_limit_up_t1": False},
            {"plan_trade_date": "20260703", "truth_status": "pending", "actual_pct_chg": "", "is_limit_up_t1": ""},
        ]
    )

    summary = _summary(table)

    assert summary["total_plan_days"] == 3
    assert summary["verified_plan_days"] == 2
    assert summary["pending_plan_days"] == 1
    assert summary["verified_records"] == 4
    assert summary["positive_records"] == 3
    assert summary["positive_rate"] == 75.0
    assert summary["limit_up_records"] == 1
    assert summary["daily_average_pct_chg"] == 3.0
    assert summary["plan_day_win_rate"] == 100.0
    assert summary["cumulative_pct_chg"] == 6.08


def test_summary_prefers_plan_price_return_over_market_pct_change():
    table = pd.DataFrame(
        [
            {
                "plan_trade_date": "20260707",
                "truth_status": "verified",
                "actual_pct_chg": 6.0,
                "return_open_pct": -1.0,
                "return_high_pct": 4.0,
                "return_low_pct": -3.0,
                "return_close_pct": -2.0,
                "is_limit_up_t1": False,
            }
        ]
    )

    summary = _summary(table)

    assert summary["positive_records"] == 0
    assert summary["average_close_return_pct"] == -2.0
    assert summary["daily_average_pct_chg"] == -2.0
    assert summary["cumulative_pct_chg"] == -2.0


def test_tail_snapshot_window_starts_at_1435():
    assert not _in_tail_window("2026-07-14 14:34:59")
    assert _in_tail_window("2026-07-14 14:35:00")
    assert _in_tail_window("2026-07-14 14:50:00")
    assert not _in_tail_window("2026-07-14 14:50:01")


def test_summary_can_scope_records_to_current_buy_model():
    table = pd.DataFrame(
        [
            {"buy_model_version": "legacy", "plan_trade_date": "20260701", "truth_status": "verified", "return_close_pct": -5.0, "is_limit_up_t1": False},
            {"buy_model_version": "tail_profit_v1", "plan_trade_date": "20260702", "truth_status": "verified", "return_close_pct": 3.0, "is_limit_up_t1": False},
        ]
    )

    summary = _summary(table, "tail_profit_v1")

    assert summary["buy_model_version"] == "tail_profit_v1"
    assert summary["verified_records"] == 1
    assert summary["cumulative_pct_chg"] == 3.0


def test_fill_truth_reconstructs_plan_price_and_entry_returns():
    table = pd.DataFrame(
        [
            {
                "plan_trade_date": "20260707",
                "plan_time": "2026-07-07 14:20:00",
                "target_trade_date": "20260708",
                "ts_code": "000001.SZ",
                "pct_chg_plan": 10.0,
                "truth_status": "verified",
            }
        ]
    ).reindex(columns=VALIDATION_COLUMNS, fill_value="")
    source = pd.DataFrame([{"ts_code": "000001.SZ", "pre_close": 10.0}])
    target = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "open": 11.0,
                "high": 12.1,
                "low": 10.0,
                "close": 11.55,
                "pre_close": 11.0,
                "pct_chg": 5.0,
                "up_limit": 12.1,
            }
        ]
    )
    original = buy_validation._fetch_truth_by_date
    buy_validation._fetch_truth_by_date = lambda date: (source, "") if date == "20260707" else (target, "")
    try:
        result = _fill_truth(table, datetime(2026, 7, 9, 16, 0, tzinfo=CN_TZ))
    finally:
        buy_validation._fetch_truth_by_date = original

    row = result.iloc[0]
    assert row["plan_price"] == 11.0
    assert row["return_open_pct"] == 0.0
    assert row["return_high_pct"] == 10.0
    assert row["return_close_pct"] == 5.0
    assert bool(row["is_limit_up_t1"])
    assert row["truth_status"] == "verified"
