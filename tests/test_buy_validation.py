from datetime import datetime

import pandas as pd

import wp.buy_validation as buy_validation
from wp.buy_validation import VALIDATION_COLUMNS, _fill_truth, _in_tail_window, _is_truth_due, _summary, update_buy_plan_validation
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


def test_tail_snapshot_window_accepts_pre_window_fallback():
    assert not _in_tail_window("2026-07-14 14:24:59")
    assert _in_tail_window("2026-07-14 14:25:00")
    assert _in_tail_window("2026-07-14 14:35:00")
    assert _in_tail_window("2026-07-14 14:50:00")
    assert not _in_tail_window("2026-07-14 14:50:01")


def test_truth_becomes_due_only_after_target_day_close():
    assert not _is_truth_due("20260716", datetime(2026, 7, 16, 15, 4, 59, tzinfo=CN_TZ))
    assert _is_truth_due("20260716", datetime(2026, 7, 16, 15, 5, 0, tzinfo=CN_TZ))
    assert _is_truth_due("20260715", datetime(2026, 7, 16, 9, 0, 0, tzinfo=CN_TZ))
    assert not _is_truth_due("20260717", datetime(2026, 7, 16, 16, 0, 0, tzinfo=CN_TZ))


def test_tail_snapshot_keeps_latest_single_primary_stock(tmp_path):
    first_plan = pd.DataFrame(
        [
            {
                "buy_rank": 1,
                "portfolio_group": "主票",
                "ts_code": "000001.SZ",
                "name": "甲",
                "price": 10.0,
                "pct_chg": 8.8,
                "tail_profit_score": 80.0,
                "tail_profit_model_version": "tail_profit_v1",
            },
            {
                "buy_rank": 2,
                "portfolio_group": "标准",
                "ts_code": "000002.SZ",
                "name": "乙",
                "price": 20.0,
                "pct_chg": 9.0,
                "tail_profit_score": 79.0,
                "tail_profit_model_version": "tail_profit_v1",
            },
        ]
    )
    first_health = {
        "data_trade_date": "20260714",
        "market_data_time": "2026-07-14 14:36:00",
        "buy_model_version": "tail_profit_v1",
    }
    current = datetime(2026, 7, 14, 14, 40, tzinfo=CN_TZ)
    first = update_buy_plan_validation(first_plan, first_health, tmp_path, current)

    assert len(first.table) == 1
    assert first.table.iloc[0]["ts_code"] == "000001.SZ"

    latest_plan = first_plan.iloc[[1]].copy()
    latest_plan["buy_rank"] = 1
    latest_plan["portfolio_group"] = "主票"
    latest_health = dict(first_health, market_data_time="2026-07-14 14:46:00")
    latest = update_buy_plan_validation(latest_plan, latest_health, tmp_path, current)

    assert len(latest.table) == 1
    assert latest.table.iloc[0]["ts_code"] == "000002.SZ"
    assert latest.table.iloc[0]["plan_time"] == "2026-07-14 14:46:00"

    after_close_plan = first_plan.iloc[[0]].copy()
    after_close_health = dict(first_health, market_data_time="2026-07-14 15:10:00")
    after_close = update_buy_plan_validation(after_close_plan, after_close_health, tmp_path, current)

    assert len(after_close.table) == 1
    assert after_close.table.iloc[0]["ts_code"] == "000002.SZ"
    assert after_close.table.iloc[0]["plan_time"] == "2026-07-14 14:46:00"


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
