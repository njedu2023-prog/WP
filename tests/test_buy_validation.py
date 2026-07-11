import pandas as pd

from wp.buy_validation import _summary


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
