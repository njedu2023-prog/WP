import pandas as pd

from wp.report_html import render_html


def test_report_html_contains_title(tmp_path):
    path = tmp_path / "latest.html"
    render_html(pd.DataFrame(), pd.DataFrame(), {"status": "无符合条件股票", "data_time": "now"}, path)
    assert "WP Top50" in path.read_text(encoding="utf-8")


def test_report_html_groups_validation_by_plan_day(tmp_path):
    path = tmp_path / "latest.html"
    validation = pd.DataFrame(
        [
            {
                "plan_trade_date": "20260709",
                "plan_time": "2026-07-09 14:43:14",
                "target_trade_date": "20260710",
                "buy_rank": 1,
                "ts_code": "000001.SZ",
                "name": "甲",
                "pct_chg_plan": 9.2,
                "actual_pct_chg": "",
                "is_limit_up_t1": "",
                "truth_status": "pending",
            },
            {
                "plan_trade_date": "20260707",
                "plan_time": "2026-07-07 14:25:21",
                "target_trade_date": "20260708",
                "buy_rank": 1,
                "ts_code": "000002.SZ",
                "name": "乙",
                "pct_chg_plan": 8.8,
                "actual_pct_chg": 5.0,
                "is_limit_up_t1": False,
                "truth_status": "verified",
            },
        ]
    )
    summary = {
        "total_plan_days": 2,
        "verified_plan_days": 1,
        "total_records": 2,
        "verified_records": 1,
        "positive_records": 1,
        "positive_rate": 100.0,
        "limit_up_records": 0,
        "limit_up_rate": 0.0,
        "daily_average_pct_chg": 5.0,
        "cumulative_pct_chg": 5.0,
    }
    render_html(
        pd.DataFrame(),
        pd.DataFrame(),
        {"status": "ok", "data_time": "now"},
        path,
        validation=validation,
        validation_summary=summary,
    )
    page = path.read_text(encoding="utf-8")
    assert "14:20 观察名单累计验证" in page
    assert "累计组合" in page
    assert page.count('class="validation-day-details"') == 2
    assert "2026-07-09" in page
    assert "按每个计划日最后一份名单统计" in page
