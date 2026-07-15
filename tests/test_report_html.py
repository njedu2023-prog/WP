import pandas as pd

from wp.main import load_backtest_summaries
from wp.report_html import render_html
from wp.scoring_model import MODEL_VERSION
from wp.tail_profit_model import TAIL_PROFIT_MODEL_VERSION


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
                "plan_price": 10.0,
                "pct_chg_plan": 8.8,
                "actual_pct_chg": 5.0,
                "return_open_pct": 1.0,
                "return_high_pct": 8.0,
                "return_low_pct": -2.0,
                "return_close_pct": 4.0,
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
        "average_open_return_pct": 1.0,
        "average_high_return_pct": 8.0,
        "daily_average_pct_chg": 4.0,
        "cumulative_pct_chg": 4.0,
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
    assert "14:35 主票累计验证" in page
    assert "累计收盘收益" in page
    assert page.count('class="validation-day-details"') == 2
    assert "2026-07-09" in page
    assert "每日最多 1 支；无合格则空仓；按计划价验证次日收益" in page
    assert ">主票<" in page
    assert "次日收益（开 / 高 / 收）" in page
    assert "次日最低" in page


def test_report_html_contains_backtest_windows_and_data_links(tmp_path):
    path = tmp_path / "latest.html"
    render_html(
        pd.DataFrame(),
        pd.DataFrame(),
        {"status": "ok", "data_time": "now"},
        path,
        backtests=[
            {
                "model_version": "wp_rule_v2_1",
                "start_date": "20260427",
                "end_date": "20260522",
                "trade_days": 20,
                "trade_count": 736,
                "auc": 0.6209,
                "hit_top10": 0.05,
                "avg_next_day_close_pct_top10": 0.6215,
                "buy_plan_days": 20,
                "buy_trade_count": 70,
                "buy_average_count_per_day": 3.5,
                "buy_positive_close_rate": 0.5286,
                "buy_limitup_rate": 0.0429,
                "buy_daily_avg_next_day_open_pct": 0.11,
                "buy_daily_avg_next_day_high_pct": 4.2,
                "buy_daily_avg_next_day_close_pct": 0.2133,
                "buy_cumulative_next_day_close_pct": 4.34,
                "buy_strict5_plan_days": 20,
                "buy_strict5_trade_count": 100,
                "buy_strict5_positive_close_rate": 0.53,
                "buy_strict5_limitup_rate": 0.04,
                "buy_strict5_daily_avg_next_day_open_pct": 0.11,
                "buy_strict5_daily_avg_next_day_high_pct": 4.2,
                "buy_strict5_daily_avg_next_day_close_pct": 0.2133,
                "buy_strict5_cumulative_next_day_close_pct": 4.34,
            }
        ],
    )
    page = path.read_text(encoding="utf-8")
    assert "模型回测验证" in page
    assert "2026-04-27 至 2026-05-22" in page
    assert "0.6209" in page
    assert "观察日" in page
    assert "+4.34%" in page
    assert "../backtests/20260427_20260522/buy_trades.csv" in page
    assert "主票明细" in page
    assert "收盘日线代理；不替代 14:35 真实快照" in page
    assert "../backtests/20260427_20260522/monthly_summary.csv" in page


def test_backtest_summary_list_hides_contained_windows(tmp_path):
    root = tmp_path / "outputs"
    windows = [
        ("20260313_20260709", "20260313", "20260709"),
        ("20260427_20260522", "20260427", "20260522"),
        ("20260622_20260709", "20260622", "20260709"),
    ]
    for folder, start, end in windows:
        path = root / "backtests" / folder / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            f'{{"model_version":"{MODEL_VERSION}","buy_model_version":"{TAIL_PROFIT_MODEL_VERSION}","start_date":"{start}","end_date":"{end}"}}',
            encoding="utf-8",
        )

    summaries = load_backtest_summaries(root)

    assert [(item["start_date"], item["end_date"]) for item in summaries] == [("20260313", "20260709")]
