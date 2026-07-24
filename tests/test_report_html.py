import pandas as pd

from wp.main import load_backtest_summaries
from wp.report_html import render_html
from wp.scoring_model import MODEL_VERSION
from wp.tail_profit_model import TAIL_PROFIT_MODEL_VERSION


def test_report_html_contains_title(tmp_path):
    path = tmp_path / "latest.html"
    render_html(pd.DataFrame(), pd.DataFrame(), {"status": "无符合条件股票", "data_time": "now"}, path)
    assert "WP Top50" in path.read_text(encoding="utf-8")


def test_report_html_polls_manifest_and_keeps_stale_buy_plan_visible(tmp_path):
    path = tmp_path / "latest.html"
    buy_plan = pd.DataFrame(
        [
            {
                "buy_rank": 1,
                "portfolio_group": "主票",
                "ts_code": "000001.SZ",
                "name": "甲",
                "pct_chg": 9.0,
            }
        ]
    )
    render_html(
        pd.DataFrame(),
        pd.DataFrame(),
        {
            "status": "ok",
            "market_data_time": "2026-07-16 14:35:00",
            "wp_run_time": "2026-07-16 14:36:00",
        },
        path,
        buy_plan=buy_plan,
    )

    page = path.read_text(encoding="utf-8")
    assert 'const MANIFEST_URL = "/WP/outputs/json/wp_manifest.json"' in page
    assert "window.setInterval(checkManifest, POLL_INTERVAL_MS)" in page
    assert "manifest.data_revision || manifest.market_data_time" in page
    assert "dataChanged || reportChanged" in page
    assert 'id="stale-data-banner"' in page
    assert 'id="buy-plan-table-wrap"' in page
    assert "市场数据已超过20分钟，名单仍显示，请核对数据时间" in page
    assert "tableWrap.hidden = stale" not in page
    assert "applyTailWindowVisibility(now)" in page
    assert "minuteOfDay < 900" in page
    assert 'http-equiv="refresh"' not in page


def test_report_html_renders_persistent_tail_observation_quality_order(tmp_path):
    path = tmp_path / "latest.html"
    observation_pool = pd.DataFrame(
        [
            {
                "quality_rank": 1,
                "observation_status": "当前主票",
                "rank_change": "升1",
                "first_seen": "2026-07-17 14:25:00",
                "last_seen": "2026-07-17 14:35:00",
                "ts_code": "000002.SZ",
                "name": "乙",
                "pct_chg": 9.4,
                "sector_name": "设备",
                "tail_profit_score": 91.2,
                "risk_penalty_score": 18.0,
                "amount_ratio_5d": 1.8,
                "limit_rule_pct": 10.0,
            },
            {
                "quality_rank": 2,
                "observation_status": "观察票",
                "rank_change": "降1",
                "first_seen": "2026-07-17 14:20:00",
                "last_seen": "2026-07-17 14:35:00",
                "ts_code": "000001.SZ",
                "name": "甲",
                "pct_chg": 8.9,
                "sector_name": "电气",
                "tail_profit_score": 86.7,
                "risk_penalty_score": 24.1,
                "amount_ratio_5d": 2.0,
                "limit_rule_pct": 10.0,
            },
        ]
    )

    render_html(
        pd.DataFrame(),
        pd.DataFrame(),
        {"status": "ok", "data_time": "2026-07-17 14:35:00"},
        path,
        observation_pool=observation_pool,
    )

    page = path.read_text(encoding="utf-8")
    assert "质量排名" in page
    assert "涨跌停规则" in page
    assert "当前主票" in page
    assert "观察票" in page
    assert page.index("000002.SZ") < page.index("000001.SZ")


def test_report_html_groups_validation_by_plan_day(tmp_path):
    path = tmp_path / "latest.html"
    validation = pd.DataFrame(
        [
            {
                "plan_trade_date": "20260716",
                "plan_time": "2026-07-16 14:43:14",
                "target_trade_date": "20260717",
                "buy_rank": 1,
                "ts_code": "000001.SZ",
                "name": "甲",
                "pct_chg_plan": 9.2,
                "actual_pct_chg": "",
                "is_limit_up_t1": "",
                "truth_status": "pending",
            },
            {
                "plan_trade_date": "20260715",
                "plan_time": "2026-07-15 14:25:21",
                "target_trade_date": "20260716",
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
    assert "14:20–14:50 主票累计验证" in page
    assert "累计收盘收益" in page
    assert page.count('class="validation-day-details"') == 2
    assert "2026-07-16" in page
    assert "自 2026-07-15 起；保留窗口内每次实际出现的主票" in page
    assert ">观察记录<" in page
    assert "次日收益（开 / 高 / 收）" in page
    assert "次日最低" in page


def test_report_html_shows_missing_sampling_days_and_closed_message(tmp_path):
    path = tmp_path / "latest.html"
    render_html(
        pd.DataFrame(),
        pd.DataFrame(),
        {"status": "ok", "tail_window_state": "market_closed"},
        path,
        validation_summary={
            "sampling_days": [
                {
                    "plan_trade_date": "20260723",
                    "target_trade_date": "20260724",
                    "sample_status": "missing",
                    "note": "当日未取得14:20-14:50合法窗口快照",
                }
            ]
        },
    )

    page = path.read_text(encoding="utf-8")
    assert "15:00已收盘，停止生成尾盘名单" in page
    assert "2026-07-23" in page
    assert "采样缺失" in page
    assert "采样缺失1日" in page
    assert "当日未取得14:20-14:50合法窗口快照" in page


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


def test_report_html_renders_v2_manual_boundary_and_ohlc_intervals(tmp_path):
    path = tmp_path / "latest.html"
    render_html(
        pd.DataFrame(),
        pd.DataFrame(),
        {"status": "ok", "data_time": "2026-07-20 14:35:00"},
        path,
        decision_support={
            "action": "建议关注买入",
            "candidate_name": "甲",
            "candidate_code": "600001.SH",
            "forecast_mode": "混合先验",
            "forecast_confidence": 60,
            "forecast_open_q10_pct": -2,
            "forecast_open_q50_pct": 0,
            "forecast_open_q90_pct": 2,
            "forecast_high_q10_pct": 1,
            "forecast_high_q50_pct": 4,
            "forecast_high_q90_pct": 9,
            "forecast_low_q10_pct": -5,
            "forecast_low_q50_pct": -2,
            "forecast_low_q90_pct": 0,
            "forecast_close_q10_pct": -3,
            "forecast_close_q50_pct": 1,
            "forecast_close_q90_pct": 5,
            "forecast_profit_probability": 55,
            "forecast_touch_plus3_probability": 65,
            "forecast_touch_minus3_probability": 30,
            "reason": "检查通过",
        },
        market_regime={"state": "允许寻找机会", "score": 62, "reason": "市场较强"},
    )
    page = path.read_text(encoding="utf-8")
    assert "WP V2 人工决策辅助" in page
    assert "仅辅助人工下单，不接入券商、不读取账户、不自动交易" not in page
    assert "次日开盘 Q10 / Q50 / Q90" in page
    assert "-2.00% / +0.00% / +2.00%" in page
    assert "T+1 人工卖出建议" in page
