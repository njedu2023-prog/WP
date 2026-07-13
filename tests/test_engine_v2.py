import pandas as pd

import wp.backtest as backtest
from wp.backtest import _buy_monthly_summary, _buy_portfolio_metrics, _data_trade_dates, build_label_frame
from wp.buy_decision import build_buy_decision
from wp.ranking import build_ranked_pool


def test_buy_pool_is_strictly_limited_to_top50():
    scored = pd.DataFrame({"ts_code": [f"S{index:03d}" for index in range(1, 61)], "p_limitup_t1": range(60, 0, -1)})
    full_rank = pd.DataFrame({"ts_code": scored["ts_code"], "rank": range(1, 61)})
    pool = build_ranked_pool(scored, full_rank, 50)
    assert len(pool) == 50
    assert pool["rank"].max() == 50
    assert "S051" not in set(pool["ts_code"])


def test_buy_decision_uses_execution_quality_not_probability_alone():
    frame = pd.DataFrame(
        [
            {"ts_code": "A", "name": "高概率弱结构", "rank": 1, "sector_name": "板块A", "p_limitup_t1": 10, "wp_score": 45, "acceptance_score": 50, "sector_strength_score": 20, "stock_strength_score": 30, "momentum_score": 20, "capital_score": 20, "model_confidence": 55, "risk_penalty_score": 60, "close_position": 60, "pct_chg": 9, "price": 10, "amount": 200000000},
            {"ts_code": "B", "name": "优质结构", "rank": 2, "sector_name": "板块B", "p_limitup_t1": 8, "wp_score": 80, "acceptance_score": 90, "sector_strength_score": 90, "stock_strength_score": 90, "momentum_score": 90, "capital_score": 90, "model_confidence": 90, "risk_penalty_score": 0, "close_position": 90, "pct_chg": 9, "price": 10, "amount": 200000000},
        ]
    )
    result = build_buy_decision(frame, {"buy_max_count": 2})
    assert result.buy_plan.iloc[0]["ts_code"] == "B"
    assert result.buy_plan.iloc[0]["decision_score"] > result.buy_plan.iloc[1]["decision_score"]


def test_missing_market_truth_is_not_counted_as_a_miss():
    today = pd.DataFrame([{"ts_code": "A", "rank": 1}])
    next_day = pd.DataFrame([{"ts_code": "A", "next_day_high": None, "next_day_limitup_price": 11.0}])
    labeled = build_label_frame(today, next_day)
    assert not bool(labeled.loc[0, "label_available"])
    assert pd.isna(labeled.loc[0, "label_t1_limitup"])


def test_buy_portfolio_metrics_equal_weight_each_plan_day():
    trades = pd.DataFrame(
        [
            {"backtest_trade_date": "20260701", "next_day_open_pct": 1, "next_day_max_pct": 5, "next_day_drawdown_pct": -2, "next_day_close_pct": 2},
            {"backtest_trade_date": "20260701", "next_day_open_pct": 3, "next_day_max_pct": 7, "next_day_drawdown_pct": -4, "next_day_close_pct": 4},
            {"backtest_trade_date": "20260702", "next_day_open_pct": -2, "next_day_max_pct": 2, "next_day_drawdown_pct": -5, "next_day_close_pct": -1},
            {"backtest_trade_date": "20260702", "next_day_open_pct": 0, "next_day_max_pct": 4, "next_day_drawdown_pct": -3, "next_day_close_pct": -3},
        ]
    )

    metrics = _buy_portfolio_metrics(trades)

    assert metrics["buy_plan_days"] == 2
    assert metrics["buy_average_count_per_day"] == 2.0
    assert metrics["buy_daily_avg_next_day_open_pct"] == 0.5
    assert metrics["buy_daily_avg_next_day_close_pct"] == 0.5
    assert metrics["buy_plan_day_win_rate"] == 0.5
    assert metrics["buy_cumulative_next_day_close_pct"] == 0.94
    assert metrics["buy_strict5_plan_days"] == 0


def test_strict5_metrics_exclude_underfilled_plan_days():
    trades = pd.DataFrame(
        [
            {
                "backtest_trade_date": date,
                "ts_code": f"{date}-{index}",
                "next_day_open_pct": close_return - 0.5,
                "next_day_max_pct": close_return + 3,
                "next_day_drawdown_pct": close_return - 3,
                "next_day_close_pct": close_return,
                "label_t1_limitup": int(date == "20260701" and index == 0),
            }
            for date, close_return in (("20260701", 1.0), ("20260702", -1.0))
            for index in range(5)
        ]
        + [
            {
                "backtest_trade_date": "20260703",
                "ts_code": "single",
                "next_day_open_pct": 20.0,
                "next_day_max_pct": 20.0,
                "next_day_drawdown_pct": -2.0,
                "next_day_close_pct": 20.0,
                "label_t1_limitup": 1,
            }
        ]
    )

    metrics = _buy_portfolio_metrics(trades)
    monthly = _buy_monthly_summary(trades).iloc[0]

    assert metrics["buy_cumulative_next_day_close_pct"] == 19.988
    assert metrics["buy_strict5_plan_days"] == 2
    assert metrics["buy_strict5_trade_count"] == 10
    assert metrics["buy_strict5_plan_day_win_rate"] == 0.5
    assert metrics["buy_strict5_cumulative_next_day_close_pct"] == -0.01
    assert monthly["strict5_plan_days"] == 2
    assert monthly["strict5_trade_count"] == 10
    assert monthly["strict5_cumulative_close_pct"] == -0.01


def test_data_trade_dates_exclude_empty_holiday_directories():
    original_available = backtest._available_trade_dates
    original_reader = backtest._read_remote_csv
    backtest._available_trade_dates = lambda start, end: ["20260403", "20260406", "20260407"]
    backtest._read_remote_csv = lambda path, cache_root=None: pd.DataFrame() if "20260406" in path else pd.DataFrame([{"ts_code": "A"}])
    try:
        dates = _data_trade_dates("20260403", "20260407")
    finally:
        backtest._available_trade_dates = original_available
        backtest._read_remote_csv = original_reader

    assert dates == ["20260403", "20260407"]
