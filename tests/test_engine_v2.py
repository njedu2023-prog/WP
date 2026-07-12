import pandas as pd

from wp.backtest import _buy_portfolio_metrics, build_label_frame
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
