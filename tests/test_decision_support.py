import pandas as pd

from wp.decision_support import build_decision_support


def _observation(qualified_runs=3, leader_runs=3, probability=65.0):
    return pd.DataFrame(
        [
            {
                "qualification_status": "合格",
                "ts_code": "600001.SH",
                "name": "甲",
                "sector_name": "电力",
                "price": 10,
                "pct_chg": 8.5,
                "tail_profit_score": 82,
                "risk_penalty_score": 20,
                "qualified_runs": qualified_runs,
                "leader_runs": leader_runs,
                "forecast_mode": "实时因果样本",
                "forecast_actionable": True,
                "forecast_confidence": 75,
                "forecast_live_sample_count": 40,
                "forecast_live_day_count": 40,
                "forecast_effective_sample_count": 25,
                "forecast_profit_probability": probability,
                "forecast_profit_probability_lower": 55,
                "forecast_expected_net_return_pct": 1.0,
                "forecast_downside_q10_pct": -2.0,
            },
            {
                "qualification_status": "合格",
                "ts_code": "600002.SH",
                "name": "乙",
                "tail_profit_score": 75,
                "risk_penalty_score": 22,
                "qualified_runs": 2,
                "leader_runs": 0,
                "forecast_mode": "实时因果样本",
                "forecast_actionable": True,
                "forecast_confidence": 70,
                "forecast_live_sample_count": 40,
                "forecast_live_day_count": 40,
                "forecast_effective_sample_count": 20,
                "forecast_profit_probability": 60,
                "forecast_profit_probability_lower": 51,
                "forecast_expected_net_return_pct": 0.5,
                "forecast_downside_q10_pct": -3.0,
            },
        ]
    )


def test_decision_support_can_lock_one_profitable_candidate():
    result = build_decision_support(
        _observation(),
        {"state": "允许寻找机会", "score": 66, "reason": "市场较强"},
        "2026-07-20 14:45:00",
    )
    assert result.summary["action"] == "买入"
    assert result.summary["action_code"] == "BUY"
    assert result.summary["is_final"] is True
    assert result.summary["candidate_code"] == "600001.SH"
    assert result.table["is_current_choice"].sum() == 1
    assert result.summary["order_routing_enabled"] is False


def test_decision_support_waits_then_allows_no_trade():
    early = build_decision_support(
        _observation(qualified_runs=1, leader_runs=1, probability=45),
        {"state": "允许寻找机会", "score": 60},
        "2026-07-20 14:25:00",
    )
    final = build_decision_support(
        _observation(qualified_runs=1, leader_runs=1, probability=45),
        {"state": "允许寻找机会", "score": 60},
        "2026-07-20 14:55:00",
    )
    assert early.summary["action"] == "继续观察"
    assert final.summary["action"] == "NO_TRADE"
    assert final.summary["is_final"] is True


def test_decision_support_never_buys_in_avoid_regime():
    result = build_decision_support(_observation(), {"state": "回避", "score": 20}, "2026-07-20 14:55:00")
    assert result.summary["action"] == "NO_TRADE"
    assert result.summary["broker_connection"] == "disabled"


def test_decision_support_never_generates_a_candidate_after_market_close():
    result = build_decision_support(
        _observation(),
        {"state": "允许寻找机会", "score": 80},
        "2026-07-20 15:00:00",
    )

    assert result.summary["action"] == "已收盘"
    assert result.summary["candidate_code"] == ""
    assert result.table.empty
