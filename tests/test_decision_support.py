import pandas as pd

from wp.decision_support import build_decision_support


def _observation(qualified_runs=3, leader_runs=3, utility=1.2):
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
                "forecast_mode": "混合先验",
                "forecast_confidence": 60,
                "forecast_risk_adjusted_utility": utility,
                "forecast_profit_probability": 58,
            },
            {
                "qualification_status": "合格",
                "ts_code": "600002.SH",
                "name": "乙",
                "tail_profit_score": 75,
                "risk_penalty_score": 22,
                "qualified_runs": 2,
                "leader_runs": 0,
                "forecast_mode": "混合先验",
                "forecast_confidence": 55,
                "forecast_risk_adjusted_utility": 0.3,
            },
        ]
    )


def test_decision_support_can_recommend_one_human_review_candidate():
    result = build_decision_support(
        _observation(),
        {"state": "允许寻找机会", "score": 66, "reason": "市场较强"},
        "2026-07-20 14:35:00",
    )
    assert result.summary["action"] == "建议关注买入"
    assert result.summary["candidate_code"] == "600001.SH"
    assert result.table["is_current_choice"].sum() == 1
    assert result.summary["order_routing_enabled"] is False


def test_decision_support_waits_then_allows_no_trade():
    early = build_decision_support(
        _observation(qualified_runs=1, leader_runs=1, utility=-1),
        {"state": "允许寻找机会", "score": 60},
        "2026-07-20 14:25:00",
    )
    final = build_decision_support(
        _observation(qualified_runs=1, leader_runs=1, utility=-1),
        {"state": "允许寻找机会", "score": 60},
        "2026-07-20 14:50:00",
    )
    assert early.summary["action"] == "继续观察"
    assert final.summary["action"] == "建议空仓"


def test_decision_support_never_buys_in_avoid_regime():
    result = build_decision_support(_observation(), {"state": "回避", "score": 20}, "2026-07-20 14:40:00")
    assert result.summary["action"] == "建议空仓"
    assert result.summary["broker_connection"] == "disabled"
