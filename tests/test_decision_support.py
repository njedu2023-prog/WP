import pandas as pd

from wp.decision_support import build_decision_support


def _observation() -> pd.DataFrame:
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
                "qualified_runs": 3,
                "leader_runs": 3,
                "forecast_mode": "实时因果样本",
                "forecast_actionable": True,
                "forecast_confidence": 75,
                "forecast_live_sample_count": 40,
                "forecast_live_day_count": 40,
                "forecast_effective_sample_count": 25,
                "forecast_profit_probability": 65,
                "forecast_profit_probability_lower": 55,
                "forecast_expected_net_return_pct": 1.0,
                "forecast_downside_q10_pct": -2.0,
            },
            {
                "qualification_status": "合格",
                "ts_code": "600002.SH",
                "name": "乙",
                "sector_name": "机械",
                "price": 20,
                "pct_chg": 8.2,
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


def test_decision_support_publishes_every_qualified_candidate():
    result = build_decision_support(
        _observation(),
        {"state": "允许寻找机会", "score": 66, "reason": "市场较强"},
        "2026-07-20 14:25:00",
    )

    assert result.summary["action_code"] == "QUALIFIED_SET"
    assert result.summary["qualified_count"] == 2
    assert result.summary["candidate_code"] == ""
    assert result.table["support_action"].eq("合格").all()
    assert not result.table["is_current_choice"].any()
    assert result.summary["order_routing_enabled"] is False


def test_decision_support_applies_gates_per_candidate():
    pool = _observation()
    pool.loc[1, "forecast_profit_probability_lower"] = 40

    result = build_decision_support(
        pool,
        {"state": "允许寻找机会", "score": 60},
        "2026-07-20 14:35:00",
    )

    assert result.summary["qualified_count"] == 1
    assert result.table["support_action"].tolist() == ["合格", "观察"]
    assert "概率下界" in result.table.iloc[1]["checks_failed"]


def test_decision_support_freezes_after_1450():
    result = build_decision_support(
        _observation(),
        {"state": "允许寻找机会", "score": 60},
        "2026-07-20 14:51:00",
    )

    assert result.summary["action_code"] == "FROZEN"
    assert result.summary["is_final"] is True
    assert result.table.empty


def test_decision_support_never_qualifies_in_avoid_regime():
    result = build_decision_support(
        _observation(),
        {"state": "回避", "score": 20},
        "2026-07-20 14:30:00",
    )

    assert result.summary["qualified_count"] == 0
    assert result.table["support_action"].eq("观察").all()
    assert result.summary["broker_connection"] == "disabled"


def test_decision_support_never_generates_a_candidate_after_market_close():
    result = build_decision_support(
        _observation(),
        {"state": "允许寻找机会", "score": 80},
        "2026-07-20 15:00:00",
    )

    assert result.summary["action"] == "已收盘"
    assert result.summary["action_code"] == "CLOSED"
    assert result.table.empty
