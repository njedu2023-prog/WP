import pandas as pd

from wp.buy_decision import build_buy_decision
from wp.ranking import rank_candidates
from wp.tail_profit_model import TAIL_PROFIT_MODEL_VERSION, add_tail_profit_scores


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": "20260714",
                "ts_code": "000001.SZ",
                "name": "稳健主票",
                "pct_chg": 8.4,
                "capital_score": 90,
                "sector_strength_score": 85,
                "risk_penalty_score": 15,
                "close_position": 88,
                "amount_ratio_5d": 1.5,
                "amount": 300_000_000,
                "sector_name": "板块A",
                "price": 10,
                "pre_day_limitup": 0,
                "today_limitup": 0,
            },
            {
                "trade_date": "20260714",
                "ts_code": "000002.SZ",
                "name": "涨幅过热",
                "pct_chg": 13.0,
                "capital_score": 100,
                "sector_strength_score": 100,
                "risk_penalty_score": 0,
                "close_position": 95,
                "amount_ratio_5d": 1.2,
                "amount": 500_000_000,
                "sector_name": "板块B",
                "price": 20,
                "pre_day_limitup": 0,
                "today_limitup": 0,
            },
            {
                "trade_date": "20260714",
                "ts_code": "000003.SZ",
                "name": "爆量风险",
                "pct_chg": 8.2,
                "capital_score": 80,
                "sector_strength_score": 80,
                "risk_penalty_score": 20,
                "close_position": 80,
                "amount_ratio_5d": 3.0,
                "amount": 250_000_000,
                "sector_name": "板块C",
                "price": 15,
                "pre_day_limitup": 0,
                "today_limitup": 0,
            },
        ]
    )


def test_tail_profit_model_applies_hard_risk_filters():
    scored = add_tail_profit_scores(_candidates())
    eligible = scored.set_index("ts_code")["tail_profit_eligible"].to_dict()

    assert eligible == {"000001.SZ": True, "000002.SZ": False, "000003.SZ": False}
    assert "涨幅过热" in scored.loc[scored["ts_code"].eq("000002.SZ"), "tail_profit_filter_reason"].iloc[0]
    assert "放量过度" in scored.loc[scored["ts_code"].eq("000003.SZ"), "tail_profit_filter_reason"].iloc[0]
    assert scored["tail_profit_model_version"].eq(TAIL_PROFIT_MODEL_VERSION).all()


def test_top50_and_buy_plan_share_tail_profit_order():
    scored = add_tail_profit_scores(_candidates())
    top50, _ = rank_candidates(scored, "2026-07-14 14:35:00", top_n=50)
    decision = build_buy_decision(top50)

    assert top50.iloc[0]["ts_code"] == "000001.SZ"
    assert bool(top50.iloc[0]["tail_profit_eligible"])
    assert len(decision.buy_plan) == 1
    assert decision.buy_plan.iloc[0]["ts_code"] == "000001.SZ"
    assert decision.buy_plan.iloc[0]["tail_profit_score"] == top50.iloc[0]["tail_profit_score"]
    assert decision.summary["max_buy_count"] == 1


def test_missing_required_tail_field_forces_cash():
    frame = _candidates().drop(columns=["amount_ratio_5d"])
    decision = build_buy_decision(frame)

    assert decision.buy_plan.empty
    assert decision.decision_table["skip_reason"].eq("关键字段缺失").all()
