from datetime import datetime

import pandas as pd

from wp.calendar import CN_TZ
from wp.legacy_main import _qualified_buy_plan
from wp.strategy_ledger import (
    STRATEGY_VERSION,
    strategy_validation_rows,
    update_strategy_ledger,
)


def _qualified_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "name": "甲",
                "sector_name": "电力",
                "price": 10.0,
                "forecast_profit_probability": 64.0,
                "forecast_profit_probability_lower": 53.0,
                "forecast_expected_net_return_pct": 0.8,
                "forecast_live_sample_count": 40,
                "forecast_live_day_count": 40,
                "forecast_effective_sample_count": 25,
                "decision_reason": "全部通过",
            },
            {
                "ts_code": "600002.SH",
                "name": "乙",
                "sector_name": "机械",
                "price": 20.0,
                "forecast_profit_probability": 63.0,
                "forecast_profit_probability_lower": 52.0,
                "forecast_expected_net_return_pct": 0.7,
                "forecast_live_sample_count": 40,
                "forecast_live_day_count": 40,
                "forecast_effective_sample_count": 24,
                "decision_reason": "全部通过",
            },
        ]
    )


def test_strategy_ledger_locks_every_candidate_first_signal_and_price(tmp_path):
    health = {
        "data_trade_date": "20260720",
        "market_data_time": "2026-07-20 14:25:00",
    }
    decision = {
        "action_code": "QUALIFIED_SET",
        "is_final": False,
        "reason": "发布全部合格票",
    }
    first = update_strategy_ledger(
        _qualified_plan(),
        decision,
        pd.DataFrame(),
        health,
        tmp_path,
        datetime(2026, 7, 20, 14, 25, 20, tzinfo=CN_TZ),
        {"forecast_round_trip_cost_pct": 0.25},
    )

    refreshed = _qualified_plan()
    refreshed.loc[0, "price"] = 11.0
    second = update_strategy_ledger(
        refreshed.iloc[[0]],
        decision,
        pd.DataFrame(),
        dict(health, market_data_time="2026-07-20 14:30:00"),
        tmp_path,
        datetime(2026, 7, 20, 14, 30, 20, tzinfo=CN_TZ),
        {"forecast_round_trip_cost_pct": 0.25},
    )

    assert len(first.table[first.table["strategy_version"].eq(STRATEGY_VERSION)]) == 2
    assert len(second.table[second.table["strategy_version"].eq(STRATEGY_VERSION)]) == 2
    row = second.table[second.table["ts_code"].eq("600001.SH")].iloc[0]
    assert row["plan_price"] == 10.0
    assert row["first_signal_time"] == "2026-07-20 14:25:00"
    assert row["last_signal_time"] == "2026-07-20 14:30:00"
    assert row["appearance_count"] == 2

    truth = pd.DataFrame(
        [
            {
                "plan_trade_date": "20260720",
                "plan_time": "2026-07-20 14:25:00",
                "target_trade_date": "20260721",
                "ts_code": "600001.SH",
                "actual_trade_date": "20260721",
                "actual_close": 10.1,
                "truth_status": "verified",
            }
        ]
    )
    verified = update_strategy_ledger(
        pd.DataFrame(),
        {},
        truth,
        health,
        tmp_path,
        datetime(2026, 7, 21, 16, 0, tzinfo=CN_TZ),
        {"forecast_round_trip_cost_pct": 0.25},
    )
    row = verified.table[verified.table["ts_code"].eq("600001.SH")].iloc[0]
    assert row["gross_return_pct"] == 1.0
    assert row["net_return_pct"] == 0.75
    assert bool(row["is_net_profit"])
    assert verified.summary["verified_signals"] == 1
    assert verified.summary["pending_signals"] == 1


def test_strategy_ledger_records_no_signal_day_without_fake_return(tmp_path):
    result = update_strategy_ledger(
        pd.DataFrame(),
        {
            "action_code": "FROZEN",
            "is_final": True,
            "reason": "没有股票通过全部门槛",
        },
        pd.DataFrame(),
        {
            "data_trade_date": "20260720",
            "market_data_time": "2026-07-20 14:55:00",
        },
        tmp_path,
        datetime(2026, 7, 20, 14, 55, tzinfo=CN_TZ),
    )

    assert result.table.iloc[0]["action"] == "NO_SIGNAL"
    assert result.table.iloc[0]["truth_status"] == "not_applicable"
    assert result.summary["decision_days"] == 1
    assert result.summary["candidate_days"] == 0
    assert result.summary["no_signal_days"] == 1


def test_qualified_buy_plan_keeps_all_and_never_selects_for_user():
    support = pd.DataFrame(
        [
            {
                "support_rank": 1,
                "support_action": "合格",
                "is_current_choice": False,
                "ts_code": "600001.SH",
                "name": "甲",
                "price": 10.0,
                "forecast_profit_probability_lower": 55.0,
                "decision_reason": "全部通过",
            },
            {
                "support_rank": 2,
                "support_action": "合格",
                "is_current_choice": False,
                "ts_code": "600002.SH",
                "name": "乙",
                "price": 20.0,
                "forecast_profit_probability_lower": 54.0,
                "decision_reason": "全部通过",
            },
            {
                "support_rank": 3,
                "support_action": "观察",
                "is_current_choice": False,
                "ts_code": "600003.SH",
                "name": "丙",
                "price": 30.0,
            },
        ]
    )

    plan = _qualified_buy_plan(support)

    assert plan["ts_code"].tolist() == ["600001.SH", "600002.SH"]
    assert plan["portfolio_group"].eq("合格候选").all()
    assert plan["tail_profit_model_version"].eq(STRATEGY_VERSION).all()


def test_strategy_validation_uses_first_qualified_contract_not_research_entry(tmp_path):
    health = {
        "data_trade_date": "20260720",
        "market_data_time": "2026-07-20 14:30:00",
    }
    result = update_strategy_ledger(
        _qualified_plan().iloc[[0]],
        {"action_code": "QUALIFIED_SET", "is_final": False},
        pd.DataFrame(),
        health,
        tmp_path,
        datetime(2026, 7, 20, 14, 30, 20, tzinfo=CN_TZ),
    )
    research = pd.DataFrame(
        [
            {
                "plan_trade_date": "20260720",
                "plan_time": "2026-07-20 14:20:00",
                "market_data_time": "2026-07-20 14:20:00",
                "target_trade_date": "20260721",
                "ts_code": "600001.SH",
                "plan_price": 9.5,
            }
        ]
    )

    scoped = strategy_validation_rows(result.table, research)

    assert scoped.iloc[0]["plan_time"] == "2026-07-20 14:30:00"
    assert scoped.iloc[0]["plan_price"] == 10.0
