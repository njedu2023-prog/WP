from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from wp.calendar import CN_TZ
from wp.main import _apply_locked_decision, _official_buy_plan
from wp.strategy_ledger import (
    STRATEGY_VERSION,
    locked_decision_for_date,
    update_strategy_ledger,
)


def _buy_plan() -> pd.DataFrame:
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
            }
        ]
    )


def test_strategy_ledger_locks_one_buy_per_day_and_uses_net_return(tmp_path):
    health = {
        "data_trade_date": "20260720",
        "market_data_time": "2026-07-20 14:46:00",
    }
    decision = {
        "action_code": "BUY",
        "is_final": True,
        "reason": "全部通过",
    }
    current = datetime(2026, 7, 20, 14, 47, tzinfo=CN_TZ)
    first = update_strategy_ledger(
        _buy_plan(),
        decision,
        pd.DataFrame(),
        health,
        tmp_path,
        current,
        {"forecast_round_trip_cost_pct": 0.25},
    )
    changed = _buy_plan()
    changed.loc[0, "ts_code"] = "600002.SH"
    second = update_strategy_ledger(
        changed,
        decision,
        pd.DataFrame(),
        health,
        tmp_path,
        current,
        {"forecast_round_trip_cost_pct": 0.25},
    )

    assert len(first.table) == 1
    assert len(second.table) == 1
    assert second.table.iloc[0]["ts_code"] == "600001.SH"
    assert second.table.iloc[0]["strategy_version"] == STRATEGY_VERSION

    truth = pd.DataFrame(
        [
            {
                "plan_trade_date": "20260720",
                "plan_time": "2026-07-20 14:46:00",
                "target_trade_date": "20260721",
                "ts_code": "600001.SH",
                "actual_trade_date": "20260721",
                "actual_close": 10.1,
                "return_close_pct": 9.99,
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
        datetime(2026, 7, 21, 15, 10, tzinfo=CN_TZ),
        {"forecast_round_trip_cost_pct": 0.25},
    )
    row = verified.table.iloc[0]
    assert row["gross_return_pct"] == 1.0
    assert row["net_return_pct"] == 0.75
    assert bool(row["is_net_profit"])
    assert verified.summary["net_win_rate"] == 100.0
    assert locked_decision_for_date(verified.table, "2026-07-20")["ts_code"] == "600001.SH"


def test_strategy_ledger_records_final_no_trade_without_fake_return(tmp_path):
    result = update_strategy_ledger(
        pd.DataFrame(),
        {
            "action_code": "NO_TRADE",
            "is_final": True,
            "reason": "概率下界不足",
        },
        pd.DataFrame(),
        {
            "data_trade_date": "20260720",
            "market_data_time": "2026-07-20 14:55:00",
        },
        tmp_path,
        datetime(2026, 7, 20, 14, 56, tzinfo=CN_TZ),
    )

    assert result.table.iloc[0]["action"] == "NO_TRADE"
    assert result.table.iloc[0]["truth_status"] == "not_applicable"
    assert result.summary["decision_days"] == 1
    assert result.summary["trade_days"] == 0
    assert result.summary["no_trade_days"] == 1


def test_locked_buy_remains_the_only_displayed_choice():
    support = SimpleNamespace(
        summary={"action_code": "WATCH", "reason": "later refresh"},
        table=pd.DataFrame(
            [
                {"ts_code": "600001.SH", "is_current_choice": False, "support_action": "研究候选"},
                {"ts_code": "600002.SH", "is_current_choice": True, "support_action": "继续观察"},
            ]
        ),
    )
    locked = {
        "action": "BUY",
        "published_at": "2026-07-20 14:45:20",
        "ts_code": "600001.SH",
        "name": "甲",
        "sector_name": "电力",
        "plan_price": 10.0,
        "forecast_profit_probability": 64.0,
        "forecast_profit_probability_lower": 53.0,
        "forecast_expected_net_return_pct": 0.8,
        "forecast_live_sample_count": 40,
        "forecast_live_day_count": 40,
        "forecast_effective_sample_count": 25,
    }

    _apply_locked_decision(support, locked)
    plan = _official_buy_plan(support.summary, pd.DataFrame(), pd.DataFrame(), locked)

    assert support.summary["action_code"] == "BUY"
    assert support.summary["candidate_code"] == "600001.SH"
    assert support.table.loc[support.table["ts_code"].eq("600001.SH"), "is_current_choice"].item()
    assert plan.iloc[0]["ts_code"] == "600001.SH"
    assert plan.iloc[0]["price"] == 10.0
