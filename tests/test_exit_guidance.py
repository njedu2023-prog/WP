import pandas as pd

from wp.exit_guidance import build_exit_guidance


def _history():
    return pd.DataFrame(
        [
            {
                "target_trade_date": "20260720",
                "plan_time": "2026-07-17 14:35:00",
                "ts_code": "600001.SH",
                "name": "甲",
                "plan_price": 10.0,
            }
        ]
    )


def test_exit_guidance_prioritizes_minus_three_risk_line():
    market = pd.DataFrame([{"ts_code": "600001.SH", "open": 9.6, "high": 10.2, "low": 9.5, "price": 9.7}])
    result = build_exit_guidance(_history(), market, "20260720", "2026-07-20 09:40:00")
    assert result.table.iloc[0]["guidance_action"] == "建议风险退出"
    assert result.table.iloc[0]["holding_confirmation"] == "待人工确认实际持仓"
    assert result.summary["order_routing_enabled"] is False


def test_exit_guidance_sells_after_1040_when_not_sealed():
    market = pd.DataFrame([{"ts_code": "600001.SH", "open": 10.1, "high": 10.5, "low": 9.9, "price": 10.2}])
    result = build_exit_guidance(_history(), market, "20260720", "2026-07-20 10:40:00")
    assert result.table.iloc[0]["guidance_action"] == "建议人工择机卖出"


def test_exit_guidance_refuses_conclusion_without_market_price():
    result = build_exit_guidance(_history(), pd.DataFrame(columns=["ts_code"]), "20260720", "2026-07-20 10:40:00")
    assert result.table.iloc[0]["guidance_action"] == "行情数据不足"


def test_exit_guidance_holds_sealed_limit_up_for_manual_t2_review():
    market = pd.DataFrame(
        [{"ts_code": "600001.SH", "open": 10.2, "high": 11.0, "low": 10.1, "price": 11.0, "up_limit": 11.0, "open_board_count": 0}]
    )
    result = build_exit_guidance(_history(), market, "20260720", "2026-07-20 15:00:00")
    row = result.table.iloc[0]
    assert row["guidance_action"] == "确认未炸板，建议继续持有"
    assert "T+2 09:31" in row["next_checkpoint"]
    assert bool(row["order_routing_enabled"]) is False


def test_exit_guidance_does_not_treat_resealed_stock_as_unbroken_limit():
    market = pd.DataFrame(
        [{"ts_code": "600001.SH", "open": 10.2, "high": 11.0, "low": 10.1, "price": 11.0, "up_limit": 11.0, "open_board_count": 1}]
    )
    result = build_exit_guidance(_history(), market, "20260720", "2026-07-20 15:00:00")
    assert result.table.iloc[0]["guidance_action"] == "涨停已炸板，建议保护利润"
