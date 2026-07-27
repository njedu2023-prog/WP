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


def test_exit_guidance_does_not_change_fixed_contract_on_intraday_loss():
    market = pd.DataFrame([{"ts_code": "600001.SH", "open": 9.6, "high": 10.2, "low": 9.5, "price": 9.7}])
    result = build_exit_guidance(_history(), market, "20260720", "2026-07-20 09:40:00")
    assert result.table.iloc[0]["guidance_action"] == "按合同持有"
    assert result.table.iloc[0]["holding_confirmation"] == "待人工确认实际持仓"
    assert result.summary["order_routing_enabled"] is False


def test_exit_guidance_waits_until_fixed_close_window():
    market = pd.DataFrame([{"ts_code": "600001.SH", "open": 10.1, "high": 10.5, "low": 9.9, "price": 10.2}])
    result = build_exit_guidance(_history(), market, "20260720", "2026-07-20 10:40:00")
    assert result.table.iloc[0]["guidance_action"] == "按合同持有"

    close_window = build_exit_guidance(_history(), market, "20260720", "2026-07-20 14:50:00")
    assert close_window.table.iloc[0]["guidance_action"] == "执行T+1收盘卖出"


def test_exit_guidance_refuses_conclusion_without_market_price():
    result = build_exit_guidance(_history(), pd.DataFrame(columns=["ts_code"]), "20260720", "2026-07-20 10:40:00")
    assert result.table.iloc[0]["guidance_action"] == "行情数据不足"


def test_exit_guidance_never_changes_to_t2_after_close():
    market = pd.DataFrame(
        [{"ts_code": "600001.SH", "open": 10.2, "high": 11.0, "low": 10.1, "price": 11.0, "up_limit": 11.0, "open_board_count": 0}]
    )
    result = build_exit_guidance(_history(), market, "20260720", "2026-07-20 15:00:00")
    row = result.table.iloc[0]
    assert row["guidance_action"] == "T+1收盘合同已结束"
    assert "收盘真值" in row["next_checkpoint"]
    assert bool(row["order_routing_enabled"]) is False


def test_exit_guidance_uses_same_close_contract_after_reseal():
    market = pd.DataFrame(
        [{"ts_code": "600001.SH", "open": 10.2, "high": 11.0, "low": 10.1, "price": 11.0, "up_limit": 11.0, "open_board_count": 1}]
    )
    result = build_exit_guidance(_history(), market, "20260720", "2026-07-20 15:00:00")
    assert result.table.iloc[0]["guidance_action"] == "T+1收盘合同已结束"
