from __future__ import annotations

import pandas as pd

from wp.v3.contracts import V3Config
from wp.v3.dashboard import render_v3_dashboard
from wp.v3.ledger import empty_shadow_ledger
from wp.v3.registry import empty_registry


def test_closed_dashboard_has_no_live_buy_list(tmp_path):
    path = tmp_path / "latest.html"
    predictions = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "name": "测试",
                "signal_slot": "14:50",
                "signal_price": 10.0,
                "passes_policy": True,
                "p_net_positive": 0.70,
                "p_net_positive_lower": 0.60,
                "expected_net_return_pct": 0.8,
                "downside_q10_pct": -2.0,
            }
        ]
    )
    render_v3_dashboard(
        path,
        manifest={
            "source_trade_date": "20260727",
            "signal_slot": "14:50",
            "session_phase": "CLOSED",
            "v3_state": "SHADOW",
        },
        predictions=predictions,
        ledger=empty_shadow_ledger(),
        registry=empty_registry(),
        config=V3Config(),
    )
    text = path.read_text(encoding="utf-8")
    assert "交易窗口已关闭" in text
    assert "当前合格候选" not in text
    assert "当日冻结候选台账" in text

