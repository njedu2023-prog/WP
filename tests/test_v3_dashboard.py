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


def test_live_dashboard_shows_current_and_locked_signal_semantics(tmp_path):
    path = tmp_path / "latest.html"
    predictions = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "name": "测试",
                "signal_slot": "14:30",
                "signal_price": 10.3,
                "passes_policy": True,
                "candidate_state": "SHADOW_QUALIFIED",
                "p_net_positive": 0.70,
                "p_net_positive_lower": 0.60,
                "expected_net_return_pct": 0.8,
                "downside_q10_pct": -2.0,
            }
        ]
    )
    ledger = empty_shadow_ledger()
    ledger["sessions"] = [
        {
            "trade_date": "20260727",
            "covered_slots": ["14:20", "14:25", "14:30"],
            "integrity_status": "COLLECTING",
            "candidates": [
                {
                    "trade_date": "20260727",
                    "ts_code": "600001.SH",
                    "name": "测试",
                    "status": "SHADOW_QUALIFIED",
                    "first_signal_time": "14:20",
                    "first_signal_price": 10.0,
                    "last_signal_time": "14:30",
                    "appearance_count": 3,
                    "p_net_positive": 0.66,
                    "p_net_positive_lower": 0.55,
                    "expected_net_return_pct": 0.6,
                    "downside_q10_pct": -2.5,
                }
            ],
        }
    ]
    render_v3_dashboard(
        path,
        manifest={
            "source_trade_date": "20260727",
            "signal_slot": "14:30",
            "session_phase": "SIGNAL",
            "session_integrity_status": "COLLECTING",
            "covered_slots": ["14:20", "14:25", "14:30"],
            "v3_state": "SHADOW",
            "tail_universe_coverage": 0.96,
            "market_data_p95_age_seconds": 120,
        },
        predictions=predictions,
        ledger=ledger,
        registry=empty_registry(),
        config=V3Config(),
    )
    text = path.read_text(encoding="utf-8")
    assert "当前合格候选" in text
    assert "今日累计锁定信号" in text
    assert "当前信号价" in text
    assert "首次信号价" in text
    assert "14:20" in text
    assert "3/7 收集中" in text
