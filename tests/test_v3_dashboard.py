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
                "p_entry_fill": 0.99,
                "p_exit_fill_given_entry": 0.995,
                "p_net_positive": 0.70,
                "p_net_positive_lower": 0.60,
                "expected_utility_pct": 0.8,
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
    assert "已收盘，不再买入" in text
    assert "当前合格候选" not in text
    assert "今日候选记录" in text
    assert "下一交易日 14:20" in text
    assert "模型研究与上线条件" in text


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
                "p_entry_fill": 0.99,
                "p_exit_fill_given_entry": 0.995,
                "p_net_positive": 0.70,
                "p_net_positive_lower": 0.60,
                "expected_utility_pct": 0.8,
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
                    "p_entry_fill": 0.99,
                    "p_exit_fill_given_entry": 0.995,
                    "p_net_positive": 0.66,
                    "p_net_positive_lower": 0.55,
                    "expected_utility_pct": 0.6,
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
    assert "今日已经出现过" in text
    assert "当前信号价" in text
    assert "首次信号价" in text
    assert "14:20" in text
    assert "仅观察，不实盘" in text
    assert "14:35 更新" in text
    assert "candidate-card" in text


def test_production_dashboard_makes_no_trade_decision_explicit(tmp_path):
    path = tmp_path / "latest.html"
    render_v3_dashboard(
        path,
        manifest={
            "source_trade_date": "20260727",
            "signal_slot": "14:40",
            "session_phase": "SIGNAL",
            "health_status": "ok",
            "live_display_allowed": True,
            "v3_state": "PRODUCTION",
        },
        predictions=pd.DataFrame(),
        ledger=empty_shadow_ledger(),
        registry=empty_registry(),
        config=V3Config(),
    )
    text = path.read_text(encoding="utf-8")
    assert "暂不买入" in text
    assert "空仓是正式决策" in text
    assert "不会为了产生名单而降低标准" in text
    assert "14:45 更新" in text
    assert "当前没有候选" in text


def test_dashboard_discloses_legacy_backfill_without_counting_it_as_v7_truth(
    tmp_path,
):
    path = tmp_path / "latest.html"
    render_v3_dashboard(
        path,
        manifest={
            "source_trade_date": "20260728",
            "session_phase": "CLOSED",
            "v3_state": "SHADOW",
        },
        predictions=pd.DataFrame(),
        ledger=empty_shadow_ledger(),
        registry=empty_registry(),
        config=V3Config(),
        legacy_audit={
            "records": [
                {
                    "plan_trade_date": "20260721",
                    "valid_preclose_snapshot_count": 2,
                    "invalid_late_snapshot_count": 2,
                    "audit_reason": "旧版盘中正式意见为空仓",
                },
                {
                    "plan_trade_date": "20260723",
                    "valid_preclose_snapshot_count": 0,
                    "invalid_late_snapshot_count": 1,
                    "audit_reason": "盘后名单无效",
                },
            ]
        },
    )

    text = path.read_text(encoding="utf-8")
    assert "7 月 21–24 日旧系统证据回补" in text
    assert "2026-07-21" in text
    assert "2026-07-23" in text
    assert "有盘中证据，无合格票" in text
    assert "无合法盘中名单" in text
    assert "不计入 V7 收益" in text


def test_not_ready_dashboard_does_not_report_research_as_failed(tmp_path):
    path = tmp_path / "latest.html"
    render_v3_dashboard(
        path,
        manifest={
            "source_trade_date": "20260729",
            "session_phase": "PRE_SIGNAL",
            "v3_state": "MODEL_NOT_READY",
        },
        predictions=pd.DataFrame(),
        ledger=empty_shadow_ledger(),
        registry=empty_registry(),
        config=V3Config(),
    )

    text = path.read_text(encoding="utf-8")
    assert "V7 三年滚动样本外研究尚未发布完成" in text
    assert "旧模型结果当作 V7 结论" in text
    assert "回测未通过" not in text
