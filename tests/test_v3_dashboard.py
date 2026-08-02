from __future__ import annotations

import pandas as pd

from wp.v3.contracts import V3Config
from wp.v3.dashboard import render_v3_dashboard
from wp.v3.ledger import empty_shadow_ledger
from wp.v3.registry import empty_registry


def candidate(
    code: str,
    *,
    cohort: str,
    passes: bool,
    probability_lower: float,
) -> dict[str, object]:
    return {
        "ts_code": code,
        "name": f"测试{code}",
        "signal_slot": "14:30",
        "signal_price": 10.0,
        "passes_policy": passes,
        "candidate_cohort": cohort,
        "candidate_state": (
            "SHADOW_QUALIFIED"
            if cohort == "QUALIFIED"
            else "RESEARCH_OBSERVATION"
        ),
        "passes_execution": True,
        "passes_freshness": True,
        "p_entry_fill": 0.99,
        "p_exit_fill_given_entry": 0.995,
        "p_net_positive": 0.70,
        "p_net_positive_lower": probability_lower,
        "expected_utility_pct": 0.8,
        "expected_utility_lower_pct": 0.4,
        "selection_score": probability_lower,
        "downside_q10_pct": -2.0,
    }


def render(
    path,
    *,
    phase: str,
    predictions: pd.DataFrame | None = None,
    ledger: dict | None = None,
    state: str = "SHADOW",
    health_status: str = "ok",
    retrospective: dict | None = None,
    research_seed: dict | None = None,
) -> str:
    render_v3_dashboard(
        path,
        manifest={
            "source_trade_date": "20260803",
            "signal_slot": "14:30",
            "session_phase": phase,
            "health_status": health_status,
            "live_display_allowed": phase != "CLOSED",
            "v3_state": state,
            "observation_selection_status": "COMPLETE",
            "tail_universe_coverage": 0.96,
            "market_data_p95_age_seconds": 120,
        },
        predictions=(
            predictions if predictions is not None else pd.DataFrame()
        ),
        ledger=ledger or empty_shadow_ledger(),
        registry=empty_registry(),
        config=V3Config(),
        retrospective=retrospective,
        research_seed=research_seed,
    )
    return path.read_text(encoding="utf-8")


def test_closed_dashboard_has_no_actionable_list(tmp_path) -> None:
    path = tmp_path / "latest.html"
    text = render(
        path,
        phase="CLOSED",
        predictions=pd.DataFrame(
            [
                candidate(
                    "600001.SH",
                    cohort="QUALIFIED",
                    passes=True,
                    probability_lower=0.60,
                )
            ]
        ),
    )

    assert "已收盘，不再显示可买名单" in text
    assert "15:00 后禁止新增候选" in text
    assert "<h2 class=\"section-title\">合格候选</h2>" not in text
    assert "<h2 class=\"section-title\">研究观察</h2>" not in text
    assert "T+1 真实验证" in text


def test_research_ready_is_not_reported_as_integrity_failure(tmp_path) -> None:
    path = tmp_path / "latest.html"
    text = render(
        path,
        phase="CLOSED",
        state="SHADOW_OBSERVATION",
        health_status="research_ready",
    )

    assert "数据或账本异常" not in text
    assert "完整性检查未通过" not in text
    assert "已收盘，不再显示可买名单" in text
    assert "研究回测就绪" in text


def test_live_dashboard_separates_all_qualified_and_five_observations(
    tmp_path,
) -> None:
    path = tmp_path / "latest.html"
    rows = [
        candidate(
            "600001.SH",
            cohort="QUALIFIED",
            passes=True,
            probability_lower=0.60,
        ),
        candidate(
            "600002.SH",
            cohort="QUALIFIED",
            passes=True,
            probability_lower=0.58,
        ),
    ]
    rows.extend(
        candidate(
            f"00000{index}.SZ",
            cohort="OBSERVATION",
            passes=False,
            probability_lower=0.53 - index * 0.01,
        )
        for index in range(1, 6)
    )
    text = render(
        path,
        phase="SIGNAL",
        predictions=pd.DataFrame(rows),
    )

    assert "<h2 class=\"section-title\">合格候选</h2>" in text
    assert "<h2 class=\"section-title\">研究观察</h2>" in text
    assert "全部固定门槛均通过；数量可以为 0，不设人为配额" in text
    assert "固定展示 5 支最接近门槛" in text
    assert "2 支合格候选" in text
    assert "5 / 5" in text
    assert "人工决定是否买入" in text
    assert "不记录人工是否买入" in text


def test_zero_qualified_is_explicit_valid_result(tmp_path) -> None:
    path = tmp_path / "latest.html"
    observations = pd.DataFrame(
        [
            candidate(
                f"00000{index}.SZ",
                cohort="OBSERVATION",
                passes=False,
                probability_lower=0.53 - index * 0.01,
            )
            for index in range(1, 6)
        ]
    )
    text = render(path, phase="SIGNAL", predictions=observations)

    assert "当前无合格候选" in text
    assert "当前没有合格候选" in text
    assert "零支是正常结果" in text
    assert "不会为了产生名单而降低固定门槛" in text
    assert "5 / 5" in text


def test_dashboard_keeps_retrospective_and_live_evidence_separate(
    tmp_path,
) -> None:
    path = tmp_path / "latest.html"
    text = render(
        path,
        phase="CLOSED",
        retrospective={
            "status": "COMPLETE",
            "qualified": {
                "metrics": {
                    "events": 20,
                    "trade_days": 12,
                    "win_rate": 0.60,
                    "mean_net_return_pct": 0.30,
                }
            },
        },
    )

    assert "2026 年 5–7 月新合同回测" in text
    assert "2026 年 8 月起真实影子运行" in text
    assert "历史回测和旧系统回补均不计入 150 个交易日" in text
    assert "只使用当日 14:30 前可见信息" in text
    assert "20" in text


def test_dashboard_shows_full_retrospective_observation_metrics(
    tmp_path,
) -> None:
    path = tmp_path / "latest.html"
    text = render(
        path,
        phase="CLOSED",
        retrospective={
            "status": "INCOMPLETE",
            "source": {"expected_trade_days": 62},
            "qualified": {
                "metrics": {
                    "events": 0,
                    "trade_days": 0,
                    "win_rate": 0.0,
                    "mean_net_return_pct": None,
                    "profit_factor": 0.0,
                }
            },
            "observations": {
                "metrics": {
                    "events": 305,
                    "trade_days": 61,
                    "win_rate": 0.44262295,
                    "mean_net_return_pct": -0.356252,
                    "profit_factor": 0.653054,
                    "stress": {
                        "50bps": {
                            "mean_net_return_pct": -0.504777,
                        }
                    },
                }
            },
            "monthly": [
                {
                    "month": "202605",
                    "evaluated_trade_days": 18,
                    "qualified": {"events": 0, "trade_days": 0},
                    "observations": {
                        "events": 90,
                        "trade_days": 18,
                        "win_rate": 0.311111,
                        "mean_net_return_pct": -0.804051,
                        "profit_factor": 0.406896,
                        "stress": {
                            "50bps": {
                                "mean_net_return_pct": -0.950718,
                            }
                        },
                    },
                }
            ],
        },
    )

    assert "2026 年 8 月起 T+1 真实验证" in text
    assert "这里只统计 8 月 3 日起盘中实时形成的影子记录" in text
    assert "合格样本为 0 不是数据缺失" in text
    assert "62 个覆盖交易日内" in text
    assert "研究观察" in text
    assert "305" in text
    assert "61 / 62" in text
    assert "44.3%" in text
    assert "-0.356%" in text
    assert "0.65" in text
    assert "-0.505%" in text
    assert "90" in text
    assert "18 / 18" in text
    assert "31.1%" in text
    assert "-0.804%" in text


def test_dashboard_uses_independent_short_cohort_tabs(tmp_path) -> None:
    path = tmp_path / "latest.html"
    text = render(
        path,
        phase="CLOSED",
        retrospective={
            "status": "COMPLETE",
            "source": {"expected_trade_days": 2},
            "qualified": {"metrics": {"events": 0, "trade_days": 0}},
            "observations": {"metrics": {"events": 10, "trade_days": 2}},
        },
    )

    assert '<section class="section" data-tab-group>' in text
    assert '<div class="retrospective-tabs" data-tab-group>' in text
    assert text.count('data-tab="QUALIFIED">合格</button>') == 2
    assert text.count('data-tab="OBSERVATION">观察</button>') == 2
    assert (
        '<div role="tabpanel" data-tab-panel="QUALIFIED" hidden>'
        in text
    )
    assert '<div role="tabpanel" data-tab-panel="OBSERVATION">' in text
    assert "group.querySelectorAll('[data-tab]')" in text
    assert "group.querySelectorAll('[data-tab-panel]')" in text
    assert "data-cohort-tab" not in text
    assert "data-cohort-panel" not in text
    assert ".status-title {\n      margin-top: 4px;\n      font-size: 15px;" in text
    assert (
        ".value { margin-top: 4px; font-size: 15px; font-weight: 700; }"
        in text
    )
    assert ".value.small" not in text


def test_v15_is_disclosed_as_seed_not_v40_performance(tmp_path) -> None:
    path = tmp_path / "latest.html"
    text = render(
        path,
        phase="PRE_SIGNAL",
        state="MODEL_NOT_READY",
        research_seed={
            "schema_version": "wp_v15_forward_risk_validation_1",
            "evidence_snapshot": {
                "challenger": {
                    "events": 55,
                    "trade_days": 29,
                    "win_rate": 0.5636,
                    "mean_net_return_pct": 0.455,
                }
            },
        },
    )

    assert "今天不授权候选" in text
    assert "固定 14:30 新合同仍在建立可部署模型和回测证据" in text
    assert "V15" in text
    assert "不能直接当作新系统盈利证据" in text


def test_legacy_rows_are_audit_only(tmp_path) -> None:
    path = tmp_path / "latest.html"
    render_v3_dashboard(
        path,
        manifest={
            "source_trade_date": "20260803",
            "session_phase": "CLOSED",
            "v3_state": "SHADOW",
        },
        predictions=pd.DataFrame(),
        ledger=empty_shadow_ledger(),
        registry=empty_registry(),
        config=V3Config(),
        legacy_audit={"records": [{"plan_trade_date": "20260723"}]},
    )
    text = path.read_text(encoding="utf-8")

    assert "旧系统历史说明" in text
    assert "不计入固定 14:30 新合同" in text
    assert "不计入 2026 年 8 月起的真实影子统计" in text
