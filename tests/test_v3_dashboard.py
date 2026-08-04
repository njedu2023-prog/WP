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
    rank: int | None = None,
) -> dict[str, object]:
    return {
        "ts_code": code,
        "name": f"测试{code}",
        "signal_slot": "14:00",
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
        "meta_p_positive": 0.70,
        "meta_p_positive_lower": probability_lower,
        "cohort_rank": rank,
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
            "source_trade_date": "20260805",
            "signal_slot": "14:00",
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
    assert "尾盘候选与真实验证" not in text
    assert "14:00 一次决策 · 14:05 可成交基准" not in text
    assert "冻结记录仍保留用于 T+1 收盘真值验证" not in text
    assert "15:00 后禁止新增候选" not in text
    assert "<h2 class=\"section-title\">合格候选</h2>" not in text
    assert "<h2 class=\"section-title\">研究观察</h2>" not in text
    assert "T+1 真实验证" in text


def test_data_status_has_a_wide_non_wrapping_column(tmp_path) -> None:
    path = tmp_path / "latest.html"
    text = render(path, phase="CLOSED")

    assert "minmax(260px, .9fr)" in text
    assert ".status-fact:last-child .value {" in text
    assert "white-space: nowrap;" in text


def test_closed_dashboard_labels_frozen_evidence_with_signal_date(
    tmp_path,
) -> None:
    path = tmp_path / "latest.html"
    ledger = {
        "schema_version": "wp_candidate_ledger_v4",
        "sessions": [
            {
                "trade_date": "20260805",
                "candidates": [],
                "observations": [
                    {
                        **candidate(
                            "600001.SH",
                            cohort="OBSERVATION",
                            passes=False,
                            probability_lower=0.53,
                        ),
                        "trade_date": "20260805",
                        "first_signal_time": "14:00",
                        "first_signal_price": 10.0,
                        "entry_price": 10.01,
                    }
                ],
            }
        ],
    }

    text = render(path, phase="CLOSED", ledger=ledger)

    assert "D 日冻结证据 · 2026-08-05" in text
    assert "今日冻结证据" not in text


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
    assert "影子观察" in text
    assert "研究影子观察" not in text


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
            rank=index,
        )
        for index in range(1, 6)
    )
    text = render(
        path,
        phase="SIGNAL",
        predictions=pd.DataFrame(rows),
        retrospective={
            "summary": {
                "observations": {
                    "rank_evidence": {"status": "NOT_CONFIRMED"}
                }
            }
        },
    )

    assert "<h2 class=\"section-title\">合格候选</h2>" in text
    assert "<h2 class=\"section-title\">研究观察</h2>" in text
    assert "全部固定门槛均通过；数量可以为 0，不设人为配额" in text
    assert "固定展示 5 支最接近门槛" in text
    assert "2 支合格候选" in text
    assert "5 / 5" in text
    assert 'class="table-wrap dense-table"' in text
    assert "校准概率 / 保守概率" in text
    assert "70.0%" in text
    assert "观察 1" in text
    assert "样本外排序未确认" in text
    assert "不代表用户实际成交" in text


def test_zero_qualified_is_explicit_valid_result(tmp_path) -> None:
    path = tmp_path / "latest.html"
    observations = pd.DataFrame(
        [
            candidate(
                f"00000{index}.SZ",
                cohort="OBSERVATION",
                passes=False,
                probability_lower=0.53 - index * 0.01,
                rank=index,
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


def test_dashboard_omits_research_evidence_block_from_frontend(
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

    assert "证据与上线边界" not in text
    assert "2026 年 5–7 月新合同回测" not in text
    assert "2026-08-05 起合格票真实影子运行" not in text
    assert "2026-08-05 起观察票真实影子运行" not in text


def test_dashboard_keeps_live_validation_contract_without_research_block(
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

    assert '<h2 class="section-title">T+1 真实验证</h2>' in text
    assert "2026 年 8 月起 T+1 真实验证" not in text
    assert "实时统计始于 2026-08-05" in text
    assert "T 日 14:00 产生信号" in text
    assert "14:05 影子入场" in text
    assert "T+1 收盘验证" in text
    assert "证据与上线边界" not in text
    assert "305" not in text


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

    assert '<section class="section dense-section" data-tab-group>' in text
    assert '<div class="retrospective-tabs" data-tab-group>' not in text
    assert text.count('data-tab="QUALIFIED">合格</button>') == 1
    assert text.count('data-tab="OBSERVATION">观察</button>') == 1
    assert text.count(
        'class="active" type="button" role="tab" '
        'aria-selected="true" data-tab="OBSERVATION"'
    ) == 1
    assert (
        "实时统计始于 2026-08-05 · T 日 14:00 产生信号 · "
        "14:05 影子入场 · T+1 收盘验证 · "
        "旧合同单独标记且不合并统计"
        in text
    )
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
    assert ".dense-table td {\n      padding: 9px 10px;" in text


def test_validation_groups_by_month_and_day_and_keeps_stock_inline(
    tmp_path,
) -> None:
    path = tmp_path / "latest.html"

    def observation(
        trade_date: str,
        target_date: str,
        code: str,
        name: str,
        net_return: float,
        *,
        rank: int = 1,
        signal_slot: str = "14:00",
        entry_slot: str = "14:05",
    ) -> dict[str, object]:
        return {
            "trade_date": trade_date,
            "target_trade_date": target_date,
            "candidate_cohort": "OBSERVATION",
            "cohort_rank": rank,
            "ts_code": code,
            "name": name,
            "first_signal_time": signal_slot,
            "entry_benchmark_slot": entry_slot,
            "entry_price": 10.0,
            "t1_close": 10.0 * (1.0 + net_return / 100.0),
            "truth_status": "verified",
            "net_return_pct": net_return,
            "net_positive": net_return > 0,
            "prospective_eligible": signal_slot == "14:00",
        }

    august = [
        observation(
            "20260805",
            "20260806",
            f"60008{index}.SH",
            "特变电工" if index == 9 else f"测试{index}",
            1.0,
            rank=index - 4,
        )
        for index in range(5, 10)
    ]
    september = [
        observation(
            "20260901",
            "20260902",
            f"00000{index}.SZ",
            f"九月{index}",
            2.0,
            rank=index,
        )
        for index in range(1, 6)
    ]
    old_contract = [
        observation(
            "20260803",
            "20260804",
            "601919.SH",
            "中远海控",
            3.0,
            signal_slot="14:30",
            entry_slot="14:35",
        )
    ]
    ledger = {
        "schema_version": "wp_candidate_ledger_v4",
        "sessions": [
            {
                "trade_date": "20260803",
                "prospective_eligible": False,
                "candidates": [],
                "observations": old_contract,
            },
            {
                "trade_date": "20260805",
                "prospective_eligible": True,
                "candidates": [],
                "observations": august,
            },
            {
                "trade_date": "20260901",
                "prospective_eligible": True,
                "candidates": [],
                "observations": september,
            },
        ],
    }

    text = render(path, phase="CLOSED", ledger=ledger)

    assert 'data-month-tab="202609-1400-1405"' in text
    assert "2026-09 · 14:00" in text
    assert "2026-08 · 14:00" in text
    assert "2026-08 · 14:30 旧" in text
    assert "旧 14:30 信号 / 14:35 入场合同" in text
    assert "当月总收益" in text
    assert "历史总收益" in text
    assert "+2.000%" in text
    assert "+3.020%" in text
    assert "当日组合净收益" in text
    assert "5 / 5" in text
    assert 'aria-label="展开当日逐票验证"' in text
    assert "data-collapse-panel hidden" in text
    assert (
        '<div class="stock-inline">'
        '<span class="stock-code">600089.SH</span>'
        '<span class="stock-name-inline">特变电工</span>'
        "</div>"
        in text
    )
    assert "group.querySelectorAll('[data-month-tab]')" in text


def test_retrospective_backtest_is_not_rendered(tmp_path) -> None:
    path = tmp_path / "latest.html"
    text = render(
        path,
        phase="CLOSED",
        retrospective={
            "status": "COMPLETE",
            "qualified": {"metrics": {"events": 0, "trade_days": 0}},
            "observations": {"metrics": {"events": 5, "trade_days": 1}},
        },
    )

    assert "retrospective-backtest-content" not in text
    assert "2026 年 5–7 月新合同回测" not in text


def test_v15_seed_is_not_rendered_on_frontend(tmp_path) -> None:
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
    assert "固定 14:00 新合同仍在建立可部署模型和回测证据" in text
    assert "V15" not in text


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
    assert "不计入固定 14:00 新合同" in text
    assert "不计入 2026 年 8 月起的真实影子统计" in text
