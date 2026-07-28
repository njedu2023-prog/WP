from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import V3Config


def render_v3_dashboard(
    path: str | Path,
    *,
    manifest: dict[str, Any],
    predictions: pd.DataFrame,
    ledger: dict[str, Any],
    registry: dict[str, Any],
    config: V3Config,
    replay: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    phase = str(manifest.get("session_phase") or "UNKNOWN")
    state = str(manifest.get("v3_state") or "MODEL_NOT_READY")
    trade_date = str(manifest.get("source_trade_date") or "")
    session = next(
        (
            item
            for item in ledger.get("sessions", [])
            if str(item.get("trade_date")) == trade_date
        ),
        {},
    )
    model = next(
        (
            item
            for item in registry.get("models", [])
            if item.get("fingerprint") == manifest.get("v3_model_fingerprint")
        ),
        {},
    )
    passed = predictions.loc[
        predictions.get("passes_policy", pd.Series(False, index=predictions.index)).fillna(False)
    ].copy()
    passed = _attach_locked_candidate_fields(passed, session)
    if not passed.empty:
        passed_sort = [
            column
            for column in (
                "selection_score",
                "p_net_positive_lower",
                "expected_net_return_pct",
            )
            if column in passed
        ]
        if passed_sort:
            passed = passed.sort_values(
                passed_sort,
                ascending=False,
                kind="stable",
            )
    session_candidates = pd.DataFrame(session.get("candidates", []))
    near = _near_candidates(predictions)
    historical = [
        candidate
        for historical_session in ledger.get("sessions", [])
        for candidate in historical_session.get("candidates", [])
    ]
    replay = replay or {}
    replay_candidates = replay.get("candidates", [])
    backtest = model.get("backtest", {})
    shadow = model.get("shadow", {})
    live_visible = bool(manifest.get("live_display_allowed", phase == "SIGNAL"))
    title_state = {
        "PRODUCTION": "生产模型",
        "SHADOW": "影子模型",
        "SHADOW_OBSERVATION": "影子观察（回测未通过）",
        "MODEL_NOT_READY": "模型尚未就绪",
        "MODEL_NOT_DESIGNATED": "研究模型未指定",
    }.get(state, state)
    status_class = "good" if state == "PRODUCTION" else "warn"
    if live_visible and len(passed):
        live_message = (
            f"当前共有 {len(passed)} 支通过全部门槛，最终买哪一支由人工决定。"
        )
    elif live_visible:
        live_message = "当前时点没有股票通过全部固定门槛，保持空仓是有效结果。"
    elif phase == "PRE_SIGNAL":
        live_message = "模型已就绪，等待 14:20 开始尾盘候选观察。"
    elif manifest.get("health_status") == "v3_input_not_ready":
        live_message = "交易窗口内缺少合法时点快照，候选输出已关闭并等待自愈。"
    else:
        live_message = (
            "交易窗口已关闭。下方仅展示冻结台账和验证结果，不构成可买名单。"
        )
    alert_title = (
        "交易窗口内"
        if live_visible
        else (
            "数据完整性故障"
            if manifest.get("health_status") == "v3_input_not_ready"
            else ("等待尾盘窗口" if phase == "PRE_SIGNAL" else "仅供复盘")
        )
    )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WP V5 尾盘 T+1 决策台</title>
<style>
:root {{
  --ink:#171a1f; --muted:#66707b; --line:#d9dee5; --paper:#ffffff;
  --soft:#f4f6f8; --green:#087a43; --green-bg:#e8f5ee;
  --red:#c62828; --amber:#9a5b00; --amber-bg:#fff4dd; --blue:#145da0;
}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--soft);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
font-size:14px;letter-spacing:0}}
header{{background:#18212b;color:#fff;border-bottom:4px solid #2f9e62}}
.header-inner{{max-width:1440px;margin:auto;padding:18px 24px;display:flex;gap:24px;
align-items:flex-end;justify-content:space-between;flex-wrap:wrap}}
h1{{font-size:24px;line-height:1.2;margin:0;font-weight:750}} .sub{{color:#bac5cf;margin-top:7px}}
.run-state{{text-align:right}} .run-state strong{{font-size:18px}}
main{{max-width:1440px;margin:auto;background:var(--paper);min-height:100vh}}
.band{{padding:18px 24px;border-bottom:1px solid var(--line)}}
.alert{{display:flex;gap:12px;align-items:flex-start;background:{'#fff4dd' if not live_visible else '#e8f5ee'};
border-left:5px solid {'#9a5b00' if not live_visible else '#087a43'};padding:13px 16px}}
.alert strong{{display:block;margin-bottom:3px}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));border-top:1px solid var(--line);
border-left:1px solid var(--line)}}
.metric{{padding:14px 16px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);
min-height:82px}} .metric span{{display:block;color:var(--muted);font-size:12px;margin-bottom:6px}}
.metric strong{{font-size:21px;line-height:1.1}} .good{{color:var(--green)}} .warn{{color:var(--amber)}}
.bad{{color:var(--red)}} h2{{font-size:18px;margin:0 0 12px}} h3{{font-size:15px;margin:0 0 9px}}
.section-head{{display:flex;justify-content:space-between;gap:14px;align-items:center;flex-wrap:wrap}}
.tag{{display:inline-flex;align-items:center;border:1px solid var(--line);padding:3px 7px;border-radius:4px;
font-size:12px;background:#fff}} .tag.good{{border-color:#9bcdb3;background:var(--green-bg)}}
.tag.warn{{border-color:#e3bd75;background:var(--amber-bg)}}
.table-wrap{{overflow:auto;border:1px solid var(--line)}} table{{border-collapse:collapse;width:100%;min-width:920px}}
th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{position:sticky;top:0;background:#f0f2f4;color:#48515b;font-size:12px;z-index:1}}
tbody tr:hover{{background:#f7faf8}} td.num{{font-variant-numeric:tabular-nums;text-align:right}}
.empty{{padding:24px;border:1px dashed #b8c0c8;color:var(--muted);text-align:center}}
.two-col{{display:grid;grid-template-columns:1.25fr .75fr;gap:24px}}
.gate-list{{display:grid;grid-template-columns:repeat(2,minmax(160px,1fr));gap:1px;background:var(--line);
border:1px solid var(--line)}} .gate{{padding:10px 12px;background:#fff;display:flex;justify-content:space-between;gap:8px}}
.gate .yes{{color:var(--green)}} .gate .no{{color:var(--red)}}
.progress{{height:10px;background:#e3e7eb;overflow:hidden;border-radius:4px;margin-top:8px}}
.progress div{{height:100%;background:#2f9e62;width:{min(100, 100 * int(shadow.get('trading_days', 0) or 0) / config.promotion.minimum_shadow_trading_days):.1f}%}}
.foot{{font-size:12px;color:var(--muted);line-height:1.7}}
details summary{{cursor:pointer;font-weight:650;padding:8px 0}}
@media(max-width:980px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.two-col{{grid-template-columns:1fr}}
.header-inner{{padding:16px}}.band{{padding:16px}}.run-state{{text-align:left}}}}
</style>
</head>
<body>
<header><div class="header-inner">
  <div><h1>WP V5 尾盘 T+1 决策台</h1>
  <div class="sub">目标：14:20–14:50 可成交买入，固定 T+1 收盘卖出后净收益为正</div></div>
  <div class="run-state"><div>{_e(trade_date)} · {_e(manifest.get('signal_slot', ''))}</div>
  <strong class="{status_class}">{_e(title_state)}</strong></div>
</div></header>
<main>
<section class="band"><div class="alert"><div><strong>{_e(alert_title)}</strong>
{_e(live_message)}</div></div></section>
<section class="band metrics">
  {_metric('模型状态', title_state, status_class)}
  {_metric('当前合格', str(len(passed)) if live_visible else '0', 'good' if len(passed) and live_visible else '')}
  {_metric('今日累计入账', str(len(session.get('candidates', []))), '')}
  {_metric('时点完整性', _session_coverage(manifest, config), _integrity_class(manifest))}
  {_metric('影子交易日', f"{int(shadow.get('trading_days', 0) or 0)} / {config.promotion.minimum_shadow_trading_days}", 'warn')}
  {_metric('三年 OOS 候选', str(backtest.get('candidate_events', '—')), '')}
  {_metric('尾盘覆盖率', _pct(manifest.get('tail_universe_coverage')), _coverage_class(manifest, config))}
  {_metric('行情 P95 年龄', _seconds(manifest.get('market_data_p95_age_seconds')), _age_class(manifest, config))}
</section>
<section class="band">
  <div class="section-head"><h2>{'当前合格候选' if live_visible else '当日冻结候选台账'}</h2>
	  <span class="tag {'good' if state == 'PRODUCTION' else 'warn'}">{_e('正式资格' if state == 'PRODUCTION' else ('影子资格，不授权实盘' if state == 'SHADOW' else '研究观察，不授权实盘'))}</span></div>
  {_candidate_table(passed if live_visible else session_candidates, live=live_visible)}
</section>
{_session_history_section(session_candidates) if live_visible else ''}
<section class="band two-col">
  <div><h2>模型证据与晋级门槛</h2>
    <div class="metrics" style="grid-template-columns:repeat(3,1fr)">
      {_metric('OOS 胜率', _pct(backtest.get('win_rate')), _return_class(backtest.get('win_rate'), 0.60))}
      {_metric('按日聚类 95% 下界', _pct(backtest.get('win_rate_day_clustered_lower')), _return_class(backtest.get('win_rate_day_clustered_lower'), 0.52))}
      {_metric('平均净收益', _pct(backtest.get('mean_net_return_pct'), already_percent=True), _return_class(backtest.get('mean_net_return_pct'), 0.30))}
      {_metric('Profit Factor', _number(backtest.get('profit_factor')), _return_class(backtest.get('profit_factor'), 1.30))}
      {_metric('校准误差 ECE', _pct(backtest.get('ece')), _inverse_class(backtest.get('ece'), 0.05))}
      {_metric('50bp 压力', _stress(backtest), 'good' if _stress(backtest) == '通过' else 'bad')}
    </div>
  </div>
  <div><h2>150 日影子进度</h2>
    <strong>{int(shadow.get('trading_days', 0) or 0)} / {config.promotion.minimum_shadow_trading_days} 个交易日</strong>
    <div class="progress"><div></div></div>
    <p class="foot">即使三年回测通过，也必须完成至少 150 个交易日的前瞻影子运行。同一政策下允许按固定协议滚动重训；特征、门槛、成本、样本或交易合同发生实质变化后重新计时。</p>
    {_promotion_checks(model.get('promotion', {}).get('checks', {}))}
  </div>
</section>
{_diagnostic_section(backtest)}
<section class="band"><div class="section-head"><h2>接近门槛但未通过</h2>
<span class="tag">用于解释，不是候选名单</span></div>{_near_table(near)}</section>
<section class="band"><h2>候选真值账本</h2>{_validation_table(historical)}</section>
<section class="band"><div class="section-head"><h2>历史因果回放</h2>
<span class="tag warn">RECONSTRUCTED_OOS，不冒充实时信号</span></div>
{_validation_table(replay_candidates)}</section>
<section class="band"><details><summary>固定交易与统计合同</summary>
	<div class="foot">信号时点：{', '.join(config.strategy.signal_slots)}；入场冲击：{config.execution.entry_slippage_bps:.0f}bp；
	费用及退出影响：{config.execution.round_trip_cost_bps:.0f}bp；基准全成本：{config.execution.baseline_all_in_cost_bps:.0f}bp；参考订单：{config.execution.reference_order_notional:,.0f} 元；
退出：下一 A 股交易日收盘；同一股票当日首次通过即锁定首次信号价；人工实际成交与模型候选统计相互独立。</div>
</details></section>
<section class="band foot">报告版本 {_e(manifest.get('report_revision', ''))} · 政策指纹 {_e(manifest.get('v3_policy_fingerprint') or '无')} · 模型指纹 {_e(manifest.get('v3_model_fingerprint') or '无')}</section>
</main>
</body></html>"""
    target.write_text(html_text, encoding="utf-8")


def _candidate_table(frame: pd.DataFrame, *, live: bool) -> str:
    if frame is None or frame.empty:
        return '<div class="empty">没有通过全部固定门槛的候选。</div>'
    rows = []
    for row in frame.to_dict(orient="records"):
        common = (
            "<tr>"
            f"<td><strong>{_e(row.get('name', ''))}</strong><br><span class='foot'>{_e(row.get('ts_code', ''))}</span></td>"
        )
        if live:
            timing = (
                f"<td>{_e(row.get('signal_slot', ''))}</td>"
                f"<td class='num'>{_number(row.get('signal_price'), 2)}</td>"
                f"<td>{_e(row.get('first_signal_time', ''))}</td>"
                f"<td class='num'>{_number(row.get('first_signal_price'), 2)}</td>"
            )
        else:
            timing = (
                f"<td>{_e(row.get('first_signal_time', ''))}</td>"
                f"<td class='num'>{_number(row.get('first_signal_price'), 2)}</td>"
                f"<td>{_e(row.get('last_signal_time', ''))}</td>"
            )
        rows.append(
            common
            + timing
            + f"<td class='num'>{int(row.get('appearance_count') or 1)}</td>"
            f"<td class='num good'>{_pct(row.get('p_net_positive'))}</td>"
            f"<td class='num'>{_pct(row.get('p_net_positive_lower'))}</td>"
            f"<td class='num'>{_pct(row.get('selection_rank_pct'))}</td>"
            f"<td class='num'>{_pct(row.get('expected_net_return_pct'), already_percent=True)}</td>"
            f"<td class='num'>{_pct(row.get('downside_q10_pct'), already_percent=True)}</td>"
            f"<td>{_e(_candidate_status(row))}</td>"
            "</tr>"
        )
    timing_headers = (
        "<th>当前时点</th><th>当前信号价</th><th>首次时点</th><th>首次信号价</th>"
        if live
        else "<th>首次时点</th><th>首次信号价</th><th>最后出现</th>"
    )
    return (
        '<div class="table-wrap"><table><thead><tr><th>股票</th>'
        + timing_headers
        + "<th>出现次数</th><th>净盈利概率</th><th>保守下界</th><th>同槽选择分位</th><th>期望净收益</th>"
        "<th>下行 Q10</th><th>状态</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _attach_locked_candidate_fields(
    predictions: pd.DataFrame,
    session: dict[str, Any],
) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    locked = {
        str(candidate.get("ts_code")): candidate
        for candidate in session.get("candidates", [])
    }
    result = predictions.copy()
    for index, row in result.iterrows():
        candidate = locked.get(str(row.get("ts_code")))
        if not candidate:
            continue
        for field in (
            "first_signal_time",
            "first_signal_price",
            "last_signal_time",
            "appearance_count",
            "status",
            "selection_rank_pct",
            "selection_score",
        ):
            result.at[index, field] = candidate.get(field)
    return result


def _session_history_section(frame: pd.DataFrame) -> str:
    return (
        '<section class="band"><div class="section-head">'
        "<h2>今日累计锁定信号</h2>"
        '<span class="tag">出现过即保留并逐票做 T+1 真值</span></div>'
        + _candidate_table(frame, live=False)
        + "</section>"
    )


def _candidate_status(row: dict[str, Any]) -> str:
    state = str(row.get("candidate_state") or row.get("status") or "")
    if state == "QUALIFIED":
        return "正式合格"
    if state == "SHADOW_QUALIFIED":
        return "影子合格"
    return state or "已锁定"


def _near_candidates(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions is None or predictions.empty or "p_net_positive" not in predictions:
        return pd.DataFrame()
    eligible = predictions.loc[
        predictions.get("execution_eligible", pd.Series(False, index=predictions.index)).fillna(False)
        & ~predictions.get("passes_policy", pd.Series(False, index=predictions.index)).fillna(False)
    ].copy()
    sort_columns = [
        column
        for column in (
            "selection_score",
            "selection_rank_pct",
            "p_net_positive",
        )
        if column in eligible
    ]
    return eligible.sort_values(
        sort_columns,
        ascending=False,
        kind="stable",
    ).head(20)


def _near_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<div class="empty">当前没有可解释的近门槛样本。</div>'
    rows = []
    for row in frame.to_dict(orient="records"):
        failed = [
            label
            for column, label in (
                ("passes_probability", "概率"),
                ("passes_probability_lower", "下界"),
                ("passes_expected_return", "期望收益"),
                ("passes_downside", "下行风险"),
                ("passes_selection_rank", "同槽排序"),
                ("passes_sample", "校准样本"),
                ("passes_empirical_lower", "经验下界"),
                ("passes_stability", "稳定性"),
                ("passes_freshness", "数据时效"),
            )
            if not bool(row.get(column))
        ]
        rows.append(
            f"<tr><td>{_e(row.get('name', ''))}<br><span class='foot'>{_e(row.get('ts_code', ''))}</span></td>"
            f"<td class='num'>{_pct(row.get('p_net_positive'))}</td>"
            f"<td class='num'>{_pct(row.get('p_net_positive_lower'))}</td>"
            f"<td>{_e('、'.join(failed) or '执行门槛')}</td></tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>股票</th><th>概率</th>'
        "<th>下界</th><th>未通过原因</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _validation_table(records: list[dict[str, Any]]) -> str:
    if not records:
        return '<div class="empty">尚无完成 T+1 真值验证的候选。</div>'
    rows = []
    for row in records[-100:][::-1]:
        net_return = row.get("net_return_pct")
        rows.append(
            f"<tr><td>{_e(row.get('trade_date', ''))}</td><td>{_e(row.get('target_trade_date', ''))}</td>"
            f"<td>{_e(row.get('name', ''))} {_e(row.get('ts_code', ''))}</td>"
            f"<td>{_e(row.get('first_signal_time', ''))}</td>"
            f"<td class='num'>{_number(row.get('first_signal_price'), 2)}</td>"
            f"<td>{_e('已验证' if row.get('truth_status') == 'verified' else '待验证')}</td>"
            f"<td class='num {_return_class(net_return, 0)}'>{_pct(net_return, already_percent=True)}</td></tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>计划日</th><th>验证日</th>'
        "<th>股票</th><th>首次时点</th><th>入场信号价</th><th>状态</th><th>净收益</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _promotion_checks(checks: dict[str, Any]) -> str:
    if not checks:
        return '<p class="foot">尚未形成可评估的晋级记录。</p>'
    return '<div class="gate-list">' + "".join(
        f"<div class='gate'><span>{_e(name)}</span><strong class='{'yes' if passed else 'no'}'>{'通过' if passed else '未通过'}</strong></div>"
        for name, passed in checks.items()
    ) + "</div>"


def _diagnostic_section(backtest: dict[str, Any]) -> str:
    diagnostics = backtest.get("diagnostics", {})
    if not diagnostics:
        return ""
    quality = diagnostics.get("score_quality", {})
    probability = quality.get("probability", {})
    composite = quality.get("selection_score", {})
    funnel = diagnostics.get("policy_funnel", [])
    top_rows = [
        row
        for row in diagnostics.get("top_n_per_slot", [])
        if row.get("score") in {"probability", "selection_score"}
        and int(row.get("top_n") or 0) in {1, 3, 5}
    ]
    funnel_rows = "".join(
        "<tr>"
        f"<td>{_e(row.get('gate', ''))}</td>"
        f"<td class='num'>{int(row.get('independent_pass_count') or 0):,}</td>"
        f"<td class='num'>{_pct(row.get('independent_pass_rate'))}</td>"
        f"<td class='num'>{int(row.get('cumulative_pass_count') or 0):,}</td>"
        f"<td class='num'>{_pct(row.get('cumulative_pass_rate'))}</td>"
        "</tr>"
        for row in funnel
    )
    top_table_rows = "".join(
        "<tr>"
        f"<td>{_e(row.get('score', ''))}</td>"
        f"<td class='num'>{int(row.get('top_n') or 0)}</td>"
        f"<td class='num'>{int(row.get('events') or 0):,}</td>"
        f"<td class='num'>{_pct(row.get('win_rate'))}</td>"
        f"<td class='num'>{_pct(row.get('win_rate_wilson_lower'))}</td>"
        f"<td class='num'>{_pct(row.get('mean_net_return_pct'), already_percent=True)}</td>"
        "</tr>"
        for row in top_rows
    )
    return (
        '<section class="band"><details><summary>样本外模型诊断（不改变交易门槛）</summary>'
        '<div class="metrics" style="grid-template-columns:repeat(4,1fr)">'
        + _metric("概率 ROC AUC", _number(probability.get("roc_auc"), 3))
        + _metric(
            "概率-收益秩相关",
            _number(probability.get("rank_correlation_to_net_return"), 3),
        )
        + _metric("选择分 ROC AUC", _number(composite.get("roc_auc"), 3))
        + _metric(
            "选择分-收益秩相关",
            _number(composite.get("rank_correlation_to_net_return"), 3),
        )
        + "</div>"
        '<h3 style="margin-top:16px">逐层淘汰漏斗</h3><div class="table-wrap"><table>'
        "<thead><tr><th>门槛</th><th>单项通过</th><th>单项比例</th>"
        "<th>累计通过</th><th>累计比例</th></tr></thead><tbody>"
        + funnel_rows
        + "</tbody></table></div>"
        '<h3 style="margin-top:16px">每时点 Top-N 诊断</h3>'
        '<p class="foot">仅检验排序能力；这些诊断组合没有被用来回填或改写固定策略。</p>'
        '<div class="table-wrap"><table><thead><tr><th>评分</th><th>Top-N</th>'
        "<th>事件</th><th>胜率</th><th>Wilson 下界</th><th>平均净收益</th>"
        "</tr></thead><tbody>"
        + top_table_rows
        + "</tbody></table></div></details></section>"
    )


def _metric(label: str, value: str, css: str = "") -> str:
    return f"<div class='metric'><span>{_e(label)}</span><strong class='{css}'>{_e(value)}</strong></div>"


def _pct(value: Any, *, already_percent: bool = False) -> str:
    try:
        numeric = float(value)
        if not already_percent:
            numeric *= 100
        return f"{numeric:+.2f}%" if already_percent else f"{numeric:.2f}%"
    except (TypeError, ValueError):
        return "—"


def _number(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _return_class(value: Any, threshold: float) -> str:
    try:
        return "good" if float(value) >= threshold else "bad"
    except (TypeError, ValueError):
        return ""


def _inverse_class(value: Any, threshold: float) -> str:
    try:
        return "good" if float(value) <= threshold else "bad"
    except (TypeError, ValueError):
        return ""


def _stress(backtest: dict[str, Any]) -> str:
    passed = backtest.get("stress", {}).get("50bps", {}).get("positive_total_return")
    return "通过" if passed else "未通过"


def _seconds(value: Any) -> str:
    try:
        return f"{float(value):.0f} 秒"
    except (TypeError, ValueError):
        return "—"


def _session_coverage(manifest: dict[str, Any], config: V3Config) -> str:
    covered = len(manifest.get("covered_slots") or [])
    total = len(config.strategy.signal_slots)
    missing = len(manifest.get("missing_slots") or [])
    integrity = str(manifest.get("session_integrity_status") or "COLLECTING")
    if integrity == "INCOMPLETE":
        return f"{covered}/{total} 缺 {missing} 槽"
    if integrity == "COMPLETE":
        return f"{covered}/{total} 完整"
    return f"{covered}/{total} 收集中"


def _integrity_class(manifest: dict[str, Any]) -> str:
    integrity = str(manifest.get("session_integrity_status") or "")
    if integrity == "COMPLETE":
        return "good"
    if integrity == "INCOMPLETE":
        return "bad"
    return ""


def _coverage_class(manifest: dict[str, Any], config: V3Config) -> str:
    try:
        coverage = float(manifest.get("tail_universe_coverage"))
    except (TypeError, ValueError):
        return ""
    return (
        "good"
        if coverage >= config.history.minimum_minute_universe_coverage
        else "bad"
    )


def _age_class(manifest: dict[str, Any], config: V3Config) -> str:
    try:
        age = float(manifest.get("market_data_p95_age_seconds"))
    except (TypeError, ValueError):
        return ""
    return (
        "good"
        if age <= config.execution.max_market_data_age_seconds
        else "bad"
    )


def _e(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value or ""))
