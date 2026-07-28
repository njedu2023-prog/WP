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
    decision = _decision_presentation(
        phase=phase,
        state=state,
        live_visible=live_visible,
        candidate_count=len(passed),
        health_status=str(manifest.get("health_status") or ""),
    )
    visible_candidates = passed if live_visible else session_candidates
    candidate_title = "当前合格候选" if live_visible else "今日候选记录"
    candidate_note = (
        "通过固定门槛的股票会全部列出，由人工决定是否以及买哪一支。"
        if live_visible and state == "PRODUCTION"
        else (
            "影子候选仅用于验证模型，不是实盘买入建议。"
            if live_visible
            else "交易窗口结束后只保留冻结记录，不再新增或替换候选。"
        )
    )
    candidate_empty_message = _candidate_empty_message(
        phase=phase,
        state=state,
        live_visible=live_visible,
    )
    current_count = len(passed) if live_visible else 0
    data_status, data_status_class = _data_status(manifest, phase)
    next_checkpoint = _next_checkpoint(
        phase=phase,
        signal_slot=str(manifest.get("signal_slot") or ""),
        config=config,
    )
    authorization_label = (
        "正式模型"
        if state == "PRODUCTION"
        else "仅研究观察"
    )
    authorization_class = "good" if state == "PRODUCTION" else "warn"
    has_backtest_candidates = int(backtest.get("candidate_events") or 0) > 0
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WP V5 尾盘候选助手</title>
<style>
:root {{
  --ink:#151a20; --muted:#65707c; --line:#d9dee4; --paper:#ffffff;
  --soft:#f1f4f6; --soft-2:#f7f8fa; --green:#087a43; --green-bg:#e8f5ee;
  --red:#b42318; --red-bg:#fef0ee; --amber:#8a5300; --amber-bg:#fff4dd;
  --blue:#175e92; --blue-bg:#eaf3f9; --header:#18212b;
}}
*{{box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{margin:0;background:var(--soft);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
font-size:14px;letter-spacing:0;line-height:1.55}}
header{{background:var(--header);color:#fff;border-bottom:4px solid #2f9e62}}
.header-inner{{max-width:1360px;margin:auto;padding:17px 24px;display:flex;gap:28px;
align-items:center;justify-content:space-between}}
.brand-kicker{{color:#88caa6;font-size:12px;font-weight:700;margin-bottom:2px}}
h1{{font-size:24px;line-height:1.25;margin:0;font-weight:760}}
.sub{{color:#bdc7d0;margin-top:5px}}
.header-meta{{text-align:right;min-width:220px}}
.header-meta-row{{display:flex;align-items:center;justify-content:flex-end;gap:8px}}
.phase-chip{{display:inline-flex;border:1px solid #6d7883;border-radius:4px;padding:3px 7px;
font-size:12px;color:#e6ebef}}
.header-state{{display:block;font-size:16px;margin-top:5px}}
.header-update{{display:block;color:#9eabb6;font-size:12px;margin-top:3px}}
main{{max-width:1360px;margin:auto;background:var(--paper);min-height:100vh;box-shadow:0 0 0 1px #e6e9ed}}
.decision-band{{padding:20px 24px;background:var(--soft-2);border-bottom:1px solid var(--line)}}
.decision-panel{{display:grid;grid-template-columns:minmax(320px,1.15fr) minmax(420px,.85fr);
border:1px solid var(--line);border-left-width:6px;background:#fff}}
.decision-panel.good{{border-left-color:var(--green);background:var(--green-bg)}}
.decision-panel.warn{{border-left-color:var(--amber);background:var(--amber-bg)}}
.decision-panel.bad{{border-left-color:var(--red);background:var(--red-bg)}}
.decision-copy{{padding:18px 20px;display:flex;gap:13px;align-items:flex-start}}
.status-dot{{width:12px;height:12px;border-radius:50%;background:currentColor;margin-top:7px;flex:0 0 auto}}
.decision-kicker{{display:block;color:var(--muted);font-size:12px;font-weight:700;margin-bottom:3px}}
.decision-title{{font-size:24px;line-height:1.25;margin:0 0 6px}}
.decision-text{{margin:0;color:#3e4852;max-width:720px}}
.decision-facts{{display:grid;grid-template-columns:repeat(2,minmax(150px,1fr));border-left:1px solid var(--line)}}
.fact{{padding:14px 16px;border-bottom:1px solid var(--line);min-height:76px}}
.fact:nth-child(odd){{border-right:1px solid var(--line)}}
.fact:nth-last-child(-n+2){{border-bottom:0}}
.fact span{{display:block;color:var(--muted);font-size:12px;margin-bottom:4px}}
.fact strong{{font-size:18px;line-height:1.25;font-variant-numeric:tabular-nums}}
.band{{padding:22px 24px;border-bottom:1px solid var(--line);scroll-margin-top:12px}}
.primary-band{{padding-top:24px;padding-bottom:26px}}
.section-head{{display:flex;justify-content:space-between;gap:14px;align-items:flex-start;flex-wrap:wrap}}
h2{{font-size:19px;line-height:1.3;margin:0 0 6px}}
h3{{font-size:15px;margin:0 0 9px}}
.section-note{{margin:0 0 14px;color:var(--muted)}}
.tag{{display:inline-flex;align-items:center;border:1px solid var(--line);padding:3px 7px;border-radius:4px;
font-size:12px;background:#fff;white-space:nowrap}}
.tag.good{{border-color:#9bcdb3;background:var(--green-bg);color:var(--green)}}
.tag.warn{{border-color:#e3bd75;background:var(--amber-bg);color:var(--amber)}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));border-top:1px solid var(--line);
border-left:1px solid var(--line)}}
.metric{{padding:14px 16px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);
min-height:82px;background:#fff}}
.metric span{{display:block;color:var(--muted);font-size:12px;margin-bottom:6px}}
.metric strong{{font-size:20px;line-height:1.2}}
.good{{color:var(--green)}} .warn{{color:var(--amber)}} .bad{{color:var(--red)}}
.table-wrap{{overflow:auto;border:1px solid var(--line)}}
table{{border-collapse:collapse;width:100%;min-width:920px}}
th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{position:sticky;top:0;background:#f0f2f4;color:#48515b;font-size:12px;z-index:1}}
tbody tr:hover{{background:#f7faf8}}
td.num{{font-variant-numeric:tabular-nums;text-align:right}}
.empty{{padding:28px 18px;border:1px dashed #b8c0c8;background:var(--soft-2);color:#48515b;text-align:center}}
.empty strong{{display:block;color:var(--ink);font-size:16px;margin-bottom:4px}}
.mobile-list{{display:none}}
.candidate-card,.validation-card{{border:1px solid var(--line);border-radius:8px;background:#fff;padding:14px}}
.candidate-card-head,.validation-card-head{{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;
padding-bottom:10px;border-bottom:1px solid var(--line)}}
.candidate-card-head strong,.validation-card-head strong{{font-size:17px}}
.code{{display:block;color:var(--muted);font-size:12px;margin-top:1px}}
.card-primary{{display:flex;align-items:flex-end;justify-content:space-between;gap:14px;padding:12px 0}}
.card-primary span{{display:block;color:var(--muted);font-size:12px}}
.card-primary strong{{font-size:24px;font-variant-numeric:tabular-nums}}
.card-primary small{{color:var(--muted)}}
.card-stats{{display:grid;grid-template-columns:repeat(2,1fr);margin:0;border-top:1px solid var(--line);
border-left:1px solid var(--line)}}
.card-stats div{{padding:9px 10px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}}
.card-stats dt{{color:var(--muted);font-size:12px}} .card-stats dd{{margin:2px 0 0;font-weight:700}}
.truth-summary{{margin:12px 0 14px}}
.two-col{{display:grid;grid-template-columns:1.15fr .85fr;gap:24px}}
.gate-list{{display:grid;grid-template-columns:repeat(2,minmax(180px,1fr));gap:1px;background:var(--line);
border:1px solid var(--line)}}
.gate{{padding:10px 12px;background:#fff;display:flex;justify-content:space-between;gap:10px}}
.gate .yes{{color:var(--green)}} .gate .no{{color:var(--red)}}
.progress{{height:10px;background:#e3e7eb;overflow:hidden;border-radius:4px;margin-top:8px}}
.progress div{{height:100%;background:#2f9e62;width:{min(100, 100 * int(shadow.get('trading_days', 0) or 0) / config.promotion.minimum_shadow_trading_days):.1f}%}}
.research-verdict{{padding:12px 14px;margin:4px 0 16px;border-left:4px solid var(--amber);
background:var(--amber-bg);color:#4d3a19}}
.foot{{font-size:12px;color:var(--muted);line-height:1.7}}
.disclosure{{border:0}}
.disclosure>summary{{cursor:pointer;list-style:none;display:flex;align-items:center;justify-content:space-between;
gap:16px;padding:2px 0;font-weight:700}}
.disclosure>summary::-webkit-details-marker{{display:none}}
.disclosure>summary::after{{content:"＋";font-size:20px;color:var(--muted);font-weight:400}}
.disclosure[open]>summary::after{{content:"－"}}
.disclosure-title{{display:block;font-size:18px;color:var(--ink)}}
.disclosure-sub{{display:block;color:var(--muted);font-size:12px;font-weight:400;margin-top:2px}}
.disclosure-body{{padding-top:16px}}
.tech-footer{{font-size:12px;color:var(--muted);line-height:1.7;background:var(--soft-2)}}
@media(max-width:980px){{
  .header-inner{{padding:15px 16px;align-items:flex-start}}
  .decision-band,.band{{padding-left:16px;padding-right:16px}}
  .decision-panel{{grid-template-columns:1fr}}
  .decision-facts{{border-left:0;border-top:1px solid var(--line)}}
  .two-col{{grid-template-columns:1fr}}
  .metrics{{grid-template-columns:repeat(2,1fr)}}
  .header-meta{{min-width:0}}
}}
@media(max-width:720px){{
  body{{font-size:14px}}
  .header-inner{{display:block}}
  h1{{font-size:21px}}
  .sub{{font-size:12px}}
  .header-meta{{text-align:left;margin-top:12px}}
  .header-meta-row{{justify-content:flex-start}}
  .decision-copy{{padding:16px}}
  .decision-title{{font-size:21px}}
  .decision-facts{{grid-template-columns:repeat(2,1fr)}}
  .fact{{padding:11px 12px;min-height:68px}}
  .fact strong{{font-size:16px}}
  .band{{padding-top:19px;padding-bottom:20px}}
  .desktop-only{{display:none}}
  .mobile-list{{display:grid;gap:10px}}
  .metrics{{grid-template-columns:repeat(2,1fr)}}
  .metric{{min-height:76px;padding:12px}}
  .metric strong{{font-size:17px}}
  .gate-list{{grid-template-columns:1fr}}
  .disclosure-title{{font-size:16px}}
}}
@media(max-width:420px){{
  .decision-facts,.metrics{{grid-template-columns:1fr 1fr}}
  .fact strong{{font-size:15px}}
  .card-stats{{grid-template-columns:1fr 1fr}}
}}
</style>
</head>
<body>
<header><div class="header-inner">
  <div>
    <div class="brand-kicker">WP V5 · T+1 固定收盘退出</div>
    <h1>尾盘候选助手</h1>
    <div class="sub">14:20–14:50 观察全部合格票；没有合格票时明确保持空仓</div>
  </div>
  <div class="header-meta">
    <div class="header-meta-row"><strong>{_e(trade_date)}</strong><span class="phase-chip">{_e(_phase_label(phase))}</span></div>
    <strong class="header-state {status_class}">{_e(title_state)}</strong>
    <span class="header-update">更新于 {_e(manifest.get('report_revision', ''))}</span>
  </div>
</div></header>
<main>
<section class="decision-band" aria-labelledby="decision-title">
  <div class="decision-panel {decision['tone']}">
    <div class="decision-copy">
      <span class="status-dot {decision['tone']}" aria-hidden="true"></span>
      <div>
        <span class="decision-kicker">现在该怎么做</span>
        <h2 class="decision-title" id="decision-title">{_e(decision['title'])}</h2>
        <p class="decision-text">{_e(decision['message'])}</p>
      </div>
    </div>
    <div class="decision-facts" aria-label="当前交易摘要">
      {_fact('现在可用候选', str(current_count), 'good' if current_count else '')}
      {_fact('今日曾出现', str(len(session_candidates)), '')}
      {_fact('数据状态', data_status, data_status_class)}
      {_fact('下一节点', next_checkpoint, '')}
    </div>
  </div>
</section>
<section class="band primary-band" id="candidates">
  <div class="section-head">
    <div><h2>{_e(candidate_title)}</h2><p class="section-note">{_e(candidate_note)}</p></div>
    <span class="tag {authorization_class}">{_e(authorization_label)}</span>
  </div>
  {_candidate_table(
      visible_candidates,
      live=live_visible,
      empty_message=candidate_empty_message,
  )}
</section>
{_session_history_section(session_candidates) if live_visible else ''}
<section class="band" id="truth">
  <div class="section-head">
    <div><h2>T+1 真值结果</h2><p class="section-note">只使用锁定的首次信号价，按下一交易日收盘价并扣除固定成本计算。</p></div>
    <span class="tag">模型候选，不等于人工成交</span>
  </div>
  {_validation_summary(historical)}
  {_validation_table(historical)}
</section>
<section class="band" id="research">
  <details class="disclosure">
    <summary>
      <span><span class="disclosure-title">模型研究与上线条件</span>
      <span class="disclosure-sub">三年样本外结果、150 日影子进度和全部晋级门槛</span></span>
    </summary>
    <div class="disclosure-body">
      <div class="research-verdict">{_e(_research_verdict(state, backtest))}</div>
      <div class="two-col">
        <div><h3>三年样本外证据</h3>
          <div class="metrics" style="grid-template-columns:repeat(3,1fr)">
            {_metric('候选事件', str(int(backtest.get('candidate_events') or 0)), '')}
            {_metric('候选胜率', _pct(backtest.get('win_rate')) if has_backtest_candidates else '—', _return_class(backtest.get('win_rate'), 0.60) if has_backtest_candidates else '')}
            {_metric('按日聚类 95% 下界', _pct(backtest.get('win_rate_day_clustered_lower')) if has_backtest_candidates else '—', _return_class(backtest.get('win_rate_day_clustered_lower'), 0.52) if has_backtest_candidates else '')}
            {_metric('平均净收益', _pct(backtest.get('mean_net_return_pct'), already_percent=True) if has_backtest_candidates else '—', _return_class(backtest.get('mean_net_return_pct'), 0.30) if has_backtest_candidates else '')}
            {_metric('Profit Factor', _number(backtest.get('profit_factor')) if has_backtest_candidates else '—', _return_class(backtest.get('profit_factor'), 1.30) if has_backtest_candidates else '')}
            {_metric('50bp 压力测试', _stress(backtest), 'good' if _stress(backtest) == '通过' else ('bad' if has_backtest_candidates else ''))}
          </div>
        </div>
        <div><h3>前瞻影子进度</h3>
          <strong>{int(shadow.get('trading_days', 0) or 0)} / {config.promotion.minimum_shadow_trading_days} 个交易日</strong>
          <div class="progress" aria-label="150 日影子进度"><div></div></div>
          <p class="foot">只有三年回测和至少 150 个真实交易日的影子验证全部通过，模型才有资格进入生产。策略合同发生实质变化后重新计时。</p>
          {_promotion_checks(model.get('promotion', {}).get('checks', {}))}
        </div>
      </div>
    </div>
  </details>
</section>
{_diagnostic_section(backtest)}
{_near_section(near)}
<section class="band">
  <details class="disclosure">
    <summary><span><span class="disclosure-title">历史因果回放</span>
    <span class="disclosure-sub">重建的样本外结果，仅用于研究，不冒充实时信号</span></span></summary>
    <div class="disclosure-body">{_validation_table(replay_candidates)}</div>
  </details>
</section>
<section class="band">
  <details class="disclosure">
    <summary><span><span class="disclosure-title">交易与统计口径</span>
    <span class="disclosure-sub">查看信号时点、成本、入场价和退出规则</span></span></summary>
    <div class="disclosure-body foot">信号时点：{', '.join(config.strategy.signal_slots)}；入场冲击：{config.execution.entry_slippage_bps:.0f}bp；
    费用及退出影响：{config.execution.round_trip_cost_bps:.0f}bp；基准全成本：{config.execution.baseline_all_in_cost_bps:.0f}bp；参考订单：{config.execution.reference_order_notional:,.0f} 元；
    退出：下一 A 股交易日收盘；同一股票当日首次通过即锁定首次信号价；人工实际成交与模型候选统计相互独立。</div>
  </details>
</section>
<section class="band tech-footer">报告版本 {_e(manifest.get('report_revision', ''))} · 政策指纹 {_e(manifest.get('v3_policy_fingerprint') or '无')} · 模型指纹 {_e(manifest.get('v3_model_fingerprint') or '无')}</section>
</main>
</body></html>"""
    target.write_text(html_text, encoding="utf-8")


def _candidate_table(
    frame: pd.DataFrame,
    *,
    live: bool,
    empty_message: str = "没有通过全部固定门槛的候选。",
) -> str:
    if frame is None or frame.empty:
        return (
            '<div class="empty"><strong>当前没有候选</strong>'
            f"{_e(empty_message)}</div>"
        )
    rows = []
    cards = []
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
        cards.append(_candidate_card(row, live=live))
    timing_headers = (
        "<th>当前时点</th><th>当前信号价</th><th>首次时点</th><th>首次信号价</th>"
        if live
        else "<th>首次时点</th><th>首次信号价</th><th>最后出现</th>"
    )
    return (
        '<div class="table-wrap desktop-only"><table><thead><tr><th>股票</th>'
        + timing_headers
        + "<th>出现次数</th><th>净盈利概率</th><th>保守下界</th><th>同槽选择分位</th><th>期望净收益</th>"
        "<th>下行 Q10</th><th>状态</th></tr></thead><tbody>"
        + "".join(rows)
        + '</tbody></table></div><div class="mobile-list">'
        + "".join(cards)
        + "</div>"
    )


def _candidate_card(row: dict[str, Any], *, live: bool) -> str:
    primary_time = (
        row.get("signal_slot")
        if live
        else row.get("first_signal_time")
    )
    primary_price = (
        row.get("signal_price")
        if live
        else row.get("first_signal_price")
    )
    locked_note = (
        f"首次 {_e(row.get('first_signal_time', ''))} · "
        f"{_number(row.get('first_signal_price'), 2)}"
        if live
        else f"最后出现 {_e(row.get('last_signal_time', ''))}"
    )
    return (
        '<article class="candidate-card">'
        '<div class="candidate-card-head"><div>'
        f"<strong>{_e(row.get('name', ''))}</strong>"
        f"<span class='code'>{_e(row.get('ts_code', ''))}</span></div>"
        f"<span class='tag'>{_e(_candidate_status(row))}</span></div>"
        '<div class="card-primary"><div>'
        f"<span>{'当前信号价' if live else '首次信号价'}</span>"
        f"<strong>{_number(primary_price, 2)}</strong></div>"
        f"<small>{_e(primary_time)}<br>{locked_note}</small></div>"
        '<dl class="card-stats">'
        f"<div><dt>盈利概率</dt><dd class='good'>{_pct(row.get('p_net_positive'))}</dd></div>"
        f"<div><dt>保守概率</dt><dd>{_pct(row.get('p_net_positive_lower'))}</dd></div>"
        f"<div><dt>期望净收益</dt><dd>{_pct(row.get('expected_net_return_pct'), already_percent=True)}</dd></div>"
        f"<div><dt>下行 Q10</dt><dd>{_pct(row.get('downside_q10_pct'), already_percent=True)}</dd></div>"
        f"<div><dt>同槽分位</dt><dd>{_pct(row.get('selection_rank_pct'))}</dd></div>"
        f"<div><dt>出现次数</dt><dd>{int(row.get('appearance_count') or 1)}</dd></div>"
        "</dl></article>"
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
        '<div><h2>今日已经出现过</h2>'
        '<p class="section-note">候选一旦出现即锁定首次时间和价格；后续消失也不会从验证台账中删除。</p></div>'
        '<span class="tag">逐票等待 T+1 真值</span></div>'
        + _candidate_table(
            frame,
            live=False,
            empty_message="今天尚未出现过通过全部固定门槛的股票。",
        )
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


def _near_section(frame: pd.DataFrame) -> str:
    return (
        '<section class="band"><details class="disclosure"><summary><span>'
        '<span class="disclosure-title">为什么没有成为候选</span>'
        '<span class="disclosure-sub">查看接近门槛但未通过的股票及具体原因；这些股票不是候选名单</span>'
        '</span></summary><div class="disclosure-body">'
        + _near_table(frame)
        + "</div></details></section>"
    )


def _validation_table(records: list[dict[str, Any]]) -> str:
    if not records:
        return (
            '<div class="empty"><strong>暂无候选需要验证</strong>'
            "候选出现后，会在下一交易日收盘后自动补齐净收益结果。</div>"
        )
    rows = []
    cards = []
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
        cards.append(_validation_card(row))
    return (
        '<div class="table-wrap desktop-only"><table><thead><tr><th>计划日</th><th>验证日</th>'
        "<th>股票</th><th>首次时点</th><th>入场信号价</th><th>状态</th><th>净收益</th></tr></thead><tbody>"
        + "".join(rows)
        + '</tbody></table></div><div class="mobile-list">'
        + "".join(cards)
        + "</div>"
    )


def _validation_card(row: dict[str, Any]) -> str:
    verified = row.get("truth_status") == "verified"
    net_return = row.get("net_return_pct")
    return (
        '<article class="validation-card">'
        '<div class="validation-card-head"><div>'
        f"<strong>{_e(row.get('name', ''))}</strong>"
        f"<span class='code'>{_e(row.get('ts_code', ''))}</span></div>"
        f"<span class='tag {'good' if verified else 'warn'}'>{'已验证' if verified else '待验证'}</span></div>"
        '<div class="card-primary"><div><span>扣费后净收益</span>'
        f"<strong class='{_return_class(net_return, 0)}'>{_pct(net_return, already_percent=True)}</strong></div>"
        f"<small>计划 {_e(row.get('trade_date', ''))}<br>验证 {_e(row.get('target_trade_date', ''))}</small></div>"
        '<dl class="card-stats">'
        f"<div><dt>首次时点</dt><dd>{_e(row.get('first_signal_time', ''))}</dd></div>"
        f"<div><dt>首次信号价</dt><dd>{_number(row.get('first_signal_price'), 2)}</dd></div>"
        "</dl></article>"
    )


def _validation_summary(records: list[dict[str, Any]]) -> str:
    verified = [
        record
        for record in records
        if record.get("truth_status") == "verified"
        and record.get("net_return_pct") is not None
    ]
    pending = sum(
        record.get("truth_status") != "verified"
        for record in records
    )
    if not verified:
        return ""
    returns = [float(record["net_return_pct"]) for record in verified]
    wins = sum(value > 0 for value in returns)
    summary = (
        '<div class="metrics truth-summary" style="grid-template-columns:repeat(4,1fr)">'
        + _metric("已验证候选", str(len(verified)))
        + _metric("净收益为正", f"{wins} / {len(verified)}", "good" if wins else "")
        + _metric(
            "候选胜率",
            _pct(wins / len(verified)),
            "good" if wins / len(verified) >= 0.50 else "bad",
        )
        + _metric(
            "平均净收益",
            _pct(sum(returns) / len(returns), already_percent=True),
            _return_class(sum(returns) / len(returns), 0),
        )
        + "</div>"
    )
    if pending:
        summary += f"<p class='foot'>另有 {pending} 支待验证。</p>"
    return summary


def _promotion_checks(checks: dict[str, Any]) -> str:
    if not checks:
        return '<p class="foot">尚未形成可评估的晋级记录。</p>'
    labels = {
        "backtest_gate": "三年回测",
        "shadow_trading_days": "影子交易日",
        "shadow_candidate_days": "影子候选日",
        "shadow_candidates": "影子候选数量",
        "shadow_win_rate": "影子胜率",
        "shadow_win_rate_lower": "胜率置信下界",
        "shadow_clustered_win_rate_lower": "按日胜率下界",
        "shadow_mean_net_return": "平均净收益",
        "shadow_clustered_mean_return_lower": "按日收益下界",
        "shadow_median_net_return": "净收益中位数",
        "shadow_profit_factor": "盈亏因子",
        "shadow_ece": "概率校准误差",
        "shadow_50bps_stress": "50bp 压力测试",
    }
    return '<div class="gate-list">' + "".join(
        f"<div class='gate'><span>{_e(labels.get(name, name))}</span><strong class='{'yes' if passed else 'no'}'>{'通过' if passed else '未通过'}</strong></div>"
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
    gate_labels = {
        "execution": "可成交与流动性",
        "prior_oos_policy_authorized": "此前样本外政策已授权",
        "all_candidate_gates": "全部候选门槛",
        "final_policy": "最终政策",
    }
    score_labels = {
        "probability": "盈利概率",
        "selection_score": "综合选择分",
    }
    funnel_rows = "".join(
        "<tr>"
        f"<td>{_e(gate_labels.get(row.get('gate'), row.get('gate', '')))}</td>"
        f"<td class='num'>{int(row.get('independent_pass_count') or 0):,}</td>"
        f"<td class='num'>{_pct(row.get('independent_pass_rate'))}</td>"
        f"<td class='num'>{int(row.get('cumulative_pass_count') or 0):,}</td>"
        f"<td class='num'>{_pct(row.get('cumulative_pass_rate'))}</td>"
        "</tr>"
        for row in funnel
    )
    top_table_rows = "".join(
        "<tr>"
        f"<td>{_e(score_labels.get(row.get('score'), row.get('score', '')))}</td>"
        f"<td class='num'>{int(row.get('top_n') or 0)}</td>"
        f"<td class='num'>{int(row.get('events') or 0):,}</td>"
        f"<td class='num'>{_pct(row.get('win_rate'))}</td>"
        f"<td class='num'>{_pct(row.get('win_rate_wilson_lower'))}</td>"
        f"<td class='num'>{_pct(row.get('mean_net_return_pct'), already_percent=True)}</td>"
        "</tr>"
        for row in top_rows
    )
    return (
        '<section class="band"><details class="disclosure"><summary><span>'
        '<span class="disclosure-title">样本外模型诊断</span>'
        '<span class="disclosure-sub">排序能力、逐层淘汰漏斗和每时点 Top-N；不会改变固定门槛</span>'
        '</span></summary><div class="disclosure-body">'
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
        + "</tbody></table></div></div></details></section>"
    )


def _decision_presentation(
    *,
    phase: str,
    state: str,
    live_visible: bool,
    candidate_count: int,
    health_status: str,
) -> dict[str, str]:
    if health_status == "v3_input_not_ready" and phase != "CLOSED":
        return {
            "tone": "bad",
            "title": "暂停使用",
            "message": "合法行情快照缺失，系统已停止输出候选并等待自动修复。",
        }
    if live_visible and state != "PRODUCTION":
        return {
            "tone": "warn",
            "title": "仅观察，不实盘",
            "message": (
                "当前模型尚未通过完整回测与影子验证。页面只记录影子候选，"
                "不提供实盘买入资格。"
            ),
        }
    if live_visible and candidate_count:
        return {
            "tone": "good",
            "title": f"有 {candidate_count} 支合格候选",
            "message": "这些股票已通过全部固定门槛；系统列出全部候选，由人工决定是否以及买哪一支。",
        }
    if live_visible:
        return {
            "tone": "warn",
            "title": "暂不买入",
            "message": "当前时点没有股票通过全部门槛。空仓是正式决策，不会为了产生名单而降低标准。",
        }
    if phase == "PRE_SIGNAL":
        return {
            "tone": "warn",
            "title": "等待 14:20",
            "message": "候选窗口尚未开始。14:20 起每 5 分钟更新一次，14:50 后不再新增。",
        }
    if phase == "CLOSED":
        return {
            "tone": "warn",
            "title": "已收盘，不再买入",
            "message": "14:50 后禁止新增候选，15:00 后页面只用于查看冻结记录和 T+1 验证结果。",
        }
    return {
        "tone": "warn",
        "title": "暂不操作",
        "message": "系统正在准备尾盘数据，候选窗口开始前不提供买入名单。",
    }


def _phase_label(phase: str) -> str:
    return {
        "PRE_SIGNAL": "等待开窗",
        "WARMUP": "数据准备",
        "SIGNAL": "候选窗口",
        "FREEZE": "冻结候选",
        "CLOSED": "收盘复盘",
    }.get(phase, phase or "状态未知")


def _data_status(
    manifest: dict[str, Any],
    phase: str,
) -> tuple[str, str]:
    health = str(manifest.get("health_status") or "")
    if health == "v3_input_not_ready":
        return "数据异常", "bad"
    if phase == "CLOSED":
        return "记录已冻结", ""
    if health == "ok":
        return "行情正常", "good"
    return "准备中", "warn"


def _next_checkpoint(
    *,
    phase: str,
    signal_slot: str,
    config: V3Config,
) -> str:
    if phase == "CLOSED":
        return "下一交易日 14:20"
    if phase in {"PRE_SIGNAL", "WARMUP"}:
        return "14:20 开始"
    slots = list(config.strategy.signal_slots)
    if signal_slot in slots:
        index = slots.index(signal_slot)
        if index + 1 < len(slots):
            return f"{slots[index + 1]} 更新"
        return f"{config.strategy.candidate_freeze_time} 冻结"
    return "等待下一时点"


def _candidate_empty_message(
    *,
    phase: str,
    state: str,
    live_visible: bool,
) -> str:
    if live_visible and state != "PRODUCTION":
        return "模型仍处于研究观察状态，不会发布具有实盘资格的候选。"
    if live_visible:
        return "本时点没有股票同时通过盈利概率、下行风险、流动性和数据新鲜度门槛。"
    if phase == "CLOSED":
        return "今天 14:20–14:50 没有股票通过全部固定门槛。"
    return "候选窗口尚未开始。"


def _research_verdict(state: str, backtest: dict[str, Any]) -> str:
    gate_passed = bool(backtest.get("backtest_gate", {}).get("passed"))
    if state == "PRODUCTION" and gate_passed:
        return "三年样本外与前瞻影子门槛均已通过，当前模型具有生产资格。"
    if int(backtest.get("candidate_events") or 0) == 0:
        return (
            "回测未通过：三年滚动样本外研究没有找到可在固定成本后稳定盈利、"
            "并通过独立确认的候选政策。"
        )
    return "回测未通过：现有候选的收益、胜率或压力测试尚未达到生产门槛。"


def _fact(label: str, value: str, css: str = "") -> str:
    return (
        "<div class='fact'>"
        f"<span>{_e(label)}</span><strong class='{css}'>{_e(value)}</strong>"
        "</div>"
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
    if int(backtest.get("candidate_events") or 0) == 0:
        return "无候选"
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
