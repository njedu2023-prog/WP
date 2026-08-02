from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from .cohorts import CohortSelection, select_live_cohorts
from .contracts import V3Config
from .ledger import session_records


HEALTHY_MANIFEST_STATUSES = {
    "",
    "ok",
    "research_ready",
    "无符合条件股票",
}


def render_v3_dashboard(
    path: str | Path,
    *,
    manifest: dict[str, Any],
    predictions: pd.DataFrame,
    ledger: dict[str, Any],
    registry: dict[str, Any],
    config: V3Config,
    replay: dict[str, Any] | None = None,
    legacy_audit: dict[str, Any] | None = None,
    retrospective: dict[str, Any] | None = None,
    research_seed: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    replay = replay or {}
    retrospective = retrospective or {}
    research_seed = research_seed or {}
    trade_date = _compact_date(manifest.get("source_trade_date"))
    phase = str(manifest.get("session_phase") or "PRE_SIGNAL")
    current_session = _session_for_date(ledger, trade_date)
    selection = select_live_cohorts(predictions, config)
    qualified = _current_cohort(
        current_session,
        selection,
        cohort="QUALIFIED",
    )
    observations = _current_cohort(
        current_session,
        selection,
        cohort="OBSERVATION",
    )
    live_visible = bool(
        manifest.get(
            "live_display_allowed",
            phase in {"SIGNAL", "NO_NEW_SIGNAL", "FROZEN"},
        )
    ) and phase != "CLOSED"
    records = [
        record
        for session in ledger.get("sessions", [])
        for record in session_records(session)
    ]
    qualified_stats = _cohort_stats(records, "QUALIFIED")
    observation_stats = _cohort_stats(records, "OBSERVATION")
    qualified_shadow_stats = _cohort_stats(
        [
            row
            for row in records
            if _compact_date(row.get("trade_date"))
            >= config.evidence.live_shadow_start_date
        ],
        "QUALIFIED",
    )
    state = str(manifest.get("v3_state") or "MODEL_NOT_READY")
    decision = _decision_copy(
        phase=phase,
        state=state,
        qualified_count=len(qualified),
        live_visible=live_visible,
        health=str(manifest.get("health_status") or "ok"),
    )
    evidence_contract = (
        f"{_display_date(config.evidence.retrospective_start_date)} - "
        f"{_display_date(config.evidence.retrospective_end_date)} 回测；"
        f"{_display_date(config.evidence.live_shadow_start_date)} 起真实影子统计"
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>WP 尾盘决策台</title>
  <style>
    :root {{
      --page: #f5f5f7;
      --surface: #ffffff;
      --surface-soft: #fafafa;
      --text: #1d1d1f;
      --muted: #6e6e73;
      --faint: #86868b;
      --line: #d2d2d7;
      --line-soft: #e8e8ed;
      --blue: #0071e3;
      --green: #16823b;
      --green-soft: #eff9f2;
      --amber: #9a6700;
      --amber-soft: #fff8e8;
      --red: #c9342f;
      --red-soft: #fff2f1;
      --shadow: 0 12px 30px rgba(0, 0, 0, .045);
    }}
    * {{ box-sizing: border-box; }}
    html {{ background: var(--page); }}
    body {{
      margin: 0;
      color: var(--text);
      background: var(--page);
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
        "PingFang SC", "Helvetica Neue", Arial, sans-serif;
      font-size: 15px;
      line-height: 1.5;
      letter-spacing: 0;
      -webkit-font-smoothing: antialiased;
    }}
    button {{ font: inherit; letter-spacing: 0; }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(245, 245, 247, .88);
      border-bottom: 1px solid rgba(210, 210, 215, .85);
      backdrop-filter: saturate(180%) blur(18px);
    }}
    .topbar-inner {{
      width: min(1240px, calc(100% - 36px));
      min-height: 48px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    .brand {{ font-size: 14px; font-weight: 700; }}
    .top-meta {{
      display: flex;
      align-items: center;
      gap: 16px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .page {{
      width: min(1240px, calc(100% - 36px));
      margin: 0 auto;
      padding: 28px 0 54px;
    }}
    .headline {{
      display: flex;
      align-items: center;
      justify-content: flex-end;
      padding: 4px 2px 14px;
    }}
    h2, h3, p {{ margin: 0; }}
    .mode {{
      min-width: 180px;
      text-align: right;
    }}
    .mode-label {{ color: var(--muted); font-size: 12px; }}
    .mode-value {{ margin-top: 2px; font-weight: 680; font-size: 15px; }}
    .status-panel {{
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) repeat(3, minmax(145px, .55fr));
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .status-main {{
      padding: 23px 24px;
      border-right: 1px solid var(--line-soft);
    }}
    .status-kicker {{
      color: {_e(decision["color"])};
      font-size: 12px;
      font-weight: 700;
    }}
    .status-title {{
      margin-top: 4px;
      font-size: 15px;
      line-height: 1.22;
      font-weight: 730;
    }}
    .status-copy {{
      max-width: 660px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .status-fact {{
      padding: 20px 18px;
      border-right: 1px solid var(--line-soft);
    }}
    .status-fact:last-child {{ border-right: 0; }}
    .label {{ color: var(--muted); font-size: 12px; }}
    .value {{ margin-top: 4px; font-size: 15px; font-weight: 700; }}
    .section {{
      margin-top: 18px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .section-head {{
      min-height: 64px;
      padding: 16px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      border-bottom: 1px solid var(--line-soft);
    }}
    .section-title {{ font-size: 17px; font-weight: 720; }}
    .section-sub {{ margin-top: 2px; color: var(--muted); font-size: 12px; }}
    .count {{
      min-width: 38px;
      text-align: right;
      font-size: 19px;
      font-weight: 730;
    }}
    .notice {{
      margin: 16px 20px;
      padding: 13px 14px;
      border-left: 3px solid var(--blue);
      background: #f4f8fd;
      color: #36536f;
      font-size: 13px;
    }}
    .notice.amber {{
      border-left-color: #d6a229;
      background: var(--amber-soft);
      color: #6d5213;
    }}
    .notice.red {{
      border-left-color: var(--red);
      background: var(--red-soft);
      color: #7f2a27;
    }}
    .empty {{
      padding: 34px 20px 38px;
      text-align: center;
      color: var(--muted);
    }}
    .empty strong {{
      display: block;
      color: var(--text);
      font-size: 16px;
      margin-bottom: 5px;
    }}
    .table-wrap {{ overflow-x: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 900px;
    }}
    th {{
      padding: 11px 14px;
      color: var(--muted);
      background: var(--surface-soft);
      border-bottom: 1px solid var(--line-soft);
      text-align: left;
      font-size: 12px;
      font-weight: 650;
      white-space: nowrap;
    }}
    td {{
      padding: 14px;
      border-bottom: 1px solid var(--line-soft);
      vertical-align: middle;
      white-space: nowrap;
    }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    tbody tr:hover {{ background: #fbfbfd; }}
    .stock {{ font-weight: 700; }}
    .stock-name {{ color: var(--muted); font-size: 12px; }}
    .positive {{ color: var(--red); font-weight: 700; }}
    .negative {{ color: var(--green); font-weight: 700; }}
    .neutral {{ color: var(--muted); }}
    .tag {{
      display: inline-block;
      padding: 2px 7px;
      border: 1px solid var(--line);
      border-radius: 5px;
      color: var(--muted);
      background: var(--surface);
      font-size: 11px;
      font-weight: 650;
    }}
    .tag.good {{
      color: var(--green);
      border-color: #a8d8b5;
      background: var(--green-soft);
    }}
    .tag.warn {{
      color: var(--amber);
      border-color: #e7ca82;
      background: var(--amber-soft);
    }}
    .reason {{
      max-width: 320px;
      white-space: normal;
      color: var(--muted);
      font-size: 12px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
    }}
    .metric {{
      padding: 18px 20px;
      border-right: 1px solid var(--line-soft);
    }}
    .metric:last-child {{ border-right: 0; }}
    .metric-value {{
      margin-top: 3px;
      font-size: 21px;
      font-weight: 730;
    }}
    .split {{
      display: grid;
      grid-template-columns: 1fr 1fr;
    }}
    .split > div {{ padding: 20px; }}
    .split > div:first-child {{ border-right: 1px solid var(--line-soft); }}
    .evidence-title {{ font-size: 14px; font-weight: 700; }}
    .evidence-copy {{ margin-top: 7px; color: var(--muted); font-size: 13px; }}
    .evidence-list {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin-top: 14px;
      border-top: 1px solid var(--line-soft);
    }}
    .evidence-item {{ padding: 12px 12px 0 0; }}
    .evidence-item strong {{ display: block; margin-top: 2px; }}
    .evidence-main {{ padding: 20px; }}
    .collapsible-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    .collapse-toggle {{
      width: 30px;
      height: 30px;
      flex: 0 0 30px;
      display: inline-grid;
      place-items: center;
      padding: 0;
      border: 1px solid var(--line);
      border-radius: 50%;
      color: var(--text);
      background: var(--surface);
      cursor: pointer;
      font-size: 20px;
      line-height: 1;
    }}
    .collapse-toggle:hover {{ background: var(--surface-soft); }}
    .collapse-toggle:focus-visible {{
      outline: 2px solid var(--blue);
      outline-offset: 2px;
    }}
    .retrospective-cohort {{
      margin-top: 18px;
      padding-top: 18px;
      border-top: 1px solid var(--line-soft);
    }}
    .retrospective-switch {{
      display: flex;
      justify-content: flex-end;
      margin-top: 14px;
    }}
    .cohort-heading {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
    }}
    .cohort-heading .evidence-copy {{ max-width: 760px; }}
    .compact-table {{
      margin-top: 15px;
      border: 1px solid var(--line-soft);
      border-radius: 6px;
    }}
    .compact-table table {{ min-width: 820px; }}
    .evidence-shadow {{
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(360px, .75fr);
      gap: 28px;
      padding: 20px;
      border-top: 1px solid var(--line-soft);
    }}
    .evidence-shadow .evidence-list {{ margin-top: 0; }}
    .segment {{
      display: inline-flex;
      padding: 2px;
      gap: 2px;
      background: #ececf0;
      border-radius: 7px;
    }}
    .segment button {{
      border: 0;
      border-radius: 5px;
      padding: 6px 10px;
      color: var(--muted);
      background: transparent;
      cursor: pointer;
      font-size: 12px;
    }}
    .segment button.active {{
      color: var(--text);
      background: var(--surface);
      box-shadow: 0 1px 4px rgba(0, 0, 0, .08);
    }}
    .footer {{
      padding: 24px 4px 0;
      color: var(--faint);
      font-size: 12px;
      text-align: center;
    }}
    @media (max-width: 900px) {{
      .status-panel {{ grid-template-columns: 1fr 1fr; }}
      .status-main {{ grid-column: 1 / -1; border-right: 0; border-bottom: 1px solid var(--line-soft); }}
      .status-fact {{ border-bottom: 1px solid var(--line-soft); }}
      .status-fact:nth-child(3) {{ border-right: 0; }}
      .status-fact:last-child {{ grid-column: 1 / -1; border-bottom: 0; }}
      .metric-grid {{ grid-template-columns: repeat(2, 1fr); }}
      .metric {{ border-bottom: 1px solid var(--line-soft); }}
      .metric:nth-child(2n) {{ border-right: 0; }}
      .split {{ grid-template-columns: 1fr; }}
      .split > div:first-child {{ border-right: 0; border-bottom: 1px solid var(--line-soft); }}
      .evidence-shadow {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 640px) {{
      .topbar-inner, .page {{ width: min(100% - 24px, 1240px); }}
      .top-meta span:first-child {{ display: none; }}
      .page {{ padding-top: 20px; }}
      .status-panel {{ grid-template-columns: 1fr; }}
      .status-main, .status-fact {{
        grid-column: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line-soft);
      }}
      .status-fact:last-child {{ grid-column: auto; }}
      .section-head {{ align-items: flex-start; }}
      .metric-grid {{ grid-template-columns: 1fr 1fr; }}
      .metric {{ padding: 15px 14px; }}
      .metric-value {{ font-size: 18px; }}
      .evidence-list {{ grid-template-columns: 1fr; }}
      .segment {{ width: 100%; }}
      .segment button {{ flex: 1; }}
    }}
  </style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand">WP · T+1 尾盘决策</div>
      <div class="top-meta">
        <span>固定决策 14:30</span>
        <span>{_e(_display_datetime(manifest.get("report_revision")))}</span>
      </div>
    </div>
  </header>

  <main class="page">
    <section class="headline">
      <div class="mode">
        <div class="mode-label">当前运行状态</div>
        <div class="mode-value">{_e(_state_label(state))}</div>
      </div>
    </section>

    <section class="status-panel">
      <div class="status-main">
        <div class="status-kicker">{_e(decision["kicker"])}</div>
        <div class="status-title">{_e(decision["title"])}</div>
        {_status_message(decision["message"])}
      </div>
      {_status_fact("合格候选", str(len(qualified)) if live_visible else "已冻结")}
      {_status_fact("研究观察", f"{len(observations)} / {config.strategy.observation_count}" if live_visible else "历史可查")}
      {_status_fact("数据状态", _data_status(manifest))}
    </section>

    {_live_sections(
        qualified=qualified,
        observations=observations,
        phase=phase,
        live_visible=live_visible,
        config=config,
        manifest=manifest,
    )}

    {_closed_day_section(
        current_session,
        show=not live_visible,
    )}

    <section class="section" data-tab-group>
      <div class="section-head">
        <div>
          <h2 class="section-title">T+1 真实验证</h2>
          <p class="section-sub">这里只统计 8 月 3 日起盘中实时形成的影子记录，不记录人工是否买入；5–7 月历史回测见下方</p>
        </div>
        <div class="segment" role="tablist" aria-label="验证组别">
          <button class="active" type="button" role="tab" aria-selected="true" data-tab="QUALIFIED">合格</button>
          <button type="button" role="tab" aria-selected="false" data-tab="OBSERVATION">观察</button>
        </div>
      </div>
      <div role="tabpanel" data-tab-panel="QUALIFIED">
        {_metric_strip(qualified_stats)}
        {_validation_table(records, "QUALIFIED")}
      </div>
      <div role="tabpanel" data-tab-panel="OBSERVATION" hidden>
        {_metric_strip(observation_stats)}
        <div class="notice amber">研究观察用于检验门槛附近股票的真实结果，不与正式策略胜率、收益或晋级门槛合并。</div>
        {_validation_table(records, "OBSERVATION")}
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2 class="section-title">证据与上线边界</h2>
          <p class="section-sub">{_e(evidence_contract)}</p>
        </div>
      </div>
      <div class="evidence-main">
        <div class="collapsible-head">
          <h3 class="evidence-title">2026 年 5–7 月新合同回测</h3>
          <button class="collapse-toggle" type="button"
                  data-collapse-toggle
                  aria-controls="retrospective-backtest-content"
                  aria-expanded="false"
                  aria-label="展开 2026 年 5–7 月新合同回测"
                  title="展开回测详情">
            <span aria-hidden="true" data-collapse-icon>+</span>
          </button>
        </div>
        <div id="retrospective-backtest-content" data-collapse-panel hidden>
          {_retrospective_evidence(retrospective, config)}
        </div>
      </div>
      <div class="evidence-shadow">
        <div>
          <h3 class="evidence-title">2026 年 8 月起真实影子运行</h3>
          <p class="evidence-copy">只累计盘中实时形成、绑定模型指纹并完成真值验证的记录。历史回测和旧系统回补均不计入 150 个交易日。</p>
        </div>
        <div class="evidence-list">
          {_evidence_item("已运行交易日", _number(qualified_shadow_stats["trading_days"]))}
          {_evidence_item("合格候选日", _number(qualified_shadow_stats["candidate_days"]))}
          {_evidence_item("已验证候选", _number(qualified_shadow_stats["verified"]))}
        </div>
      </div>
      {_research_seed_note(research_seed)}
    </section>

    {_system_contract(config, manifest, registry, replay)}
    {_legacy_note(legacy_audit)}

    <p class="footer">本页展示模型候选与影子真值，不代表用户实际成交，也不承诺收益。</p>
  </main>
  <script>
    document.querySelectorAll('[data-tab-group]').forEach(function (group) {{
      var buttons = group.querySelectorAll('[data-tab]');
      var panels = group.querySelectorAll('[data-tab-panel]');
      buttons.forEach(function (button) {{
        button.addEventListener('click', function () {{
          var target = button.getAttribute('data-tab');
          buttons.forEach(function (item) {{
            var selected = item === button;
            item.classList.toggle('active', selected);
            item.setAttribute('aria-selected', selected ? 'true' : 'false');
          }});
          panels.forEach(function (panel) {{
            panel.hidden = panel.getAttribute('data-tab-panel') !== target;
          }});
        }});
      }});
    }});
    document.querySelectorAll('[data-collapse-toggle]').forEach(function (button) {{
      var panel = document.getElementById(button.getAttribute('aria-controls'));
      if (!panel) return;
      button.addEventListener('click', function () {{
        var expanded = button.getAttribute('aria-expanded') !== 'true';
        button.setAttribute('aria-expanded', expanded ? 'true' : 'false');
        button.setAttribute(
          'aria-label',
          (expanded ? '收起 ' : '展开 ') + '2026 年 5–7 月新合同回测'
        );
        button.setAttribute('title', expanded ? '收起回测详情' : '展开回测详情');
        panel.hidden = !expanded;
        var icon = button.querySelector('[data-collapse-icon]');
        if (icon) icon.textContent = expanded ? '−' : '+';
      }});
    }});
  </script>
</body>
</html>
"""
    target.write_text(html, encoding="utf-8")


def _live_sections(
    *,
    qualified: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    phase: str,
    live_visible: bool,
    config: V3Config,
    manifest: dict[str, Any],
) -> str:
    if not live_visible:
        return ""
    qualified_body = (
        _cohort_table(qualified, "QUALIFIED")
        if qualified
        else (
            '<div class="empty"><strong>当前没有合格候选</strong>'
            "零支是正常结果，系统不会为了产生名单而降低固定门槛。</div>"
        )
    )
    observation_status = str(
        manifest.get("observation_selection_status") or ""
    )
    if observations:
        observation_body = _cohort_table(observations, "OBSERVATION")
    else:
        observation_body = (
            '<div class="empty"><strong>研究观察暂不可用</strong>'
            "模型分数或可成交股票池不足，系统没有伪造 5 支股票。</div>"
        )
    observation_notice_class = (
        "notice amber"
        if len(observations) == config.strategy.observation_count
        else "notice red"
    )
    observation_notice = (
        f"固定展示 {config.strategy.observation_count} 支最接近门槛、"
        "但未全部通过的可成交股票。仅用于研究比较，不属于合格候选。"
        if len(observations) == config.strategy.observation_count
        else (
            f"观察组完整性异常：期望 {config.strategy.observation_count} 支，"
            f"实际 {len(observations)} 支；状态 {observation_status or '未知'}。"
        )
    )
    return f"""
    <section class="section">
      <div class="section-head">
        <div>
          <h2 class="section-title">合格候选</h2>
          <p class="section-sub">全部固定门槛均通过；数量可以为 0，不设人为配额</p>
        </div>
        <div class="count">{len(qualified)}</div>
      </div>
      {qualified_body}
    </section>
    <section class="section">
      <div class="section-head">
        <div>
          <h2 class="section-title">研究观察</h2>
          <p class="section-sub">最接近门槛的非合格股票，独立做影子真值</p>
        </div>
        <div class="count">{len(observations)} / {config.strategy.observation_count}</div>
      </div>
      <div class="{observation_notice_class}">{_e(observation_notice)}</div>
      {observation_body}
    </section>
    """


def _closed_day_section(
    session: dict[str, Any],
    *,
    show: bool,
) -> str:
    if not show or not session:
        return ""
    rows = session_records(session)
    if not rows:
        body = (
            '<div class="empty"><strong>当日无合格信号</strong>'
            "若观察组也为空，表示当日模型或数据完整性不足。</div>"
        )
    else:
        body = _cohort_table(rows, "HISTORY")
    return f"""
    <section class="section">
      <div class="section-head">
        <div>
          <h2 class="section-title">今日冻结证据</h2>
          <p class="section-sub">已收盘，不再显示可买名单；以下仅供审计与 T+1 验证</p>
        </div>
        <div class="count">{len(rows)}</div>
      </div>
      {body}
    </section>
    """


def _cohort_table(
    records: list[dict[str, Any]],
    cohort: str,
) -> str:
    if not records:
        return ""
    rows = []
    for row in records:
        record_cohort = str(row.get("candidate_cohort") or "QUALIFIED")
        status = (
            '<span class="tag good">合格</span>'
            if record_cohort == "QUALIFIED"
            else '<span class="tag warn">研究观察</span>'
        )
        reason = (
            "全部固定门槛通过"
            if record_cohort == "QUALIFIED"
            else str(
                row.get("failed_gate_labels")
                or row.get("rejection_reasons")
                or "未通过全部固定门槛"
            )
        )
        rows.append(
            "<tr>"
            f"<td>{status}</td>"
            f"<td><div class='stock'>{_e(row.get('ts_code'))}</div>"
            f"<div class='stock-name'>{_e(row.get('name'))}</div></td>"
            f"<td>{_e(row.get('first_signal_time') or '14:30')}</td>"
            f"<td>{_price(row.get('first_signal_price'))}</td>"
            f"<td>{_price(row.get('entry_price'))}</td>"
            f"<td>{_pct(row.get('p_net_positive_lower'))}</td>"
            f"<td>{_return_pct(row.get('expected_utility_lower_pct'))}</td>"
            f"<td>{_return_pct(row.get('downside_q10_pct'))}</td>"
            f"<td class='reason'>{_e(reason)}</td>"
            "</tr>"
        )
    heading = "组别" if cohort == "HISTORY" else "状态"
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"<th>{heading}</th><th>股票</th><th>信号</th><th>信号价</th>"
        "<th>14:35 基准价</th><th>盈利概率下界</th>"
        "<th>净收益下界</th><th>下行 10% 分位</th><th>判定依据</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _validation_table(
    records: list[dict[str, Any]],
    cohort: str,
) -> str:
    selected = [
        row
        for row in records
        if str(row.get("candidate_cohort") or "QUALIFIED") == cohort
    ]
    selected.sort(
        key=lambda row: (
            _compact_date(row.get("trade_date")),
            str(row.get("ts_code") or ""),
        ),
        reverse=True,
    )
    if not selected:
        return (
            '<div class="empty"><strong>暂无验证记录</strong>'
            "新合同尚未形成该组的可验证样本。</div>"
        )
    rows = []
    for row in selected[:80]:
        verified = str(row.get("truth_status") or "") == "verified"
        status = (
            '<span class="tag good">已验证</span>'
            if verified
            else '<span class="tag">待 T+1 收盘</span>'
        )
        outcome = (
            "盈利"
            if row.get("net_positive") is True
            else ("未盈利" if verified else "待验证")
        )
        rows.append(
            "<tr>"
            f"<td>{_e(_display_date(row.get('trade_date')))}</td>"
            f"<td>{_e(_display_date(row.get('target_trade_date')))}</td>"
            f"<td><div class='stock'>{_e(row.get('ts_code'))}</div>"
            f"<div class='stock-name'>{_e(row.get('name'))}</div></td>"
            f"<td>{_price(row.get('entry_price'))}</td>"
            f"<td>{_price(row.get('t1_close'))}</td>"
            f"<td>{_return_pct(row.get('net_return_pct'))}</td>"
            f"<td>{_e(outcome)}</td><td>{status}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>信号日</th><th>验证日</th><th>股票</th><th>入场价</th>"
        "<th>T+1 收盘</th><th>净收益</th><th>结果</th><th>状态</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _metric_strip(stats: dict[str, Any]) -> str:
    return (
        '<div class="metric-grid">'
        + _metric("样本", _number(stats["records"]))
        + _metric("已验证", _number(stats["verified"]))
        + _metric("覆盖交易日", _number(stats["trading_days"]))
        + _metric("胜率", _ratio(stats["win_rate"]))
        + _metric(
            "平均净收益",
            _signed_pct(stats["mean_net_return_pct"]),
            _return_class(stats["mean_net_return_pct"]),
        )
        + "</div>"
    )


def _cohort_stats(
    records: list[dict[str, Any]],
    cohort: str,
) -> dict[str, Any]:
    selected = [
        row
        for row in records
        if str(row.get("candidate_cohort") or "QUALIFIED") == cohort
    ]
    verified = [
        row
        for row in selected
        if str(row.get("truth_status") or "") == "verified"
    ]
    returns = pd.to_numeric(
        pd.Series(
            [row.get("net_return_pct") for row in verified],
            dtype="object",
        ),
        errors="coerce",
    ).dropna()
    wins = sum(row.get("net_positive") is True for row in verified)
    return {
        "records": len(selected),
        "verified": len(verified),
        "trading_days": len(
            {
                _compact_date(row.get("trade_date"))
                for row in selected
                if _compact_date(row.get("trade_date"))
            }
        ),
        "candidate_days": len(
            {
                _compact_date(row.get("trade_date"))
                for row in selected
                if _compact_date(row.get("trade_date"))
            }
        ),
        "win_rate": wins / len(verified) if verified else None,
        "mean_net_return_pct": (
            float(returns.mean()) if not returns.empty else None
        ),
    }


def _retrospective_evidence(
    retrospective: dict[str, Any],
    config: V3Config,
) -> str:
    summary = retrospective.get("summary") or retrospective
    status = str(summary.get("status") or "").upper()
    if not summary or status in {"", "PENDING", "NOT_STARTED"}:
        return (
            '<p class="evidence-copy">新合同回测尚未形成可确认结果。'
            "页面不会把 V15 多时点结果冒充为固定 14:30 结果。</p>"
            '<div class="notice amber">状态：等待严格点时回放与真值计算</div>'
        )
    qualified_metrics = (
        summary.get("qualified", {}).get("metrics", {})
        or summary.get("metrics", {})
        or summary
    )
    observation_metrics = (
        summary.get("observations", {}).get("metrics", {}) or {}
    )
    monthly = summary.get("monthly", [])
    covered_days = (
        summary.get("source", {}).get("expected_trade_days")
        or summary.get("integrity", {}).get("evaluated_trade_days")
        or sum(
            int(_float(row.get("evaluated_trade_days")) or 0)
            for row in monthly
        )
    )
    qualified_events = (
        qualified_metrics.get("events")
        if qualified_metrics.get("events") is not None
        else summary.get("qualified_events")
    )
    gate = summary.get("backtest_gate", {})
    failed = list(gate.get("failed_gates", []))
    if status == "INCOMPLETE":
        notice = (
            '<div class="notice amber">回测选择已冻结，但仍有 T+1 真值待闭合。'
            "待真值只补收益，不重选股票、不重训模型。</div>"
        )
    elif gate.get("passed"):
        notice = (
            '<div class="notice amber">历史门槛已通过，但仍不是盈利保证。'
            "必须继续完成 150 个交易日的真实影子验证。</div>"
        )
    else:
        notice = (
            '<div class="notice red">历史生产门槛未全部通过'
            f"（{len(failed)} 项失败），不得授权正式交易。</div>"
        )
    zero_qualified_note = ""
    if (_float(qualified_events) or 0.0) == 0.0 and covered_days:
        zero_qualified_note = (
            '<div class="notice">合格样本为 0 不是数据缺失：'
            f"{_number(covered_days)} 个覆盖交易日内，没有股票同时通过全部固定门槛。"
            "研究观察有独立样本和收益统计，见下方。</div>"
        )
    return (
        '<p class="evidence-copy">只使用当日 14:30 前可见信息，'
        "按统一 14:35 入场和 T+1 收盘合同重放。</p>"
        + '<div class="retrospective-tabs" data-tab-group>'
        '<div class="retrospective-switch">'
        '<div class="segment" role="tablist" aria-label="回测组别">'
        '<button type="button" role="tab" aria-selected="false" '
        'data-tab="QUALIFIED">合格</button>'
        '<button class="active" type="button" role="tab" aria-selected="true" '
        'data-tab="OBSERVATION">观察</button>'
        "</div></div>"
        '<div role="tabpanel" data-tab-panel="QUALIFIED" hidden>'
        + zero_qualified_note
        + _retrospective_cohort(
            title="合格",
            badge="决策组",
            description=(
                "必须同时通过盈利概率、成交、风险和稳定性固定门槛；"
                "数量可以为 0。"
            ),
            metrics=qualified_metrics,
            monthly=monthly,
            cohort_key="qualified",
            covered_days=covered_days,
        )
        + "</div>"
        '<div role="tabpanel" data-tab-panel="OBSERVATION">'
        + _retrospective_cohort(
            title="观察",
            badge="比较组",
            description=(
                f"每个覆盖交易日固定选择 {config.strategy.observation_count} 支"
                "最接近门槛的非合格股票；用于研究，不是买入建议。"
            ),
            metrics=observation_metrics,
            monthly=monthly,
            cohort_key="observations",
            covered_days=covered_days,
        )
        + "</div></div>"
        + notice
    )


def _retrospective_cohort(
    *,
    title: str,
    badge: str,
    description: str,
    metrics: dict[str, Any],
    monthly: list[dict[str, Any]],
    cohort_key: str,
    covered_days: Any,
) -> str:
    events = metrics.get("events")
    trade_days = metrics.get("trade_days")
    has_events = (_float(events) or 0.0) > 0.0
    win_rate = metrics.get("win_rate") if has_events else None
    mean_return = (
        metrics.get("mean_net_return_pct") if has_events else None
    )
    profit_factor = metrics.get("profit_factor") if has_events else None
    stress_50 = (
        metrics.get("stress", {})
        .get("50bps", {})
        .get("mean_net_return_pct")
        if has_events
        else None
    )
    day_coverage = (
        f"{_number(trade_days)} / {_number(covered_days)}"
        if covered_days is not None
        else _number(trade_days)
    )
    badge_css = "good" if cohort_key == "qualified" else "warn"
    return (
        '<div class="retrospective-cohort">'
        '<div class="cohort-heading"><div>'
        f'<h3 class="evidence-title">{_e(title)}</h3>'
        f'<p class="evidence-copy">{_e(description)}</p>'
        f'</div><span class="tag {badge_css}">{_e(badge)}</span></div>'
        '<div class="evidence-list">'
        + _evidence_item("样本", _number(events))
        + _evidence_item("统计日 / 覆盖日", day_coverage)
        + _evidence_item("胜率", _ratio(win_rate))
        + _evidence_item(
            "平均净收益",
            _signed_pct(mean_return),
            _return_class(mean_return),
        )
        + _evidence_item("Profit Factor", _number(profit_factor))
        + _evidence_item(
            "50bp 压力收益",
            _signed_pct(stress_50),
            _return_class(stress_50),
        )
        + "</div>"
        + _retrospective_monthly_table(monthly, cohort_key)
        + "</div>"
    )


def _retrospective_monthly_table(
    monthly: list[dict[str, Any]],
    cohort_key: str,
) -> str:
    rows = []
    for row in monthly:
        metrics = row.get(cohort_key, {}) or {}
        events = metrics.get("events")
        has_events = (_float(events) or 0.0) > 0.0
        covered_days = row.get("evaluated_trade_days")
        trade_days = metrics.get("trade_days")
        stress_50 = (
            metrics.get("stress", {})
            .get("50bps", {})
            .get("mean_net_return_pct")
            if has_events
            else None
        )
        day_coverage = (
            f"{_number(trade_days)} / {_number(covered_days)}"
            if covered_days is not None
            else _number(trade_days)
        )
        rows.append(
            "<tr>"
            f"<td>{_e(_month_label(row.get('month')))}</td>"
            f"<td>{_number(events)}</td>"
            f"<td>{day_coverage}</td>"
            f"<td>{_ratio(metrics.get('win_rate') if has_events else None)}</td>"
            "<td>"
            f"{_signed_pct(metrics.get('mean_net_return_pct') if has_events else None)}"
            "</td>"
            f"<td>{_number(metrics.get('profit_factor') if has_events else None)}</td>"
            f"<td>{_signed_pct(stress_50)}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        '<div class="table-wrap compact-table"><table><thead><tr>'
        "<th>月份</th><th>样本</th><th>统计日 / 覆盖日</th><th>胜率</th>"
        "<th>平均净收益</th><th>Profit Factor</th><th>50bp 压力收益</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _research_seed_note(seed: dict[str, Any]) -> str:
    snapshot = seed.get("evidence_snapshot") or {}
    challenger = snapshot.get("challenger") or {}
    if not challenger:
        return ""
    return (
        '<div class="notice amber"><strong>V15 仅作为研究种子：</strong> '
        f"{_number(challenger.get('events'))} 支 / "
        f"{_number(challenger.get('trade_days'))} 日，"
        f"平均净收益 {_signed_pct(challenger.get('mean_net_return_pct'))}。"
        "V15 使用多个早段时点和每日 Top3，与当前固定 14:30、合格不限数量的合同不同，"
        "不能直接当作新系统盈利证据。</div>"
    )


def _system_contract(
    config: V3Config,
    manifest: dict[str, Any],
    registry: dict[str, Any],
    replay: dict[str, Any],
) -> str:
    return f"""
    <section class="section">
      <div class="section-head">
        <div>
          <h2 class="section-title">系统合同</h2>
          <p class="section-sub">所有时间、价格和统计口径在信号形成前固定</p>
        </div>
      </div>
      <div class="split">
        <div>
          <h3 class="evidence-title">执行合同</h3>
          <p class="evidence-copy">14:30 形成候选；14:35 五分钟收盘价加 {_number(config.execution.entry_slippage_bps)}bp 作为影子入场价；T+1 参与收盘集合竞价卖出；往返成本 {_number(config.execution.round_trip_cost_bps)}bp。</p>
        </div>
        <div>
          <h3 class="evidence-title">统计合同</h3>
          <p class="evidence-copy">合格候选可以为 0，全部通过者均展示。研究观察固定 {config.strategy.observation_count} 支，只做比较研究。两组共享真值合同但不合并统计，用户实盘选择不进入系统。</p>
        </div>
      </div>
    </section>
    """


def _legacy_note(legacy_audit: dict[str, Any] | None) -> str:
    if not legacy_audit:
        return ""
    return (
        '<section class="section"><div class="section-head"><div>'
        '<h2 class="section-title">旧系统历史说明</h2>'
        '<p class="section-sub">旧版盘中记录保留审计，但不计入固定 14:30 新合同</p>'
        "</div></div>"
        '<div class="notice amber">旧系统回补、盘后生成名单以及不同入场合同的记录，'
        "均不计入 2026 年 8 月起的真实影子统计。</div></section>"
    )


def _current_cohort(
    session: dict[str, Any],
    selection: CohortSelection,
    *,
    cohort: str,
) -> list[dict[str, Any]]:
    key = "candidates" if cohort == "QUALIFIED" else "observations"
    locked = list(session.get(key, [])) if session else []
    if locked:
        return locked
    frame = (
        selection.qualified
        if cohort == "QUALIFIED"
        else selection.observations
    )
    return frame.to_dict(orient="records")


def _session_for_date(
    ledger: dict[str, Any],
    trade_date: str,
) -> dict[str, Any]:
    return next(
        (
            session
            for session in ledger.get("sessions", [])
            if _compact_date(session.get("trade_date")) == trade_date
        ),
        {},
    )


def _decision_copy(
    *,
    phase: str,
    state: str,
    qualified_count: int,
    live_visible: bool,
    health: str,
) -> dict[str, str]:
    if health not in HEALTHY_MANIFEST_STATUSES:
        return {
            "kicker": "数据或账本异常",
            "title": "暂不使用本次名单",
            "message": "完整性检查未通过。系统不会用缺失数据补足候选或研究观察。",
            "color": "#c9342f",
        }
    if state in {
        "MODEL_NOT_READY",
        "MODEL_ARTIFACT_INVALID",
        "POLICY_MISMATCH",
        "MODEL_NOT_DESIGNATED",
    }:
        return {
            "kicker": "模型尚未就绪",
            "title": "今天不授权候选",
            "message": "固定 14:30 新合同仍在建立可部署模型和回测证据，当前不输出伪候选。",
            "color": "#9a6700",
        }
    if phase == "PRE_SIGNAL":
        return {
            "kicker": "等待决策",
            "title": "14:30 生成一次名单",
            "message": "14:00–14:25 只积累因果快照，不提前泄露或反复改写候选。",
            "color": "#0071e3",
        }
    if not live_visible or phase == "CLOSED":
        return {
            "kicker": "今日决策已结束",
            "title": "已收盘，不再显示可买名单",
            "message": "",
            "color": "#6e6e73",
        }
    if qualified_count:
        if state == "SHADOW_OBSERVATION":
            return {
                "kicker": "研究筛选通过",
                "title": f"{qualified_count} 支影子合格候选",
                "message": (
                    "这些股票通过固定筛选门槛，但新合同尚未通过完整历史与"
                    "未来影子证据；仅供人工研究判断。"
                ),
                "color": "#9a6700",
            }
        return {
            "kicker": "固定门槛已通过",
            "title": f"{qualified_count} 支合格候选",
            "message": "系统展示全部合格股票，不替用户决定买哪一支，也不记录用户是否成交。",
            "color": "#16823b",
        }
    return {
        "kicker": "NO SIGNAL",
        "title": "当前无合格候选",
        "message": "没有股票同时通过盈利概率、成交、风险、稳定性与数据新鲜度门槛。",
        "color": "#6e6e73",
    }


def _state_label(state: str) -> str:
    labels = {
        "PRODUCTION": "生产授权",
        "SHADOW": "真实影子验证",
        "SHADOW_OBSERVATION": "影子观察",
        "MODEL_NOT_READY": "等待模型包",
        "MODEL_ARTIFACT_INVALID": "模型包异常",
        "POLICY_MISMATCH": "模型合同不匹配",
        "MODEL_NOT_DESIGNATED": "模型未指定",
    }
    return labels.get(state, state or "未知")


def _data_status(manifest: dict[str, Any]) -> str:
    health = str(manifest.get("health_status") or "未知")
    coverage = _float(manifest.get("tail_universe_coverage"))
    age = _float(manifest.get("market_data_p95_age_seconds"))
    health_label = {
        "ok": "正常",
        "research_ready": "研究回测就绪",
        "无符合条件股票": "正常",
    }.get(health, health)
    parts = [health_label]
    if coverage is not None:
        parts.append(f"覆盖 {coverage:.1%}")
    if age is not None:
        parts.append(f"P95 {age:.0f}秒")
    return " · ".join(parts)


def _status_fact(label: str, value: str) -> str:
    return (
        '<div class="status-fact">'
        f'<div class="label">{_e(label)}</div>'
        f'<div class="value">{_e(value)}</div>'
        "</div>"
    )


def _status_message(message: Any) -> str:
    text = str(message or "").strip()
    return f'<p class="status-copy">{_e(text)}</p>' if text else ""


def _metric(label: str, value: str, css: str = "") -> str:
    return (
        '<div class="metric">'
        f'<div class="label">{_e(label)}</div>'
        f'<div class="metric-value {css}">{_e(value)}</div>'
        "</div>"
    )


def _evidence_item(label: str, value: str, css: str = "") -> str:
    return (
        '<div class="evidence-item">'
        f'<span class="label">{_e(label)}</span>'
        f'<strong class="{css}">{_e(value)}</strong>'
        "</div>"
    )


def _compact_date(value: Any) -> str:
    text = str(value or "").strip().replace("-", "")
    return text[:8] if len(text) >= 8 and text[:8].isdigit() else ""


def _display_date(value: Any) -> str:
    text = _compact_date(value)
    return f"{text[:4]}-{text[4:6]}-{text[6:]}" if text else "—"


def _display_datetime(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "等待首次更新"


def _month_label(value: Any) -> str:
    text = str(value or "").replace("-", "")
    return (
        f"{text[:4]}-{text[4:6]}"
        if len(text) >= 6 and text[:6].isdigit()
        else "—"
    )


def _price(value: Any) -> str:
    parsed = _float(value)
    return f"{parsed:.2f}" if parsed is not None else "—"


def _pct(value: Any) -> str:
    parsed = _float(value)
    return f"{parsed * 100:.1f}%" if parsed is not None else "—"


def _return_pct(value: Any) -> str:
    parsed = _float(value)
    if parsed is None:
        return "—"
    css = _return_class(parsed)
    return f'<span class="{css}">{parsed:+.2f}%</span>'


def _signed_pct(value: Any) -> str:
    parsed = _float(value)
    return f"{parsed:+.3f}%" if parsed is not None else "—"


def _ratio(value: Any) -> str:
    parsed = _float(value)
    return f"{parsed:.1%}" if parsed is not None else "—"


def _return_class(value: Any) -> str:
    parsed = _float(value)
    if parsed is None:
        return "neutral"
    if parsed > 0:
        return "positive"
    if parsed < 0:
        return "negative"
    return "neutral"


def _number(value: Any) -> str:
    parsed = _float(value)
    if parsed is None:
        return "—"
    return f"{int(parsed):,}" if parsed.is_integer() else f"{parsed:,.2f}"


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if pd.notna(parsed) else None
    except (TypeError, ValueError):
        return None


def _e(value: Any) -> str:
    return escape(str(value if value is not None else ""))
