from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


def _fmt(value, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return html.escape(str(value))


def render_html(top50: pd.DataFrame, full_rank: pd.DataFrame, health: dict, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sector_top = []
    if not full_rank.empty and "sector_name" in full_rank:
        sector_top = full_rank.groupby("sector_name").size().sort_values(ascending=False).head(10).items()
    rows = []
    for _, row in top50.iterrows():
        risk_cls = " risk-high" if float(row.get("risk_penalty_score", 0) or 0) >= 65 else ""
        top_cls = " top10" if int(row.get("rank", 999)) <= 10 else ""
        rows.append(
            "<tr class=\"{}{}\">".format(top_cls.strip(), risk_cls)
            + f"<td>{row['rank']}</td><td>{html.escape(str(row['ts_code']))}</td><td>{html.escape(str(row['name']))}</td>"
            + f"<td>{_fmt(row['price'])}</td><td>{_fmt(row['pct_chg'])}%</td><td>{html.escape(str(row['sector_name']))}</td>"
            + f"<td>{_fmt(row['sector_strength_score'])}</td><td>{_fmt(row['stock_strength_score'])}</td><td>{_fmt(row['acceptance_score'])}</td>"
            + f"<td>{_fmt(row['p_limitup_t1'])}%</td><td>{_fmt(row['wp_score'])}</td><td>{_fmt(row['model_confidence'])}</td>"
            + f"<td>{html.escape(str(row['signal_level']))}</td><td>{html.escape(str(row['core_reason']))}</td><td>{html.escape(str(row['risk_reason']))}</td><td>{html.escape(str(row['update_time']))}</td></tr>"
        )
    if not rows:
        rows.append("<tr><td colspan=\"16\" class=\"empty\">无符合条件股票</td></tr>")
    status_cls = "bad" if health.get("status") not in {"ok", "无符合条件股票"} else "ok"
    data_trade_date = html.escape(str(health.get("data_trade_date") or "-"))
    expected_trade_date = html.escape(str(health.get("expected_trade_date") or "-"))
    realtime_sources = health.get("realtime_sources") or []
    realtime_source_text = ", ".join(str(item) for item in realtime_sources) if realtime_sources else "未标记"
    status_rows = [
        ("运行状态", f"<span class=\"status {status_cls}\">{html.escape(str(health.get('status')))}</span>"),
        ("数据更新时间", html.escape(str(health.get("data_time")))),
        ("行情日期", data_trade_date),
        ("期望交易日", expected_trade_date),
        ("候选池数量", str(health.get("candidate_count", 0))),
        ("入选 Top50 数量", str(health.get("top50_count", 0))),
        ("原始数据量", str(health.get("raw_count", 0))),
        ("缺失字段", html.escape(", ".join(health.get("missing_fields", [])) or "无")),
        ("读取缓存 fallback", html.escape(str(health.get("data_load_fallback_used", health.get("fallback_used"))))),
        ("实时行情来源", html.escape(realtime_source_text)),
        ("实时 fallback", html.escape(str(health.get("realtime_fallback_used", False)))),
    ]
    status_html = "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in status_rows)
    sector_rows = "".join(
        f"<tr><td>{idx}</td><td>{html.escape(str(name))}</td><td>{count} 只</td></tr>"
        for idx, (name, count) in enumerate(sector_top, 1)
    )
    if not sector_rows:
        sector_rows = "<tr><td colspan=\"3\" class=\"empty\">暂无板块数据</td></tr>"
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WP 次日涨停概率 Top50</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif; color: #1d1d1f; background: #f5f5f7; letter-spacing: 0; }}
    header {{ padding: 30px 32px 22px; background: rgba(255, 255, 255, 0.92); border-bottom: 1px solid #d2d2d7; }}
    h1 {{ margin: 0 0 10px; font-size: 30px; line-height: 1.18; font-weight: 700; }}
    .meta {{ display: flex; gap: 10px 18px; flex-wrap: wrap; color: #6e6e73; font-size: 14px; line-height: 1.45; }}
    main {{ padding: 24px 32px 40px; }}
    section {{ margin-bottom: 22px; }}
    .summary-section {{ overflow: hidden; background: #fff; border: 1px solid #d2d2d7; border-radius: 8px; }}
    .summary-grid {{ display: grid; grid-template-columns: minmax(360px, 1fr) minmax(320px, 0.84fr); }}
    .summary-pane {{ padding: 22px 24px; }}
    .summary-pane + .summary-pane {{ border-left: 1px solid #d2d2d7; }}
    .summary-title {{ margin: 0 0 12px; font-size: 18px; line-height: 1.25; font-weight: 700; }}
    .summary-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    .summary-table th, .summary-table td {{ padding: 10px 0; border-bottom: 1px solid #f1f1f3; vertical-align: middle; white-space: nowrap; }}
    .summary-table tr:last-child th, .summary-table tr:last-child td {{ border-bottom: 0; }}
    .summary-table th {{ width: 150px; color: #6e6e73; font-weight: 500; text-align: left; }}
    .summary-table td {{ color: #1d1d1f; font-weight: 600; text-align: left; }}
    .sector-table th, .sector-table td {{ text-align: left; }}
    .sector-table th:nth-child(1), .sector-table td:nth-child(1) {{ width: 44px; color: #86868b; font-variant-numeric: tabular-nums; }}
    .sector-table th:nth-child(3), .sector-table td:nth-child(3) {{ text-align: left; font-weight: 600; }}
    .status.ok {{ color: #0b7a3b; font-weight: 700; }}
    .status.bad {{ color: #b42318; font-weight: 700; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid #d2d2d7; background: white; border-radius: 8px; }}
    .rank-table {{ border-collapse: collapse; min-width: 1680px; width: 100%; font-size: 13px; }}
    .rank-table th, .rank-table td {{ padding: 10px 11px; border-bottom: 1px solid #f1f1f3; text-align: left; white-space: nowrap; }}
    .rank-table th {{ background: #fbfbfd; color: #6e6e73; position: sticky; top: 0; font-weight: 600; }}
    .rank-table tbody tr:hover td {{ background: #f5f5f7; }}
    .rank-table tr.top10 td {{ background: #fff8dc; }}
    .rank-table tr.risk-high td {{ color: #9f1f1f; }}
    .section-block {{ background: #fff; border: 1px solid #d2d2d7; border-radius: 8px; padding: 18px 20px; color: #424245; line-height: 1.65; }}
    .section-block strong {{ display: block; color: #1d1d1f; margin-bottom: 6px; }}
    .section-block p {{ margin: 0 0 14px; }}
    .section-block p:last-child {{ margin-bottom: 0; }}
    .empty {{ text-align: center; color: #86868b; padding: 24px; }}
    @media (max-width: 820px) {{
      header {{ padding: 24px 18px 18px; }}
      h1 {{ font-size: 25px; }}
      main {{ padding: 18px 14px 30px; }}
      .summary-grid {{ grid-template-columns: 1fr; }}
      .summary-pane {{ padding: 18px 16px; }}
      .summary-pane + .summary-pane {{ border-left: 0; border-top: 1px solid #d2d2d7; }}
      .summary-table {{ font-size: 13px; }}
      .summary-table th {{ width: 128px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>WP 次日涨停概率 Top50</h1>
    <div class="meta">
      <span>筛选条件：今日涨幅超过 6%，前一日未涨停，今日未涨停。</span>
      <span>目标：预测下一交易日涨停概率。</span>
      <span>刷新频率：每 10 分钟。</span>
    </div>
  </header>
  <main>
    <section class="summary-section" aria-label="运行状态与板块热度">
      <div class="summary-grid">
        <div class="summary-pane">
          <h2 class="summary-title">今日运行状态</h2>
          <table class="summary-table">
            <tbody>{status_html}</tbody>
          </table>
        </div>
        <div class="summary-pane">
          <h2 class="summary-title">板块热度 Top10</h2>
          <table class="summary-table sector-table">
            <thead><tr><th>序号</th><th>板块</th><th>数量</th></tr></thead>
            <tbody>{sector_rows}</tbody>
          </table>
        </div>
      </div>
    </section>
    <section>
      <div class="table-wrap">
        <table class="rank-table">
          <thead><tr><th>排名</th><th>代码</th><th>名称</th><th>当前价/收盘价</th><th>今日涨幅</th><th>所属板块</th><th>板块强度</th><th>个股强度</th><th>承接分</th><th>次日涨停概率</th><th>WP评分</th><th>模型置信度</th><th>信号等级</th><th>核心理由</th><th>风险提示</th><th>更新时间</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>
    <section class="section-block">
      <strong>模型解释</strong>
      <p>第一阶段采用规则评分模型，综合强资金板块、强个股、承接质量、动量、资金和形态，并扣除高位加速、冲高回落、爆量滞涨、板块后排和流动性不足等风险。</p>
      <strong>风险提示</strong>
      <p>S级不等于一定涨停；本系统仅用于辅助决策，不构成投资建议，不做自动交易。</p>
    </section>
  </main>
</body>
</html>
"""
    output.write_text(page, encoding="utf-8")
