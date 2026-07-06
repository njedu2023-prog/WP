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
    sector_html = "".join(f"<li>{html.escape(str(k))}: {v} 只</li>" for k, v in sector_top) or "<li>暂无板块数据</li>"
    status_cls = "bad" if health.get("status") not in {"ok", "无符合条件股票"} else "ok"
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WP 次日涨停概率 Top50</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #f5f7fb; }}
    header {{ padding: 18px 24px; background: #16213e; color: white; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .meta {{ display: flex; gap: 16px; flex-wrap: wrap; font-size: 14px; }}
    main {{ padding: 18px 24px 32px; }}
    section {{ margin-bottom: 18px; }}
    .panel {{ background: white; border: 1px solid #d8deea; border-radius: 8px; padding: 14px 16px; }}
    .status.ok {{ color: #146c43; font-weight: 700; }}
    .status.bad {{ color: #b42318; font-weight: 700; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid #d8deea; background: white; border-radius: 8px; }}
    table {{ border-collapse: collapse; min-width: 1680px; width: 100%; font-size: 13px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #edf0f5; text-align: left; white-space: nowrap; }}
    th {{ background: #eef3fb; color: #22314f; position: sticky; top: 0; }}
    tr.top10 td {{ background: #fff8df; }}
    tr.risk-high td {{ color: #9f1f1f; }}
    .empty {{ text-align: center; color: #6b7280; padding: 24px; }}
    ul {{ margin: 8px 0 0 18px; padding: 0; }}
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
    <section class="panel">
      <strong>今日运行状态：</strong><span class="status {status_cls}">{html.escape(str(health.get("status")))}</span>
      <div>数据更新时间：{html.escape(str(health.get("data_time")))}</div>
      <div>候选池数量：{health.get("candidate_count", 0)}；入选 Top50 数量：{health.get("top50_count", 0)}；原始数据量：{health.get("raw_count", 0)}</div>
      <div>数据健康检查：缺失字段 {html.escape(", ".join(health.get("missing_fields", [])) or "无")}；fallback={health.get("fallback_used")}</div>
    </section>
    <section class="panel">
      <strong>板块热度 Top10</strong>
      <ul>{sector_html}</ul>
    </section>
    <section>
      <div class="table-wrap">
        <table>
          <thead><tr><th>排名</th><th>代码</th><th>名称</th><th>当前价/收盘价</th><th>今日涨幅</th><th>所属板块</th><th>板块强度</th><th>个股强度</th><th>承接分</th><th>次日涨停概率</th><th>WP评分</th><th>模型置信度</th><th>信号等级</th><th>核心理由</th><th>风险提示</th><th>更新时间</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>
    <section class="panel">
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
