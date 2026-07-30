from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from wp.v3.io import atomic_write_text


EXPERTS = (
    ("早盘结构", "14:20-14:35 的方向效率、位置和回撤结构"),
    ("尾段确认", "14:40-14:50 的价格确认与成交集中度"),
    ("市场/行业领先", "相对大盘和同行同时占优"),
    ("趋势持续", "尾盘斜率、效率与收盘位置一致"),
    ("回撤修复", "先回撤、后修复，且最近五分钟保持向上"),
    ("放量突破", "价格推进得到可成交资金量确认"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a human-readable V16 research evidence report."
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument("--frontier", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    frontier = pd.read_csv(args.frontier)
    content = render(summary, frontier)
    atomic_write_text(args.output, content)
    return 0


def render(summary: dict[str, Any], frontier: pd.DataFrame) -> str:
    metrics = summary.get("nested_oos_metrics") or {}
    readiness = summary.get("historical_readiness") or {}
    evaluation = summary.get("evaluation") or {}
    shadow = summary.get("final_shadow_candidate") or {}
    status = str(shadow.get("status") or "NOT_READY_FOR_SHADOW")
    ready = status == "READY_FOR_150_DAY_FUTURE_SHADOW"
    status_cn = (
        "历史门槛通过，等待 150 个未来交易日影子验证"
        if ready
        else "历史证据未达门槛，不进入生产"
    )
    status_class = "ready" if ready else "blocked"
    metric_items = (
        ("样本外候选", integer(metrics.get("events"))),
        ("覆盖交易日", integer(metrics.get("candidate_days"))),
        ("出票日比例", percent(metrics.get("candidate_day_rate"))),
        ("净胜率", percent(metrics.get("win_rate"))),
        ("平均净收益", signed_pct(metrics.get("mean_net_return_pct"))),
        ("Profit Factor", number(metrics.get("profit_factor"), 2)),
        (
            "额外 50bp 后",
            signed_pct(metrics.get("stress_50bps_mean_net_return_pct")),
        ),
        (
            "均值置信下界",
            signed_pct(metrics.get("clustered_mean_lower_pct")),
        ),
    )
    metric_html = "".join(
        f"<div class='metric'><span>{escape(label)}</span>"
        f"<strong>{escape(value)}</strong></div>"
        for label, value in metric_items
    )
    gate_rows = "".join(
        "<tr>"
        f"<td>{escape(gate_name(name))}</td>"
        f"<td class='gate {'pass' if passed else 'fail'}'>"
        f"{'通过' if passed else '未通过'}</td>"
        "</tr>"
        for name, passed in (readiness.get("gates") or {}).items()
    )
    expert_html = "".join(
        f"<article><h3>{escape(name)}</h3><p>{escape(description)}</p></article>"
        for name, description in EXPERTS
    )
    frontier_view = prepare_frontier(frontier)
    chart = frontier_chart(frontier_view)
    frontier_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row.get('policy_id') or ''))}</td>"
        f"<td>{percent(row.get('candidate_day_rate'))}</td>"
        f"<td>{percent(row.get('win_rate'))}</td>"
        f"<td>{signed_pct(row.get('mean_net_return_pct'))}</td>"
        f"<td>{number(row.get('profit_factor'), 2)}</td>"
        f"<td>{signed_pct(row.get('stress_50bps_mean_net_return_pct'))}</td>"
        "</tr>"
        for _, row in frontier_view.head(12).iterrows()
    )
    fold_rows = "".join(render_fold(row) for row in summary.get("folds", []))
    policy = (
        (shadow.get("selection") or {}).get("policy")
        if isinstance(shadow.get("selection"), dict)
        else None
    )
    policy_text = (
        str(policy.get("policy_id"))
        if isinstance(policy, dict)
        else "无可冻结策略"
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WP V16 量化研究证据</title>
<style>
:root {{
  color-scheme: light;
  --ink:#171a1f; --muted:#626b76; --line:#dfe3e8; --paper:#ffffff;
  --soft:#f6f7f8; --green:#147d45; --red:#b42318; --amber:#a15c00;
  --teal:#0f6674; --blue:#2855a6;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; color:var(--ink); background:var(--paper);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
  "Microsoft YaHei",sans-serif; line-height:1.5; letter-spacing:0;
}}
main {{ width:min(1280px,calc(100% - 32px)); margin:0 auto 64px; }}
header {{ padding:34px 0 24px; border-bottom:1px solid var(--line); }}
h1 {{ margin:0; font-size:30px; line-height:1.25; }}
header p {{ margin:8px 0 0; color:var(--muted); }}
.status {{
  display:flex; justify-content:space-between; gap:20px; align-items:center;
  padding:18px 20px; margin:22px 0 0; border-left:5px solid;
  background:var(--soft);
}}
.status.ready {{ border-color:var(--green); }}
.status.blocked {{ border-color:var(--red); }}
.status strong {{ font-size:18px; }}
.status span {{ color:var(--muted); text-align:right; }}
.metrics {{
  display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
  border-bottom:1px solid var(--line);
}}
.metric {{ min-height:104px; padding:20px; border-right:1px solid var(--line); }}
.metric:nth-child(4n) {{ border-right:0; }}
.metric span {{ display:block; color:var(--muted); font-size:13px; }}
.metric strong {{ display:block; margin-top:8px; font-size:25px; }}
section {{ padding:30px 0; border-bottom:1px solid var(--line); }}
h2 {{ margin:0 0 16px; font-size:21px; }}
h3 {{ margin:0 0 5px; font-size:15px; }}
p {{ margin:0; }}
.two {{ display:grid; grid-template-columns:1.05fr .95fr; gap:34px; }}
.experts {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); }}
.experts article {{ padding:18px; background:var(--paper); min-height:112px; }}
.experts p {{ color:var(--muted); font-size:14px; }}
.chart {{ min-height:300px; background:var(--soft); overflow:hidden; }}
svg {{ display:block; width:100%; height:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th,td {{ padding:11px 10px; border-bottom:1px solid var(--line); text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }}
th {{ color:var(--muted); font-weight:600; background:var(--soft); }}
.gate {{ font-weight:700; }}
.gate.pass {{ color:var(--green); }} .gate.fail {{ color:var(--red); }}
.note {{ color:var(--muted); font-size:13px; margin-top:12px; }}
.policy {{ color:var(--blue); font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  overflow-wrap:anywhere; }}
.scroll {{ overflow-x:auto; }}
@media (max-width:850px) {{
  main {{ width:min(100% - 20px,1280px); }}
  .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
  .metric:nth-child(odd) {{ border-right:1px solid var(--line); }}
  .metric:nth-child(even) {{ border-right:0; }}
  .two,.experts {{ grid-template-columns:1fr; }}
  .status {{ align-items:flex-start; flex-direction:column; }}
  .status span {{ text-align:left; }}
}}
</style>
</head>
<body>
<main>
<header>
  <h1>WP V16 量化研究证据</h1>
  <p>唯一口径：T 日 14:20-14:50 可成交信号，T+1 收盘卖出，扣除既定成本。</p>
  <div class="status {status_class}">
    <strong>{escape(status_cn)}</strong>
    <span>V15 保持冻结 · V16 生产授权：否</span>
  </div>
</header>
<div class="metrics">{metric_html}</div>
<section class="two">
  <div>
    <h2>结论门槛</h2>
    <table><tbody>{gate_rows}</tbody></table>
  </div>
  <div>
    <h2>研究边界</h2>
    <table><tbody>
      <tr><td>因果候选行</td><td>{integer(evaluation.get("frontier_rows"))}</td></tr>
      <tr><td>覆盖交易日</td><td>{integer(evaluation.get("frontier_days"))}</td></tr>
      <tr><td>样本外折数</td><td>{integer(evaluation.get("folds_scored"))} / {integer(evaluation.get("folds_total"))}</td></tr>
      <tr><td>固定卖出合同</td><td>T+1 收盘</td></tr>
      <tr><td>未来影子验证</td><td>至少 150 个交易日</td></tr>
    </tbody></table>
    <p class="note">历史门槛即使全部通过，也不能直接上线；影子期内参数、特征和阈值均不得回看改写。</p>
  </div>
</section>
<section>
  <h2>六类专家共同筛选</h2>
  <div class="experts">{expert_html}</div>
</section>
<section class="two">
  <div>
    <h2>频率与收益前沿</h2>
    <div class="chart">{chart}</div>
    <p class="note">每个点是一套预先声明的固定门槛；横轴是出票日比例，纵轴是平均净收益。红线为零收益。</p>
  </div>
  <div>
    <h2>最终影子候选</h2>
    <p class="policy">{escape(policy_text)}</p>
    <p class="note">若显示“无可冻结策略”，系统应继续输出 NO_SIGNAL，而不是降低门槛凑名单。</p>
  </div>
</section>
<section>
  <h2>帕累托规则</h2>
  <div class="scroll"><table>
    <thead><tr><th>规则</th><th>出票日</th><th>胜率</th><th>平均净收益</th><th>PF</th><th>额外 50bp 后</th></tr></thead>
    <tbody>{frontier_rows}</tbody>
  </table></div>
</section>
<section>
  <h2>逐折样本外审计</h2>
  <div class="scroll"><table>
    <thead><tr><th>折</th><th>测试区间</th><th>状态</th><th>候选</th><th>胜率</th><th>平均净收益</th></tr></thead>
    <tbody>{fold_rows}</tbody>
  </table></div>
</section>
</main>
</body>
</html>
"""


def prepare_frontier(frontier: pd.DataFrame) -> pd.DataFrame:
    if frontier.empty:
        return frontier
    if "pareto_efficient" in frontier:
        values = frontier["pareto_efficient"]
        if not pd.api.types.is_bool_dtype(values.dtype):
            values = values.astype(str).str.lower().eq("true")
        efficient = frontier.loc[values].copy()
        if not efficient.empty:
            return efficient
    return frontier.copy()


def frontier_chart(frame: pd.DataFrame) -> str:
    width, height = 640, 300
    left, right, top, bottom = 58, 22, 20, 44
    if frame.empty:
        return (
            f"<svg viewBox='0 0 {width} {height}' role='img' "
            "aria-label='暂无可绘制规则'><text x='50%' y='50%' "
            "text-anchor='middle' fill='#626b76'>暂无可绘制规则</text></svg>"
        )
    x = pd.to_numeric(frame["candidate_day_rate"], errors="coerce")
    y = pd.to_numeric(frame["mean_net_return_pct"], errors="coerce")
    valid = x.notna() & y.notna()
    x, y = x.loc[valid], y.loc[valid]
    if x.empty:
        return frontier_chart(frame.head(0))
    x_min, x_max = 0.0, max(float(x.max()) * 1.08, 0.01)
    y_min = min(float(y.min()), 0.0)
    y_max = max(float(y.max()), 0.0)
    if y_max == y_min:
        y_max = y_min + 1.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (
            width - left - right
        )

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (
            height - top - bottom
        )

    points = "".join(
        f"<circle cx='{sx(float(x.loc[index])):.1f}' "
        f"cy='{sy(float(y.loc[index])):.1f}' r='4.5' fill='#0f6674' "
        "fill-opacity='.72'/>"
        for index in x.index
    )
    zero_y = sy(0.0)
    return f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="频率与平均净收益散点图">
<rect width="{width}" height="{height}" fill="#f6f7f8"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#9aa2ad"/>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#9aa2ad"/>
<line x1="{left}" y1="{zero_y:.1f}" x2="{width-right}" y2="{zero_y:.1f}" stroke="#b42318" stroke-dasharray="5 5"/>
{points}
<text x="{left}" y="{height-12}" fill="#626b76" font-size="12">0%</text>
<text x="{width-right}" y="{height-12}" text-anchor="end" fill="#626b76" font-size="12">{x_max:.0%}</text>
<text x="8" y="{top+4}" fill="#626b76" font-size="12">{y_max:+.2f}%</text>
<text x="8" y="{height-bottom}" fill="#626b76" font-size="12">{y_min:+.2f}%</text>
</svg>"""


def render_fold(row: dict[str, Any]) -> str:
    test = row.get("test") or {}
    authorized = bool(row.get("policy_authorized"))
    status = "规则通过" if authorized else "NO_SIGNAL"
    return (
        "<tr>"
        f"<td>{escape(str(row.get('fold') or ''))}</td>"
        f"<td>{escape(str(row.get('test_start') or ''))} - "
        f"{escape(str(row.get('test_end') or ''))}</td>"
        f"<td class='gate {'pass' if authorized else 'fail'}'>{status}</td>"
        f"<td>{integer(test.get('events'))}</td>"
        f"<td>{percent(test.get('win_rate'))}</td>"
        f"<td>{signed_pct(test.get('mean_net_return_pct'))}</td>"
        "</tr>"
    )


def gate_name(value: str) -> str:
    names = {
        "minimum_nested_oos_candidates": "样本外候选至少 250 支",
        "minimum_nested_oos_candidate_days": "候选日至少 50 日",
        "minimum_win_rate": "净胜率至少 55%",
        "minimum_wilson_lower": "胜率保守下界至少 52%",
        "minimum_clustered_win_rate_lower": "交易日分块胜率下界至少 52%",
        "minimum_mean_net_return_pct": "平均净收益至少 0.20%",
        "clustered_mean_lower_positive": "交易日聚类均值下界为正",
        "minimum_profit_factor": "Profit Factor 至少 1.20",
        "real_50bps_stress_nonnegative": "额外 50bp 压力后不亏",
        "return_p10_above_minus_3pct": "收益 10% 分位不低于 -3%",
        "temporal_integrity": "训练、确认、测试严格按时间隔离",
    }
    return names.get(value, value)


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def finite(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else None


def integer(value: Any) -> str:
    parsed = finite(value)
    return f"{int(parsed):,}" if parsed is not None else "-"


def percent(value: Any) -> str:
    parsed = finite(value)
    return f"{parsed:.2%}" if parsed is not None else "-"


def signed_pct(value: Any) -> str:
    parsed = finite(value)
    return f"{parsed:+.4f}%" if parsed is not None else "-"


def number(value: Any, digits: int) -> str:
    parsed = finite(value)
    return f"{parsed:.{digits}f}" if parsed is not None else "-"


if __name__ == "__main__":
    raise SystemExit(main())
