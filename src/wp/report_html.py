from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


def _fmt(value, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return html.escape(str(value))


def _pct_cell(value) -> str:
    try:
        pct = float(value)
    except Exception:
        return "<span class=\"pct-pending\">待验证</span>"
    if pd.isna(pct):
        return "<span class=\"pct-pending\">待验证</span>"
    cls = "pct-up" if pct > 0 else "pct-down" if pct < 0 else "pct-flat"
    sign = "+" if pct > 0 else ""
    return f"<span class=\"{cls}\">{sign}{pct:.2f}%</span>"


def _date_text(value: object) -> str:
    text = str(value or "").strip()
    if len(text) == 8 and text.isdigit():
        text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return html.escape(text or "-")


def _summary_int(summary: dict, key: str) -> int:
    try:
        return int(summary.get(key, 0))
    except (TypeError, ValueError):
        return 0


def _rate(value: object) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def _backtest_rows(backtests: list[dict]) -> str:
    rows = []
    for summary in backtests:
        start = str(summary.get("start_date", ""))
        end = str(summary.get("end_date", ""))
        folder = f"{start}_{end}"
        rows.append(
            "<tr>"
            + f"<td>{_date_text(start)} 至 {_date_text(end)}</td>"
            + f"<td>{_summary_int(summary, 'trade_days')}</td>"
            + f"<td>{_summary_int(summary, 'buy_plan_days')}</td>"
            + f"<td>{_summary_int(summary, 'buy_strict5_plan_days')}</td>"
            + f"<td>{_summary_int(summary, 'buy_strict5_trade_count')}</td>"
            + f"<td>{_rate(summary.get('buy_strict5_positive_close_rate'))}</td>"
            + f"<td>{_rate(summary.get('buy_strict5_limitup_rate'))}</td>"
            + f"<td>{_pct_cell(summary.get('buy_strict5_daily_avg_next_day_open_pct'))}</td>"
            + f"<td>{_pct_cell(summary.get('buy_strict5_daily_avg_next_day_high_pct'))}</td>"
            + f"<td>{_pct_cell(summary.get('buy_strict5_daily_avg_next_day_close_pct'))}</td>"
            + f"<td>{_pct_cell(summary.get('buy_strict5_cumulative_next_day_close_pct'))}</td>"
            + f"<td>{_fmt(summary.get('auc', '-'), 4)}</td>"
            + f"<td><a href=\"../backtests/{html.escape(folder)}/summary.json\">汇总</a> · "
            + f"<a href=\"../backtests/{html.escape(folder)}/trades.csv\">Top50</a> · "
            + f"<a href=\"../backtests/{html.escape(folder)}/buy_trades.csv\">观察名单</a> · "
            + f"<a href=\"../backtests/{html.escape(folder)}/monthly_summary.csv\">分月</a></td>"
            + "</tr>"
        )
    return "".join(rows) or "<tr><td colspan=\"13\" class=\"empty\">暂无回测数据</td></tr>"


def _validation_overview(summary: dict) -> str:
    total_days = _summary_int(summary, "total_plan_days")
    verified_days = _summary_int(summary, "verified_plan_days")
    total_records = _summary_int(summary, "total_records")
    verified_records = _summary_int(summary, "verified_records")
    positive_records = _summary_int(summary, "positive_records")
    limit_up_records = _summary_int(summary, "limit_up_records")
    positive_rate = float(summary.get("positive_rate", 0.0) or 0.0)
    limit_up_rate = float(summary.get("limit_up_rate", 0.0) or 0.0)
    has_verified_days = verified_days > 0
    average_open = _pct_cell(summary.get("average_open_return_pct")) if verified_records else "<span class=\"pct-pending\">待验证</span>"
    average_high = _pct_cell(summary.get("average_high_return_pct")) if verified_records else "<span class=\"pct-pending\">待验证</span>"
    daily_average = _pct_cell(summary.get("daily_average_pct_chg")) if has_verified_days else "<span class=\"pct-pending\">待验证</span>"
    cumulative = _pct_cell(summary.get("cumulative_pct_chg")) if has_verified_days else "<span class=\"pct-pending\">待验证</span>"
    metrics = [
        ("已验证日", f"{verified_days} / {total_days}"),
        ("已验证票", f"{verified_records} / {total_records}"),
        ("上涨", f"{positive_records} / {verified_records}<small>{positive_rate:.2f}%</small>"),
        ("触及涨停", f"{limit_up_records} / {verified_records}<small>{limit_up_rate:.2f}%</small>"),
        ("平均次日开盘", average_open),
        ("平均次日最高", average_high),
        ("日均收盘收益", daily_average),
        ("累计收盘收益", cumulative),
    ]
    return "".join(
        f"<div class=\"validation-kpi\"><span>{label}</span><strong>{value}</strong></div>" for label, value in metrics
    )


def _validation_days(validation: pd.DataFrame) -> str:
    if validation.empty:
        return "<div class=\"empty\">暂无验证记录</div>"

    view = validation.copy()
    view["plan_trade_date"] = view.get("plan_trade_date", "").fillna("").astype(str)
    view["_rank_sort"] = pd.to_numeric(view.get("buy_rank"), errors="coerce").fillna(999)
    numeric_column = lambda name: pd.to_numeric(view[name], errors="coerce") if name in view.columns else pd.Series(float("nan"), index=view.index, dtype="float64")
    legacy_return = numeric_column("actual_pct_chg")
    close_return = numeric_column("return_close_pct")
    view["_close_return"] = close_return.where(close_return.notna(), legacy_return)
    view["_open_return"] = numeric_column("return_open_pct")
    view["_high_return"] = numeric_column("return_high_pct")
    view = view.sort_values(["plan_trade_date", "plan_time", "_rank_sort"], ascending=[False, False, True])
    groups = []
    for plan_date, day in view.groupby("plan_trade_date", sort=False):
        day = day.sort_values(["plan_time", "_rank_sort"], ascending=[False, True])
        verified_mask = day.get("truth_status", "").fillna("").astype(str).eq("verified")
        verified_count = int(verified_mask.sum())
        total_count = int(len(day))
        close_pct = day.loc[verified_mask, "_close_return"].dropna()
        open_pct = day.loc[verified_mask, "_open_return"].dropna()
        high_pct = day.loc[verified_mask, "_high_return"].dropna()
        returns_cell = (
            f"<span class=\"validation-return\">开 {_pct_cell(open_pct.mean())} · 高 {_pct_cell(high_pct.mean())} · 收 {_pct_cell(close_pct.mean())}</span>"
            if len(close_pct)
            else "<span class=\"pct-pending\">待验证</span>"
        )
        positive_count = int(close_pct.gt(0).sum())
        limit_up_count = int(
            day.loc[verified_mask, "is_limit_up_t1"].astype(str).str.lower().isin({"true", "1", "yes"}).sum()
        )
        if verified_count == total_count and total_count:
            status_label = "已验证"
            status_class = "verified"
        elif verified_count:
            status_label = "部分验证"
            status_class = "partial"
        else:
            status_label = "待验证"
            status_class = "pending"
        target_dates = [str(value) for value in day.get("target_trade_date", pd.Series(dtype=str)).tolist() if str(value)]
        target_date = target_dates[0] if target_dates else ""
        positive_text = f"{positive_count} / {verified_count}" if verified_count else "-"
        limit_up_text = f"{limit_up_count} / {verified_count}" if verified_count else "-"
        detail_rows = []
        for _, row in day.iterrows():
            truth_status = str(row.get("truth_status", ""))
            is_limit_up = str(row.get("is_limit_up_t1", "")).lower() in {"true", "1", "yes"}
            close_value = row.get("return_close_pct", "")
            if _fmt(close_value) in {"", "nan", "None"}:
                close_value = row.get("actual_pct_chg", "")
            detail_rows.append(
                "<tr>"
                + f"<td>{html.escape(str(row.get('plan_time', '')))}</td>"
                + f"<td>{html.escape(str(row.get('buy_rank', '')))}</td>"
                + f"<td>{html.escape(str(row.get('ts_code', '')))}</td>"
                + f"<td>{html.escape(str(row.get('name', '')))}</td>"
                + f"<td>{_fmt(row.get('plan_price', ''))}</td>"
                + f"<td>{_fmt(row.get('pct_chg_plan', 0))}%</td>"
                + f"<td>{_pct_cell(row.get('return_open_pct', ''))}</td>"
                + f"<td>{_pct_cell(row.get('return_high_pct', ''))}</td>"
                + f"<td>{_pct_cell(row.get('return_low_pct', ''))}</td>"
                + f"<td>{_pct_cell(close_value)}</td>"
                + f"<td>{'是' if is_limit_up else '否' if truth_status == 'verified' else '待验证'}</td>"
                + "</tr>"
            )
        groups.append(
            "<details class=\"validation-day-details\">"
            + "<summary class=\"validation-day-summary validation-grid\">"
            + f"<span>{_date_text(plan_date)}</span>"
            + f"<span>{_date_text(target_date)}</span>"
            + f"<span>{total_count}</span>"
            + f"<span>{returns_cell}</span>"
            + f"<span>{positive_text}</span>"
            + f"<span>{limit_up_text}</span>"
            + f"<span class=\"validation-status {status_class}\">{status_label}</span>"
            + "<span class=\"validation-day-action\" aria-hidden=\"true\"></span>"
            + "</summary>"
            + "<div class=\"validation-detail-wrap\"><table class=\"validation-detail-table\">"
            + "<thead><tr><th>计划时间</th><th>买入序</th><th>代码</th><th>名称</th><th>计划价</th><th>计划涨幅</th><th>次日开盘</th><th>次日最高</th><th>次日最低</th><th>次日收盘</th><th>触及涨停</th></tr></thead>"
            + f"<tbody>{''.join(detail_rows)}</tbody></table></div></details>"
        )
    return "".join(groups)


def render_html(
    top50: pd.DataFrame,
    full_rank: pd.DataFrame,
    health: dict,
    output_path: str | Path,
    buy_plan: pd.DataFrame | None = None,
    validation: pd.DataFrame | None = None,
    validation_summary: dict | None = None,
    backtests: list[dict] | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    buy_plan = buy_plan if buy_plan is not None else pd.DataFrame()
    validation = validation if validation is not None else pd.DataFrame()
    validation_summary = validation_summary or {}
    backtests = sorted(backtests or [], key=lambda item: str(item.get("start_date", "")))
    backtest_rows = _backtest_rows(backtests)
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
    buy_rows = []
    for _, row in buy_plan.iterrows():
        buy_rows.append(
            "<tr>"
            + f"<td>{html.escape(str(row.get('buy_rank', '')))}</td>"
            + f"<td>{html.escape(str(row.get('portfolio_group', '')))}</td>"
            + f"<td>{html.escape(str(row.get('ts_code', '')))}</td>"
            + f"<td>{html.escape(str(row.get('name', '')))}</td>"
            + f"<td>{_fmt(row.get('pct_chg', 0))}%</td>"
            + f"<td>{html.escape(str(row.get('sector_name', '')))}</td>"
            + f"<td>{_fmt(row.get('p_limitup_t1', 0))}%</td>"
            + f"<td>{_fmt(row.get('wp_score', 0))}</td>"
            + f"<td>{_fmt(row.get('decision_score', 0))}</td>"
            + f"<td>{_fmt(row.get('risk_penalty_score', 0))}</td>"
            + f"<td>{html.escape(str(row.get('confirm_before_buy', '')))}</td>"
            + f"<td>{html.escape(str(row.get('reject_if', '')))}</td>"
            + f"<td>{html.escape(str(row.get('buy_reason', '')))}</td>"
            + "</tr>"
        )
    if not buy_rows:
        buy_rows.append("<tr><td colspan=\"13\" class=\"empty\">当前无买入观察计划</td></tr>")
    validation_overview = _validation_overview(validation_summary)
    validation_days = _validation_days(validation)
    status_cls = "bad" if health.get("status") not in {"ok", "无符合条件股票"} else "ok"
    data_trade_date = html.escape(str(health.get("data_trade_date") or "-"))
    expected_trade_date = html.escape(str(health.get("expected_trade_date") or "-"))
    realtime_sources = health.get("realtime_sources") or []
    realtime_source_text = ", ".join(str(item) for item in realtime_sources) if realtime_sources else "未标记"
    market_data_time = html.escape(str(health.get("market_data_time") or health.get("data_time") or "-"))
    wp_run_time = html.escape(str(health.get("wp_run_time") or health.get("data_time") or "-"))
    status_text = html.escape(str(health.get("status")))
    status_rows = [
        ("运行状态", f"<span class=\"status {status_cls}\">{html.escape(str(health.get('status')))}</span>"),
        ("市场数据时间", f"<span class=\"market-time\">{market_data_time}</span>"),
        ("上游生成时间", html.escape(str(health.get("source_generated_at") or "-"))),
        ("报告更新时间", wp_run_time),
        ("行情日期", data_trade_date),
        ("期望交易日", expected_trade_date),
        ("候选池数量", str(health.get("candidate_count", 0))),
        ("入选 Top50 数量", str(health.get("top50_count", 0))),
        ("买入观察数量", str(health.get("buy_plan_count", 0))),
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
  <meta http-equiv="refresh" content="600">
  <title>WP Top50</title>
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ width: 100%; max-width: 100%; overflow-x: hidden; }}
    body {{ margin: 0; font-family: "SF Pro SC", "SF Pro Text", "SF Pro Display", "SF Pro Icons", -apple-system, BlinkMacSystemFont, "PingFang SC", "Helvetica Neue", Helvetica, Arial, sans-serif; color: #1d1d1f; background: #f5f5f7; letter-spacing: 0; }}
    header {{ padding: 30px 32px 22px; background: rgba(255, 255, 255, 0.92); border-bottom: 1px solid #d2d2d7; }}
    h1 {{ margin: 0 0 10px; font-size: 30px; line-height: 1.18; font-weight: 700; }}
    main {{ min-width: 0; max-width: 100%; padding: 24px 32px 40px; }}
    section {{ min-width: 0; max-width: 100%; margin-bottom: 22px; }}
    .summary-section {{ overflow: hidden; background: #fff; border: 1px solid #d2d2d7; border-radius: 8px; }}
    .summary-toggle {{ list-style: none; cursor: pointer; display: flex; align-items: center; justify-content: space-between; gap: 18px; padding: 18px 24px; user-select: none; }}
    .summary-toggle::-webkit-details-marker {{ display: none; }}
    .summary-toggle-title {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; min-width: 0; }}
    .summary-toggle-title strong {{ font-size: 18px; line-height: 1.25; font-weight: 700; color: #1d1d1f; }}
    .summary-toggle-meta {{ color: #6e6e73; font-size: 13px; line-height: 1.4; }}
    .summary-toggle-action {{ flex: 0 0 auto; width: 28px; height: 28px; border: 1px solid #d2d2d7; border-radius: 50%; background: #f5f5f7; color: #1d1d1f; display: inline-grid; place-items: center; font-size: 20px; line-height: 1; font-weight: 500; }}
    .summary-toggle-action::after {{ content: "+"; transform: translateY(-1px); }}
    .summary-details[open] .summary-toggle-action::after {{ content: "-"; transform: translateY(-2px); }}
    .summary-grid {{ border-top: 1px solid #f1f1f3; }}
    .summary-grid {{ display: grid; grid-template-columns: minmax(360px, 1fr) minmax(320px, 0.84fr); }}
    .summary-pane {{ padding: 22px 24px; }}
    .summary-pane + .summary-pane {{ border-left: 1px solid #d2d2d7; }}
    .summary-title {{ margin: 0 0 12px; font-size: 18px; line-height: 1.25; font-weight: 700; }}
    .summary-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    .summary-table th, .summary-table td {{ padding: 10px 0; border-bottom: 1px solid #f1f1f3; vertical-align: middle; white-space: nowrap; }}
    .summary-table tr:last-child th, .summary-table tr:last-child td {{ border-bottom: 0; }}
    .summary-table th {{ width: 150px; color: #6e6e73; font-weight: 500; text-align: left; }}
    .summary-table td {{ color: #1d1d1f; font-weight: 600; text-align: left; }}
    .market-time {{ color: #d70015; font-weight: 700; }}
    .pct-up {{ color: #d70015; font-weight: 700; }}
    .pct-down {{ color: #008a00; font-weight: 700; }}
    .pct-flat, .pct-pending {{ color: #86868b; font-weight: 600; }}
    .sector-pane {{ display: flex; flex-direction: column; align-items: flex-start; }}
    .sector-table {{ width: auto; min-width: 260px; max-width: 360px; table-layout: auto; }}
    .sector-table th, .sector-table td {{ padding: 8px 18px 8px 0; text-align: left; }}
    .sector-table th:nth-child(1), .sector-table td:nth-child(1) {{ width: 48px; color: #86868b; font-variant-numeric: tabular-nums; }}
    .sector-table th:nth-child(2), .sector-table td:nth-child(2) {{ min-width: 112px; }}
    .sector-table th:nth-child(3), .sector-table td:nth-child(3) {{ width: 64px; padding-right: 0; text-align: left; font-weight: 600; }}
    .status.ok {{ color: #0b7a3b; font-weight: 700; }}
    .status.bad {{ color: #b42318; font-weight: 700; }}
    .table-wrap {{ display: block; width: 100%; min-width: 0; max-width: 100%; overflow-x: auto; border: 1px solid #d2d2d7; background: white; border-radius: 8px; -webkit-overflow-scrolling: touch; }}
    .rank-table {{ border-collapse: collapse; min-width: 1680px; width: 100%; font-size: 13px; }}
    .rank-table th, .rank-table td {{ padding: 10px 11px; border-bottom: 1px solid #f1f1f3; text-align: left; white-space: nowrap; }}
    .rank-table th {{ background: #fbfbfd; color: #6e6e73; position: sticky; top: 0; font-weight: 600; }}
    .rank-table tbody tr:hover td {{ background: #f5f5f7; }}
    .rank-table tr.top10 td {{ background: #fff8dc; }}
    .rank-table tr.risk-high td {{ color: #9f1f1f; }}
    .buy-table {{ border-collapse: collapse; min-width: 1380px; width: 100%; font-size: 13px; }}
    .buy-table th, .buy-table td {{ padding: 10px 11px; border-bottom: 1px solid #f1f1f3; text-align: left; vertical-align: top; }}
    .buy-table th {{ background: #fbfbfd; color: #6e6e73; font-weight: 600; white-space: nowrap; }}
    .buy-table td:nth-child(11), .buy-table td:nth-child(12), .buy-table td:nth-child(13) {{ min-width: 150px; line-height: 1.45; }}
    .backtest-section {{ width: 100%; min-width: 0; max-width: 100%; overflow: hidden; background: #fff; border: 1px solid #d2d2d7; border-radius: 8px; }}
    .backtest-scroll {{ width: 100%; overflow-x: auto; border-top: 1px solid #e5e5ea; -webkit-overflow-scrolling: touch; }}
    .backtest-table {{ border-collapse: collapse; min-width: 1480px; width: 100%; font-size: 13px; }}
    .backtest-table th, .backtest-table td {{ padding: 10px 12px; border-bottom: 1px solid #f1f1f3; text-align: left; white-space: nowrap; }}
    .backtest-table th {{ background: #fbfbfd; color: #6e6e73; font-weight: 600; }}
    .backtest-table a {{ color: #06c; text-decoration: none; }}
    .backtest-table a:hover {{ text-decoration: underline; }}
    .validation-section {{ width: 100%; min-width: 0; max-width: 100%; overflow: hidden; background: #fff; border: 1px solid #d2d2d7; border-radius: 8px; }}
    .validation-heading {{ padding: 18px 20px 14px; display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }}
    .validation-heading strong {{ font-size: 16px; }}
    .validation-heading span {{ color: #6e6e73; font-size: 12px; }}
    .validation-kpis {{ display: grid; grid-template-columns: repeat(8, minmax(120px, 1fr)); gap: 1px; background: #e5e5ea; border-top: 1px solid #e5e5ea; border-bottom: 1px solid #e5e5ea; }}
    .validation-kpi {{ min-width: 0; padding: 14px 16px; background: #fff; }}
    .validation-kpi > span {{ display: block; margin-bottom: 5px; color: #6e6e73; font-size: 12px; font-weight: 500; }}
    .validation-kpi > strong {{ display: flex; align-items: baseline; gap: 7px; color: #1d1d1f; font-size: 18px; line-height: 1.2; white-space: nowrap; }}
    .validation-kpi small {{ color: #6e6e73; font-size: 11px; font-weight: 500; }}
    .validation-days {{ width: 100%; min-width: 0; max-width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; }}
    .validation-day-list {{ min-width: 1080px; }}
    .validation-grid {{ display: grid; grid-template-columns: 1.05fr 1.05fr 0.55fr 2.2fr 0.7fr 0.7fr 0.78fr 30px; align-items: center; column-gap: 12px; }}
    .validation-day-header {{ padding: 10px 16px; color: #6e6e73; background: #fbfbfd; border-bottom: 1px solid #e5e5ea; font-size: 12px; font-weight: 600; }}
    .validation-day-summary {{ list-style: none; min-height: 50px; padding: 10px 16px; border-bottom: 1px solid #f1f1f3; cursor: pointer; font-size: 13px; font-weight: 600; user-select: none; }}
    .validation-day-summary::-webkit-details-marker {{ display: none; }}
    .validation-day-summary:hover {{ background: #f5f5f7; }}
    .validation-day-action {{ width: 24px; height: 24px; border: 1px solid #d2d2d7; border-radius: 50%; display: inline-grid; place-items: center; color: #1d1d1f; font-size: 17px; font-weight: 500; }}
    .validation-day-action::after {{ content: "+"; transform: translateY(-1px); }}
    .validation-day-details[open] .validation-day-action::after {{ content: "-"; transform: translateY(-2px); }}
    .validation-status.verified {{ color: #0b7a3b; }}
    .validation-status.partial {{ color: #b26a00; }}
    .validation-status.pending {{ color: #86868b; }}
    .validation-detail-wrap {{ padding: 0 16px 14px; background: #fbfbfd; border-bottom: 1px solid #e5e5ea; overflow-x: auto; }}
    .validation-return {{ display: flex; align-items: center; gap: 5px; white-space: nowrap; }}
    .validation-detail-table {{ border-collapse: collapse; min-width: 1180px; width: 100%; font-size: 12px; }}
    .validation-detail-table th, .validation-detail-table td {{ padding: 9px 10px; border-bottom: 1px solid #e5e5ea; text-align: left; white-space: nowrap; }}
    .validation-detail-table th {{ color: #6e6e73; font-weight: 600; }}
    .validation-detail-table tr:last-child td {{ border-bottom: 0; }}
    .section-block {{ background: #fff; border: 1px solid #d2d2d7; border-radius: 8px; padding: 18px 20px; color: #424245; line-height: 1.65; }}
    .section-block strong {{ display: block; color: #1d1d1f; margin-bottom: 6px; }}
    .section-block p {{ margin: 0 0 14px; }}
    .section-block p:last-child {{ margin-bottom: 0; }}
    .empty {{ text-align: center; color: #86868b; padding: 24px; }}
    @media (max-width: 820px) {{
      header {{ padding: 24px 18px 18px; }}
      h1 {{ font-size: 25px; }}
      main {{ padding: 18px 14px 30px; }}
      .summary-section {{ width: 100%; min-width: 0; max-width: 100%; }}
      .summary-toggle {{ align-items: flex-start; padding: 16px; }}
      .summary-toggle-title {{ flex-direction: column; gap: 4px; }}
      .summary-grid {{ grid-template-columns: 1fr; }}
      .summary-pane {{ padding: 18px 16px; }}
      .summary-pane + .summary-pane {{ border-left: 0; border-top: 1px solid #d2d2d7; }}
      .summary-table {{ font-size: 13px; }}
      .summary-table th {{ width: 128px; }}
      .sector-table {{ min-width: 236px; max-width: 100%; }}
      .sector-table th, .sector-table td {{ padding-top: 7px; padding-bottom: 7px; }}
      .validation-heading {{ padding: 16px; }}
      .validation-kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .validation-kpi {{ padding: 12px 14px; }}
      .validation-kpi > strong {{ font-size: 16px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>WP Top50</h1>
  </header>
  <main>
    <details class="summary-section summary-details" aria-label="运行状态与板块热度">
      <summary class="summary-toggle">
        <span class="summary-toggle-title">
          <strong>运行状态与板块热度</strong>
          <span class="summary-toggle-meta">状态：<span class="status {status_cls}">{status_text}</span>；市场数据：<span class="market-time">{market_data_time}</span>；报告更新：{wp_run_time}</span>
        </span>
        <span class="summary-toggle-action" aria-hidden="true"></span>
      </summary>
      <div class="summary-grid">
        <div class="summary-pane">
          <h2 class="summary-title">今日运行状态</h2>
          <table class="summary-table">
            <tbody>{status_html}</tbody>
          </table>
        </div>
        <div class="summary-pane sector-pane">
          <h2 class="summary-title">板块热度 Top10</h2>
          <table class="summary-table sector-table">
            <thead><tr><th>序号</th><th>板块</th><th>数量</th></tr></thead>
            <tbody>{sector_rows}</tbody>
          </table>
        </div>
      </div>
    </details>
    <section>
      <div class="section-block">
        <strong>14:20 尾盘买入观察计划</strong>
      </div>
      <div class="backtest-scroll">
        <table class="buy-table">
          <thead><tr><th>买入序</th><th>组合层级</th><th>代码</th><th>名称</th><th>涨幅</th><th>板块</th><th>次日概率</th><th>WP评分</th><th>决策分</th><th>风险分</th><th>14:50确认条件</th><th>放弃条件</th><th>买入理由</th></tr></thead>
          <tbody>{''.join(buy_rows)}</tbody>
        </table>
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
    <section class="validation-section">
      <div class="validation-heading">
        <strong>14:20 观察名单累计验证</strong>
        <span>按计划价买入，统计下一交易日实际收益</span>
      </div>
      <div class="validation-kpis">{validation_overview}</div>
      <div class="validation-days">
        <div class="validation-day-list">
          <div class="validation-day-header validation-grid"><span>计划日</span><span>验证日</span><span>名单</span><span>次日收益（开 / 高 / 收）</span><span>上涨</span><span>触及涨停</span><span>状态</span><span></span></div>
          {validation_days}
        </div>
      </div>
    </section>
    <section class="backtest-section">
      <div class="validation-heading">
        <strong>模型回测验证</strong>
        <span>每日最多 5 支；历史区间使用收盘价代理买入，真实计划按计划价验证</span>
      </div>
      <div class="table-wrap">
        <table class="backtest-table">
          <thead><tr><th>区间</th><th>交易日</th><th>观察日</th><th>严格5支日</th><th>严格5支样本</th><th>上涨率</th><th>触及涨停</th><th>次日开盘</th><th>次日最高</th><th>次日收盘</th><th>累计收盘</th><th>AUC</th><th>原始数据</th></tr></thead>
          <tbody>{backtest_rows}</tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""
    output.write_text(page, encoding="utf-8")
