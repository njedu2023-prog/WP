from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from .buy_validation import VALIDATION_TRACKING_START_DATE, _summary, scope_validation_table


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


def _clock_text(value: object) -> str:
    text = str(value or "").strip()
    if len(text) >= 16 and text[10:11] in {" ", "T"}:
        text = text[11:16]
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
            + f"<td>{_summary_int(summary, 'buy_trade_count')}</td>"
            + f"<td>{_rate(summary.get('buy_positive_close_rate'))}</td>"
            + f"<td>{_rate(summary.get('buy_limitup_rate'))}</td>"
            + f"<td>{_pct_cell(summary.get('buy_daily_avg_next_day_open_pct'))}</td>"
            + f"<td>{_pct_cell(summary.get('buy_daily_avg_next_day_high_pct'))}</td>"
            + f"<td>{_pct_cell(summary.get('buy_daily_avg_next_day_close_pct'))}</td>"
            + f"<td>{_pct_cell(summary.get('buy_cumulative_next_day_close_pct'))}</td>"
            + f"<td>{_fmt(summary.get('auc', '-'), 4)}</td>"
            + f"<td><a href=\"../backtests/{html.escape(folder)}/summary.json\">汇总</a> · "
            + f"<a href=\"../backtests/{html.escape(folder)}/buy_trades.csv\">主票明细</a> · "
            + f"<a href=\"../backtests/{html.escape(folder)}/monthly_summary.csv\">分月</a></td>"
            + "</tr>"
        )
    return "".join(rows) or "<tr><td colspan=\"12\" class=\"empty\">暂无回测数据</td></tr>"


def _validation_overview(summary: dict) -> str:
    total_days = _summary_int(summary, "total_plan_days")
    verified_days = _summary_int(summary, "verified_plan_days")
    total_records = _summary_int(summary, "total_records")
    verified_records = _summary_int(summary, "verified_records")
    positive_records = _summary_int(summary, "positive_records")
    limit_up_records = _summary_int(summary, "limit_up_records")
    positive_rate = float(summary.get("positive_rate", 0.0) or 0.0)
    limit_up_rate = float(summary.get("limit_up_rate", 0.0) or 0.0)
    missing_days = sum(
        1
        for item in summary.get("sampling_days", [])
        if str(item.get("sample_status") or "") == "missing"
    )
    has_verified_days = verified_days > 0
    average_open = _pct_cell(summary.get("average_open_return_pct")) if verified_records else "<span class=\"pct-pending\">待验证</span>"
    average_high = _pct_cell(summary.get("average_high_return_pct")) if verified_records else "<span class=\"pct-pending\">待验证</span>"
    daily_average = _pct_cell(summary.get("daily_average_pct_chg")) if has_verified_days else "<span class=\"pct-pending\">待验证</span>"
    cumulative = _pct_cell(summary.get("cumulative_pct_chg")) if has_verified_days else "<span class=\"pct-pending\">待验证</span>"
    metrics = [
        ("已验证日", f"{verified_days} / {total_days}<small>采样缺失{missing_days}日</small>"),
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


def _validation_days(
    validation: pd.DataFrame,
    sampling_days: list[dict] | None = None,
) -> str:
    sampling_days = sampling_days or []
    if validation.empty and not sampling_days:
        return "<div class=\"empty\">暂无主票验证记录</div>"

    groups: list[tuple[str, str]] = []
    represented_dates: set[str] = set()
    if not validation.empty:
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
        for plan_date, day in view.groupby("plan_trade_date", sort=False):
            plan_date = str(plan_date)
            represented_dates.add(plan_date.replace("-", ""))
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
            content = (
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
            groups.append((plan_date.replace("-", ""), content))

    for sample in sampling_days:
        plan_date = str(sample.get("plan_trade_date") or "").replace("-", "")
        if (
            str(sample.get("sample_status") or "") != "missing"
            or not plan_date
            or plan_date in represented_dates
        ):
            continue
        target_date = str(sample.get("target_trade_date") or "")
        note = html.escape(str(sample.get("note") or "当日尾盘采样缺失"))
        content = (
            "<div class=\"validation-day-summary validation-grid validation-day-missing\">"
            + f"<span>{_date_text(plan_date)}</span>"
            + f"<span>{_date_text(target_date)}</span>"
            + "<span>0</span>"
            + f"<span class=\"sampling-missing-note\">{note}</span>"
            + "<span>-</span><span>-</span>"
            + "<span class=\"validation-status missing\">采样缺失</span><span></span>"
            + "</div>"
        )
        groups.append((plan_date, content))

    groups.sort(key=lambda item: item[0], reverse=True)
    return "".join(content for _, content in groups)


def _forecast_range(summary: dict, target: str) -> str:
    values = []
    for suffix in (10, 50, 90):
        value = pd.to_numeric(pd.Series([summary.get(f"forecast_{target}_q{suffix}_pct")]), errors="coerce").iloc[0]
        values.append(None if pd.isna(value) else float(value))
    if any(value is None for value in values):
        return "-"
    return " / ".join(f"{value:+.2f}%" for value in values)


def _probability(summary: dict, key: str) -> str:
    value = pd.to_numeric(pd.Series([summary.get(key)]), errors="coerce").iloc[0]
    return "-" if pd.isna(value) else f"{float(value):.1f}%"


def _decision_support_panel(decision: dict, regime: dict) -> str:
    action = str(decision.get("action") or "继续观察")
    action_class = "buy" if action == "建议关注买入" else "avoid" if action == "建议空仓" else "wait"
    candidate_name = str(decision.get("candidate_name") or "")
    candidate_code = str(decision.get("candidate_code") or "")
    candidate = f"{candidate_name} {candidate_code}".strip() or "暂无首选"
    warning = str(decision.get("forecast_warning") or "")
    reason = str(decision.get("reason") or "")
    mode = str(decision.get("forecast_mode") or "样本不足")
    confidence = _probability(decision, "forecast_confidence")
    regime_state = str(regime.get("state") or "数据不足")
    regime_score = _fmt(regime.get("score", 0))
    regime_reason = str(regime.get("reason") or "")
    next_checkpoint = str(decision.get("next_checkpoint") or "下一次数据刷新后复核")
    intervals = [
        ("次日开盘", _forecast_range(decision, "open")),
        ("次日最高", _forecast_range(decision, "high")),
        ("次日最低", _forecast_range(decision, "low")),
        ("次日收盘", _forecast_range(decision, "close")),
    ]
    interval_html = "".join(
        f"<div class=\"forecast-metric\"><span>{label} Q10 / Q50 / Q90</span><strong>{html.escape(value)}</strong></div>"
        for label, value in intervals
    )
    warning_html = f"<p class=\"model-warning\">{html.escape(warning)}</p>" if warning else ""
    return (
        "<section class=\"v2-section\">"
        "<div class=\"v2-heading\"><strong>WP V2 人工决策辅助</strong>"
        "<span>目标：提高T+1盈利质量，允许等待与空仓</span></div>"
        "<div class=\"decision-line\">"
        f"<span class=\"decision-action {action_class}\">{html.escape(action)}</span>"
        f"<strong>{html.escape(candidate)}</strong>"
        f"<span>市场：{html.escape(regime_state)} · {regime_score}分</span>"
        f"<span>预测：{html.escape(mode)} · 置信{confidence}</span>"
        "</div>"
        f"<p class=\"decision-reason\">{html.escape(reason or regime_reason)}</p>"
        f"<div class=\"forecast-grid\">{interval_html}</div>"
        "<div class=\"probability-line\">"
        f"<span>收盘盈利概率 <strong>{_probability(decision, 'forecast_profit_probability')}</strong></span>"
        f"<span>触及+3% <strong>{_probability(decision, 'forecast_touch_plus3_probability')}</strong></span>"
        f"<span>触及-3% <strong>{_probability(decision, 'forecast_touch_minus3_probability')}</strong></span>"
        f"<span>下一检查 <strong>{html.escape(next_checkpoint)}</strong></span>"
        "</div>"
        f"{warning_html}"
        f"<p class=\"regime-reason\">市场依据：{html.escape(regime_reason or '-')}</p>"
        "</section>"
    )


def _exit_guidance_panel(frame: pd.DataFrame) -> str:
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            "<tr>"
            + f"<td>{html.escape(str(row.get('ts_code', '')))}</td>"
            + f"<td>{html.escape(str(row.get('name', '')))}</td>"
            + f"<td>{_pct_cell(row.get('open_return_pct', ''))}</td>"
            + f"<td>{_pct_cell(row.get('high_return_pct', ''))}</td>"
            + f"<td>{_pct_cell(row.get('low_return_pct', ''))}</td>"
            + f"<td>{_pct_cell(row.get('current_return_pct', ''))}</td>"
            + f"<td>{html.escape(str(row.get('guidance_action', '')))}</td>"
            + f"<td>{html.escape(str(row.get('guidance_reason', '')))}</td>"
            + f"<td>{html.escape(str(row.get('next_checkpoint', '')))}</td>"
            + "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan=\"9\" class=\"empty\">今天没有需要复核的T+1系统观察记录；实际持仓始终以人工确认为准。</td></tr>")
    return (
        "<section class=\"exit-section\"><div class=\"v2-heading\">"
        "<strong>T+1 人工卖出建议</strong><span>系统记录不等于实际持仓</span></div>"
        "<div class=\"backtest-scroll\"><table class=\"exit-table\">"
        "<thead><tr><th>代码</th><th>名称</th><th>开盘收益</th><th>最高收益</th><th>最低收益</th><th>当前收益</th><th>建议</th><th>依据</th><th>下一检查</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def render_html(
    top50: pd.DataFrame,
    full_rank: pd.DataFrame,
    health: dict,
    output_path: str | Path,
    buy_plan: pd.DataFrame | None = None,
    observation_pool: pd.DataFrame | None = None,
    validation: pd.DataFrame | None = None,
    validation_summary: dict | None = None,
    backtests: list[dict] | None = None,
    decision_support: dict | None = None,
    market_regime: dict | None = None,
    t1_forecasts: pd.DataFrame | None = None,
    exit_guidance: pd.DataFrame | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    buy_plan = buy_plan if buy_plan is not None else pd.DataFrame()
    observation_pool = observation_pool if observation_pool is not None else buy_plan
    validation = validation if validation is not None else pd.DataFrame()
    validation_summary = validation_summary or {}
    sampling_days = list(validation_summary.get("sampling_days") or [])
    validation_model = str(validation_summary.get("buy_model_version") or "")
    validation = scope_validation_table(validation, validation_model, VALIDATION_TRACKING_START_DATE)
    validation_summary = _summary(validation, validation_model, VALIDATION_TRACKING_START_DATE)
    validation_summary["sampling_days"] = sampling_days
    backtests = sorted(backtests or [], key=lambda item: str(item.get("start_date", "")))
    decision_support = decision_support or {}
    market_regime = market_regime or {}
    t1_forecasts = t1_forecasts if t1_forecasts is not None else pd.DataFrame()
    exit_guidance = exit_guidance if exit_guidance is not None else pd.DataFrame()
    backtest_rows = _backtest_rows(backtests)
    decision_support_html = _decision_support_panel(decision_support, market_regime)
    exit_guidance_html = _exit_guidance_panel(exit_guidance)
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
            + f"<td>{_fmt(row.get('tail_profit_score', 0))}</td><td>{'可观察' if bool(row.get('tail_profit_eligible', False)) else '仅排名'}</td>"
            + f"<td>{_fmt(row['sector_strength_score'])}</td><td>{_fmt(row['stock_strength_score'])}</td><td>{_fmt(row['acceptance_score'])}</td>"
            + f"<td>{_fmt(row['p_limitup_t1'])}%</td><td>{_fmt(row['wp_score'])}</td><td>{_fmt(row['model_confidence'])}</td>"
            + f"<td>{html.escape(str(row['signal_level']))}</td><td>{html.escape(str(row['core_reason']))}</td><td>{html.escape(str(row['risk_reason']))}</td><td>{html.escape(str(row['update_time']))}</td></tr>"
        )
    if not rows:
        rows.append("<tr><td colspan=\"18\" class=\"empty\">无符合条件股票</td></tr>")
    buy_rows = []
    for _, row in observation_pool.iterrows():
        observation_status = str(row.get("observation_status") or row.get("portfolio_group") or "观察票")
        status_class = {
            "当前主票": "current-primary",
            "已封板": "sealed",
            "资格复核": "under-review",
        }.get(observation_status, "observed")
        quality_rank = row.get("quality_rank", row.get("buy_rank", ""))
        reason = str(row.get("qualification_reason") or row.get("buy_reason") or "")
        buy_rows.append(
            f"<tr class=\"{status_class}\">"
            + f"<td>{html.escape(str(quality_rank))}</td>"
            + f"<td><span class=\"observation-status {status_class}\">{html.escape(observation_status)}</span></td>"
            + f"<td>{html.escape(str(row.get('rank_change', '-')))}</td>"
            + f"<td>{_clock_text(row.get('first_seen', ''))}</td>"
            + f"<td>{html.escape(str(row.get('ts_code', '')))}</td>"
            + f"<td>{html.escape(str(row.get('name', '')))}</td>"
            + f"<td>{_fmt(row.get('pct_chg', 0))}%</td>"
            + f"<td>{html.escape(str(row.get('sector_name', '')))}</td>"
            + f"<td>{_fmt(row.get('tail_profit_score', 0))}</td>"
            + f"<td>{_fmt(row.get('risk_penalty_score', 0))}</td>"
            + f"<td>{_fmt(row.get('amount_ratio_5d', 0))}</td>"
            + f"<td>{_fmt(row.get('limit_rule_pct', 0), 0)}%</td>"
            + f"<td>{_clock_text(row.get('last_seen', ''))}</td>"
            + f"<td>{html.escape(str(row.get('confirm_before_buy', '')))}</td>"
            + f"<td>{html.escape(str(row.get('reject_if', '')))}</td>"
            + f"<td>{html.escape(reason)}</td>"
            + "</tr>"
        )
    if not buy_rows:
        empty_message = (
            "15:00已收盘，停止生成尾盘名单"
            if str(health.get("tail_window_state") or health.get("tail_observation_state") or "") == "market_closed"
            else "当前无具备资格的尾盘观察票"
        )
        buy_rows.append(f"<tr><td colspan=\"16\" class=\"empty\">{empty_message}</td></tr>")
    validation_overview = _validation_overview(validation_summary)
    validation_days = _validation_days(validation, sampling_days)
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
        ("当前主票数量", str(health.get("buy_plan_count", 0))),
        ("尾盘观察池数量", str(health.get("tail_observation_count", len(observation_pool)))),
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
    refresh_script = r"""  <script>
    (() => {
      const MANIFEST_URL = "/WP/outputs/json/wp_manifest.json";
      const INITIAL_REPORT_REVISION = __INITIAL_REPORT_REVISION__;
      const INITIAL_DATA_REVISION = __INITIAL_DATA_REVISION__;
      const EMBEDDED_MARKET_TIME = __EMBEDDED_MARKET_TIME__;
      const STALE_AFTER_MINUTES = 20;
      const POLL_INTERVAL_MS = 60_000;

      function beijingParts(now) {
        const parts = new Intl.DateTimeFormat("en-US", {
          timeZone: "Asia/Shanghai",
          weekday: "short",
          hour: "2-digit",
          minute: "2-digit",
          hour12: false,
        }).formatToParts(now);
        return Object.fromEntries(parts.map((part) => [part.type, part.value]));
      }

      function isTradingWindow(now) {
        const parts = beijingParts(now);
        if (!["Mon", "Tue", "Wed", "Thu", "Fri"].includes(parts.weekday)) return false;
        const minuteOfDay = Number(parts.hour) * 60 + Number(parts.minute);
        return (minuteOfDay >= 565 && minuteOfDay <= 695) || (minuteOfDay >= 775 && minuteOfDay <= 910);
      }

      function applyTailWindowVisibility(now) {
        const parts = beijingParts(now);
        const minuteOfDay = Number(parts.hour) * 60 + Number(parts.minute);
        const visible = ["Mon", "Tue", "Wed", "Thu", "Fri"].includes(parts.weekday)
          && minuteOfDay >= 860
          && minuteOfDay < 900;
        const section = document.getElementById("buy-plan-section");
        if (section) section.hidden = !visible;
      }

      function parseBeijingTime(value) {
        const match = String(value || "").match(/^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})/);
        if (!match) return null;
        return new Date(`${match[1]}-${match[2]}-${match[3]}T${match[4]}:${match[5]}:${match[6]}+08:00`);
      }

      function applyFreshness(marketDataTime) {
        const now = new Date();
        applyTailWindowVisibility(now);
        const dataTime = parseBeijingTime(marketDataTime);
        const ageMinutes = dataTime ? (now.getTime() - dataTime.getTime()) / 60_000 : Infinity;
        const stale = isTradingWindow(now) && (!dataTime || ageMinutes > STALE_AFTER_MINUTES || ageMinutes < -5);
        const banner = document.getElementById("stale-data-banner");
        const freshness = document.getElementById("live-freshness");
        document.body.classList.toggle("data-stale", stale);
        if (banner) banner.hidden = !stale;
        if (freshness) {
          freshness.textContent = stale ? "；数据过期" : "";
          freshness.className = stale ? "status bad" : "";
        }
      }

      async function checkManifest() {
        try {
          const url = `${MANIFEST_URL}?v=${Date.now()}`;
          const response = await fetch(url, { cache: "no-store" });
          if (!response.ok) throw new Error(`manifest HTTP ${response.status}`);
          const manifest = await response.json();
          const nextDataRevision = String(manifest.data_revision || manifest.market_data_time || "");
          const nextReportRevision = String(manifest.report_revision || manifest.wp_run_time || manifest.latest_update || "");
          applyFreshness(manifest.market_data_time || EMBEDDED_MARKET_TIME);
          const dataChanged = nextDataRevision && nextDataRevision !== INITIAL_DATA_REVISION;
          const reportChanged = nextReportRevision && nextReportRevision !== INITIAL_REPORT_REVISION;
          if (dataChanged || reportChanged) {
            const reportKey = `${nextDataRevision}-${nextReportRevision}`.replace(/\D/g, "");
            const currentUrl = new URL(window.location.href);
            if (currentUrl.searchParams.get("report") !== reportKey) {
              currentUrl.searchParams.set("report", reportKey);
              window.location.replace(currentUrl.toString());
            }
          }
        } catch (_error) {
          applyFreshness(EMBEDDED_MARKET_TIME);
        }
      }

      applyFreshness(EMBEDDED_MARKET_TIME);
      checkManifest();
      window.setInterval(checkManifest, POLL_INTERVAL_MS);
      document.addEventListener("visibilitychange", () => {
        if (!document.hidden) checkManifest();
      });
    })();
  </script>"""
    refresh_script = refresh_script.replace("__INITIAL_REPORT_REVISION__", json.dumps(str(health.get("report_revision") or health.get("wp_run_time") or health.get("data_time") or ""), ensure_ascii=False))
    refresh_script = refresh_script.replace("__INITIAL_DATA_REVISION__", json.dumps(str(health.get("market_data_time") or health.get("data_time") or ""), ensure_ascii=False))
    refresh_script = refresh_script.replace("__EMBEDDED_MARKET_TIME__", json.dumps(str(health.get("market_data_time") or health.get("data_time") or ""), ensure_ascii=False))
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
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
    .summary-toggle-title strong {{ font-size: 16px; line-height: 1.25; font-weight: 700; color: #1d1d1f; }}
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
    .buy-table {{ border-collapse: collapse; min-width: 1720px; width: 100%; font-size: 13px; }}
    .buy-table th, .buy-table td {{ padding: 10px 11px; border-bottom: 1px solid #f1f1f3; text-align: left; vertical-align: top; }}
    .buy-table th {{ background: #fbfbfd; color: #6e6e73; font-weight: 600; white-space: nowrap; }}
    .buy-table td:nth-child(14), .buy-table td:nth-child(15), .buy-table td:nth-child(16) {{ min-width: 150px; line-height: 1.45; }}
    .buy-table tr.current-primary td {{ background: #fff8dc; }}
    .buy-table tr.sealed td {{ background: #fff2f2; }}
    .buy-table tr.under-review td {{ color: #6e6e73; background: #fafafa; }}
    .observation-status {{ font-weight: 700; white-space: nowrap; }}
    .observation-status.current-primary {{ color: #9a5b00; }}
    .observation-status.sealed {{ color: #b42318; }}
    .observation-status.under-review {{ color: #6e6e73; }}
    .observation-status.observed {{ color: #0b7a3b; }}
    .stale-data-banner {{ margin-top: 10px; padding: 12px 14px; border: 1px solid #f1a7a7; border-radius: 6px; color: #b42318; background: #fff2f2; font-size: 13px; font-weight: 600; }}
    .stale-data-banner[hidden] {{ display: none; }}
    .v2-section, .exit-section {{ width: 100%; overflow: hidden; background: #fff; border: 1px solid #d2d2d7; border-radius: 8px; }}
    .v2-heading {{ padding: 16px 20px 12px; display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }}
    .v2-heading strong {{ font-size: 16px; line-height: 1.25; }}
    .v2-heading span {{ color: #6e6e73; font-size: 12px; }}
    .decision-line {{ padding: 16px 20px 10px; display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
    .decision-line > strong {{ font-size: 17px; }}
    .decision-line > span:not(.decision-action) {{ color: #6e6e73; font-size: 13px; }}
    .decision-action {{ padding: 5px 9px; border-radius: 5px; font-size: 13px; font-weight: 700; }}
    .decision-action.buy {{ color: #9f1f1f; background: #fff2f2; }}
    .decision-action.wait {{ color: #7a4b00; background: #fff8e6; }}
    .decision-action.avoid {{ color: #0b6332; background: #edf8f1; }}
    .decision-reason, .model-warning, .regime-reason {{ margin: 0; padding: 0 20px 12px; color: #424245; font-size: 13px; line-height: 1.55; }}
    .model-warning {{ color: #8a4b08; }}
    .regime-reason {{ color: #6e6e73; padding-bottom: 16px; }}
    .forecast-grid {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); border-top: 1px solid #e5e5ea; border-bottom: 1px solid #e5e5ea; }}
    .forecast-metric {{ min-width: 0; padding: 13px 16px; }}
    .forecast-metric + .forecast-metric {{ border-left: 1px solid #e5e5ea; }}
    .forecast-metric span {{ display: block; color: #6e6e73; font-size: 11px; margin-bottom: 5px; }}
    .forecast-metric strong {{ display: block; font-size: 14px; white-space: nowrap; }}
    .probability-line {{ padding: 12px 20px; display: flex; gap: 22px; flex-wrap: wrap; font-size: 12px; color: #6e6e73; }}
    .probability-line strong {{ color: #1d1d1f; }}
    .exit-table {{ border-collapse: collapse; min-width: 1180px; width: 100%; font-size: 12px; }}
    .exit-table th, .exit-table td {{ padding: 10px 12px; border-top: 1px solid #f1f1f3; text-align: left; white-space: nowrap; }}
    .exit-table th {{ color: #6e6e73; background: #fbfbfd; font-weight: 600; }}
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
    .validation-status.missing {{ color: #b42318; }}
    .validation-day-missing {{ cursor: default; background: #fff8f7; }}
    .sampling-missing-note {{ color: #b42318; font-weight: 500; }}
    .validation-detail-wrap {{ padding: 0 16px 14px; background: #fbfbfd; border-bottom: 1px solid #e5e5ea; overflow-x: auto; }}
    .validation-return {{ display: flex; align-items: center; gap: 5px; white-space: nowrap; }}
    .validation-detail-table {{ border-collapse: collapse; min-width: 1180px; width: 100%; font-size: 12px; }}
    .validation-detail-table th, .validation-detail-table td {{ padding: 9px 10px; border-bottom: 1px solid #e5e5ea; text-align: left; white-space: nowrap; }}
    .validation-detail-table th {{ color: #6e6e73; font-weight: 600; }}
    .validation-detail-table tr:last-child td {{ border-bottom: 0; }}
    .section-block {{ background: #fff; border: 1px solid #d2d2d7; border-radius: 8px; padding: 18px 20px; color: #424245; line-height: 1.65; }}
    .section-block > strong {{ font-size: 16px; line-height: 1.25; }}
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
      .forecast-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .forecast-metric:nth-child(3) {{ border-left: 0; border-top: 1px solid #e5e5ea; }}
      .forecast-metric:nth-child(4) {{ border-top: 1px solid #e5e5ea; }}
      .decision-line {{ align-items: flex-start; }}
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
          <span class="summary-toggle-meta">状态：<span class="status {status_cls}">{status_text}</span>；市场数据：<span class="market-time">{market_data_time}</span>；报告更新：{wp_run_time}<span id="live-freshness"></span></span>
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
    {decision_support_html}
    <section id="buy-plan-section">
      <div class="section-block">
        <strong>尾盘观察</strong>
        <div id="stale-data-banner" class="stale-data-banner" role="alert" hidden>市场数据已超过20分钟，名单仍显示，请核对数据时间。</div>
      </div>
      <div id="buy-plan-table-wrap" class="backtest-scroll">
        <table class="buy-table">
          <thead><tr><th>质量排名</th><th>状态</th><th>排名变化</th><th>首次出现</th><th>代码</th><th>名称</th><th>涨幅</th><th>板块</th><th>质量分</th><th>风险分</th><th>5日量能比</th><th>涨跌停规则</th><th>最近确认</th><th>14:50确认</th><th>放弃</th><th>理由</th></tr></thead>
          <tbody>{''.join(buy_rows)}</tbody>
        </table>
      </div>
    </section>
    {exit_guidance_html}
    <section>
      <div class="table-wrap">
        <table class="rank-table">
          <thead><tr><th>排名</th><th>代码</th><th>名称</th><th>当前价/收盘价</th><th>今日涨幅</th><th>所属板块</th><th>尾盘收益分</th><th>观察资格</th><th>板块强度</th><th>个股强度</th><th>承接分</th><th>次日涨停概率</th><th>WP评分</th><th>模型置信度</th><th>信号等级</th><th>核心理由</th><th>风险提示</th><th>更新时间</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>
    <section class="validation-section">
      <div class="validation-heading">
        <strong>14:20–14:50 主票累计验证</strong>
        <span>自 2026-07-15 起；保留窗口内每次实际出现的主票</span>
      </div>
      <div class="validation-kpis">{validation_overview}</div>
      <div class="validation-days">
        <div class="validation-day-list">
          <div class="validation-day-header validation-grid"><span>计划日</span><span>验证日</span><span>观察记录</span><span>次日收益（开 / 高 / 收）</span><span>上涨</span><span>触及涨停</span><span>状态</span><span></span></div>
          {validation_days}
        </div>
      </div>
    </section>
    <section class="backtest-section">
      <div class="validation-heading">
        <strong>模型回测验证</strong>
        <span>收盘日线代理；不替代 14:35 真实快照</span>
      </div>
      <div class="table-wrap">
        <table class="backtest-table">
          <thead><tr><th>区间</th><th>交易日</th><th>观察日</th><th>样本</th><th>上涨率</th><th>触及涨停</th><th>次日开盘</th><th>次日最高</th><th>次日收盘</th><th>累计收盘</th><th>AUC</th><th>原始数据</th></tr></thead>
          <tbody>{backtest_rows}</tbody>
        </table>
      </div>
    </section>
  </main>
{refresh_script}
</body>
</html>
"""
    output.write_text(page, encoding="utf-8")
