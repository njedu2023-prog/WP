from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .candidate_filter import filter_candidates
from .feature_engineering import add_feature_scores
from .ranking import rank_candidates
from .scoring_model import add_scores
from .utils import ensure_dir, write_json


RAW_BASE_URL = "https://raw.githubusercontent.com/njedu2023-prog/a-share-top3-data/main"
API_BASE_URL = "https://api.github.com/repos/njedu2023-prog/a-share-top3-data/contents"
SCHEMA = [
    "trade_date", "update_time", "ts_code", "name", "price", "open", "high", "low",
    "close", "pre_close", "pct_chg", "amount", "volume", "turnover_rate",
    "volume_ratio", "sector_name", "sector_rank", "sector_limitup_count",
    "sector_gt6_count", "sector_amount_ratio", "pre_day_limitup", "today_limitup",
    "today_limit_up_price", "prev_limit_up_price", "ret_5d", "ret_20d",
]


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    daily_summary: pd.DataFrame
    summary: dict


def _date_key(value: str) -> str:
    text = str(value).strip().replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"invalid date: {value}")
    return text


def _date_dash(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def _to_num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _read_remote_csv(path: str, cache_root: Path | None = None) -> pd.DataFrame:
    if cache_root is not None:
        cache_file = cache_root / path
        if cache_file.exists() and cache_file.stat().st_size > 0:
            return pd.read_csv(cache_file, encoding="utf-8-sig")
    url = f"{RAW_BASE_URL}/{path}"
    try:
        with urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8-sig")
        if cache_root is not None:
            cache_file = cache_root / path
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(text, encoding="utf-8")
        return pd.read_csv(StringIO(text))
    except (HTTPError, URLError, OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _available_trade_dates(start_date: str, end_date: str) -> list[str]:
    start = _date_key(start_date)
    end = _date_key(end_date)
    years = range(int(start[:4]), int(end[:4]) + 1)
    dates: list[str] = []
    for year in years:
        url = f"{API_BASE_URL}/data/raw/{year}?ref=main"
        try:
            req = Request(url, headers={"Accept": "application/vnd.github+json"})
            with urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            dates.extend(item["name"] for item in payload if item.get("type") == "dir")
        except (HTTPError, URLError, OSError, json.JSONDecodeError):
            continue
    dates = sorted(date for date in dates if start <= date <= end)
    if dates:
        return dates

    cur = datetime.strptime(start, "%Y%m%d")
    last = datetime.strptime(end, "%Y%m%d")
    while cur <= last:
        if cur.weekday() < 5:
            dates.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return dates


def _next_available_date(trade_date: str) -> str | None:
    start = (datetime.strptime(trade_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
    end = (datetime.strptime(trade_date, "%Y%m%d") + timedelta(days=14)).strftime("%Y%m%d")
    dates = _available_trade_dates(start, end)
    return dates[0] if dates else None


def _previous_available_date(trade_date: str) -> str | None:
    start = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=14)).strftime("%Y%m%d")
    end = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
    dates = _available_trade_dates(start, end)
    return dates[-1] if dates else None


def _limitup_codes(trade_date: str, cache_root: Path | None = None) -> set[str]:
    frame = _read_remote_csv(f"data/raw/{trade_date[:4]}/{trade_date}/limit_list_d.csv", cache_root)
    if frame.empty or "ts_code" not in frame.columns:
        return set()
    return set(frame["ts_code"].dropna().astype(str).str.strip())


def build_rank_input_for_date(trade_date: str, cache_root: Path | None = None) -> pd.DataFrame:
    trade_date = _date_key(trade_date)
    base = f"data/raw/{trade_date[:4]}/{trade_date}"
    daily = _read_remote_csv(f"{base}/daily.csv", cache_root)
    if daily.empty:
        return pd.DataFrame(columns=SCHEMA)

    daily_basic = _read_remote_csv(f"{base}/daily_basic.csv", cache_root)
    stock_basic = _read_remote_csv(f"{base}/stock_basic.csv", cache_root)
    stk_limit = _read_remote_csv(f"{base}/stk_limit.csv", cache_root)
    limit_list = _read_remote_csv(f"{base}/limit_list_d.csv", cache_root)
    hot_boards = _read_remote_csv(f"{base}/hot_boards.csv", cache_root)
    intraday = _read_remote_csv(f"{base}/intraday_features.csv", cache_root)

    out = daily.copy()
    out["ts_code"] = out["ts_code"].astype(str).str.strip()
    out["trade_date"] = out.get("trade_date", trade_date).fillna(trade_date).astype(str)

    for extra, cols in [
        (daily_basic, ["ts_code", "turnover_rate", "volume_ratio", "total_mv", "float_mv"]),
        (stock_basic, ["ts_code", "name", "industry", "market", "list_date"]),
        (stk_limit, ["ts_code", "up_limit", "down_limit"]),
        (intraday, ["ts_code", "limit_touch_count", "open_board_count", "limitup_quality_score", "intraday_risk_score"]),
    ]:
        if not extra.empty:
            keep = [col for col in cols if col in extra.columns]
            out = out.merge(extra[keep].drop_duplicates("ts_code"), on="ts_code", how="left")

    close = _to_num(out, "close")
    pct_chg = _to_num(out, "pct_chg")
    out["pre_close"] = np.where((1 + pct_chg / 100) > 0, close / (1 + pct_chg / 100), np.nan)
    out["price"] = close
    out["volume"] = _to_num(out, "vol")
    out["amount"] = _to_num(out, "amount") * 1000
    out["sector_name"] = out.get("industry", pd.Series("未分类", index=out.index)).fillna("未分类").astype(str)

    current_limit_codes = set()
    if not limit_list.empty and "ts_code" in limit_list.columns:
        current_limit_codes = set(limit_list["ts_code"].dropna().astype(str).str.strip())
    prev_date = _previous_available_date(trade_date)
    prev_limit_codes = _limitup_codes(prev_date, cache_root) if prev_date else set()
    up_limit = _to_num(out, "up_limit")
    out["today_limit_up_price"] = up_limit
    out["prev_limit_up_price"] = np.nan
    out["today_limitup"] = np.where(out["ts_code"].isin(current_limit_codes) | ((up_limit > 0) & (close >= up_limit * 0.999)), 1, 0)
    out["pre_day_limitup"] = np.where(out["ts_code"].isin(prev_limit_codes), 1, 0)

    sector_gt6 = out.assign(_gt6=pct_chg > 6).groupby("sector_name")["_gt6"].sum()
    sector_amount = out.groupby("sector_name")["amount"].sum()
    amount_median = float(sector_amount.median()) if len(sector_amount) else 0.0
    sector_metrics = pd.DataFrame({
        "sector_name": sector_gt6.index,
        "sector_gt6_count": sector_gt6.values,
        "sector_amount_ratio": [(sector_amount.get(name, 0.0) / amount_median) if amount_median > 0 else 1.0 for name in sector_gt6.index],
    })
    if not hot_boards.empty and "industry" in hot_boards.columns:
        boards = hot_boards.rename(columns={"industry": "sector_name", "rank": "sector_rank", "limit_up_count": "sector_limitup_count"})
        keep = [col for col in ["sector_name", "sector_rank", "sector_limitup_count"] if col in boards.columns]
        sector_metrics = sector_metrics.merge(boards[keep].drop_duplicates("sector_name"), on="sector_name", how="left")
    out = out.merge(sector_metrics, on="sector_name", how="left")

    out["sector_rank"] = _to_num(out, "sector_rank", 99)
    out["sector_limitup_count"] = _to_num(out, "sector_limitup_count", 0)
    out["sector_gt6_count"] = _to_num(out, "sector_gt6_count", 0)
    out["sector_amount_ratio"] = _to_num(out, "sector_amount_ratio", 1)
    out["ret_5d"] = pct_chg
    out["ret_20d"] = pct_chg
    out["update_time"] = _date_dash(trade_date)
    for col in SCHEMA:
        if col not in out.columns:
            out[col] = np.nan
    return out[SCHEMA]


def build_label_frame(today_rank: pd.DataFrame, next_day: pd.DataFrame) -> pd.DataFrame:
    if today_rank.empty:
        return pd.DataFrame()
    merged = today_rank.merge(next_day, on="ts_code", how="left", suffixes=("", "_next"))
    if "next_day_limitup_price" in merged and "next_day_high" in merged:
        merged["label_t1_limitup"] = (merged["next_day_high"] >= merged["next_day_limitup_price"] * 0.999).astype(int)
    else:
        merged["label_t1_limitup"] = 0
    return merged


def add_t1_labels(today_rank: pd.DataFrame, trade_date: str, cache_root: Path | None = None) -> pd.DataFrame:
    if today_rank.empty:
        return today_rank.copy()
    next_date = _next_available_date(_date_key(trade_date))
    out = today_rank.copy()
    out["next_trade_date"] = next_date or ""
    out["label_t1_limitup"] = 0
    if not next_date:
        return out
    base = f"data/raw/{next_date[:4]}/{next_date}"
    daily_next = _read_remote_csv(f"{base}/daily.csv", cache_root)
    limit_next = _read_remote_csv(f"{base}/stk_limit.csv", cache_root)
    if daily_next.empty or limit_next.empty:
        return out
    next_frame = daily_next[["ts_code", "high"]].merge(limit_next[["ts_code", "up_limit"]], on="ts_code", how="left")
    next_frame["ts_code"] = next_frame["ts_code"].astype(str).str.strip()
    next_frame["next_high"] = pd.to_numeric(next_frame["high"], errors="coerce")
    next_frame["next_up_limit"] = pd.to_numeric(next_frame["up_limit"], errors="coerce")
    merged = out.merge(next_frame[["ts_code", "next_high", "next_up_limit"]], on="ts_code", how="left")
    merged["label_t1_limitup"] = ((merged["next_up_limit"] > 0) & (merged["next_high"] >= merged["next_up_limit"] * 0.999)).astype(int)
    return merged


def _hit_rate(frame: pd.DataFrame, n: int) -> float:
    if frame.empty or "label_t1_limitup" not in frame.columns:
        return 0.0
    top = frame.head(n)
    if top.empty:
        return 0.0
    return round(float(top["label_t1_limitup"].mean()), 4)


def _period_hit_rate(frame: pd.DataFrame, n: int) -> float:
    if frame.empty or "label_t1_limitup" not in frame.columns or "backtest_trade_date" not in frame.columns:
        return 0.0
    samples = []
    for _, group in frame.sort_values(["backtest_trade_date", "rank"]).groupby("backtest_trade_date"):
        samples.append(group.head(n))
    if not samples:
        return 0.0
    combined = pd.concat(samples, ignore_index=True)
    if combined.empty:
        return 0.0
    return round(float(combined["label_t1_limitup"].mean()), 4)


def run_backtest(start_date: str, end_date: str, output_root: str | Path, top_n: int = 50) -> BacktestResult:
    start = _date_key(start_date)
    end = _date_key(end_date)
    output_root = Path(output_root)
    cache_root = output_root.parent / "data" / "cache" / "history"
    dates = _available_trade_dates(start, end)
    all_trades: list[pd.DataFrame] = []
    daily_rows: list[dict] = []

    for trade_date in dates:
        rank_input = build_rank_input_for_date(trade_date, cache_root)
        candidates = filter_candidates(rank_input)
        scored = add_scores(add_feature_scores(candidates))
        top50, full_rank = rank_candidates(scored, _date_dash(trade_date), top_n=top_n)
        labeled = add_t1_labels(top50, trade_date, cache_root)
        if not labeled.empty:
            labeled["backtest_trade_date"] = trade_date
            all_trades.append(labeled)
        daily_rows.append({
            "trade_date": trade_date,
            "raw_count": int(len(rank_input)),
            "candidate_count": int(len(candidates)),
            "top_count": int(len(top50)),
            "hit_top10": _hit_rate(labeled, 10),
            "hit_top20": _hit_rate(labeled, 20),
            "hit_top50": _hit_rate(labeled, 50),
            "next_trade_date": str(labeled["next_trade_date"].iloc[0]) if not labeled.empty and "next_trade_date" in labeled.columns else "",
        })

    trades = pd.concat(all_trades, ignore_index=True, sort=False) if all_trades else pd.DataFrame()
    daily_summary = pd.DataFrame(daily_rows)
    summary = {
        "mode": "backtest",
        "start_date": start,
        "end_date": end,
        "trade_days": int(len(dates)),
        "trade_count": int(len(trades)),
        "hit_top10": _period_hit_rate(trades, 10),
        "hit_top20": _period_hit_rate(trades, 20),
        "hit_top50": round(float(trades["label_t1_limitup"].mean()), 4) if not trades.empty and "label_t1_limitup" in trades.columns else 0.0,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    out_dir = output_root / "backtests" / f"{start}_{end}"
    ensure_dir(out_dir)
    trades.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    daily_summary.to_csv(out_dir / "daily_summary.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "summary.json", summary)
    write_json(output_root / "json" / "wp_backtest_latest.json", summary)
    render_backtest_html(summary, daily_summary, trades, output_root / "html_reports" / "backtest_latest.html")
    return BacktestResult(trades=trades, daily_summary=daily_summary, summary=summary)


def render_backtest_html(summary: dict, daily_summary: pd.DataFrame, trades: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    daily_rows = "".join(
        f"<tr><td>{row.trade_date}</td><td>{row.raw_count}</td><td>{row.candidate_count}</td><td>{row.top_count}</td><td>{row.hit_top10:.2%}</td><td>{row.hit_top20:.2%}</td><td>{row.hit_top50:.2%}</td><td>{row.next_trade_date}</td></tr>"
        for row in daily_summary.itertuples(index=False)
    ) or "<tr><td colspan='8' class='empty'>无历史数据</td></tr>"
    trade_preview = trades.sort_values(["backtest_trade_date", "rank"]).head(100) if not trades.empty else pd.DataFrame()
    trade_rows = "".join(
        f"<tr><td>{row.backtest_trade_date}</td><td>{row.rank}</td><td>{row.ts_code}</td><td>{row.name}</td><td>{float(row.p_limitup_t1):.2f}%</td><td>{float(row.wp_score):.2f}</td><td>{int(row.label_t1_limitup)}</td></tr>"
        for row in trade_preview.itertuples(index=False)
    ) or "<tr><td colspan='7' class='empty'>无交易样本</td></tr>"
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WP 历史区间测试</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #f5f7fb; }}
    header {{ padding: 18px 24px; background: #16213e; color: white; }}
    main {{ padding: 18px 24px 32px; }}
    .panel {{ background: white; border: 1px solid #d8deea; border-radius: 8px; padding: 14px 16px; margin-bottom: 18px; }}
    .metric {{ display: flex; gap: 18px; flex-wrap: wrap; }}
    table {{ border-collapse: collapse; min-width: 900px; width: 100%; font-size: 13px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #edf0f5; text-align: left; white-space: nowrap; }}
    th {{ background: #eef3fb; }}
    .table-wrap {{ overflow-x: auto; }}
    .empty {{ text-align: center; color: #6b7280; padding: 24px; }}
  </style>
</head>
<body>
  <header><h1>WP 历史区间测试</h1><div>{summary["start_date"]} 至 {summary["end_date"]}</div></header>
  <main>
    <section class="panel metric">
      <div>交易日：<strong>{summary["trade_days"]}</strong></div>
      <div>样本数：<strong>{summary["trade_count"]}</strong></div>
      <div>Top10 命中：<strong>{summary["hit_top10"]:.2%}</strong></div>
      <div>Top20 命中：<strong>{summary["hit_top20"]:.2%}</strong></div>
      <div>Top50 命中：<strong>{summary["hit_top50"]:.2%}</strong></div>
    </section>
    <section class="panel"><h2>每日汇总</h2><div class="table-wrap"><table><thead><tr><th>日期</th><th>原始数</th><th>候选数</th><th>Top数</th><th>Top10</th><th>Top20</th><th>Top50</th><th>下一交易日</th></tr></thead><tbody>{daily_rows}</tbody></table></div></section>
    <section class="panel"><h2>样本预览</h2><div class="table-wrap"><table><thead><tr><th>日期</th><th>排名</th><th>代码</th><th>名称</th><th>概率</th><th>评分</th><th>T+1涨停</th></tr></thead><tbody>{trade_rows}</tbody></table></div></section>
  </main>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")
