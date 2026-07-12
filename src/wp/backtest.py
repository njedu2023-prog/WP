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
from .buy_decision import build_buy_decision
from .feature_engineering import add_feature_scores
from .ranking import build_ranked_pool, rank_candidates
from .scoring_model import MODEL_VERSION, add_scores
from .utils import ensure_dir, write_json


RAW_BASE_URL = "https://raw.githubusercontent.com/njedu2023-prog/a-share-top3-data/main"
API_BASE_URL = "https://api.github.com/repos/njedu2023-prog/a-share-top3-data/contents"
SCHEMA = [
    "trade_date", "update_time", "ts_code", "name", "price", "open", "high", "low",
    "close", "pre_close", "pct_chg", "amount", "volume", "turnover_rate",
    "volume_ratio", "sector_name", "sector_rank", "sector_limitup_count",
    "sector_gt6_count", "sector_amount_ratio", "sector_net_inflow",
    "sector_turnover", "sector_hot_score", "pre_day_limitup", "today_limitup",
    "today_limit_up_price", "prev_limit_up_price", "ret_5d", "ret_20d",
    "ret_3d", "ret_10d", "amount_ratio_5d", "amount_ratio_20d", "turnover_rate_5d_avg",
    "close_position", "intraday_pullback_pct", "open_to_close_pct",
    "gap_open_pct", "amplitude", "high_20d_break", "platform_break_20d",
    "stage_high_20d", "ma5_position", "ma10_position", "ma20_position",
    "intraday_vwap_position", "late_pullback_pct", "late_price_change_pct",
    "late_volume_ratio", "tail_lift_flag", "dragon_tiger_flag", "dragon_tiger_net_rate",
    "dragon_tiger_reason", "limit_touch_count", "open_board_count",
    "limitup_quality_score", "intraday_risk_score", "announcement_flag",
    "hot_topic_flag", "auction_price", "auction_vol", "auction_amount",
    "auction_pct_chg", "auction_amount_ratio", "auction_strength_score",
    "realtime_source", "stock_age_days", "suspended_flag", "delist_flag",
    "data_quality_flag",
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


def _add_history_features(out: pd.DataFrame, trade_date: str, cache_root: Path | None = None) -> pd.DataFrame:
    start = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=45)).strftime("%Y%m%d")
    dates = _available_trade_dates(start, trade_date)[-24:]
    frames = []
    for date in dates:
        frame = _read_remote_csv(f"data/raw/{date[:4]}/{date}/daily.csv", cache_root)
        if frame.empty:
            continue
        keep = [col for col in ["ts_code", "trade_date", "close", "high", "low", "amount", "pct_chg"] if col in frame.columns]
        frame = frame[keep].copy()
        frame["trade_date"] = frame.get("trade_date", date)
        frames.append(frame)
    if not frames:
        return out
    hist = pd.concat(frames, ignore_index=True, sort=False)
    hist["ts_code"] = hist["ts_code"].astype(str).str.strip()
    hist["trade_date"] = hist["trade_date"].astype(str).str.replace("-", "", regex=False)
    for col in ["close", "high", "low", "amount", "pct_chg"]:
        if col in hist.columns:
            hist[col] = pd.to_numeric(hist[col], errors="coerce")
    hist = hist.sort_values(["ts_code", "trade_date"])
    current = hist[hist["trade_date"] == trade_date].set_index("ts_code")
    prev = hist[hist["trade_date"] < trade_date]
    if current.empty or prev.empty:
        return out

    rows = []
    for ts_code, group in prev.groupby("ts_code"):
        group = group.sort_values("trade_date")
        cur = current.loc[ts_code] if ts_code in current.index else None
        if cur is None:
            continue
        close = float(cur.get("close", np.nan))
        high = float(cur.get("high", np.nan))
        amount = float(cur.get("amount", np.nan))
        tail3 = group["close"].tail(3)
        amount_5 = group["amount"].tail(5).mean()
        amount_20 = group["amount"].tail(20).mean()
        close_3 = group["close"].tail(3).iloc[0] if len(tail3) else np.nan
        close_5 = group["close"].tail(5).iloc[0] if len(group.tail(5)) else np.nan
        close_10 = group["close"].tail(10).iloc[0] if len(group.tail(10)) else np.nan
        close_20 = group["close"].tail(20).iloc[0] if len(group.tail(20)) else np.nan
        ma5 = group["close"].tail(5).mean()
        ma10 = group["close"].tail(10).mean()
        ma20 = group["close"].tail(20).mean()
        high_20 = group["high"].tail(20).max()
        close_high_20 = group["close"].tail(20).max()
        rows.append({
            "ts_code": ts_code,
            "amount_ratio_5d": amount / amount_5 if amount_5 and amount_5 > 0 else np.nan,
            "amount_ratio_20d": amount / amount_20 if amount_20 and amount_20 > 0 else np.nan,
            "ret_3d": (close / close_3 - 1) * 100 if close_3 and close_3 > 0 else np.nan,
            "ret_5d": (close / close_5 - 1) * 100 if close_5 and close_5 > 0 else np.nan,
            "ret_10d": (close / close_10 - 1) * 100 if close_10 and close_10 > 0 else np.nan,
            "ret_20d": (close / close_20 - 1) * 100 if close_20 and close_20 > 0 else np.nan,
            "ma5_position": (close / ma5 - 1) * 100 if ma5 and ma5 > 0 else np.nan,
            "ma10_position": (close / ma10 - 1) * 100 if ma10 and ma10 > 0 else np.nan,
            "ma20_position": (close / ma20 - 1) * 100 if ma20 and ma20 > 0 else np.nan,
            "stage_high_20d": high_20,
            "high_20d_break": int(high >= high_20 * 0.999) if high_20 and high_20 > 0 else 0,
            "platform_break_20d": int(close >= close_high_20 * 1.005) if close_high_20 and close_high_20 > 0 else 0,
        })
    if not rows:
        return out
    return out.merge(pd.DataFrame(rows), on="ts_code", how="left")


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
    top_list = _read_remote_csv(f"{base}/top_list.csv", cache_root)

    out = daily.copy()
    out["ts_code"] = out["ts_code"].astype(str).str.strip()
    out["trade_date"] = out.get("trade_date", trade_date).fillna(trade_date).astype(str)

    for extra, cols in [
        (daily_basic, ["ts_code", "turnover_rate", "volume_ratio", "total_mv", "float_mv"]),
        (stock_basic, ["ts_code", "name", "industry", "market", "list_date"]),
        (stk_limit, ["ts_code", "up_limit", "down_limit"]),
        (intraday, [
            "ts_code", "limit_touch_count", "open_board_count", "limitup_quality_score",
            "intraday_risk_score", "late_volume_ratio", "late_price_weakness",
            "max_drawdown_after_limit", "intraday_vwap_position",
        ]),
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
    if "list_date" in out.columns:
        list_date = pd.to_datetime(out["list_date"].astype(str), format="%Y%m%d", errors="coerce")
        trade_dt = pd.to_datetime(trade_date, format="%Y%m%d", errors="coerce")
        out["stock_age_days"] = (trade_dt - list_date).dt.days
    else:
        out["stock_age_days"] = np.nan
    out["delist_flag"] = out.get("name", pd.Series("", index=out.index)).fillna("").astype(str).str.contains("退|退市", regex=True).astype(int)
    out["suspended_flag"] = np.where((close <= 0) | (_to_num(out, "amount") <= 0) | (_to_num(out, "vol") <= 0), 1, 0)

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
    out["close_position"] = np.where(_to_num(out, "high") > _to_num(out, "low"), (close - _to_num(out, "low")) / (_to_num(out, "high") - _to_num(out, "low")) * 100, 50)
    out["intraday_pullback_pct"] = np.where(close > 0, (_to_num(out, "high") / close - 1) * 100, 0)
    out["open_to_close_pct"] = np.where(_to_num(out, "open") > 0, (close / _to_num(out, "open") - 1) * 100, 0)
    out["gap_open_pct"] = np.where(_to_num(out, "pre_close") > 0, (_to_num(out, "open") / _to_num(out, "pre_close") - 1) * 100, 0)
    out["amplitude"] = np.where(_to_num(out, "pre_close") > 0, (_to_num(out, "high") - _to_num(out, "low")) / _to_num(out, "pre_close") * 100, 0)
    out["late_pullback_pct"] = _to_num(out, "max_drawdown_after_limit", np.nan).fillna(out["intraday_pullback_pct"])
    out["late_price_change_pct"] = -_to_num(out, "late_price_weakness", 0)
    out["late_volume_ratio"] = _to_num(out, "late_volume_ratio", 1)
    typical_price = (_to_num(out, "high") + _to_num(out, "low") + close) / 3
    out["intraday_vwap_position"] = _to_num(out, "intraday_vwap_position", np.nan)
    out["intraday_vwap_position"] = out["intraday_vwap_position"].where(out["intraday_vwap_position"].notna(), np.where(typical_price > 0, (close / typical_price - 1) * 100, 0))
    out["tail_lift_flag"] = np.where((out["late_volume_ratio"] >= 1.8) & (out["close_position"] >= 82) & (out["open_to_close_pct"] >= 3), 1, 0)
    out["announcement_flag"] = 0

    sector_gt6 = out.assign(_gt6=pct_chg > 8).groupby("sector_name")["_gt6"].sum()
    sector_amount = out.groupby("sector_name")["amount"].sum()
    sector_turnover = out.groupby("sector_name")["turnover_rate"].mean() if "turnover_rate" in out.columns else pd.Series(dtype="float64")
    amount_median = float(sector_amount.median()) if len(sector_amount) else 0.0
    sector_metrics = pd.DataFrame({
        "sector_name": sector_gt6.index,
        "sector_gt6_count": sector_gt6.values,
        "sector_amount_ratio": [(sector_amount.get(name, 0.0) / amount_median) if amount_median > 0 else 1.0 for name in sector_gt6.index],
        "sector_turnover": [sector_turnover.get(name, np.nan) for name in sector_gt6.index],
    })
    if not hot_boards.empty and "industry" in hot_boards.columns:
        boards = hot_boards.rename(columns={"industry": "sector_name", "rank": "sector_rank", "limit_up_count": "sector_limitup_count"})
        keep = [col for col in ["sector_name", "sector_rank", "sector_limitup_count"] if col in boards.columns]
        sector_metrics = sector_metrics.merge(boards[keep].drop_duplicates("sector_name"), on="sector_name", how="left")
    out = out.merge(sector_metrics, on="sector_name", how="left")
    out["hot_topic_flag"] = np.where(_to_num(out, "sector_rank", 99) <= 10, 1, 0)
    out["sector_net_inflow"] = 0
    out["sector_hot_score"] = np.maximum(0, 100 - _to_num(out, "sector_rank", 99) * 4) + _to_num(out, "sector_gt6_count", 0) * 5 + _to_num(out, "sector_limitup_count", 0) * 8

    if not top_list.empty and "ts_code" in top_list.columns:
        top = top_list.copy()
        top["dragon_tiger_flag"] = 1
        top = top.rename(columns={"net_rate": "dragon_tiger_net_rate", "reason": "dragon_tiger_reason"})
        keep = [c for c in ["ts_code", "dragon_tiger_flag", "dragon_tiger_net_rate", "dragon_tiger_reason"] if c in top.columns]
        out = out.merge(top[keep].drop_duplicates("ts_code"), on="ts_code", how="left")

    out["sector_rank"] = _to_num(out, "sector_rank", 99)
    out["sector_limitup_count"] = _to_num(out, "sector_limitup_count", 0)
    out["sector_gt6_count"] = _to_num(out, "sector_gt6_count", 0)
    out["sector_amount_ratio"] = _to_num(out, "sector_amount_ratio", 1)
    out = _add_history_features(out, trade_date, cache_root)
    out["ret_20d"] = _to_num(out, "ret_20d", 0).replace(0, np.nan).fillna(pct_chg)
    out["ret_5d"] = _to_num(out, "ret_5d", 0).replace(0, np.nan).fillna(pct_chg)
    out["ret_3d"] = _to_num(out, "ret_3d", 0).replace(0, np.nan).fillna(out["ret_5d"])
    out["ret_10d"] = _to_num(out, "ret_10d", 0).replace(0, np.nan).fillna(out["ret_20d"])
    out["amount_ratio_5d"] = _to_num(out, "amount_ratio_5d", 1).replace([np.inf, -np.inf], np.nan).fillna(_to_num(out, "volume_ratio", 1))
    out["amount_ratio_20d"] = _to_num(out, "amount_ratio_20d", 1).replace([np.inf, -np.inf], np.nan).fillna(out["amount_ratio_5d"])
    out["turnover_rate_5d_avg"] = _to_num(out, "turnover_rate_5d_avg", 0)
    out["high_20d_break"] = _to_num(out, "high_20d_break", 0)
    out["platform_break_20d"] = _to_num(out, "platform_break_20d", 0)
    out["stage_high_20d"] = _to_num(out, "stage_high_20d", 0)
    out["ma5_position"] = _to_num(out, "ma5_position", 0)
    out["ma10_position"] = _to_num(out, "ma10_position", 0)
    out["ma20_position"] = _to_num(out, "ma20_position", 0)
    out["dragon_tiger_flag"] = _to_num(out, "dragon_tiger_flag", 0)
    out["dragon_tiger_net_rate"] = _to_num(out, "dragon_tiger_net_rate", 0)
    out["sector_net_inflow"] = _to_num(out, "sector_net_inflow", 0)
    out["sector_turnover"] = _to_num(out, "sector_turnover", 0)
    out["sector_hot_score"] = _to_num(out, "sector_hot_score", 0).clip(0, 100)
    out["announcement_flag"] = _to_num(out, "announcement_flag", 0)
    out["hot_topic_flag"] = _to_num(out, "hot_topic_flag", 0)
    required_core = ["ts_code", "close", "pre_close", "pct_chg", "amount", "today_limit_up_price"]
    out["data_quality_flag"] = np.where(out[required_core].isna().any(axis=1), 1, 0)
    out.loc[(_to_num(out, "close") <= 0) | (_to_num(out, "pre_close") <= 0) | (_to_num(out, "amount") <= 0) | (_to_num(out, "today_limit_up_price") <= 0), "data_quality_flag"] = 1
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
        available = merged["next_day_high"].notna() & merged["next_day_limitup_price"].notna()
        merged["label_available"] = available
        merged["label_t1_limitup"] = np.where(
            available,
            (merged["next_day_high"] >= merged["next_day_limitup_price"] * 0.999).astype(int),
            np.nan,
        )
    else:
        merged["label_available"] = False
        merged["label_t1_limitup"] = np.nan
    return merged


def add_t1_labels(today_rank: pd.DataFrame, trade_date: str, cache_root: Path | None = None) -> pd.DataFrame:
    if today_rank.empty:
        return today_rank.copy()
    next_date = _next_available_date(_date_key(trade_date))
    out = today_rank.copy()
    out["next_trade_date"] = next_date or ""
    out["label_t1_limitup"] = np.nan
    out["label_available"] = False
    out["next_day_open"] = np.nan
    out["next_day_high"] = np.nan
    out["next_day_close"] = np.nan
    out["next_day_low"] = np.nan
    out["next_day_limitup_price"] = np.nan
    out["next_day_open_pct"] = np.nan
    out["next_day_max_pct"] = np.nan
    out["next_day_close_pct"] = np.nan
    out["next_day_drawdown_pct"] = np.nan
    if not next_date:
        return out
    base = f"data/raw/{next_date[:4]}/{next_date}"
    daily_next = _read_remote_csv(f"{base}/daily.csv", cache_root)
    limit_next = _read_remote_csv(f"{base}/stk_limit.csv", cache_root)
    if daily_next.empty or limit_next.empty:
        return out
    keep_daily = [col for col in ["ts_code", "open", "high", "close", "low"] if col in daily_next.columns]
    next_frame = daily_next[keep_daily].merge(limit_next[["ts_code", "up_limit"]], on="ts_code", how="left")
    next_frame["ts_code"] = next_frame["ts_code"].astype(str).str.strip()
    next_frame["next_open"] = pd.to_numeric(next_frame["open"], errors="coerce") if "open" in next_frame.columns else np.nan
    next_frame["next_high"] = pd.to_numeric(next_frame["high"], errors="coerce")
    next_frame["next_close"] = pd.to_numeric(next_frame["close"], errors="coerce") if "close" in next_frame.columns else np.nan
    next_frame["next_low"] = pd.to_numeric(next_frame["low"], errors="coerce") if "low" in next_frame.columns else np.nan
    next_frame["next_up_limit"] = pd.to_numeric(next_frame["up_limit"], errors="coerce")
    merged = out.merge(next_frame[["ts_code", "next_open", "next_high", "next_close", "next_low", "next_up_limit"]], on="ts_code", how="left")
    available = (merged["next_up_limit"] > 0) & merged["next_high"].notna()
    merged["label_available"] = available
    merged["label_t1_limitup"] = np.where(
        available,
        (merged["next_high"] >= merged["next_up_limit"] * 0.999).astype(int),
        np.nan,
    )
    price = pd.to_numeric(merged.get("price", merged.get("close", 0)), errors="coerce")
    merged["next_day_open"] = merged["next_open"]
    merged["next_day_high"] = merged["next_high"]
    merged["next_day_close"] = merged["next_close"]
    merged["next_day_low"] = merged["next_low"]
    merged["next_day_limitup_price"] = merged["next_up_limit"]
    merged["next_day_open_pct"] = np.where(price > 0, (merged["next_day_open"] / price - 1) * 100, np.nan)
    merged["next_day_max_pct"] = np.where(price > 0, (merged["next_day_high"] / price - 1) * 100, np.nan)
    merged["next_day_close_pct"] = np.where(price > 0, (merged["next_day_close"] / price - 1) * 100, np.nan)
    merged["next_day_drawdown_pct"] = np.where(price > 0, (merged["next_day_low"] / price - 1) * 100, np.nan)
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


def _period_top_samples(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if frame.empty or "backtest_trade_date" not in frame.columns:
        return pd.DataFrame()
    return pd.concat(
        [group.sort_values("rank").head(n) for _, group in frame.groupby("backtest_trade_date")],
        ignore_index=True,
    )


def _auc_score(frame: pd.DataFrame) -> float | None:
    if frame.empty or "label_t1_limitup" not in frame.columns or "p_limitup_t1" not in frame.columns:
        return None
    y = pd.to_numeric(frame["label_t1_limitup"], errors="coerce")
    score = pd.to_numeric(frame["p_limitup_t1"], errors="coerce")
    valid = y.notna() & score.notna()
    y = y[valid]
    score = score[valid]
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    ranks = score.rank(method="average")
    auc = (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
    return round(float(auc), 4)


def _brier_score(frame: pd.DataFrame) -> float | None:
    if frame.empty or "label_t1_limitup" not in frame.columns or "p_limitup_t1" not in frame.columns:
        return None
    y = pd.to_numeric(frame["label_t1_limitup"], errors="coerce")
    p = pd.to_numeric(frame["p_limitup_t1"], errors="coerce") / 100
    valid = y.notna() & p.notna()
    if not valid.any():
        return None
    return round(float(((p[valid] - y[valid]) ** 2).mean()), 6)


def _mean_metric(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return round(float(values.mean()), 4)


def _min_metric(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return round(float(values.min()), 4)


def _buy_portfolio_metrics(frame: pd.DataFrame) -> dict:
    empty = {
        "buy_plan_days": 0,
        "buy_average_count_per_day": 0.0,
        "buy_daily_avg_next_day_open_pct": None,
        "buy_daily_avg_next_day_high_pct": None,
        "buy_daily_avg_next_day_low_pct": None,
        "buy_daily_avg_next_day_close_pct": None,
        "buy_plan_day_win_rate": 0.0,
        "buy_cumulative_next_day_close_pct": 0.0,
    }
    if frame.empty or "backtest_trade_date" not in frame.columns:
        return empty
    metric_columns = [
        "next_day_open_pct",
        "next_day_max_pct",
        "next_day_drawdown_pct",
        "next_day_close_pct",
    ]
    available = [column for column in metric_columns if column in frame.columns]
    if "next_day_close_pct" not in available:
        return empty
    work = frame[["backtest_trade_date", *available]].copy()
    for column in available:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    daily = work.groupby("backtest_trade_date", sort=True)[available].mean()
    close_return = daily["next_day_close_pct"].dropna()
    if close_return.empty:
        return empty
    counts = frame.groupby("backtest_trade_date").size()
    metric_map = {
        "next_day_open_pct": "buy_daily_avg_next_day_open_pct",
        "next_day_max_pct": "buy_daily_avg_next_day_high_pct",
        "next_day_drawdown_pct": "buy_daily_avg_next_day_low_pct",
        "next_day_close_pct": "buy_daily_avg_next_day_close_pct",
    }
    result = empty.copy()
    result["buy_plan_days"] = int(len(close_return))
    result["buy_average_count_per_day"] = round(float(counts.mean()), 2)
    for source, target in metric_map.items():
        if source in daily.columns and daily[source].notna().any():
            result[target] = round(float(daily[source].dropna().mean()), 4)
    result["buy_plan_day_win_rate"] = round(float(close_return.gt(0).mean()), 4)
    result["buy_cumulative_next_day_close_pct"] = round(float(((1 + close_return / 100).prod() - 1) * 100), 4)
    return result


def run_backtest(start_date: str, end_date: str, output_root: str | Path, top_n: int = 50) -> BacktestResult:
    start = _date_key(start_date)
    end = _date_key(end_date)
    output_root = Path(output_root)
    cache_root = output_root.parent / "data" / "cache" / "history"
    dates = _available_trade_dates(start, end)
    all_trades: list[pd.DataFrame] = []
    all_buy_trades: list[pd.DataFrame] = []
    daily_rows: list[dict] = []

    for trade_date in dates:
        rank_input = build_rank_input_for_date(trade_date, cache_root)
        candidates = filter_candidates(rank_input)
        scored = add_scores(add_feature_scores(candidates), calibration_enabled=False)
        top50, full_rank = rank_candidates(scored, _date_dash(trade_date), top_n=top_n)
        buy_pool = build_ranked_pool(scored, full_rank, top_n)
        buy_plan = build_buy_decision(buy_pool).buy_plan
        labeled = add_t1_labels(top50, trade_date, cache_root)
        verified = labeled[labeled.get("label_available", pd.Series(False, index=labeled.index)).fillna(False)].copy()
        if not verified.empty:
            verified["backtest_trade_date"] = trade_date
            verified["backtest_data_mode"] = "eod_proxy"
            verified["calibration_eligible"] = False
            all_trades.append(verified)
        buy_labeled = add_t1_labels(buy_plan, trade_date, cache_root)
        buy_verified = buy_labeled[
            buy_labeled.get("label_available", pd.Series(False, index=buy_labeled.index)).fillna(False)
        ].copy()
        if not buy_verified.empty:
            buy_verified["backtest_trade_date"] = trade_date
            buy_verified["backtest_data_mode"] = "eod_proxy"
            buy_verified["calibration_eligible"] = False
            all_buy_trades.append(buy_verified)
        daily_rows.append({
            "trade_date": trade_date,
            "raw_count": int(len(rank_input)),
            "candidate_count": int(len(candidates)),
            "top_count": int(len(top50)),
            "verified_count": int(len(verified)),
            "buy_count": int(len(buy_plan)),
            "buy_verified_count": int(len(buy_verified)),
            "buy_limitup_rate": _hit_rate(buy_verified, len(buy_verified)),
            "buy_avg_close_pct": _mean_metric(buy_verified, "next_day_close_pct"),
            "hit_top10": _hit_rate(verified, 10),
            "hit_top20": _hit_rate(verified, 20),
            "hit_top50": _hit_rate(verified, 50),
            "next_trade_date": str(labeled["next_trade_date"].iloc[0]) if not labeled.empty and "next_trade_date" in labeled.columns else "",
        })

    trades = pd.concat(all_trades, ignore_index=True, sort=False) if all_trades else pd.DataFrame()
    buy_trades = pd.concat(all_buy_trades, ignore_index=True, sort=False) if all_buy_trades else pd.DataFrame()
    daily_summary = pd.DataFrame(daily_rows)
    buy_portfolio_metrics = _buy_portfolio_metrics(buy_trades)
    summary = {
        "mode": "backtest",
        "model_version": MODEL_VERSION,
        "backtest_data_mode": "eod_proxy",
        "calibration_eligible": False,
        "warning": "使用收盘日线代理14:20状态，仅用于方向审计，不进入实时概率校准。",
        "start_date": start,
        "end_date": end,
        "trade_days": int(len(dates)),
        "trade_count": int(len(trades)),
        "hit_top10": _period_hit_rate(trades, 10),
        "hit_top20": _period_hit_rate(trades, 20),
        "hit_top50": round(float(trades["label_t1_limitup"].mean()), 4) if not trades.empty and "label_t1_limitup" in trades.columns else 0.0,
        "precision_at_10": _period_hit_rate(trades, 10),
        "precision_at_20": _period_hit_rate(trades, 20),
        "precision_at_50": _period_hit_rate(trades, 50),
        "auc": _auc_score(trades),
        "brier": _brier_score(trades),
        "buy_trade_count": int(len(buy_trades)),
        "buy_limitup_rate": round(float(buy_trades["label_t1_limitup"].mean()), 4) if not buy_trades.empty else 0.0,
        "buy_positive_close_rate": round(float(pd.to_numeric(buy_trades.get("next_day_close_pct"), errors="coerce").gt(0).mean()), 4) if not buy_trades.empty else 0.0,
        "buy_avg_next_day_max_pct": _mean_metric(buy_trades, "next_day_max_pct"),
        "buy_avg_next_day_close_pct": _mean_metric(buy_trades, "next_day_close_pct"),
        "buy_worst_drawdown_pct": _min_metric(buy_trades, "next_day_drawdown_pct"),
        **buy_portfolio_metrics,
        "avg_next_day_max_pct_top10": _mean_metric(_period_top_samples(trades, 10), "next_day_max_pct"),
        "avg_next_day_close_pct_top10": _mean_metric(_period_top_samples(trades, 10), "next_day_close_pct"),
        "max_drawdown_risk_top10": _min_metric(_period_top_samples(trades, 10), "next_day_drawdown_pct"),
        "avg_next_day_max_pct_top50": _mean_metric(trades, "next_day_max_pct"),
        "avg_next_day_close_pct_top50": _mean_metric(trades, "next_day_close_pct"),
        "max_drawdown_risk_top50": _min_metric(trades, "next_day_drawdown_pct"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    out_dir = output_root / "backtests" / f"{start}_{end}"
    ensure_dir(out_dir)
    trades.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    buy_trades.to_csv(out_dir / "buy_trades.csv", index=False, encoding="utf-8-sig")
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
      <div>AUC：<strong>{summary["auc"] if summary["auc"] is not None else "-"}</strong></div>
      <div>Brier：<strong>{summary["brier"] if summary["brier"] is not None else "-"}</strong></div>
      <div>买入观察样本：<strong>{summary["buy_trade_count"]}</strong></div>
      <div>买入观察计划日：<strong>{summary["buy_plan_days"]}</strong></div>
      <div>买入观察涨停：<strong>{summary["buy_limitup_rate"]:.2%}</strong></div>
      <div>买入观察上涨：<strong>{summary["buy_positive_close_rate"]:.2%}</strong></div>
      <div>观察组合次日开盘：<strong>{f'{summary["buy_daily_avg_next_day_open_pct"]:.2f}%' if summary["buy_daily_avg_next_day_open_pct"] is not None else "-"}</strong></div>
      <div>观察组合次日最高：<strong>{f'{summary["buy_daily_avg_next_day_high_pct"]:.2f}%' if summary["buy_daily_avg_next_day_high_pct"] is not None else "-"}</strong></div>
      <div>观察组合次日收盘：<strong>{f'{summary["buy_daily_avg_next_day_close_pct"]:.2f}%' if summary["buy_daily_avg_next_day_close_pct"] is not None else "-"}</strong></div>
      <div>观察组合累计收盘：<strong>{summary["buy_cumulative_next_day_close_pct"]:.2f}%</strong></div>
      <div>Top10 平均次日最高涨幅：<strong>{f'{summary["avg_next_day_max_pct_top10"]:.2f}%' if summary["avg_next_day_max_pct_top10"] is not None else "-"}</strong></div>
      <div>Top10 平均次日收盘涨幅：<strong>{f'{summary["avg_next_day_close_pct_top10"]:.2f}%' if summary["avg_next_day_close_pct_top10"] is not None else "-"}</strong></div>
      <div>Top10 最大回撤风险：<strong>{f'{summary["max_drawdown_risk_top10"]:.2f}%' if summary["max_drawdown_risk_top10"] is not None else "-"}</strong></div>
    </section>
    <section class="panel"><h2>每日汇总</h2><div class="table-wrap"><table><thead><tr><th>日期</th><th>原始数</th><th>候选数</th><th>Top数</th><th>Top10</th><th>Top20</th><th>Top50</th><th>下一交易日</th></tr></thead><tbody>{daily_rows}</tbody></table></div></section>
    <section class="panel"><h2>样本预览</h2><div class="table-wrap"><table><thead><tr><th>日期</th><th>排名</th><th>代码</th><th>名称</th><th>概率</th><th>评分</th><th>T+1涨停</th></tr></thead><tbody>{trade_rows}</tbody></table></div></section>
  </main>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")
