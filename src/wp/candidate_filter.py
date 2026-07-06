from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import first_existing, numeric_series, text_series


def estimate_limit_up_pct(ts_code: pd.Series, name: pd.Series) -> pd.Series:
    code = ts_code.fillna("").astype(str)
    stock_name = name.fillna("").astype(str)
    is_st = stock_name.str.contains("ST", case=False, regex=False)
    is_chuang = code.str.startswith(("300", "301"))
    is_kechuang = code.str.startswith("688")
    is_bj = code.str.startswith(("8", "4", "920"))
    return np.select([is_st, is_bj, is_chuang | is_kechuang], [5.0, 30.0, 20.0], default=10.0)


def enrich_basic_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ts_code"] = text_series(out, ["ts_code", "code", "证券代码", "代码"])
    out["name"] = text_series(out, ["name", "stock_name", "证券简称", "名称"])
    out["trade_date"] = text_series(out, ["trade_date", "date", "交易日期"])
    out["price"] = numeric_series(out, ["price", "close", "最新价", "收盘价"])
    out["close"] = numeric_series(out, ["close", "price", "收盘价", "最新价"])
    out["pre_close"] = numeric_series(out, ["pre_close", "prev_close", "昨收"])
    pct_col = first_existing(out, ["pct_chg", "change_pct", "涨跌幅"])
    if pct_col:
        out["pct_chg"] = pd.to_numeric(out[pct_col], errors="coerce").fillna(0.0)
    else:
        out["pct_chg"] = np.where(out["pre_close"] > 0, (out["close"] / out["pre_close"] - 1) * 100, 0.0)
    out["amount"] = numeric_series(out, ["amount", "成交额", "turnover_amount"])
    out["sector_name"] = text_series(out, ["sector_name", "industry", "板块", "所属板块"], "未知板块")
    out["limit_up_pct"] = estimate_limit_up_pct(out["ts_code"], out["name"])
    out["today_limit_up_price"] = numeric_series(out, ["today_limit_up_price", "limit_up_price", "涨停价"])
    out["prev_limit_up_price"] = numeric_series(out, ["prev_limit_up_price", "pre_limit_up_price", "昨日涨停价"])
    out.loc[out["today_limit_up_price"] <= 0, "today_limit_up_price"] = out["pre_close"] * (1 + out["limit_up_pct"] / 100)
    out.loc[out["prev_limit_up_price"] <= 0, "prev_limit_up_price"] = numeric_series(out, ["prev_pre_close", "pre_pre_close"], out["pre_close"]) * (1 + out["limit_up_pct"] / 100)
    out["is_st"] = out["name"].str.contains("ST", case=False, regex=False)
    return out


def flag_limitup(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev_flag = first_existing(out, ["pre_day_limitup", "prev_is_limit_up", "is_limit_up_yesterday", "前一日涨停"])
    today_flag = first_existing(out, ["today_limitup", "is_limit_up_today", "is_limit_up", "今日涨停"])
    if prev_flag:
        out["pre_day_limitup"] = pd.to_numeric(out[prev_flag], errors="coerce").fillna(0).astype(int)
    else:
        prev_close = numeric_series(out, ["prev_close", "yesterday_close", "前收盘"], out["pre_close"])
        out["pre_day_limitup"] = (prev_close >= out["prev_limit_up_price"] * 0.999).astype(int)
    if today_flag:
        out["today_limitup"] = pd.to_numeric(out[today_flag], errors="coerce").fillna(0).astype(int)
    else:
        out["today_limitup"] = (out["close"] >= out["today_limit_up_price"] * 0.999).astype(int)
    return out


def filter_candidates(df: pd.DataFrame, min_pct_chg: float = 6.0, min_amount: float = 100000000, exclude_st: bool = True) -> pd.DataFrame:
    if df.empty:
        return enrich_basic_fields(df)
    out = flag_limitup(enrich_basic_fields(df))
    mask = (out["pct_chg"] >= min_pct_chg) & (out["pre_day_limitup"] != 1) & (out["today_limitup"] != 1)
    if exclude_st:
        mask &= ~out["is_st"]
    if min_amount > 0:
        mask &= out["amount"].fillna(0) >= min_amount
    return out.loc[mask].copy()
