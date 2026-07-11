from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import first_existing, numeric_series, text_series


TRUE_FLAGS = {"1", "true", "yes", "y", "是", "涨停", "停牌", "退市"}


def _flag_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.fillna("").astype(str).str.strip().str.lower()
    values = numeric.notna() & numeric.ne(0)
    values |= numeric.isna() & text.isin(TRUE_FLAGS)
    return values.astype(int)


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
    out["stock_age_days"] = numeric_series(out, ["stock_age_days"], np.nan)
    if out["stock_age_days"].isna().all() and "list_date" in out.columns:
        list_date = pd.to_datetime(out["list_date"].astype(str), format="%Y%m%d", errors="coerce")
        trade_date = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        out["stock_age_days"] = (trade_date - list_date).dt.days
    out["suspended_flag"] = numeric_series(out, ["suspended_flag", "is_suspended", "停牌"], 0).astype(int)
    out.loc[(out["price"] <= 0) | (out["close"] <= 0) | (out["amount"].fillna(0) <= 0), "suspended_flag"] = 1
    volume_col = first_existing(out, ["volume", "vol", "成交量"])
    if volume_col is not None:
        out.loc[pd.to_numeric(out[volume_col], errors="coerce").fillna(0) <= 0, "suspended_flag"] = 1
    out["delist_flag"] = numeric_series(out, ["delist_flag", "is_delist"], 0).astype(int)
    out.loc[out["name"].str.contains("退|退市", regex=True, na=False), "delist_flag"] = 1
    out["data_quality_flag"] = numeric_series(out, ["data_quality_flag"], 0).astype(int)
    required = ["ts_code", "name", "close", "pre_close", "pct_chg", "amount", "today_limit_up_price"]
    out.loc[out[required].isna().any(axis=1), "data_quality_flag"] = 1
    out.loc[(out["close"] <= 0) | (out["pre_close"] <= 0) | (out["amount"] <= 0) | (out["today_limit_up_price"] <= 0), "data_quality_flag"] = 1
    out.loc[out["ts_code"].str.strip().eq("") | out["name"].str.strip().eq(""), "data_quality_flag"] = 1
    return out


def flag_limitup(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev_flag = first_existing(out, ["pre_day_limitup", "prev_is_limit_up", "is_limit_up_yesterday", "前一日涨停"])
    today_flag = first_existing(out, ["today_limitup", "is_limit_up_today", "is_limit_up", "今日涨停"])
    if prev_flag:
        out["pre_day_limitup"] = _flag_series(out[prev_flag])
    else:
        prev_close = numeric_series(out, ["prev_close", "yesterday_close", "前收盘"], out["pre_close"])
        out["pre_day_limitup"] = (prev_close >= out["prev_limit_up_price"] * 0.999).astype(int)
    if today_flag:
        out["today_limitup"] = _flag_series(out[today_flag])
    else:
        out["today_limitup"] = (out["close"] >= out["today_limit_up_price"] * 0.999).astype(int)
    return out


def filter_candidates(
    df: pd.DataFrame,
    min_pct_chg: float = 8.0,
    min_amount: float = 100000000,
    exclude_st: bool = True,
    exclude_suspended: bool = True,
    exclude_new_stock_days: int = 10,
) -> pd.DataFrame:
    if df.empty:
        return enrich_basic_fields(df)
    out = flag_limitup(enrich_basic_fields(df))
    sort_columns = ["ts_code"]
    ascending = [True]
    if "update_time" in out.columns:
        out["_sort_update_time"] = out["update_time"].fillna("").astype(str)
        sort_columns.append("_sort_update_time")
        ascending.append(True)
    sort_columns.append("amount")
    ascending.append(True)
    out = out.sort_values(sort_columns, ascending=ascending, kind="mergesort").drop_duplicates("ts_code", keep="last")
    out = out.drop(columns=["_sort_update_time"], errors="ignore")
    mask = (out["pct_chg"] > min_pct_chg) & (out["pre_day_limitup"] != 1) & (out["today_limitup"] != 1)
    if exclude_st:
        mask &= ~out["is_st"]
    if exclude_suspended:
        mask &= out["suspended_flag"] != 1
    if exclude_new_stock_days > 0:
        mask &= out["stock_age_days"].isna() | (out["stock_age_days"] >= exclude_new_stock_days)
    mask &= out["delist_flag"] != 1
    mask &= out["data_quality_flag"] != 1
    if min_amount > 0:
        mask &= out["amount"].fillna(0) >= min_amount
    return out.loc[mask].copy()
