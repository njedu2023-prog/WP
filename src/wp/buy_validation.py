from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time
from io import StringIO
from pathlib import Path

import pandas as pd

from .calendar import CN_TZ, next_trading_day_str
from .data_loader import _read_remote_text
from .tail_profit_model import TAIL_PROFIT_MODEL_VERSION


VALIDATION_COLUMNS = [
    "plan_trade_date",
    "plan_time",
    "market_data_time",
    "target_trade_date",
    "buy_rank",
    "portfolio_group",
    "ts_code",
    "name",
    "plan_price",
    "pct_chg_plan",
    "sector_name",
    "p_limitup_t1",
    "wp_score",
    "decision_score",
    "tail_profit_score",
    "buy_model_version",
    "risk_penalty_score",
    "actual_trade_date",
    "actual_open",
    "actual_high",
    "actual_low",
    "actual_close",
    "actual_pct_chg",
    "return_open_pct",
    "return_high_pct",
    "return_low_pct",
    "return_close_pct",
    "is_limit_up_t1",
    "truth_status",
    "truth_error",
    "truth_updated_at",
]

VALIDATION_TRACKING_START_DATE = "20260715"
TAIL_WINDOW_START = time(14, 20)
TAIL_WINDOW_END = time(14, 50)


@dataclass
class BuyValidationResult:
    table: pd.DataFrame
    summary: dict


def _parse_dt(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    result = parsed.to_pydatetime()
    return result if result.tzinfo else result.replace(tzinfo=CN_TZ)


def _in_tail_window(value: object) -> bool:
    parsed = _parse_dt(value)
    if parsed is None:
        return False
    return TAIL_WINDOW_START <= parsed.time() <= TAIL_WINDOW_END


def _limit_threshold(ts_code: str) -> float:
    code = str(ts_code).split(".", 1)[0]
    if str(ts_code).endswith(".BJ") or code.startswith(("8", "9")):
        return 29.5
    if code.startswith(("300", "301", "688")):
        return 19.5
    return 9.5


def _is_truth_due(target_trade_date: str, current: datetime) -> bool:
    today = current.strftime("%Y%m%d")
    if target_trade_date < today:
        return True
    return target_trade_date == today and current.time() >= time(15, 5)


def _existing_table(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            frame = pd.read_csv(
                path,
                dtype={"plan_trade_date": str, "target_trade_date": str, "actual_trade_date": str},
                keep_default_na=False,
            )
            for col in VALIDATION_COLUMNS:
                if col not in frame.columns:
                    frame[col] = ""
            return frame[VALIDATION_COLUMNS].copy()
        except Exception:
            pass
    return pd.DataFrame(columns=VALIDATION_COLUMNS)


def _new_snapshot_rows(buy_plan: pd.DataFrame, health: dict, current: datetime) -> pd.DataFrame:
    if buy_plan.empty:
        return pd.DataFrame(columns=VALIDATION_COLUMNS)
    market_data_time = str(health.get("market_data_time") or health.get("data_time") or "")
    if not _in_tail_window(market_data_time):
        return pd.DataFrame(columns=VALIDATION_COLUMNS)
    plan_trade_date = str(health.get("data_trade_date") or "").replace("-", "")
    if len(plan_trade_date) != 8:
        return pd.DataFrame(columns=VALIDATION_COLUMNS)
    target_trade_date = next_trading_day_str(plan_trade_date)
    snapshot_plan = buy_plan.copy()
    if "portfolio_group" in snapshot_plan.columns:
        primary = snapshot_plan["portfolio_group"].fillna("").astype(str).eq("主票")
        if primary.any():
            snapshot_plan = snapshot_plan[primary].copy()
    rows = []
    for _, row in snapshot_plan.iterrows():
        rows.append(
            {
                "plan_trade_date": plan_trade_date,
                "plan_time": market_data_time,
                "market_data_time": market_data_time,
                "target_trade_date": target_trade_date,
                "buy_rank": row.get("buy_rank", ""),
                "portfolio_group": row.get("portfolio_group", ""),
                "ts_code": row.get("ts_code", ""),
                "name": row.get("name", ""),
                "plan_price": row.get("price", ""),
                "pct_chg_plan": row.get("pct_chg", ""),
                "sector_name": row.get("sector_name", ""),
                "p_limitup_t1": row.get("p_limitup_t1", ""),
                "wp_score": row.get("wp_score", ""),
                "decision_score": row.get("decision_score", ""),
                "tail_profit_score": row.get("tail_profit_score", ""),
                "buy_model_version": row.get("tail_profit_model_version", ""),
                "risk_penalty_score": row.get("risk_penalty_score", ""),
                "actual_trade_date": target_trade_date,
                "actual_open": "",
                "actual_high": "",
                "actual_low": "",
                "actual_close": "",
                "actual_pct_chg": "",
                "return_open_pct": "",
                "return_high_pct": "",
                "return_low_pct": "",
                "return_close_pct": "",
                "is_limit_up_t1": "",
                "truth_status": "pending",
                "truth_error": "",
                "truth_updated_at": "",
            }
        )
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def _fetch_truth_by_date(trade_date: str) -> tuple[pd.DataFrame | None, str]:
    upstream_truth, upstream_error = _fetch_upstream_truth_by_date(trade_date)
    if upstream_truth is not None:
        return upstream_truth, ""

    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        return None, f"upstream daily truth unavailable: {upstream_error}; TUSHARE_TOKEN not configured"
    try:
        import tushare as ts

        ts.set_token(token)
        pro = ts.pro_api()
        frame = pro.daily(trade_date=trade_date)
        if frame is None or frame.empty:
            return None, f"no daily truth for {trade_date}"
        try:
            limit_frame = pro.stk_limit(trade_date=trade_date)
            frame = _merge_limit_prices(frame, limit_frame)
        except Exception:
            if "up_limit" not in frame.columns:
                frame["up_limit"] = pd.NA
        return frame, ""
    except Exception as exc:
        return None, str(exc)


def _merge_limit_prices(frame: pd.DataFrame, limit_frame: pd.DataFrame | None) -> pd.DataFrame:
    out = frame.copy()
    if limit_frame is None or limit_frame.empty or not {"ts_code", "up_limit"}.issubset(limit_frame.columns):
        if "up_limit" not in out.columns:
            out["up_limit"] = pd.NA
        return out
    limits = limit_frame[["ts_code", "up_limit"]].drop_duplicates("ts_code").copy()
    limits["ts_code"] = limits["ts_code"].astype(str).str.strip()
    out["ts_code"] = out["ts_code"].astype(str).str.strip()
    if "up_limit" in out.columns:
        out = out.drop(columns=["up_limit"])
    return out.merge(limits, on="ts_code", how="left")


def _fetch_upstream_truth_by_date(trade_date: str) -> tuple[pd.DataFrame | None, str]:
    if len(str(trade_date)) != 8:
        return None, f"invalid trade_date {trade_date!r}"
    template = os.environ.get("WP_TRUTH_DAILY_URL_TEMPLATE", "").strip()
    if template:
        url = template.format(trade_date=trade_date, year=trade_date[:4])
        limit_template = os.environ.get("WP_TRUTH_LIMIT_URL_TEMPLATE", "").strip()
        limit_url = limit_template.format(trade_date=trade_date, year=trade_date[:4]) if limit_template else url.replace("/daily.csv", "/stk_limit.csv")
    else:
        repo = os.environ.get("WP_TRUTH_REPO", "njedu2023-prog/a-share-top3-data").strip()
        url = f"https://raw.githubusercontent.com/{repo}/main/data/raw/{trade_date[:4]}/{trade_date}/daily.csv"
        limit_url = f"https://raw.githubusercontent.com/{repo}/main/data/raw/{trade_date[:4]}/{trade_date}/stk_limit.csv"
    try:
        text = _read_remote_text(url, timeout=30)
        frame = pd.read_csv(StringIO(text), dtype={"trade_date": str})
        if frame.empty:
            return None, f"empty upstream daily truth for {trade_date}"
        if "ts_code" not in frame.columns:
            return None, f"upstream daily truth missing ts_code for {trade_date}"
        if "pct_chg" not in frame.columns and {"close", "pre_close"}.issubset(frame.columns):
            close = pd.to_numeric(frame["close"], errors="coerce")
            pre_close = pd.to_numeric(frame["pre_close"], errors="coerce")
            frame["pct_chg"] = (close / pre_close.replace(0, pd.NA) - 1) * 100
        if "pct_chg" not in frame.columns:
            return None, f"upstream daily truth missing pct_chg for {trade_date}"
        if "close" not in frame.columns:
            frame["close"] = ""
        frame["ts_code"] = frame["ts_code"].astype(str).str.strip()
        try:
            limit_text = _read_remote_text(limit_url, timeout=30)
            limit_frame = pd.read_csv(StringIO(limit_text), dtype={"trade_date": str})
        except Exception:
            limit_frame = pd.DataFrame()
        frame = _merge_limit_prices(frame, limit_frame)
        return frame, ""
    except Exception as exc:
        return None, str(exc)


def _number(value: object) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else None


def _return_pct(value: object, entry_price: float | None) -> float | str:
    exit_price = _number(value)
    if entry_price is None or entry_price <= 0 or exit_price is None:
        return ""
    return round((exit_price / entry_price - 1) * 100, 4)


def _reconstruct_plan_price(row: pd.Series, source_truth: pd.DataFrame | None) -> float | None:
    stored = _number(row.get("plan_price"))
    if stored is not None and stored > 0:
        return stored
    if source_truth is None or source_truth.empty:
        return None
    match = source_truth[source_truth["ts_code"].astype(str).eq(str(row.get("ts_code", "")))]
    if match.empty:
        return None
    pre_close = _number(match.iloc[0].get("pre_close"))
    pct_chg_plan = _number(row.get("pct_chg_plan"))
    if pre_close is None or pre_close <= 0 or pct_chg_plan is None:
        return None
    return pre_close * (1 + pct_chg_plan / 100)


def _fill_truth(table: pd.DataFrame, current: datetime) -> pd.DataFrame:
    if table.empty:
        return table
    out = table.copy()
    plan_price = pd.to_numeric(out["plan_price"], errors="coerce")
    close_return = pd.to_numeric(out["return_close_pct"], errors="coerce")
    needs_refresh = (
        out["truth_status"].fillna("").astype(str).ne("verified")
        | plan_price.isna()
        | plan_price.le(0)
        | close_return.isna()
    )
    due_rows = out.loc[needs_refresh].copy()
    due_rows = due_rows[
        due_rows["target_trade_date"].astype(str).map(
            lambda value: len(value) == 8 and _is_truth_due(value, current)
        )
    ]
    if due_rows.empty:
        return out

    dates = sorted(
        {
            str(value)
            for column in ("plan_trade_date", "target_trade_date")
            for value in due_rows[column].dropna().tolist()
            if len(str(value)) == 8
        }
    )
    truth_cache = {trade_date: _fetch_truth_by_date(trade_date) for trade_date in dates}
    for idx, row in due_rows.iterrows():
        target = str(row.get("target_trade_date", ""))
        target_truth, target_error = truth_cache.get(target, (None, f"missing truth date {target}"))
        if target_truth is None:
            if str(row.get("truth_status", "")) != "verified":
                out.at[idx, "truth_status"] = "pending"
            out.at[idx, "truth_error"] = target_error
            continue

        code = str(row.get("ts_code", ""))
        match = target_truth[target_truth["ts_code"].astype(str).eq(code)]
        if match.empty:
            if str(row.get("truth_status", "")) != "verified":
                out.at[idx, "truth_status"] = "pending"
            out.at[idx, "truth_error"] = f"missing {code} on {target}"
            continue

        plan_date = str(row.get("plan_trade_date", ""))
        source_truth = truth_cache.get(plan_date, (None, ""))[0]
        entry_price = _reconstruct_plan_price(row, source_truth)
        actual = match.iloc[0]
        pct_chg = _number(actual.get("pct_chg"))
        if entry_price is not None and entry_price > 0:
            out.at[idx, "plan_price"] = round(entry_price, 4)
        for target_column, source_column in (
            ("actual_open", "open"),
            ("actual_high", "high"),
            ("actual_low", "low"),
            ("actual_close", "close"),
        ):
            out.at[idx, target_column] = actual.get(source_column, "")
        out.at[idx, "actual_pct_chg"] = "" if pct_chg is None else round(pct_chg, 4)
        for target_column, source_column in (
            ("return_open_pct", "open"),
            ("return_high_pct", "high"),
            ("return_low_pct", "low"),
            ("return_close_pct", "close"),
        ):
            out.at[idx, target_column] = _return_pct(actual.get(source_column), entry_price)

        high = _number(actual.get("high"))
        up_limit = _number(actual.get("up_limit"))
        if high is not None and up_limit is not None and up_limit > 0:
            out.at[idx, "is_limit_up_t1"] = bool(high >= up_limit * 0.999)
        elif high is not None:
            pre_close = _number(actual.get("pre_close"))
            threshold = _limit_threshold(code)
            out.at[idx, "is_limit_up_t1"] = bool(pre_close and (high / pre_close - 1) * 100 >= threshold)

        close_return_value = _number(out.at[idx, "return_close_pct"])
        if entry_price is not None and entry_price > 0 and close_return_value is not None:
            out.at[idx, "truth_status"] = "verified"
            out.at[idx, "truth_error"] = ""
            out.at[idx, "truth_updated_at"] = current.strftime("%Y-%m-%d %H:%M:%S")
        else:
            out.at[idx, "truth_status"] = "pending"
            out.at[idx, "truth_error"] = "missing plan price or next-day close"
    return out


def scope_validation_table(
    table: pd.DataFrame,
    model_version: str = "",
    start_date: str = VALIDATION_TRACKING_START_DATE,
) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    scoped = table.copy()
    if model_version and "buy_model_version" in scoped.columns:
        scoped = scoped[scoped["buy_model_version"].fillna("").astype(str).eq(model_version)].copy()
    if "plan_trade_date" in scoped.columns and start_date:
        plan_dates = scoped["plan_trade_date"].fillna("").astype(str).str.replace("-", "", regex=False).str.strip()
        scoped = scoped[plan_dates.ge(start_date)].copy()
    if "plan_time" in scoped.columns:
        scoped = scoped[scoped["plan_time"].map(_in_tail_window)].copy()
    return scoped


def _summary(
    table: pd.DataFrame,
    model_version: str = "",
    start_date: str = "",
) -> dict:
    empty_summary = {
        "buy_model_version": model_version,
        "total_plan_days": 0,
        "verified_plan_days": 0,
        "pending_plan_days": 0,
        "total_records": 0,
        "verified_records": 0,
        "pending_records": 0,
        "positive_records": 0,
        "positive_rate": 0.0,
        "limit_up_records": 0,
        "limit_up_rate": 0.0,
        "average_open_return_pct": 0.0,
        "average_high_return_pct": 0.0,
        "average_low_return_pct": 0.0,
        "average_close_return_pct": 0.0,
        "average_pct_chg": 0.0,
        "daily_average_pct_chg": 0.0,
        "positive_plan_days": 0,
        "plan_day_win_rate": 0.0,
        "cumulative_pct_chg": 0.0,
        "tracking_start_date": start_date,
    }
    if model_version or start_date:
        table = scope_validation_table(table, model_version, start_date)
    if table.empty:
        return empty_summary

    normalized = table.copy()
    normalized["plan_trade_date"] = normalized["plan_trade_date"].fillna("").astype(str).str.strip()
    numeric_column = lambda name: pd.to_numeric(normalized[name], errors="coerce") if name in normalized.columns else pd.Series(float("nan"), index=normalized.index, dtype="float64")
    legacy_return = numeric_column("actual_pct_chg")
    close_return = numeric_column("return_close_pct")
    normalized["_close_return"] = close_return.where(close_return.notna(), legacy_return)
    normalized["_open_return"] = numeric_column("return_open_pct")
    normalized["_high_return"] = numeric_column("return_high_pct")
    normalized["_low_return"] = numeric_column("return_low_pct")
    verified = normalized[normalized["truth_status"].fillna("").astype(str).eq("verified")].copy()
    verified_pct = verified["_close_return"].dropna()
    limit_up = verified[verified["is_limit_up_t1"].astype(str).str.lower().isin({"true", "1", "yes"})]
    positive = verified[verified["_close_return"].gt(0)]

    valid_dates = normalized[normalized["plan_trade_date"].ne("")]
    total_plan_days = int(valid_dates["plan_trade_date"].nunique())
    daily_returns: list[float] = []
    for _, day in valid_dates.groupby("plan_trade_date", sort=True):
        day_verified = day["truth_status"].fillna("").astype(str).eq("verified")
        day_pct = day.loc[day_verified, "_close_return"].dropna()
        if len(day) and day_verified.all() and len(day_pct) == len(day):
            daily_returns.append(float(day_pct.mean()))

    verified_plan_days = len(daily_returns)
    daily_series = pd.Series(daily_returns, dtype="float64")
    positive_plan_days = int(daily_series.gt(0).sum())
    cumulative_pct_chg = float(((1 + daily_series / 100).prod() - 1) * 100) if verified_plan_days else 0.0
    return {
        "buy_model_version": model_version,
        "total_plan_days": total_plan_days,
        "verified_plan_days": verified_plan_days,
        "pending_plan_days": max(total_plan_days - verified_plan_days, 0),
        "total_records": int(len(normalized)),
        "verified_records": int(len(verified)),
        "pending_records": int(len(normalized) - len(verified)),
        "positive_records": int(len(positive)),
        "positive_rate": round(len(positive) / len(verified) * 100, 2) if len(verified) else 0.0,
        "limit_up_records": int(len(limit_up)),
        "limit_up_rate": round(len(limit_up) / len(verified) * 100, 2) if len(verified) else 0.0,
        "average_open_return_pct": round(float(verified["_open_return"].dropna().mean()), 2) if verified["_open_return"].notna().any() else 0.0,
        "average_high_return_pct": round(float(verified["_high_return"].dropna().mean()), 2) if verified["_high_return"].notna().any() else 0.0,
        "average_low_return_pct": round(float(verified["_low_return"].dropna().mean()), 2) if verified["_low_return"].notna().any() else 0.0,
        "average_close_return_pct": round(float(verified_pct.mean()), 2) if len(verified_pct) else 0.0,
        "average_pct_chg": round(float(verified_pct.mean()), 2) if len(verified_pct) else 0.0,
        "daily_average_pct_chg": round(float(daily_series.mean()), 2) if verified_plan_days else 0.0,
        "positive_plan_days": positive_plan_days,
        "plan_day_win_rate": round(positive_plan_days / verified_plan_days * 100, 2) if verified_plan_days else 0.0,
        "cumulative_pct_chg": round(cumulative_pct_chg, 2),
        "return_basis": "plan_price_to_next_trade_day",
        "tracking_start_date": start_date,
    }


def update_buy_plan_validation(buy_plan: pd.DataFrame, health: dict, output_root: Path, current: datetime) -> BuyValidationResult:
    csv_path = output_root / "csv" / "wp_buy_plan_validation.csv"
    existing = _existing_table(csv_path)
    snapshot = _new_snapshot_rows(buy_plan, health, current)
    current_model = str(health.get("buy_model_version") or "")
    if not buy_plan.empty and "tail_profit_model_version" in buy_plan.columns:
        versions = buy_plan["tail_profit_model_version"].dropna().astype(str)
        if not versions.empty:
            current_model = str(versions.iloc[0])
    if not snapshot.empty:
        plan_dates = set(snapshot["plan_trade_date"].astype(str).tolist())
        snapshot_models = set(snapshot["buy_model_version"].fillna("").astype(str).tolist())
        locked_dates = set(
            existing.loc[
                existing["plan_trade_date"].astype(str).isin(plan_dates)
                & existing["buy_model_version"].fillna("").astype(str).isin(snapshot_models)
                & existing["truth_status"].fillna("").astype(str).eq("verified"),
                "plan_trade_date",
            ].astype(str).tolist()
        )
        snapshot = snapshot[~snapshot["plan_trade_date"].astype(str).isin(locked_dates)].copy()
    if not snapshot.empty:
        snapshot_keys = set(
            zip(
                snapshot["buy_model_version"].fillna("").astype(str),
                snapshot["plan_trade_date"].astype(str),
                snapshot["plan_time"].astype(str),
            )
        )
        existing_keys = list(
            zip(
                existing["buy_model_version"].fillna("").astype(str),
                existing["plan_trade_date"].astype(str),
                existing["plan_time"].astype(str),
            )
        )
        # Preserve every dynamic primary-list snapshot in the 14:20-14:50
        # window. Re-running the exact same market timestamp replaces only that
        # timestamp, leaving earlier and later observations intact.
        existing = existing[[key not in snapshot_keys for key in existing_keys]].copy()
        table = snapshot.copy() if existing.empty else pd.concat([existing, snapshot], ignore_index=True)
    else:
        table = existing
    if not table.empty:
        key_cols = ["buy_model_version", "plan_trade_date", "plan_time", "ts_code"]
        table = table.drop_duplicates(key_cols, keep="last")
        table["_buy_rank_sort"] = pd.to_numeric(table["buy_rank"], errors="coerce").fillna(999)
        table = table.sort_values(["plan_trade_date", "plan_time", "_buy_rank_sort"]).drop(columns=["_buy_rank_sort"])
    table = _fill_truth(table, current)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    display_table = scope_validation_table(table, current_model, VALIDATION_TRACKING_START_DATE)
    return BuyValidationResult(
        table=display_table,
        summary=_summary(table, current_model, VALIDATION_TRACKING_START_DATE),
    )
