from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCHEMA_VERSION = "wp_v30_limit_event_data_probe_1"
SIGNAL_SLOTS = (
    "14:20",
    "14:25",
    "14:30",
    "14:35",
    "14:40",
    "14:45",
    "14:50",
)
KPL_TAGS = ("涨停", "炸板", "跌停")
KPL_FIELDS = (
    "ts_code,name,trade_date,lu_time,ld_time,open_time,last_time,"
    "lu_desc,tag,theme,net_change,bid_amount,status,bid_change,"
    "bid_turnover,lu_bid_vol,pct_chg,bid_pct_chg,rt_pct_chg,"
    "limit_order,amount,turnover_rate,free_float,lu_limit_order"
)
CURRENT_DAY_OUTPUT_COLUMNS = (
    "trade_date",
    "signal_slot",
    "market_limit_hit_count",
    "market_limit_open_count",
    "market_limit_down_count",
    "market_limit_hit_last_10m",
    "market_limit_open_last_10m",
    "market_net_sealed_count",
    "market_open_to_hit_ratio",
)
CURRENT_DAY_FORBIDDEN_COLUMNS = (
    "tag",
    "theme",
    "status",
    "lu_desc",
    "last_time",
    "net_change",
    "limit_order",
    "amount",
    "turnover_rate",
    "pct_chg",
    "rt_pct_chg",
    "lu_limit_order",
    "free_float",
    "bid_amount",
    "bid_change",
    "bid_turnover",
    "lu_bid_vol",
)
_A_SHARE_CODE = re.compile(r"^\d{6}\.(?:SH|SZ|BJ)$")
_SESSION_START = "09:25:00"
_SESSION_END = "15:00:00"


def parse_event_time(value: Any, trade_date: str) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    try:
        if pd.isna(value):
            return pd.NaT
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, np.integer)):
        raw = str(int(value))
    elif isinstance(value, (float, np.floating)) and np.isfinite(value):
        raw = str(int(value)) if float(value).is_integer() else str(value)
    else:
        raw = str(value).strip()
    if not raw or raw.lower() in {"0", "none", "nan", "nat", "--"}:
        return pd.NaT

    candidates: list[tuple[str, str]] = []
    if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", raw):
        fmt = "%H:%M:%S" if raw.count(":") == 2 else "%H:%M"
        candidates.append((f"{trade_date} {raw}", f"%Y%m%d {fmt}"))
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 4:
        candidates.append((f"{trade_date}{digits}", "%Y%m%d%H%M"))
    elif len(digits) == 6:
        candidates.append((f"{trade_date}{digits}", "%Y%m%d%H%M%S"))
    elif len(digits) == 12:
        candidates.append((digits, "%Y%m%d%H%M"))
    elif len(digits) == 14:
        candidates.append((digits, "%Y%m%d%H%M%S"))
    for candidate, fmt in candidates:
        try:
            return pd.Timestamp(datetime.strptime(candidate, fmt))
        except ValueError:
            continue
    parsed = pd.to_datetime(raw, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    timestamp = pd.Timestamp(parsed)
    if timestamp.strftime("%Y%m%d") != str(trade_date):
        return pd.NaT
    return timestamp


def normalize_kpl_frame(
    frame: pd.DataFrame,
    *,
    trade_date: str,
    requested_tag: str,
) -> pd.DataFrame:
    result = frame.reindex(columns=KPL_FIELDS.split(",")).copy()
    result["trade_date"] = result["trade_date"].astype(str)
    result["ts_code"] = result["ts_code"].astype(str)
    result["_requested_tag"] = requested_tag
    for source, target in (
        ("lu_time", "_lu_timestamp"),
        ("open_time", "_open_timestamp"),
        ("ld_time", "_ld_timestamp"),
        ("last_time", "_last_timestamp"),
    ):
        result[target] = pd.to_datetime(
            [
                parse_event_time(value, trade_date)
                for value in result[source]
            ],
            errors="coerce",
        )
    return result


def audit_kpl_frame(
    frame: pd.DataFrame,
    *,
    trade_date: str,
    requested_tag: str,
) -> dict[str, Any]:
    required = set(KPL_FIELDS.split(","))
    schema_ok = required.issubset(frame.columns)
    if not schema_ok:
        return {
            "status": "ok",
            "trade_date": trade_date,
            "requested_tag": requested_tag,
            "rows": int(len(frame)),
            "schema_ok": False,
            "coverage_pass": False,
        }
    normalized = normalize_kpl_frame(
        frame,
        trade_date=trade_date,
        requested_tag=requested_tag,
    )
    relevant = (
        "_ld_timestamp" if requested_tag == "跌停" else "_lu_timestamp"
    )
    raw_relevant = (
        normalized["ld_time"]
        if requested_tag == "跌停"
        else normalized["lu_time"]
    )
    supplied = _supplied_time(raw_relevant)
    parsed = normalized[relevant].notna()
    relevant_coverage = float(parsed.mean()) if len(normalized) else 1.0
    supplied_parse_rate = (
        float(parsed.loc[supplied].mean()) if supplied.any() else 1.0
    )
    all_supplied_parse = all(
        _time_parse_rate(normalized, source, target) >= 1.0
        for source, target in (
            ("lu_time", "_lu_timestamp"),
            ("open_time", "_open_timestamp"),
            ("ld_time", "_ld_timestamp"),
            ("last_time", "_last_timestamp"),
        )
    )
    timestamp_columns = (
        "_lu_timestamp",
        "_open_timestamp",
        "_ld_timestamp",
        "_last_timestamp",
    )
    session_start = pd.Timestamp(f"{trade_date} {_SESSION_START}")
    session_end = pd.Timestamp(f"{trade_date} {_SESSION_END}")
    valid_session = all(
        normalized[column]
        .dropna()
        .between(session_start, session_end, inclusive="both")
        .all()
        for column in timestamp_columns
    )
    time_diagnostics: dict[str, dict[str, Any]] = {}
    for source, target in (
        ("lu_time", "_lu_timestamp"),
        ("open_time", "_open_timestamp"),
        ("ld_time", "_ld_timestamp"),
        ("last_time", "_last_timestamp"),
    ):
        values = normalized[target].dropna()
        in_session = values.between(
            session_start,
            session_end,
            inclusive="both",
        )
        time_diagnostics[source] = {
            "min": (
                values.min().strftime("%H:%M:%S")
                if not values.empty
                else None
            ),
            "max": (
                values.max().strftime("%H:%M:%S")
                if not values.empty
                else None
            ),
            "out_of_session": int((~in_session).sum()),
        }
    open_order = normalized.loc[
        normalized["_lu_timestamp"].notna()
        & normalized["_open_timestamp"].notna()
    ]
    open_after_touch = bool(
        open_order.empty
        or open_order["_open_timestamp"].ge(
            open_order["_lu_timestamp"]
        ).all()
    )
    date_ok = bool(
        normalized.empty
        or normalized["trade_date"].eq(trade_date).all()
    )
    code_ok = bool(
        normalized["ts_code"].str.match(_A_SHARE_CODE).all()
        if len(normalized)
        else True
    )
    duplicates = bool(normalized["ts_code"].duplicated().any())
    coverage_pass = bool(
        date_ok
        and code_ok
        and not duplicates
        and relevant_coverage >= 0.90
        and supplied_parse_rate >= 1.0
        and all_supplied_parse
        and valid_session
        and open_after_touch
    )
    return {
        "status": "ok",
        "trade_date": trade_date,
        "requested_tag": requested_tag,
        "rows": int(len(normalized)),
        "unique_codes": int(normalized["ts_code"].nunique()),
        "schema_ok": True,
        "date_ok": date_ok,
        "a_share_code_ok": code_ok,
        "duplicate_codes": duplicates,
        "relevant_time_coverage": relevant_coverage,
        "supplied_relevant_time_parse_rate": supplied_parse_rate,
        "all_supplied_times_parseable": all_supplied_parse,
        "times_within_session": valid_session,
        "time_diagnostics": time_diagnostics,
        "open_not_before_first_touch": open_after_touch,
        "coverage_pass": coverage_pass,
    }


def build_causal_event_projection(
    frames: Iterable[pd.DataFrame],
    *,
    trade_date: str,
    signal_slots: tuple[str, ...] = SIGNAL_SLOTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.concat(list(frames), ignore_index=True)
    if data.empty:
        raise RuntimeError(f"V30 event source is empty for {trade_date}")
    if any(column not in data for column in ("_requested_tag", "ts_code")):
        raise RuntimeError("V30 event source is not normalized")

    up = data.loc[data["_requested_tag"].isin(("涨停", "炸板"))].copy()
    up_by_stock = (
        up.groupby("ts_code", sort=True)
        .agg(
            first_limit_touch=("_lu_timestamp", "min"),
            first_limit_open=("_open_timestamp", "min"),
        )
        .reset_index()
    )
    down = data.loc[data["_requested_tag"].eq("跌停")].copy()
    down_by_stock = (
        down.groupby("ts_code", sort=True)
        .agg(first_limit_down=("_ld_timestamp", "min"))
        .reset_index()
    )
    stocks = up_by_stock.merge(
        down_by_stock,
        on="ts_code",
        how="outer",
        validate="one_to_one",
    )
    stocks["trade_date"] = trade_date

    records: list[dict[str, Any]] = []
    for slot in signal_slots:
        cutoff = pd.Timestamp(f"{trade_date} {slot}:00")
        lower = cutoff - pd.Timedelta(minutes=10)
        hit = stocks["first_limit_touch"].le(cutoff)
        opened = stocks["first_limit_open"].le(cutoff)
        down_hit = stocks["first_limit_down"].le(cutoff)
        recent_hit = stocks["first_limit_touch"].between(
            lower,
            cutoff,
            inclusive="right",
        )
        recent_open = stocks["first_limit_open"].between(
            lower,
            cutoff,
            inclusive="right",
        )
        hit_count = int(hit.sum())
        open_count = int(opened.sum())
        records.append(
            {
                "trade_date": trade_date,
                "signal_slot": slot,
                "market_limit_hit_count": hit_count,
                "market_limit_open_count": open_count,
                "market_limit_down_count": int(down_hit.sum()),
                "market_limit_hit_last_10m": int(recent_hit.sum()),
                "market_limit_open_last_10m": int(recent_open.sum()),
                "market_net_sealed_count": max(hit_count - open_count, 0),
                "market_open_to_hit_ratio": (
                    float(open_count / hit_count) if hit_count else 0.0
                ),
            }
        )
    projection = pd.DataFrame.from_records(
        records,
        columns=CURRENT_DAY_OUTPUT_COLUMNS,
    )
    if any(
        column in projection
        for column in CURRENT_DAY_FORBIDDEN_COLUMNS
    ):
        raise RuntimeError("V30 causal projection contains final-state fields")
    return projection, stocks


def attach_candidate_event_state(
    candidates: pd.DataFrame,
    stock_events: pd.DataFrame,
    market_projection: pd.DataFrame,
) -> pd.DataFrame:
    required = {"trade_date", "signal_slot", "ts_code"}
    if not required.issubset(candidates.columns):
        raise RuntimeError("V30 candidates are missing identity columns")
    result = candidates.loc[:, sorted(required)].copy()
    result["trade_date"] = result["trade_date"].astype(str)
    result["signal_slot"] = result["signal_slot"].astype(str)
    result["ts_code"] = result["ts_code"].astype(str)
    result = result.merge(
        stock_events[
            [
                "trade_date",
                "ts_code",
                "first_limit_touch",
                "first_limit_open",
                "first_limit_down",
            ]
        ],
        on=["trade_date", "ts_code"],
        how="left",
        validate="many_to_one",
    )
    result = result.merge(
        market_projection,
        on=["trade_date", "signal_slot"],
        how="left",
        validate="many_to_one",
    )
    cutoff = pd.to_datetime(
        result["trade_date"] + " " + result["signal_slot"] + ":00",
        format="%Y%m%d %H:%M:%S",
        errors="coerce",
    )
    result["candidate_limit_hit_before_signal"] = (
        result["first_limit_touch"].le(cutoff).fillna(False)
    )
    result["candidate_limit_open_before_signal"] = (
        result["first_limit_open"].le(cutoff).fillna(False)
    )
    result["candidate_limit_down_before_signal"] = (
        result["first_limit_down"].le(cutoff).fillna(False)
    )
    result["candidate_minutes_since_limit_touch"] = (
        (cutoff - result["first_limit_touch"]).dt.total_seconds() / 60.0
    ).where(result["candidate_limit_hit_before_signal"])
    result["candidate_minutes_since_limit_open"] = (
        (cutoff - result["first_limit_open"]).dt.total_seconds() / 60.0
    ).where(result["candidate_limit_open_before_signal"])
    result.drop(
        columns=[
            "first_limit_touch",
            "first_limit_open",
            "first_limit_down",
        ],
        inplace=True,
    )
    if any(column in result for column in CURRENT_DAY_FORBIDDEN_COLUMNS):
        raise RuntimeError("V30 candidate projection contains final-state fields")
    return result


def _supplied_time(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return ~normalized.isin({"", "0", "none", "nan", "nat", "--"})


def _time_parse_rate(
    frame: pd.DataFrame,
    source_column: str,
    parsed_column: str,
) -> float:
    supplied = _supplied_time(frame[source_column])
    if not supplied.any():
        return 1.0
    return float(frame.loc[supplied, parsed_column].notna().mean())
