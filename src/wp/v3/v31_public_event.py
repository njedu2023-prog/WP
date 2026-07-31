from __future__ import annotations

import re
from typing import Any, Mapping

import pandas as pd


SCHEMA_VERSION = "wp_v31_public_event_data_probe_1"
LOOKBACK_TRADE_DAYS = 5
PROBE_DATES = (
    "20230825",
    "20231229",
    "20240315",
    "20240927",
    "20250115",
    "20250723",
    "20260115",
    "20260723",
)
SOURCE_SPECS: dict[str, dict[str, str]] = {
    "forecast": {
        "date_arg": "ann_date",
        "date_column": "ann_date",
        "fields": (
            "ts_code,ann_date,end_date,type,p_change_min,p_change_max,"
            "net_profit_min,net_profit_max,last_parent_net,first_ann_date"
        ),
    },
    "express": {
        "date_arg": "ann_date",
        "date_column": "ann_date",
        "fields": (
            "ts_code,ann_date,end_date,revenue,operate_profit,total_profit,"
            "n_income,diluted_eps,diluted_roe,yoy_net_profit,yoy_sales"
        ),
    },
    "repurchase": {
        "date_arg": "ann_date",
        "date_column": "ann_date",
        "fields": (
            "ts_code,ann_date,end_date,proc,exp_date,vol,amount,"
            "high_limit,low_limit"
        ),
    },
    "stk_holdertrade": {
        "date_arg": "ann_date",
        "date_column": "ann_date",
        "fields": (
            "ts_code,ann_date,holder_name,holder_type,in_de,change_vol,"
            "change_ratio,after_share,after_ratio,avg_price,total_share,"
            "begin_date,close_date"
        ),
    },
    "share_float": {
        "date_arg": "ann_date",
        "date_column": "ann_date",
        "fields": (
            "ts_code,ann_date,float_date,float_share,float_ratio,"
            "holder_name,share_type"
        ),
    },
    "block_trade": {
        "date_arg": "trade_date",
        "date_column": "trade_date",
        "fields": (
            "ts_code,trade_date,price,vol,amount,buyer,seller"
        ),
    },
}
_A_SHARE_CODE = re.compile(r"^\d{6}\.(?:SH|SZ|BJ)$")


def build_lookback_map(
    open_dates: list[str],
    target_dates: tuple[str, ...] = PROBE_DATES,
    *,
    lookback: int = LOOKBACK_TRADE_DAYS,
) -> dict[str, list[str]]:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    ordered = sorted({str(value) for value in open_dates})
    mapping: dict[str, list[str]] = {}
    for target in target_dates:
        previous = [value for value in ordered if value < target]
        if len(previous) < lookback:
            raise RuntimeError(f"insufficient lookback before {target}")
        mapping[target] = previous[-lookback:]
    return mapping


def audit_event_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    requested_date: str,
) -> dict[str, Any]:
    spec = SOURCE_SPECS[source]
    fields = spec["fields"].split(",")
    required = set(fields)
    schema_ok = required.issubset(frame.columns)
    if not schema_ok:
        return {
            "status": "ok",
            "source": source,
            "requested_date": requested_date,
            "rows": int(len(frame)),
            "schema_ok": False,
            "coverage_pass": False,
        }
    normalized = frame.reindex(columns=fields).copy()
    date_column = spec["date_column"]
    normalized[date_column] = normalized[date_column].astype(str)
    normalized["ts_code"] = normalized["ts_code"].astype(str)
    date_ok = bool(
        normalized.empty
        or normalized[date_column].eq(requested_date).all()
    )
    code_ok = bool(
        normalized.empty
        or normalized["ts_code"].str.match(_A_SHARE_CODE).all()
    )
    exact_duplicates = bool(normalized.duplicated().any())
    passed = bool(
        schema_ok and date_ok and code_ok and not exact_duplicates
    )
    return {
        "status": "ok",
        "source": source,
        "requested_date": requested_date,
        "rows": int(len(normalized)),
        "unique_codes": int(normalized["ts_code"].nunique()),
        "schema_ok": schema_ok,
        "date_ok": date_ok,
        "a_share_code_ok": code_ok,
        "exact_duplicates": exact_duplicates,
        "coverage_pass": passed,
    }


def normalize_event_frame(
    frame: pd.DataFrame,
    *,
    source: str,
) -> pd.DataFrame:
    spec = SOURCE_SPECS[source]
    fields = spec["fields"].split(",")
    result = frame.reindex(columns=fields).copy()
    result["ts_code"] = result["ts_code"].astype(str)
    result["event_date"] = result[spec["date_column"]].astype(str)
    result["event_source"] = source
    return result


def build_candidate_event_presence(
    candidates: pd.DataFrame,
    events_by_source: Mapping[str, pd.DataFrame],
    lookback_map: Mapping[str, list[str]],
) -> pd.DataFrame:
    required = {"trade_date", "ts_code"}
    if not required.issubset(candidates.columns):
        raise RuntimeError("V31 candidates are missing identity columns")
    identities = candidates.loc[:, ["trade_date", "ts_code"]].copy()
    identities["trade_date"] = identities["trade_date"].astype(str)
    identities["ts_code"] = identities["ts_code"].astype(str)
    identities = identities.drop_duplicates().sort_values(
        ["trade_date", "ts_code"],
        kind="stable",
    )
    records: list[dict[str, Any]] = []
    for row in identities.itertuples(index=False):
        allowed_dates = set(lookback_map[str(row.trade_date)])
        record: dict[str, Any] = {
            "trade_date": str(row.trade_date),
            "ts_code": str(row.ts_code),
        }
        for source in SOURCE_SPECS:
            source_frame = events_by_source.get(source, pd.DataFrame())
            if source_frame.empty:
                present = False
            else:
                present = bool(
                    (
                        source_frame["event_date"].isin(allowed_dates)
                        & source_frame["ts_code"].eq(str(row.ts_code))
                    ).any()
                )
            record[f"event_{source}"] = present
        records.append(record)
    result = pd.DataFrame.from_records(records)
    event_columns = [f"event_{source}" for source in SOURCE_SPECS]
    result["event_any_source"] = result[event_columns].any(axis=1)
    return result


def causal_dates_valid(
    events_by_source: Mapping[str, pd.DataFrame],
    lookback_map: Mapping[str, list[str]],
) -> bool:
    permitted = {
        date
        for dates in lookback_map.values()
        for date in dates
    }
    for frame in events_by_source.values():
        if frame.empty:
            continue
        dates = set(frame["event_date"].astype(str))
        if not dates.issubset(permitted):
            return False
    return all(
        all(date < target for date in dates)
        for target, dates in lookback_map.items()
    )
