from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .v31_public_event import (
    LOOKBACK_TRADE_DAYS,
    PROBE_DATES,
    SOURCE_SPECS,
    build_candidate_event_presence,
    build_lookback_map,
    causal_dates_valid,
)


SCHEMA_VERSION = "wp_v32_a_share_public_event_data_probe_1"
SOURCE_V31_PROBE_RUN_ID = 30_663_984_930
SOURCE_V31_ARTIFACT_ID = 8_806_378_449
SOURCE_V31_ARTIFACT_DIGEST = (
    "sha256:"
    "11f60e1b87609524cae09405221a6c096eec9c3319e5b550ba87f69188b8fec4"
)
_A_SHARE_CODE = re.compile(
    r"^(?:6\d{5}\.SH|[03]\d{5}\.SZ|[489]\d{5}\.BJ)$"
)


def a_share_mask(frame: pd.DataFrame) -> pd.Series:
    if "ts_code" not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return (
        frame["ts_code"]
        .astype(str)
        .str.match(_A_SHARE_CODE, na=False)
        .astype(bool)
    )


def audit_a_share_event_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    requested_date: str,
) -> dict[str, Any]:
    spec = SOURCE_SPECS[source]
    fields = spec["fields"].split(",")
    schema_ok = set(fields).issubset(frame.columns)
    if not schema_ok:
        return {
            "status": "ok",
            "source": source,
            "requested_date": requested_date,
            "raw_rows": int(len(frame)),
            "rows": 0,
            "excluded_non_a_share_rows": 0,
            "schema_ok": False,
            "coverage_pass": False,
        }

    raw = frame.reindex(columns=fields).copy()
    date_column = spec["date_column"]
    raw[date_column] = raw[date_column].astype(str)
    raw["ts_code"] = raw["ts_code"].astype(str)
    date_ok = bool(raw.empty or raw[date_column].eq(requested_date).all())
    mask = a_share_mask(raw)
    retained = raw.loc[mask].copy()
    excluded_rows = int((~mask).sum())
    exact_duplicates = bool(retained.duplicated().any())
    passed = bool(schema_ok and date_ok and not exact_duplicates)
    return {
        "status": "ok",
        "source": source,
        "requested_date": requested_date,
        "raw_rows": int(len(raw)),
        "rows": int(len(retained)),
        "excluded_non_a_share_rows": excluded_rows,
        "unique_codes": int(retained["ts_code"].nunique()),
        "schema_ok": schema_ok,
        "date_ok": date_ok,
        "a_share_universe_normalized": True,
        "exact_duplicates": exact_duplicates,
        "coverage_pass": passed,
    }


def normalize_a_share_event_frame(
    frame: pd.DataFrame,
    *,
    source: str,
) -> pd.DataFrame:
    spec = SOURCE_SPECS[source]
    fields = spec["fields"].split(",")
    result = frame.reindex(columns=fields).copy()
    result["ts_code"] = result["ts_code"].astype(str)
    result = result.loc[a_share_mask(result)].copy()
    result["event_date"] = result[spec["date_column"]].astype(str)
    result["event_source"] = source
    return result


__all__ = [
    "LOOKBACK_TRADE_DAYS",
    "PROBE_DATES",
    "SCHEMA_VERSION",
    "SOURCE_SPECS",
    "SOURCE_V31_ARTIFACT_DIGEST",
    "SOURCE_V31_ARTIFACT_ID",
    "SOURCE_V31_PROBE_RUN_ID",
    "a_share_mask",
    "audit_a_share_event_frame",
    "build_candidate_event_presence",
    "build_lookback_map",
    "causal_dates_valid",
    "normalize_a_share_event_frame",
]
