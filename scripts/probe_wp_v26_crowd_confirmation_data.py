from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tushare as ts

from wp.v3.history import TushareHistoryClient
from wp.v3.io import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATES = (
    "20230825",
    "20231229",
    "20240315",
    "20240927",
    "20250115",
    "20250723",
    "20260115",
    "20260723",
)
SIGNAL_TIMES = (
    "14:20:00",
    "14:25:00",
    "14:30:00",
    "14:35:00",
    "14:40:00",
    "14:45:00",
    "14:50:00",
)
HOT_FIELDS = (
    "trade_date,data_type,ts_code,ts_name,rank,pct_change,"
    "current_price,rank_time"
)
MEMBER_FIELDS = (
    "l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,"
    "ts_code,name,in_date,out_date,is_new"
)
CLASSIFY_FIELDS = (
    "index_code,industry_name,parent_code,level,industry_code,is_pub,src"
)
SW_MINUTE_FIELDS = "ts_code,trade_time,open,close,high,low,amount,vol"
A_SHARE_CODE = re.compile(r"^\d{6}\.(?:SH|SZ|BJ)$")


def main() -> int:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for the V26 data probe")
    output = Path(
        os.getenv(
            "WP_V26_PROBE_OUTPUT",
            str(ROOT / "artifacts" / "wp_v26_data_probe"),
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    client = TushareHistoryClient(
        ts.pro_api(token),
        output / "cache",
        page_size=2_000,
        requests_per_minute=120,
        attempts=2,
    )

    ths = probe_hot_source(
        client,
        api_name="ths_hot",
        market="热股",
    )
    dc = probe_hot_source(
        client,
        api_name="dc_hot",
        market="A股市场",
        hot_type="人气榜",
    )
    industry = probe_fine_industry_source(client)

    attention_sources = [
        name
        for name, result in (("ths_hot", ths), ("dc_hot", dc))
        if result["passed"]
    ]
    if attention_sources:
        selected_family: str | None = "intraday_attention"
        selected_sources = attention_sources
    elif industry["passed"]:
        selected_family = "fine_grained_industry_index"
        selected_sources = ["index_member_all", "sw_mins"]
    else:
        selected_family = None
        selected_sources = []

    payload = {
        "schema_version": "wp_v26_crowd_confirmation_data_probe_1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "profit_outcomes_read": False,
        "selection_used_profit_outcomes": False,
        "probe_dates": list(SAMPLE_DATES),
        "intraday_attention": {
            "passed": bool(attention_sources),
            "admitted_sources": attention_sources,
            "ths_hot": ths,
            "dc_hot": dc,
        },
        "fine_grained_industry_index": industry,
        "selected_source_family": selected_family,
        "selected_sources": selected_sources,
        "full_backfill_authorized": selected_family is not None,
        "model_research_authorized": False,
        "next_gate": (
            "full_three_year_outcome_blind_coverage_audit"
            if selected_family is not None
            else "stop_v26_data_direction"
        ),
    }
    atomic_write_json(output / "wp_v26_crowd_confirmation_probe.json", payload)
    print(
        "WP_V26_PROBE_RESULT="
        + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def parse_rank_time(value: Any, trade_date: str) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    raw = str(value).strip()
    if not raw:
        return pd.NaT

    formats: list[tuple[str, str]] = []
    if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", raw):
        time_format = "%H:%M:%S" if raw.count(":") == 2 else "%H:%M"
        formats.append((f"{trade_date} {raw}", f"%Y%m%d {time_format}"))
    digits = re.sub(r"\D", "", raw)
    digit_formats = {
        4: ("%Y%m%d%H%M", f"{trade_date}{digits}"),
        6: ("%Y%m%d%H%M%S", f"{trade_date}{digits}"),
        10: ("%Y%m%d%H", digits),
        12: ("%Y%m%d%H%M", digits),
        14: ("%Y%m%d%H%M%S", digits),
    }
    if len(digits) in digit_formats:
        fmt, candidate = digit_formats[len(digits)]
        formats.append((candidate, fmt))
    for candidate, fmt in formats:
        try:
            return pd.Timestamp(datetime.strptime(candidate, fmt))
        except ValueError:
            continue
    parsed = pd.to_datetime(raw, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed)


def hot_snapshot_record(
    frame: pd.DataFrame,
    *,
    api_name: str,
    trade_date: str,
) -> dict[str, Any]:
    required = set(HOT_FIELDS.split(","))
    schema_ok = required.issubset(frame.columns)
    if not schema_ok:
        return {
            "status": "ok",
            "api_name": api_name,
            "trade_date": trade_date,
            "rows": int(len(frame)),
            "schema_ok": False,
            "coverage_pass": False,
        }

    data = frame.loc[:, HOT_FIELDS.split(",")].copy()
    data["trade_date"] = data["trade_date"].astype(str)
    data["ts_code"] = data["ts_code"].astype(str)
    data["rank_timestamp"] = [
        parse_rank_time(value, trade_date)
        for value in data["rank_time"]
    ]
    valid_times = data["rank_timestamp"].notna()
    date_consistent = bool(
        valid_times.any()
        and data.loc[valid_times, "rank_timestamp"]
        .dt.strftime("%Y%m%d")
        .eq(trade_date)
        .all()
        and data.loc[valid_times, "trade_date"].eq(trade_date).all()
    )
    lower = pd.Timestamp(f"{trade_date} 13:20:00")
    upper = pd.Timestamp(f"{trade_date} 14:50:00")
    tail = data.loc[
        valid_times
        & data["rank_timestamp"].between(lower, upper, inclusive="both")
    ]
    if tail.empty:
        return {
            "status": "ok",
            "api_name": api_name,
            "trade_date": trade_date,
            "rows": int(len(data)),
            "schema_ok": True,
            "date_consistent": date_consistent,
            "usable_snapshot_time": None,
            "usable_snapshot_rows": 0,
            "coverage_pass": False,
        }

    snapshot_time = tail["rank_timestamp"].max()
    snapshot = tail.loc[tail["rank_timestamp"].eq(snapshot_time)].copy()
    a_share = snapshot["ts_code"].str.match(A_SHARE_CODE)
    snapshot = snapshot.loc[a_share]
    rank = pd.to_numeric(snapshot["rank"], errors="coerce")
    duplicates = snapshot["ts_code"].duplicated().any()
    rank_coverage = float(rank.notna().mean()) if len(snapshot) else 0.0
    rank_range_ok = bool(
        rank.notna().any()
        and rank.dropna().between(1, 2_000, inclusive="both").all()
    )
    unique_codes = int(snapshot["ts_code"].nunique())
    passed = bool(
        date_consistent
        and unique_codes >= 50
        and not duplicates
        and rank_coverage >= 0.95
        and rank_range_ok
        and snapshot_time <= upper
    )
    return {
        "status": "ok",
        "api_name": api_name,
        "trade_date": trade_date,
        "rows": int(len(data)),
        "schema_ok": True,
        "date_consistent": date_consistent,
        "usable_snapshot_time": snapshot_time.isoformat(),
        "usable_snapshot_rows": int(len(snapshot)),
        "unique_a_share_codes": unique_codes,
        "duplicate_codes_in_snapshot": bool(duplicates),
        "rank_coverage": rank_coverage,
        "rank_range_ok": rank_range_ok,
        "coverage_pass": passed,
    }


def probe_hot_source(
    client: TushareHistoryClient,
    *,
    api_name: str,
    market: str,
    hot_type: str | None = None,
) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    for trade_date in SAMPLE_DATES:
        params: dict[str, Any] = {
            "trade_date": trade_date,
            "market": market,
            "is_new": "N",
        }
        if hot_type is not None:
            params["hot_type"] = hot_type
        try:
            frame = client.query(
                api_name,
                cache_key=f"{trade_date}_v26_probe",
                paged=True,
                fields=HOT_FIELDS,
                **params,
            )
            probes.append(
                hot_snapshot_record(
                    frame,
                    api_name=api_name,
                    trade_date=trade_date,
                )
            )
        except Exception as error:
            probes.append(
                {
                    "status": "error",
                    "api_name": api_name,
                    "trade_date": trade_date,
                    "rows": 0,
                    "coverage_pass": False,
                    "error": str(error)[:500],
                }
            )
    return {
        "passed": bool(probes and all(row["coverage_pass"] for row in probes)),
        "required_dates": len(SAMPLE_DATES),
        "passed_dates": sum(bool(row["coverage_pass"]) for row in probes),
        "probes": probes,
    }


def normalize_membership(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.reindex(columns=MEMBER_FIELDS.split(",")).copy()
    for column in ("l1_code", "l2_code", "l3_code", "ts_code", "is_new"):
        data[column] = data[column].fillna("").astype(str).str.strip()
    data["in_timestamp"] = pd.to_datetime(
        data["in_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    out_raw = data["out_date"].fillna("").astype(str).str.strip()
    data["out_timestamp"] = pd.to_datetime(
        out_raw.where(out_raw.ne("")),
        format="%Y%m%d",
        errors="coerce",
    )
    return data.drop_duplicates(
        ["ts_code", "l1_code", "l2_code", "l3_code", "in_date", "out_date"]
    )


def membership_probe_record(
    frame: pd.DataFrame,
    *,
    trade_date: str,
) -> dict[str, Any]:
    timestamp = pd.Timestamp(datetime.strptime(trade_date, "%Y%m%d"))
    valid_interval = frame["in_timestamp"].notna() & (
        frame["out_timestamp"].isna()
        | frame["out_timestamp"].ge(frame["in_timestamp"])
    )
    active = frame.loc[
        valid_interval
        & frame["in_timestamp"].le(timestamp)
        & (
            frame["out_timestamp"].isna()
            | frame["out_timestamp"].gt(timestamp)
        )
    ].copy()
    active = active.sort_values(
        ["ts_code", "in_timestamp"],
        kind="stable",
    ).drop_duplicates("ts_code", keep="last")
    unique_codes = int(active["ts_code"].nunique())
    l2_coverage = float(active["l2_code"].ne("").mean()) if len(active) else 0.0
    l3_coverage = float(active["l3_code"].ne("").mean()) if len(active) else 0.0
    passed = bool(
        unique_codes >= 3_500
        and l2_coverage >= 0.95
        and l3_coverage >= 0.95
        and valid_interval.all()
    )
    return {
        "trade_date": trade_date,
        "unique_active_codes": unique_codes,
        "l2_code_coverage": l2_coverage,
        "l3_code_coverage": l3_coverage,
        "all_intervals_valid": bool(valid_interval.all()),
        "coverage_pass": passed,
    }


def select_index_codes(frame: pd.DataFrame, *, count: int = 4) -> list[str]:
    if frame.empty:
        return []
    data = frame.copy()
    published = data["is_pub"].fillna("").astype(str).isin(("1", "Y", "True"))
    codes = sorted(
        {
            str(code).strip()
            for code in data.loc[published, "index_code"]
            if str(code).strip()
        }
    )
    if len(codes) < count:
        return []
    positions = np.linspace(0, len(codes) - 1, num=count, dtype=int)
    return [codes[int(position)] for position in positions]


def sw_minute_probe_record(
    frame: pd.DataFrame,
    *,
    ts_code: str,
    trade_date: str,
) -> dict[str, Any]:
    required = set(SW_MINUTE_FIELDS.split(","))
    schema_ok = required.issubset(frame.columns)
    if not schema_ok:
        return {
            "ts_code": ts_code,
            "trade_date": trade_date,
            "schema_ok": False,
            "coverage_pass": False,
        }
    data = frame.loc[:, SW_MINUTE_FIELDS.split(",")].copy()
    data["trade_timestamp"] = pd.to_datetime(
        data["trade_time"],
        errors="coerce",
    )
    expected = {
        pd.Timestamp(f"{trade_date} {clock}")
        for clock in SIGNAL_TIMES
    }
    observed = set(data["trade_timestamp"].dropna())
    numeric = data.loc[
        data["trade_timestamp"].isin(expected),
        ["open", "close", "high", "low", "amount", "vol"],
    ].apply(pd.to_numeric, errors="coerce")
    exact_slots = len(expected.intersection(observed))
    date_consistent = bool(
        data["trade_timestamp"].dropna().dt.strftime("%Y%m%d").eq(trade_date).all()
    )
    numeric_complete = bool(len(numeric) == len(SIGNAL_TIMES) and numeric.notna().all(axis=None))
    passed = bool(
        date_consistent
        and exact_slots == len(SIGNAL_TIMES)
        and numeric_complete
    )
    return {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "schema_ok": True,
        "rows": int(len(data)),
        "exact_signal_slots": exact_slots,
        "date_consistent": date_consistent,
        "numeric_complete": numeric_complete,
        "coverage_pass": passed,
    }


def probe_fine_industry_source(
    client: TushareHistoryClient,
) -> dict[str, Any]:
    try:
        classifications = {
            level: client.query(
                "index_classify",
                cache_key=f"sw2021_{level}_v26_probe",
                level=level,
                src="SW2021",
                fields=CLASSIFY_FIELDS,
            )
            for level in ("L2", "L3")
        }
        published_counts = {
            level: int(
                frame["is_pub"]
                .fillna("")
                .astype(str)
                .isin(("1", "Y", "True"))
                .sum()
            )
            for level, frame in classifications.items()
        }
        selected_codes = {
            level: select_index_codes(frame)
            for level, frame in classifications.items()
        }

        membership_frames = [
            client.query(
                "index_member_all",
                cache_key=f"{state}_v26_probe",
                paged=True,
                is_new=state,
                fields=MEMBER_FIELDS,
            )
            for state in ("Y", "N")
        ]
        membership = normalize_membership(
            pd.concat(membership_frames, ignore_index=True)
        )
        membership_probes = [
            membership_probe_record(membership, trade_date=trade_date)
            for trade_date in SAMPLE_DATES
        ]

        minute_probes: list[dict[str, Any]] = []
        for level in ("L2", "L3"):
            for ts_code in selected_codes[level]:
                for trade_date in SAMPLE_DATES:
                    try:
                        query_date = datetime.strptime(
                            trade_date,
                            "%Y%m%d",
                        ).strftime("%Y-%m-%d")
                        frame = client.query(
                            "sw_mins",
                            cache_key=f"{ts_code}_{trade_date}_5min_v26_probe",
                            ts_code=ts_code,
                            freq="5min",
                            start_date=f"{query_date} 13:00:00",
                            end_date=f"{query_date} 15:00:00",
                            fields=SW_MINUTE_FIELDS,
                        )
                        record = sw_minute_probe_record(
                            frame,
                            ts_code=ts_code,
                            trade_date=trade_date,
                        )
                    except Exception as error:
                        record = {
                            "ts_code": ts_code,
                            "trade_date": trade_date,
                            "coverage_pass": False,
                            "error": str(error)[:500],
                        }
                    record["level"] = level
                    minute_probes.append(record)

        classification_pass = bool(
            published_counts.get("L2", 0) >= 80
            and published_counts.get("L3", 0) >= 150
            and all(len(codes) == 4 for codes in selected_codes.values())
        )
        membership_pass = bool(
            membership_probes
            and all(row["coverage_pass"] for row in membership_probes)
        )
        minute_pass = bool(
            minute_probes
            and all(row["coverage_pass"] for row in minute_probes)
        )
        return {
            "passed": classification_pass and membership_pass and minute_pass,
            "classification_pass": classification_pass,
            "published_index_counts": published_counts,
            "selected_index_codes": selected_codes,
            "membership_pass": membership_pass,
            "membership_rows": int(len(membership)),
            "membership_probes": membership_probes,
            "minute_pass": minute_pass,
            "minute_probe_count": len(minute_probes),
            "minute_passed_count": sum(
                bool(row["coverage_pass"]) for row in minute_probes
            ),
            "minute_probes": minute_probes,
        }
    except Exception as error:
        return {
            "passed": False,
            "classification_pass": False,
            "membership_pass": False,
            "minute_pass": False,
            "error": str(error)[:500],
        }


if __name__ == "__main__":
    raise SystemExit(main())
