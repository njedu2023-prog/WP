from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import tushare as ts

from wp.v3.history import TushareHistoryClient
from wp.v3.io import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATES = (
    "20230825",
    "20231229",
    "20250115",
    "20260723",
)
SAMPLE_CODES = (
    "600000.SH",
    "000001.SZ",
    "601318.SH",
    "300750.SZ",
    "688981.SH",
)
CYQ_FIELDS = (
    "ts_code,trade_date,his_low,his_high,cost_5pct,cost_15pct,"
    "cost_50pct,cost_85pct,cost_95pct,weight_avg,winner_rate"
)
MARGIN_FIELDS = (
    "trade_date,ts_code,name,rzye,rqye,rzmre,rqyl,rzche,rqchl,"
    "rqmcl,rzrqye"
)
TOP_LIST_FIELDS = (
    "trade_date,ts_code,name,close,pct_change,turnover_rate,amount,"
    "l_sell,l_buy,l_amount,net_amount,net_rate,amount_rate,"
    "float_values,reason"
)
ANNOUNCEMENT_FIELDS = "ann_date,ts_code,name,title,url,rec_time"

CYQ_COST_COLUMNS = (
    "his_low",
    "cost_5pct",
    "cost_15pct",
    "cost_50pct",
    "cost_85pct",
    "cost_95pct",
    "his_high",
)
MARGIN_NUMERIC_COLUMNS = (
    "rzye",
    "rqye",
    "rzmre",
    "rqyl",
    "rzche",
    "rqchl",
    "rqmcl",
    "rzrqye",
)


def main() -> int:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for the V25 data probe")
    output = Path(
        os.getenv(
            "WP_V25_PROBE_OUTPUT",
            str(ROOT / "artifacts" / "wp_v25_data_probe"),
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

    cyq_rows = probe_cyq_history(client)
    margin_rows = [
        probe_margin_cross_section(client, trade_date=trade_date)
        for trade_date in SAMPLE_DATES
    ]
    top_list_rows = [
        probe_sparse_cross_section(
            client,
            api_name="top_list",
            date_column="trade_date",
            trade_date=trade_date,
            fields=TOP_LIST_FIELDS,
        )
        for trade_date in SAMPLE_DATES
    ]
    announcement_rows = [
        probe_sparse_cross_section(
            client,
            api_name="anns_d",
            date_column="ann_date",
            trade_date=trade_date,
            fields=ANNOUNCEMENT_FIELDS,
        )
        for trade_date in SAMPLE_DATES
    ]

    cyq_pass = bool(
        cyq_rows
        and all(row["coverage_pass"] for row in cyq_rows)
    )
    margin_pass = bool(
        margin_rows
        and all(row["coverage_pass"] for row in margin_rows)
    )
    top_list_pass = bool(
        top_list_rows
        and all(row["coverage_pass"] for row in top_list_rows)
    )
    announcement_pass = bool(
        announcement_rows
        and all(row["coverage_pass"] for row in announcement_rows)
    )
    admitted = [
        name
        for name, passed in (
            ("previous_day_holder_cost_distribution", cyq_pass),
            ("previous_day_margin_positioning", margin_pass),
            ("previous_day_abnormal_trading_disclosure", top_list_pass),
            ("previous_day_announcement_metadata", announcement_pass),
        )
        if passed
    ]
    payload = {
        "schema_version": "wp_v25_positioning_data_probe_1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "profit_outcomes_read": False,
        "probe_dates": list(SAMPLE_DATES),
        "probe_codes": list(SAMPLE_CODES),
        "previous_day_holder_cost_distribution": {
            "passed": cyq_pass,
            "probes": cyq_rows,
        },
        "previous_day_margin_positioning": {
            "passed": margin_pass,
            "probes": margin_rows,
        },
        "previous_day_abnormal_trading_disclosure": {
            "passed": top_list_pass,
            "optional": True,
            "probes": top_list_rows,
        },
        "previous_day_announcement_metadata": {
            "passed": announcement_pass,
            "optional": True,
            "probes": announcement_rows,
        },
        "admitted_data_families": admitted,
        "v25_positioning_backfill_authorized": (
            cyq_pass and margin_pass
        ),
    }
    atomic_write_json(output / "wp_v25_positioning_data_probe.json", payload)
    print(
        "WP_V25_PROBE_RESULT="
        + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def probe_cyq_history(
    client: TushareHistoryClient,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ts_code in SAMPLE_CODES:
        try:
            frame = client.query(
                "cyq_perf",
                cache_key=f"{ts_code}_v25_probe",
                paged=True,
                ts_code=ts_code,
                start_date=min(SAMPLE_DATES),
                end_date=max(SAMPLE_DATES),
                fields=CYQ_FIELDS,
            )
            normalized = normalize_cyq(frame)
            for trade_date in SAMPLE_DATES:
                sample = normalized.loc[
                    normalized["trade_date"].eq(trade_date)
                ]
                rows.append(
                    cyq_probe_record(
                        sample,
                        ts_code=ts_code,
                        trade_date=trade_date,
                    )
                )
        except Exception as error:
            for trade_date in SAMPLE_DATES:
                rows.append(
                    {
                        "status": "error",
                        "api_name": "cyq_perf",
                        "trade_date": trade_date,
                        "ts_code": ts_code,
                        "rows": 0,
                        "coverage_pass": False,
                        "error": str(error)[:500],
                    }
                )
    return rows


def normalize_cyq(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reindex(columns=CYQ_FIELDS.split(",")).copy()
    result["trade_date"] = result["trade_date"].astype(str)
    result["ts_code"] = result["ts_code"].astype(str)
    for column in (*CYQ_COST_COLUMNS, "weight_avg", "winner_rate"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.sort_values(["ts_code", "trade_date"], kind="stable")


def cyq_probe_record(
    frame: pd.DataFrame,
    *,
    ts_code: str,
    trade_date: str,
) -> dict[str, Any]:
    unique = not frame.duplicated(["trade_date", "ts_code"]).any()
    complete = bool(
        len(frame) == 1
        and frame[
            [*CYQ_COST_COLUMNS, "weight_avg", "winner_rate"]
        ].notna().all(axis=None)
    )
    ordered = bool(
        complete
        and frame.loc[:, CYQ_COST_COLUMNS]
        .diff(axis=1)
        .iloc[:, 1:]
        .ge(0.0)
        .all(axis=None)
    )
    weighted_positive = bool(
        complete and float(frame["weight_avg"].iloc[0]) > 0.0
    )
    winner_bounded = bool(
        complete
        and 0.0 <= float(frame["winner_rate"].iloc[0]) <= 100.0
    )
    passed = bool(
        len(frame) == 1
        and unique
        and complete
        and ordered
        and weighted_positive
        and winner_bounded
    )
    return {
        "status": "ok",
        "api_name": "cyq_perf",
        "trade_date": trade_date,
        "ts_code": ts_code,
        "rows": int(len(frame)),
        "unique_stock_date": unique,
        "numeric_complete": complete,
        "cost_percentiles_ordered": ordered,
        "weighted_cost_positive": weighted_positive,
        "winner_rate_bounded": winner_bounded,
        "coverage_pass": passed,
    }


def probe_margin_cross_section(
    client: TushareHistoryClient,
    *,
    trade_date: str,
) -> dict[str, Any]:
    try:
        frame = client.query(
            "margin_detail",
            cache_key=f"{trade_date}_v25_probe",
            paged=True,
            trade_date=trade_date,
            fields=MARGIN_FIELDS,
        )
        frame = frame.reindex(columns=MARGIN_FIELDS.split(",")).copy()
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame["ts_code"] = frame["ts_code"].astype(str)
        duplicates = frame.duplicated(["trade_date", "ts_code"]).any()
        coverage = {
            column: float(
                pd.to_numeric(frame[column], errors="coerce").notna().mean()
            )
            for column in MARGIN_NUMERIC_COLUMNS
        }
        available = set(frame["ts_code"])
        sample_present = {
            code: code in available for code in SAMPLE_CODES
        }
        passed = bool(
            len(frame) >= 1_500
            and not duplicates
            and all(value >= 0.95 for value in coverage.values())
            and sum(sample_present.values()) >= 4
        )
        return {
            "status": "ok",
            "api_name": "margin_detail",
            "trade_date": trade_date,
            "rows": int(len(frame)),
            "unique_codes": int(frame["ts_code"].nunique()),
            "duplicate_stock_date": bool(duplicates),
            "numeric_coverage": coverage,
            "sample_codes_present": sample_present,
            "coverage_pass": passed,
        }
    except Exception as error:
        return {
            "status": "error",
            "api_name": "margin_detail",
            "trade_date": trade_date,
            "rows": 0,
            "unique_codes": 0,
            "coverage_pass": False,
            "error": str(error)[:500],
        }


def probe_sparse_cross_section(
    client: TushareHistoryClient,
    *,
    api_name: str,
    date_column: str,
    trade_date: str,
    fields: str,
) -> dict[str, Any]:
    try:
        frame = client.query(
            api_name,
            cache_key=f"{trade_date}_v25_probe",
            paged=True,
            **{date_column: trade_date},
            fields=fields,
        )
        required = {date_column, "ts_code"}
        schema_ok = required.issubset(frame.columns)
        date_ok = bool(
            schema_ok
            and not frame.empty
            and frame[date_column].astype(str).eq(trade_date).all()
        )
        passed = bool(schema_ok and date_ok and frame["ts_code"].notna().any())
        return {
            "status": "ok",
            "api_name": api_name,
            "trade_date": trade_date,
            "rows": int(len(frame)),
            "unique_codes": (
                int(frame["ts_code"].astype(str).nunique())
                if "ts_code" in frame
                else 0
            ),
            "schema_ok": schema_ok,
            "date_ok": date_ok,
            "coverage_pass": passed,
        }
    except Exception as error:
        return {
            "status": "error",
            "api_name": api_name,
            "trade_date": trade_date,
            "rows": 0,
            "unique_codes": 0,
            "coverage_pass": False,
            "error": str(error)[:500],
        }


if __name__ == "__main__":
    raise SystemExit(main())

