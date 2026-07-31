from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import tushare as ts

from wp.v3.history import MINUTE_FIELDS, TushareHistoryClient
from wp.v3.io import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATES = (
    "20230828",
    "20240102",
    "20250116",
    "20260724",
)
SAMPLE_CODES = (
    "600000.SH",
    "000001.SZ",
    "601318.SH",
)
AUCTION_DATES = (
    "20240102",
    "20250116",
    "20260724",
)
MONEYFLOW_DATES = (
    "20230825",
    "20231229",
    "20250115",
    "20260723",
)
AUCTION_FIELDS = (
    "ts_code,trade_date,close,open,high,low,vol,amount,vwap"
)
MONEYFLOW_FIELDS = (
    "ts_code,trade_date,buy_sm_amount,sell_sm_amount,"
    "buy_md_amount,sell_md_amount,buy_lg_amount,sell_lg_amount,"
    "buy_elg_amount,sell_elg_amount,net_mf_amount"
)


def main() -> int:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for the V23 data probe")
    output = Path(
        os.getenv(
            "WP_V23_PROBE_OUTPUT",
            str(ROOT / "artifacts" / "wp_v23_data_probe"),
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    client = TushareHistoryClient(
        ts.pro_api(token),
        output / "cache",
        page_size=8_000,
        requests_per_minute=120,
        attempts=2,
    )

    minute_rows: list[dict[str, Any]] = []
    for trade_date in SAMPLE_DATES:
        for ts_code in SAMPLE_CODES:
            minute_rows.append(
                probe_minutes(
                    client,
                    trade_date=trade_date,
                    ts_code=ts_code,
                    frequency="1min",
                    expected_tail_rows=60,
                )
            )
    for trade_date in (SAMPLE_DATES[0], SAMPLE_DATES[-1]):
        minute_rows.append(
            probe_minutes(
                client,
                trade_date=trade_date,
                ts_code=SAMPLE_CODES[0],
                frequency="5min",
                expected_tail_rows=12,
            )
        )

    auction_rows = [
        probe_cross_section(
            client,
            api_name="stk_auction_o",
            trade_date=trade_date,
            fields=AUCTION_FIELDS,
            minimum_rows=1_000,
        )
        for trade_date in AUCTION_DATES
    ]
    moneyflow_rows = [
        probe_cross_section(
            client,
            api_name="moneyflow",
            trade_date=trade_date,
            fields=MONEYFLOW_FIELDS,
            minimum_rows=1_000,
        )
        for trade_date in MONEYFLOW_DATES
    ]

    one_minute = [
        row for row in minute_rows if row["frequency"] == "1min"
    ]
    one_minute_pass = bool(
        one_minute
        and all(
            row["status"] == "ok"
            and float(row["tail_coverage_ratio"]) >= 0.90
            and not row["duplicate_trade_time"]
            and bool(row["ohlc_consistent"])
            for row in one_minute
        )
    )
    auction_pass = bool(
        auction_rows
        and all(row["coverage_pass"] for row in auction_rows)
    )
    moneyflow_pass = bool(
        moneyflow_rows
        and all(row["coverage_pass"] for row in moneyflow_rows)
    )
    payload = {
        "schema_version": "wp_v23_point_in_time_data_probe_1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "profit_outcomes_read": False,
        "probe_dates": list(SAMPLE_DATES),
        "probe_codes": list(SAMPLE_CODES),
        "historical_1min": {
            "passed": one_minute_pass,
            "probes": minute_rows,
        },
        "opening_auction": {
            "passed": auction_pass,
            "probes": auction_rows,
        },
        "lagged_daily_l2_moneyflow": {
            "passed": moneyflow_pass,
            "causal_use": "previous_trade_day_only",
            "probes": moneyflow_rows,
        },
        "recommended_new_data_families": [
            family
            for family, passed in (
                ("historical_1min_microstructure", one_minute_pass),
                ("same_day_opening_auction", auction_pass),
                ("previous_day_l2_moneyflow", moneyflow_pass),
            )
            if passed
        ],
        "v23_backfill_authorized": one_minute_pass,
    }
    atomic_write_json(output / "wp_v23_data_probe.json", payload)
    print(
        "WP_V23_PROBE_RESULT="
        + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def probe_minutes(
    client: TushareHistoryClient,
    *,
    trade_date: str,
    ts_code: str,
    frequency: str,
    expected_tail_rows: int,
) -> dict[str, Any]:
    start = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 13:55:00"
    end = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} 15:00:00"
    try:
        frame = client.query(
            "stk_mins",
            cache_key=f"{ts_code}_{trade_date}_{frequency}_probe",
            ts_code=ts_code,
            start_date=start,
            end_date=end,
            freq=frequency,
            fields=MINUTE_FIELDS,
        )
        normalized = normalize_minute_probe(frame, trade_date=trade_date)
        tail = normalized.loc[
            normalized["trade_time"].dt.strftime("%H:%M").between(
                "14:01" if frequency == "1min" else "14:05",
                "15:00",
            )
        ].copy()
        return {
            "status": "ok",
            "trade_date": trade_date,
            "ts_code": ts_code,
            "frequency": frequency,
            "rows": int(len(normalized)),
            "tail_rows": int(len(tail)),
            "tail_coverage_ratio": min(
                len(tail) / max(expected_tail_rows, 1),
                1.0,
            ),
            "first_trade_time": (
                normalized["trade_time"].min().isoformat()
                if not normalized.empty
                else None
            ),
            "last_trade_time": (
                normalized["trade_time"].max().isoformat()
                if not normalized.empty
                else None
            ),
            "duplicate_trade_time": bool(
                normalized.duplicated(["ts_code", "trade_time"]).any()
            ),
            "ohlc_consistent": bool(ohlc_consistent(normalized)),
            "positive_amount_share": finite_positive_share(
                normalized,
                "amount",
            ),
            "positive_volume_share": finite_positive_share(
                normalized,
                "vol",
            ),
        }
    except Exception as error:
        return {
            "status": "error",
            "trade_date": trade_date,
            "ts_code": ts_code,
            "frequency": frequency,
            "rows": 0,
            "tail_rows": 0,
            "tail_coverage_ratio": 0.0,
            "duplicate_trade_time": False,
            "ohlc_consistent": False,
            "error": str(error)[:500],
        }


def probe_cross_section(
    client: TushareHistoryClient,
    *,
    api_name: str,
    trade_date: str,
    fields: str,
    minimum_rows: int,
) -> dict[str, Any]:
    try:
        frame = client.query(
            api_name,
            cache_key=f"{trade_date}_probe",
            paged=True,
            trade_date=trade_date,
            fields=fields,
        )
        codes = frame.get("ts_code", pd.Series(dtype=object)).astype(str)
        duplicates = (
            frame.duplicated(["trade_date", "ts_code"]).any()
            if {"trade_date", "ts_code"}.issubset(frame.columns)
            else True
        )
        numeric = [
            column
            for column in fields.split(",")
            if column not in {"trade_date", "ts_code"}
            and column in frame
        ]
        numeric_coverage = {
            column: float(
                pd.to_numeric(frame[column], errors="coerce").notna().mean()
            )
            for column in numeric
        }
        coverage_pass = bool(
            len(frame) >= minimum_rows
            and not duplicates
            and all(code in set(codes) for code in SAMPLE_CODES)
        )
        return {
            "status": "ok",
            "api_name": api_name,
            "trade_date": trade_date,
            "rows": int(len(frame)),
            "unique_codes": int(codes.nunique()),
            "sample_codes_present": {
                code: bool(code in set(codes)) for code in SAMPLE_CODES
            },
            "duplicate_stock_date": bool(duplicates),
            "numeric_coverage": numeric_coverage,
            "coverage_pass": coverage_pass,
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


def normalize_minute_probe(
    frame: pd.DataFrame,
    *,
    trade_date: str,
) -> pd.DataFrame:
    result = frame.reindex(columns=MINUTE_FIELDS.split(",")).copy()
    result["trade_time"] = pd.to_datetime(
        result["trade_time"],
        errors="coerce",
    )
    result = result.dropna(subset=["ts_code", "trade_time"])
    result = result.loc[
        result["trade_time"].dt.strftime("%Y%m%d").eq(trade_date)
    ].copy()
    for column in ("open", "high", "low", "close", "vol", "amount"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.sort_values(["ts_code", "trade_time"], kind="stable")


def ohlc_consistent(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    numeric = frame[["open", "high", "low", "close"]]
    finite = numeric.notna().all(axis=1)
    if not finite.all():
        return False
    return bool(
        (
            frame["high"].ge(frame[["open", "close", "low"]].max(axis=1))
            & frame["low"].le(
                frame[["open", "close", "high"]].min(axis=1)
            )
        ).all()
    )


def finite_positive_share(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.gt(0).mean())


if __name__ == "__main__":
    raise SystemExit(main())
