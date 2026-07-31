from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import tushare as ts

from wp.v3.history import TushareHistoryClient
from wp.v3.io import atomic_write_json, file_sha256
from wp.v3.v28_industry_peer import (
    MEMBER_FIELDS,
    SIGNAL_SLOTS,
    build_stock_slot_frame,
    normalize_membership,
    peer_group_count,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATES = (
    "20230728",
    "20230825",
    "20230928",
    "20231031",
    "20231130",
    "20231229",
    "20240315",
    "20240927",
    "20250115",
    "20250723",
    "20260115",
    "20260723",
)
MINUTE_COLUMNS = (
    "ts_code",
    "trade_date",
    "trade_time",
    "close",
    "amount",
)
MINIMUM_STOCKS_PER_SLOT = 1_000
MINIMUM_FIELD_COVERAGE = 0.95
MINIMUM_PRECLOSE_COVERAGE = 0.98
MINIMUM_L2_GROUPS = 80
MINIMUM_L3_GROUPS = 120


def main() -> int:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for the V28 data probe")
    history_root = Path(
        os.getenv(
            "WP_V28_HISTORY_ROOT",
            str(ROOT / "artifacts" / "wp_v3_history"),
        )
    )
    output = Path(
        os.getenv(
            "WP_V28_PROBE_OUTPUT",
            str(ROOT / "artifacts" / "wp_v28_data_probe"),
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    client = TushareHistoryClient(
        ts.pro_api(token),
        output / "query_cache",
        page_size=8_000,
        requests_per_minute=120,
        attempts=3,
    )

    membership_frames = [
        client.query(
            "index_member_all",
            cache_key=f"{state}_v28_probe",
            paged=True,
            is_new=state,
            fields=MEMBER_FIELDS,
        )
        for state in ("Y", "N")
    ]
    membership = normalize_membership(
        pd.concat(membership_frames, ignore_index=True)
    )
    minute_manifest = load_minute_manifest(history_root / "minute")
    probes: list[dict[str, Any]] = []
    partition_hashes: dict[str, str] = {}
    for trade_date in SAMPLE_DATES:
        try:
            month = trade_date[:6]
            minute_path = history_root / "minute" / (
                f"wp_v3_minutes_{month}.parquet"
            )
            if month not in partition_hashes:
                expected = str(
                    (minute_manifest.get("partitions") or {})
                    .get(month, {})
                    .get("sha256")
                    or ""
                )
                actual = file_sha256(minute_path)
                if not expected or actual != expected:
                    raise RuntimeError(
                        f"minute partition digest mismatch for {month}"
                    )
                partition_hashes[month] = actual
            minutes = pd.read_parquet(
                minute_path,
                columns=list(MINUTE_COLUMNS),
            )
            minutes = minutes.loc[
                minutes["trade_date"].astype(str).eq(trade_date)
            ].copy()
            daily = client.query(
                "daily",
                cache_key=f"{trade_date}_v28_preclose_probe",
                trade_date=trade_date,
                fields="ts_code,trade_date,pre_close",
            )
            probes.append(
                probe_date(
                    minutes,
                    daily,
                    membership,
                    trade_date=trade_date,
                )
            )
        except Exception as error:
            probes.append(
                {
                    "trade_date": trade_date,
                    "coverage_pass": False,
                    "error": str(error)[:500],
                }
            )

    passed = bool(probes and all(row["coverage_pass"] for row in probes))
    payload = {
        "schema_version": "wp_v28_fine_industry_peer_data_probe_1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "profit_outcomes_read": False,
        "future_information_allowed": False,
        "source_panel_run_id": 30_600_193_544,
        "probe_dates": list(SAMPLE_DATES),
        "passed_dates": sum(bool(row["coverage_pass"]) for row in probes),
        "probes": probes,
        "verified_minute_partitions": partition_hashes,
        "v28_full_backfill_authorized": passed,
        "model_research_authorized": False,
        "next_gate": (
            "full_three_year_outcome_blind_peer_coverage_audit"
            if passed
            else "stop_v28_data_direction"
        ),
    }
    atomic_write_json(
        output / "wp_v28_fine_industry_peer_data_probe.json",
        payload,
    )
    print(
        "WP_V28_DATA_PROBE_RESULT="
        + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def load_minute_manifest(minute_dir: Path) -> dict[str, Any]:
    path = minute_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "wp_historical_minutes_3":
        raise RuntimeError("V28 minute source schema is not immutable V9")
    return payload


def probe_date(
    minute_bars: pd.DataFrame,
    daily_preclose: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    trade_date: str,
) -> dict[str, Any]:
    stock_slots = build_stock_slot_frame(
        minute_bars,
        daily_preclose,
        membership,
        trade_date=trade_date,
    )
    timestamps = pd.to_datetime(
        stock_slots["trade_timestamp"],
        errors="coerce",
    )
    date_consistent = bool(
        timestamps.notna().all()
        and timestamps.dt.strftime("%Y%m%d").eq(trade_date).all()
    )
    duplicate_identity = bool(
        stock_slots.duplicated(["trade_date", "signal_slot", "ts_code"]).any()
    )
    slot_records: list[dict[str, Any]] = []
    for slot in SIGNAL_SLOTS:
        frame = stock_slots.loc[stock_slots["signal_slot"].eq(slot)].copy()
        rows = int(len(frame))
        preclose_coverage = (
            float(
                pd.to_numeric(frame["pre_close"], errors="coerce")
                .gt(0)
                .mean()
            )
            if rows
            else 0.0
        )
        l2_coverage = (
            float(frame["l2_code"].fillna("").astype(str).str.strip().ne("").mean())
            if rows
            else 0.0
        )
        l3_coverage = (
            float(frame["l3_code"].fillna("").astype(str).str.strip().ne("").mean())
            if rows
            else 0.0
        )
        return_coverage = (
            float(
                pd.to_numeric(
                    frame["ret_from_prev_close_pct"],
                    errors="coerce",
                )
                .notna()
                .mean()
            )
            if rows
            else 0.0
        )
        tail_coverage = (
            float(
                pd.to_numeric(frame["ret_20m_pct"], errors="coerce")
                .notna()
                .mean()
            )
            if rows
            else 0.0
        )
        l2_groups = peer_group_count(
            frame,
            level="l2_code",
            minimum_members=5,
        )
        l3_groups = peer_group_count(
            frame,
            level="l3_code",
            minimum_members=3,
        )
        passed = bool(
            rows >= MINIMUM_STOCKS_PER_SLOT
            and preclose_coverage >= MINIMUM_PRECLOSE_COVERAGE
            and l2_coverage >= MINIMUM_FIELD_COVERAGE
            and l3_coverage >= MINIMUM_FIELD_COVERAGE
            and return_coverage >= MINIMUM_FIELD_COVERAGE
            and tail_coverage >= MINIMUM_FIELD_COVERAGE
            and l2_groups >= MINIMUM_L2_GROUPS
            and l3_groups >= MINIMUM_L3_GROUPS
        )
        slot_records.append(
            {
                "signal_slot": slot,
                "rows": rows,
                "preclose_coverage": preclose_coverage,
                "l2_coverage": l2_coverage,
                "l3_coverage": l3_coverage,
                "return_coverage": return_coverage,
                "tail_20m_coverage": tail_coverage,
                "eligible_l2_groups": l2_groups,
                "eligible_l3_groups": l3_groups,
                "coverage_pass": passed,
            }
        )
    return {
        "trade_date": trade_date,
        "rows": int(len(stock_slots)),
        "date_consistent": date_consistent,
        "duplicate_identity": duplicate_identity,
        "slot_count": len(slot_records),
        "slot_records": slot_records,
        "coverage_pass": bool(
            date_consistent
            and not duplicate_identity
            and len(slot_records) == len(SIGNAL_SLOTS)
            and all(record["coverage_pass"] for record in slot_records)
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
