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
    audit_stock_slot_frame,
    build_stock_slot_frame,
    normalize_membership,
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
    return audit_stock_slot_frame(stock_slots, trade_date=trade_date)


if __name__ == "__main__":
    raise SystemExit(main())
