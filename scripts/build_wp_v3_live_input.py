from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import tushare as ts

from wp.v3.contracts import DEFAULT_SIGNAL_SLOTS, load_v3_config
from wp.v3.history import TushareHistoryClient
from wp.v3.io import atomic_write_csv, atomic_write_json
from wp.v3.ledger import load_shadow_ledger, session_records
from wp.v3.live_data import (
    build_live_feature_frame,
    capture_entry_settlement_frame,
    capture_live_minute_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
CN_TZ = ZoneInfo("Asia/Shanghai")


def due_slot(now: datetime) -> str:
    grace_seconds = int(os.environ.get("WP_SCHEDULE_GRACE_SECONDS", "120"))
    due = [
        slot
        for slot in DEFAULT_SIGNAL_SLOTS
        if datetime.combine(
            now.date(),
            datetime.strptime(slot, "%H:%M").time(),
            now.tzinfo,
        )
        + timedelta(seconds=grace_seconds)
        <= now
    ]
    if not due:
        raise RuntimeError("no V3 signal slot is due")
    return due[-1]


def build_live_input(
    *,
    root: Path = ROOT,
    env: dict[str, str] | None = None,
) -> tuple[Path, dict]:
    environment = env or os.environ
    token = environment.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for V3 live causal features")
    capture_started_at = datetime.now(CN_TZ)
    trade_date = (
        environment.get("WP_EXPECTED_TRADE_DATE", "").strip()
        or capture_started_at.strftime("%Y%m%d")
    )
    requested = environment.get("WP_V3_SIGNAL_SLOT", "").strip()
    signal_slot = (
        requested
        if requested in DEFAULT_SIGNAL_SLOTS
        else due_slot(capture_started_at)
    )
    config = load_v3_config(root / "config" / "wp_v3.yml")
    client = TushareHistoryClient(
        ts.pro_api(token),
        root / "data" / "v3" / "cache",
        page_size=config.history.tushare_page_size,
    )
    frame, manifest = build_live_feature_frame(
        client,
        trade_date=trade_date,
        signal_slot=signal_slot,
        config=config,
    )
    manifest = dict(manifest)
    manifest["capture_started_at"] = capture_started_at.isoformat()
    manifest["capture_completed_at"] = datetime.now(CN_TZ).isoformat()
    manifest["capture_contract"] = "anchored_signal_slot_snapshot"
    output_dir = root / "data" / "v3" / "latest"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "wp_v3_live_features.csv"
    json_path = output_dir / "wp_v3_live_manifest.json"
    atomic_write_csv(frame, csv_path)
    atomic_write_json(json_path, manifest)
    return csv_path, manifest


def capture_warmup_input(
    *,
    observation_slot: str,
    root: Path = ROOT,
    env: dict[str, str] | None = None,
) -> dict:
    environment = env or os.environ
    token = environment.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for V3 warmup snapshots")
    now = datetime.now(CN_TZ)
    trade_date = (
        environment.get("WP_EXPECTED_TRADE_DATE", "").strip()
        or now.strftime("%Y%m%d")
    )
    config = load_v3_config(root / "config" / "wp_v3.yml")
    client = TushareHistoryClient(
        ts.pro_api(token),
        root / "data" / "v3" / "cache",
        page_size=config.history.tushare_page_size,
    )
    return capture_live_minute_snapshot(
        client,
        trade_date=trade_date,
        observation_slot=observation_slot,
        config=config,
    )


def capture_entry_settlement_input(
    *,
    settlement_slot: str,
    root: Path = ROOT,
    env: dict[str, str] | None = None,
) -> tuple[Path, dict]:
    environment = env or os.environ
    token = environment.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for entry settlement")
    capture_started_at = datetime.now(CN_TZ)
    trade_date = (
        environment.get("WP_EXPECTED_TRADE_DATE", "").strip()
        or capture_started_at.strftime("%Y%m%d")
    )
    config = load_v3_config(root / "config" / "wp_v3.yml")
    ledger = load_shadow_ledger(
        root / "outputs" / "json" / "wp_v3_candidate_ledger.json"
    )
    session = next(
        (
            item
            for item in ledger.get("sessions", [])
            if str(item.get("trade_date")) == trade_date
        ),
        {},
    )
    codes = [
        str(candidate.get("ts_code"))
        for candidate in session_records(session)
        if str(candidate.get("entry_benchmark_slot") or "")
        == settlement_slot
        and str(candidate.get("entry_benchmark_status") or "PENDING")
        == "PENDING"
    ]
    client = TushareHistoryClient(
        ts.pro_api(token),
        root / "data" / "v3" / "cache",
        page_size=config.history.tushare_page_size,
    )
    frame, manifest = capture_entry_settlement_frame(
        client,
        trade_date=trade_date,
        settlement_slot=settlement_slot,
        ts_codes=codes,
        config=config,
    )
    manifest = {
        **manifest,
        "capture_started_at": capture_started_at.isoformat(),
        "capture_completed_at": datetime.now(CN_TZ).isoformat(),
    }
    output_dir = root / "data" / "v3" / "latest"
    csv_path = output_dir / "wp_v3_entry_settlement.csv"
    json_path = output_dir / "wp_v3_entry_settlement_manifest.json"
    atomic_write_csv(frame, csv_path)
    atomic_write_json(json_path, manifest)
    return csv_path, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build exact WP V9 live causal features.")
    parser.add_argument("--signal-slot", choices=DEFAULT_SIGNAL_SLOTS)
    parser.add_argument("--settlement-slot")
    args = parser.parse_args()
    if args.settlement_slot:
        path, manifest = capture_entry_settlement_input(
            settlement_slot=args.settlement_slot,
        )
        print(path)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.signal_slot:
        os.environ["WP_V3_SIGNAL_SLOT"] = args.signal_slot
    path, manifest = build_live_input()
    print(path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
