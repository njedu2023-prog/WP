from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import tushare as ts

from wp.v3.contracts import DEFAULT_SIGNAL_SLOTS, load_v3_config
from wp.v3.history import TushareHistoryClient
from wp.v3.live_data import build_live_feature_frame, capture_live_minute_snapshot


ROOT = Path(__file__).resolve().parents[1]
CN_TZ = ZoneInfo("Asia/Shanghai")


def due_slot(now: datetime) -> str:
    hhmm = now.strftime("%H:%M")
    due = [slot for slot in DEFAULT_SIGNAL_SLOTS if slot <= hhmm]
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
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build exact WP V5 live causal features.")
    parser.add_argument("--signal-slot", choices=DEFAULT_SIGNAL_SLOTS)
    args = parser.parse_args()
    if args.signal_slot:
        os.environ["WP_V3_SIGNAL_SLOT"] = args.signal_slot
    path, manifest = build_live_input()
    print(path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
