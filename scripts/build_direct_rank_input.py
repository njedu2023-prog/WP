from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CN_TZ = ZoneInfo("Asia/Shanghai")
REQUIRED_COLUMNS = {
    "trade_date",
    "update_time",
    "ts_code",
    "pct_chg",
    "amount",
    "pre_day_limitup",
    "today_limitup",
}
ACCEPTED_SOURCE_STATUSES = {"ok", "empty_schema_ready"}


@dataclass(frozen=True)
class DirectSourceResult:
    attempted: bool
    ok: bool
    source_path: str = ""
    manifest_path: str = ""
    error: str = ""


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def _atomic_write_json(payload: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def _compact_error(error: BaseException | str) -> str:
    text = " ".join(str(error).split())
    return text[:600]


def _validate_source(csv_path: Path, manifest_path: Path, expected_trade_date: str) -> tuple[pd.DataFrame, dict]:
    if not csv_path.is_file() or csv_path.stat().st_size <= 0:
        raise RuntimeError(f"direct rank input is missing: {csv_path}")
    if not manifest_path.is_file():
        raise RuntimeError(f"direct manifest is missing: {manifest_path}")

    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise RuntimeError(f"direct rank input is missing columns: {','.join(missing)}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    status = str(manifest.get("status") or "").strip()
    if status not in ACCEPTED_SOURCE_STATUSES:
        raise RuntimeError(f"direct source status is {status or 'missing'}")
    source_trade_date = str(manifest.get("source_trade_date") or "").replace("-", "")
    if source_trade_date and source_trade_date != expected_trade_date:
        raise RuntimeError(
            f"direct source trade date {source_trade_date} does not match {expected_trade_date}"
        )
    return frame, manifest


def build_direct_rank_input(
    *,
    root: Path = ROOT,
    upstream_root: Path | None = None,
    env: dict[str, str] | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> DirectSourceResult:
    runtime_env = dict(os.environ if env is None else env)
    token = runtime_env.get("TUSHARE_TOKEN", "").strip()
    if not token:
        return DirectSourceResult(True, False, error="TUSHARE_TOKEN is not configured in WP")

    processor_root = Path(
        upstream_root
        or runtime_env.get("WP_UPSTREAM_WORKTREE", "").strip()
        or root / "upstream"
    ).resolve()
    fetch_script = processor_root / "scripts" / "fetch_daily_snapshots.py"
    pipeline_script = processor_root / "scripts" / "wp" / "wp_pipeline.py"
    if not fetch_script.is_file() or not pipeline_script.is_file():
        return DirectSourceResult(
            True,
            False,
            error=f"upstream processor checkout is incomplete: {processor_root}",
        )

    direct_env = runtime_env.copy()
    defaults = {
        "ENABLE_AUCTION": "1",
        "ENABLE_MINUTE": "1",
        "REALTIME_MINUTE_ONLY": "1",
        "ENABLE_MARKET_MINUTE_SCAN": "1",
        "TRY_FULL_MARKET_MINUTE": "0",
        "REALTIME_QUOTE_CHUNK_SIZE": "300",
        "MINUTE_FREQ": "1min",
        "MAX_MINUTE_SYMBOLS": "6000",
        "WP_INTRADAY_MIN_PCT": "8",
    }
    for name, value in defaults.items():
        direct_env.setdefault(name, value)

    timeout_seconds = int(direct_env.get("WP_DIRECT_TIMEOUT_SECONDS", "1500"))
    try:
        command_runner(
            [sys.executable, str(fetch_script)],
            cwd=processor_root,
            env=direct_env,
            check=True,
            timeout=timeout_seconds,
        )
        command_runner(
            [sys.executable, str(pipeline_script)],
            cwd=processor_root,
            env=direct_env,
            check=True,
            timeout=timeout_seconds,
        )

        source_dir = processor_root / "data" / "wp" / "latest"
        source_csv = source_dir / "wp_latest_rank_input.csv"
        source_manifest = source_dir / "wp_manifest.json"
        expected_trade_date = (
            direct_env.get("TRADE_DATE", "").strip()
            or datetime.now(CN_TZ).strftime("%Y%m%d")
        )
        frame, manifest = _validate_source(source_csv, source_manifest, expected_trade_date)

        destination_dir = root / "data" / "direct" / "latest"
        destination_csv = destination_dir / "wp_latest_rank_input.csv"
        destination_manifest = destination_dir / "wp_manifest.json"
        _atomic_copy(source_csv, destination_csv)

        manifest.update(
            {
                "source_mode": "direct_tushare",
                "source_repository": "njedu2023-prog/a-share-top3-data",
                "processor_revision": direct_env.get("WP_UPSTREAM_PROCESSOR_REVISION", "").strip(),
                "scheduled_slot": direct_env.get("WP_TARGET_SLOT", "").strip()
                or str(manifest.get("scheduled_slot") or ""),
                "direct_row_count": int(len(frame)),
                "archive_mode": "wp_transaction_snapshot",
            }
        )
        _atomic_write_json(manifest, destination_manifest)
        return DirectSourceResult(
            True,
            True,
            source_path=str(destination_csv),
            manifest_path=str(destination_manifest),
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, pd.errors.ParserError, subprocess.SubprocessError) as exc:
        return DirectSourceResult(True, False, error=_compact_error(exc))
