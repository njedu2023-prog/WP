from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
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
    "name",
    "price",
    "pre_close",
    "pct_chg",
    "amount",
    "sector_name",
    "pre_day_limitup",
    "today_limitup",
    "today_limit_up_price",
    "ret_5d",
    "ret_20d",
    "amount_ratio_5d",
    "amount_ratio_20d",
    "ma5_position",
    "ma20_position",
    "close_position",
    "realtime_source",
    "stock_age_days",
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


def _trade_date_from_slot(value: str) -> str:
    candidate = value.strip()[:10].replace("-", "")
    return candidate if len(candidate) == 8 and candidate.isdigit() else ""


def _resolved_fetch_trade_date(processor_root: Path) -> str:
    metadata_path = processor_root / "data" / "latest" / "_meta.json"
    if not metadata_path.is_file():
        return ""
    payload = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    candidate = str(payload.get("resolved_trade_date") or "").replace("-", "")
    return candidate if len(candidate) == 8 and candidate.isdigit() else ""


def _validate_source(
    csv_path: Path,
    manifest_path: Path,
    expected_trade_date: str,
) -> tuple[pd.DataFrame, dict, dict]:
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
    if not source_trade_date:
        raise RuntimeError("direct source trade date is missing")
    if expected_trade_date and source_trade_date != expected_trade_date:
        raise RuntimeError(
            f"direct source trade date {source_trade_date} does not match {expected_trade_date}"
        )

    quality = {
        "rows": int(len(frame)),
        "source_trade_date": source_trade_date,
        "required_columns": len(REQUIRED_COLUMNS),
        "required_columns_complete": True,
    }
    if frame.empty:
        if status != "empty_schema_ready":
            raise RuntimeError("direct source is empty without empty_schema_ready status")
        return frame, manifest, quality

    row_dates = (
        frame["trade_date"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace("-", "", regex=False)
    )
    if not row_dates.eq(source_trade_date).all():
        raise RuntimeError("direct rank input contains rows from another trade date")

    codes = frame["ts_code"].fillna("").astype(str).str.strip()
    if codes.eq("").any() or codes.duplicated().any():
        raise RuntimeError("direct rank input contains blank or duplicate stock codes")

    for column in ("name", "sector_name", "realtime_source", "update_time"):
        values = frame[column].fillna("").astype(str).str.strip()
        if values.eq("").any():
            raise RuntimeError(f"direct rank input contains blank {column}")
    if pd.to_datetime(frame["update_time"], errors="coerce").isna().any():
        raise RuntimeError("direct rank input contains invalid update_time")

    numeric_columns = (
        "price",
        "pre_close",
        "pct_chg",
        "amount",
        "pre_day_limitup",
        "today_limitup",
        "today_limit_up_price",
        "ret_5d",
        "ret_20d",
        "amount_ratio_5d",
        "amount_ratio_20d",
        "ma5_position",
        "ma20_position",
        "close_position",
        "stock_age_days",
    )
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    incomplete = [column for column in numeric_columns if numeric[column].isna().any()]
    if incomplete:
        raise RuntimeError(
            f"direct rank input contains invalid numeric values: {','.join(incomplete)}"
        )
    for column in (
        "price",
        "pre_close",
        "amount",
        "today_limit_up_price",
        "amount_ratio_5d",
        "amount_ratio_20d",
    ):
        if numeric[column].le(0).any():
            raise RuntimeError(f"direct rank input contains non-positive {column}")
    if numeric["pct_chg"].le(8).any():
        raise RuntimeError("direct rank input contains pct_chg not greater than 8")
    if not numeric["close_position"].between(0, 100).all():
        raise RuntimeError("direct rank input contains invalid close_position")
    if numeric["stock_age_days"].lt(0).any():
        raise RuntimeError("direct rank input contains invalid stock_age_days")
    for column in ("pre_day_limitup", "today_limitup"):
        if not numeric[column].isin({0, 1}).all():
            raise RuntimeError(f"direct rank input contains invalid {column}")

    meaningful_history = (
        numeric["amount_ratio_5d"].sub(1).abs().gt(1e-6)
        | numeric["ret_5d"].sub(numeric["pct_chg"]).abs().gt(1e-6)
        | numeric["ma5_position"].abs().gt(1e-6)
        | numeric["ma20_position"].abs().gt(1e-6)
    )
    history_coverage = float(meaningful_history.mean() * 100)
    if len(frame) >= 3 and history_coverage < 70:
        raise RuntimeError(
            f"direct rank input historical features are degraded: {history_coverage:.2f}%"
        )

    quality.update(
        {
            "unique_codes": int(codes.nunique()),
            "market_data_time_min": str(frame["update_time"].min()),
            "market_data_time_max": str(frame["update_time"].max()),
            "pct_chg_min": float(numeric["pct_chg"].min()),
            "pct_chg_max": float(numeric["pct_chg"].max()),
            "history_feature_coverage_pct": round(history_coverage, 2),
        }
    )
    return frame, manifest, quality


def build_direct_rank_input(
    *,
    root: Path = ROOT,
    upstream_root: Path | None = None,
    env: dict[str, str] | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    monotonic: Callable[[], float] = time.monotonic,
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

    target_slot = direct_env.get("WP_TARGET_SLOT", "").strip()
    if not direct_env.get("TRADE_DATE", "").strip() and target_slot:
        slot_trade_date = _trade_date_from_slot(target_slot)
        if slot_trade_date:
            direct_env["TRADE_DATE"] = slot_trade_date

    timeout_seconds = max(1, int(direct_env.get("WP_DIRECT_TIMEOUT_SECONDS", "1500")))
    deadline = monotonic() + timeout_seconds

    def run_processor(script: Path) -> None:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(str(script), timeout_seconds)
        command_runner(
            [sys.executable, str(script)],
            cwd=processor_root,
            env=direct_env,
            check=True,
            timeout=remaining,
        )

    try:
        run_processor(fetch_script)
        resolved_trade_date = _resolved_fetch_trade_date(processor_root)
        if resolved_trade_date:
            direct_env["TRADE_DATE"] = resolved_trade_date
        run_processor(pipeline_script)

        source_dir = processor_root / "data" / "wp" / "latest"
        source_csv = source_dir / "wp_latest_rank_input.csv"
        source_manifest = source_dir / "wp_manifest.json"
        expected_trade_date = direct_env.get("TRADE_DATE", "").strip()
        frame, manifest, quality = _validate_source(
            source_csv,
            source_manifest,
            expected_trade_date,
        )

        destination_dir = root / "data" / "direct" / "latest"
        destination_csv = destination_dir / "wp_latest_rank_input.csv"
        destination_manifest = destination_dir / "wp_manifest.json"
        _atomic_copy(source_csv, destination_csv)

        manifest.update(
            {
                "source_mode": "direct_tushare",
                "source_repository": "njedu2023-prog/a-share-top3-data",
                "processor_revision": direct_env.get("WP_UPSTREAM_PROCESSOR_REVISION", "").strip(),
                "scheduled_slot": target_slot
                or str(manifest.get("scheduled_slot") or ""),
                "direct_row_count": int(len(frame)),
                "direct_quality": quality,
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
