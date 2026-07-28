from __future__ import annotations

import json
import os
import subprocess
import sys
import time as time_module
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import tushare as ts

try:
    from build_wp_v3_live_input import (
        build_live_input,
        capture_entry_settlement_input,
        capture_warmup_input,
    )
except ModuleNotFoundError:  # pragma: no cover - package import in tests
    from scripts.build_wp_v3_live_input import (
        build_live_input,
        capture_entry_settlement_input,
        capture_warmup_input,
    )


CN_TZ = ZoneInfo("Asia/Shanghai")
SCHEDULE_GRACE_SECONDS = int(os.environ.get("WP_SCHEDULE_GRACE_SECONDS", "120"))
PREP_START = time(13, 40)
RUN_START = time(14, 0)
RUN_END = time(15, 0)
WARMUP_SLOTS = {"14:00", "14:05", "14:10", "14:15"}
MAX_SLOT_LATENESS_SECONDS = int(
    os.environ.get("WP_MAX_SLOT_LATENESS_SECONDS", "420")
)
LIVE_COMMIT_PATHS = [
    "outputs/html_reports/latest.html",
    "outputs/csv/wp_buy_plan.csv",
    "outputs/csv/wp_buy_plan_validation.csv",
    "outputs/csv/wp_v3_live_predictions.csv",
    "outputs/json/wp_buy_plan_validation.json",
    "outputs/json/wp_decision_support.json",
    "outputs/json/wp_strategy_ledger.json",
    "outputs/json/wp_v3_candidate_ledger.json",
    "outputs/json/wp_model_registry_v3.json",
    "outputs/json/wp_manifest.json",
    "data/v3/latest/wp_v3_live_features.csv",
    "data/v3/latest/wp_v3_live_manifest.json",
    "data/v3/latest/wp_v3_entry_settlement.csv",
    "data/v3/latest/wp_v3_entry_settlement_manifest.json",
]


def now_cn() -> datetime:
    return datetime.now(CN_TZ)


def today_window(now: datetime) -> tuple[datetime, datetime] | None:
    today = now.date()
    prep_dt = datetime.combine(today, PREP_START, CN_TZ)
    start_dt = datetime.combine(today, RUN_START, CN_TZ)
    end_dt = datetime.combine(today, RUN_END, CN_TZ)
    if prep_dt <= now <= end_dt:
        return start_dt, end_dt
    return None


def in_run_window(now: datetime) -> bool:
    today = now.date()
    start = datetime.combine(today, RUN_START, CN_TZ)
    end = datetime.combine(today, time(15, 10), CN_TZ)
    return start <= now <= end


def is_trade_day(token: str, day: str) -> bool:
    ts.set_token(token)
    pro = ts.pro_api()
    cal = pro.trade_cal(exchange="SSE", start_date=day, end_date=day)
    return bool(len(cal) and int(cal.iloc[0].get("is_open", 0)) == 1)


def _latest_file(pattern: str) -> str | None:
    matches = [path for path in Path.cwd().glob(pattern) if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime).as_posix()


def output_commit_paths() -> list[str]:
    paths = list(LIVE_COMMIT_PATHS)
    manifest_path = Path("outputs/json/wp_manifest.json")
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        evidence_path = str(manifest.get("signal_evidence_path") or "")
        if evidence_path:
            evidence_dir = Path(evidence_path).parent
            if evidence_dir.is_dir():
                paths.append(evidence_dir.as_posix())
    latest_archive = _latest_file("outputs/html_reports/archive/*/*.html")
    if latest_archive:
        paths.append(latest_archive)
    return paths


def run_once(
    signal_slot: str | None = None,
    *,
    settlement_slot: str | None = None,
) -> None:
    env = os.environ.copy()
    env["WP_MODE"] = "live"
    if signal_slot:
        env["WP_V3_SIGNAL_SLOT"] = signal_slot
    current = now_cn()
    if (
        settlement_slot is None
        and time(14, 55) <= current.time() < time(15, 0)
    ):
        settlement_slot = "14:55"
    live_path = Path("data/v3/latest/wp_v3_live_features.csv")
    if settlement_slot:
        settlement_path, settlement_manifest = capture_entry_settlement_input(
            settlement_slot=settlement_slot,
            root=Path.cwd(),
            env=env,
        )
        env["WP_V3_ENTRY_SETTLEMENT_CSV"] = settlement_path.as_posix()
        env["WP_V3_ENTRY_SETTLEMENT_MANIFEST"] = settlement_path.with_name(
            "wp_v3_entry_settlement_manifest.json"
        ).as_posix()
        env["WP_EXPECTED_TRADE_DATE"] = str(
            settlement_manifest["trade_date"]
        )
        print(
            "WP V6 entry settlement ready: "
            f"slot={settlement_slot} "
            f"observed={settlement_manifest['observed_symbols']}/"
            f"{settlement_manifest['requested_symbols']}"
        )
    requested_signal = str(signal_slot or "")
    if (
        "14:20" <= requested_signal <= "14:50"
        or (
            not requested_signal
            and time(14, 20) <= current.time() <= time(14, 52)
        )
    ):
        source_path, source_manifest = build_live_input(root=Path.cwd(), env=env)
        env["WP_V3_SOURCE_CSV"] = source_path.as_posix()
        env["WP_EXPECTED_TRADE_DATE"] = str(source_manifest["trade_date"])
        env["WP_V3_SIGNAL_SLOT"] = str(source_manifest["signal_slot"])
        print(
            "WP V6 causal source ready: "
            f"{source_path} slot={source_manifest['signal_slot']}"
        )
    elif current.time() > time(14, 50) and live_path.exists():
        env["WP_V3_SOURCE_CSV"] = live_path.as_posix()
        print("WP V6 uses the frozen 14:50 feature snapshot for close-state rendering.")
    else:
        print("::warning::WP V6 live source is absent before the signal window.")
    manifest_path = Path("outputs/json/wp_manifest.json")
    manifest_before = manifest_path.read_bytes() if manifest_path.exists() else None
    subprocess.run([sys.executable, "-m", "wp.main"], check=True, env=env)
    manifest_after = manifest_path.read_bytes() if manifest_path.exists() else None
    if manifest_before == manifest_after:
        print("Skip GitHub output commit: WP manifest is unchanged.")
        return
    commit_paths = output_commit_paths()
    subprocess.run(
        [sys.executable, "scripts/github_commit_paths.py", "Update WP outputs", *commit_paths],
        check=True,
        env=env,
    )


def run_once_if_due() -> None:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    current = now_cn()
    trade_date = current.strftime("%Y%m%d")
    if token:
        if not is_trade_day(token, trade_date):
            print(f"Skip WP update: {trade_date} is not an A-share trading day.")
            return
    else:
        print("WP calendar fallback: TUSHARE_TOKEN is not configured; upstream data freshness will gate outputs.")

    if not in_run_window(current):
        print(f"Skip WP update outside A-share trading window: {current:%Y-%m-%d %H:%M:%S}")
        return

    print(f"WP single update started: {current:%Y-%m-%d %H:%M:%S}")
    run_once()


def run_session() -> None:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    current = now_cn()
    trade_date = current.strftime("%Y%m%d")
    if token:
        if not is_trade_day(token, trade_date):
            print(f"Skip WP session: {trade_date} is not an A-share trading day.")
            return
    else:
        print("WP session calendar fallback: TUSHARE_TOKEN is not configured; upstream data freshness will gate outputs.")

    window = today_window(current)
    if window is None:
        print(f"Skip WP session outside trading session prep/window: {current:%Y-%m-%d %H:%M:%S}")
        return

    start_dt, _ = window
    if current < start_dt:
        wait_seconds = max(0.0, (start_dt - current).total_seconds())
        print(f"Wait until WP session start: {start_dt:%Y-%m-%d %H:%M:%S}, wait={wait_seconds:.0f}s")
        time_module.sleep(wait_seconds)

    failures: list[str] = []
    schedule = [
        datetime.combine(current.date(), time(14, minute), CN_TZ)
        for minute in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55)
    ] + [datetime.combine(current.date(), time(15, 0), CN_TZ)]
    for scheduled_at in schedule:
        current = now_cn()
        capture_at = scheduled_at + timedelta(seconds=SCHEDULE_GRACE_SECONDS)
        if current < capture_at:
            wait_seconds = (capture_at - current).total_seconds()
            print(
                f"Wait for completed WP slot {scheduled_at:%H:%M}; "
                f"wait={wait_seconds:.0f}s"
            )
            time_module.sleep(wait_seconds)
        started_at = now_cn()
        lateness = (started_at - scheduled_at).total_seconds()
        if lateness > MAX_SLOT_LATENESS_SECONDS:
            message = (
                f"missed anchored slot {scheduled_at:%H:%M}; "
                f"lateness={lateness:.0f}s"
            )
            failures.append(message)
            print(f"::error::{message}")
            continue
        print(
            f"WP completed-bar iteration {scheduled_at:%H:%M} started: "
            f"{started_at:%Y-%m-%d %H:%M:%S}; lateness={lateness:.0f}s"
        )
        try:
            slot = scheduled_at.strftime("%H:%M")
            if slot in WARMUP_SLOTS:
                manifest = capture_warmup_input(
                    observation_slot=slot,
                    root=Path.cwd(),
                    env=os.environ,
                )
                print(
                    "WP V6 warmup snapshot ready: "
                    f"slot={slot} rows={manifest['row_count']} "
                    f"coverage={manifest['tail_universe_coverage']:.2%}"
                )
            elif slot == "14:55":
                run_once(settlement_slot=slot)
            else:
                run_once(
                    slot if slot <= "14:50" else None,
                    settlement_slot=(
                        slot
                        if "14:25" <= slot <= "14:50"
                        else None
                    ),
                )
        except Exception as error:
            message = f"{scheduled_at:%H:%M} failed: {error}"
            failures.append(message)
            print(f"::error::{message}")

    print(f"WP session completed: {now_cn():%Y-%m-%d %H:%M:%S}")
    if failures:
        raise RuntimeError("WP session completed with slot failures: " + "; ".join(failures))


def main() -> None:
    mode = os.environ.get("WP_RUN_MODE", "once").strip().lower()
    if mode == "session":
        run_session()
        return
    if mode == "auto":
        current = now_cn()
        if current.time() <= time(14, 57):
            run_session()
        else:
            run_once_if_due()
        return
    run_once_if_due()


if __name__ == "__main__":
    main()
