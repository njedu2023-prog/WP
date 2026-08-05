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
from wp.v3.ledger import load_shadow_ledger

try:
    from build_wp_v3_live_input import (
        build_live_input,
        capture_entry_settlement_input,
        warm_live_reference_input,
    )
except ModuleNotFoundError:  # pragma: no cover - package import in tests
    from scripts.build_wp_v3_live_input import (
        build_live_input,
        capture_entry_settlement_input,
        warm_live_reference_input,
    )


CN_TZ = ZoneInfo("Asia/Shanghai")
SCHEDULE_GRACE_SECONDS = int(os.environ.get("WP_SCHEDULE_GRACE_SECONDS", "120"))
PREP_START = time(13, 45)
RUN_START = time(14, 0)
RUN_END = time(15, 0)
LATE_RECOVERY_START = time(14, 7)
LATE_RECOVERY_END = time(16, 10)
MAX_CAPTURE_LATENESS_SECONDS = int(
    os.environ.get("WP_MAX_CAPTURE_LATENESS_SECONDS", "45")
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
    start = datetime.combine(today, time(14, 0), CN_TZ)
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


def _has_fixed_signal_session(trade_date: str) -> bool:
    ledger = load_shadow_ledger(
        Path("outputs/json/wp_v3_candidate_ledger.json")
    )
    return any(
        str(session.get("trade_date") or "") == trade_date
        and "14:00" in {
            str(slot)
            for slot in session.get("covered_slots", [])
        }
        for session in ledger.get("sessions", [])
    )


def run_once(
    signal_slot: str | None = None,
    *,
    settlement_slot: str | None = None,
    late_recovery: bool = False,
) -> None:
    env = os.environ.copy()
    env["WP_MODE"] = "live"
    if signal_slot:
        env["WP_V3_SIGNAL_SLOT"] = signal_slot
    if late_recovery:
        env["WP_V3_LATE_RECOVERY"] = "1"
        env["WP_V3_RECOVERY_REASON"] = "missed_scheduler"
    current = now_cn()
    if (
        settlement_slot is None
        and time(14, 5) <= current.time() < time(14, 10)
    ):
        settlement_slot = "14:05"
    live_path = Path("data/v3/latest/wp_v3_live_features.csv")
    manifest_path = Path("outputs/json/wp_manifest.json")
    manifest_before = (
        manifest_path.read_bytes() if manifest_path.exists() else None
    )
    trade_date = current.strftime("%Y%m%d")
    if settlement_slot and (
        not live_path.exists()
        or not _has_fixed_signal_session(trade_date)
    ):
        env["WP_V3_SIGNAL_SLOT"] = "14:00"
        source_path, source_manifest = build_live_input(
            root=Path.cwd(),
            env=env,
        )
        env["WP_V3_SOURCE_CSV"] = source_path.as_posix()
        env["WP_EXPECTED_TRADE_DATE"] = str(
            source_manifest["trade_date"]
        )
        print(
            "WP V41 late recovery rebuilt the immutable 14:00 "
            "snapshot before entry settlement."
        )
        subprocess.run(
            [sys.executable, "-m", "wp.main"],
            check=True,
            env=env,
        )
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
            "WP V41 entry settlement ready: "
            f"slot={settlement_slot} "
            f"observed={settlement_manifest['observed_symbols']}/"
            f"{settlement_manifest['requested_symbols']}"
        )
    requested_signal = str(signal_slot or "")
    if (
        requested_signal == "14:00"
        or (
            not requested_signal
            and settlement_slot is None
            and time(14, 0) <= current.time() <= time(14, 7)
        )
    ):
        source_path, source_manifest = build_live_input(root=Path.cwd(), env=env)
        env["WP_V3_SOURCE_CSV"] = source_path.as_posix()
        env["WP_EXPECTED_TRADE_DATE"] = str(source_manifest["trade_date"])
        env["WP_V3_SIGNAL_SLOT"] = str(source_manifest["signal_slot"])
        print(
            "WP V41 causal source ready: "
            f"{source_path} slot={source_manifest['signal_slot']}"
        )
    elif current.time() > time(14, 0) and live_path.exists():
        env["WP_V3_SOURCE_CSV"] = live_path.as_posix()
        print(
            "WP V41 reuses the immutable 14:00 feature snapshot for "
            "settlement and close-state rendering."
        )
    else:
        print("::warning::WP V41 live source is absent before 14:00.")
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

    same_day_recovery = bool(
        os.environ.get("GITHUB_EVENT_NAME", "").strip() == "push"
        and LATE_RECOVERY_START <= current.time() <= LATE_RECOVERY_END
        and not _has_fixed_signal_session(trade_date)
    )
    if same_day_recovery:
        print(
            "WP same-day recovery started: rebuilding immutable "
            "14:00 signal and 14:05 entry benchmark from rt_min_daily."
        )
        run_once("14:00", late_recovery=True)
        run_once(settlement_slot="14:05", late_recovery=True)
        print(
            "WP same-day recovery completed as non-prospective evidence."
        )
        return

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

    if token:
        warmed = warm_live_reference_input(
            root=Path.cwd(),
            env=os.environ.copy(),
        )
        print(
            "WP compact runtime references ready: "
            f"{warmed['prior_trade_date_start']}.."
            f"{warmed['prior_trade_date_end']} "
            f"dates={warmed['prior_trade_date_count']} "
            f"stocks={warmed['stock_basic_rows']}"
        )

    failures: list[str] = []
    schedule = [
        datetime.combine(current.date(), slot, CN_TZ)
        for slot in (time(14, 0), time(14, 5), time(14, 10), time(15, 0))
    ]
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
        capture_lateness = (started_at - capture_at).total_seconds()
        if capture_lateness > MAX_CAPTURE_LATENESS_SECONDS:
            message = (
                f"missed anchored slot {scheduled_at:%H:%M}; "
                f"capture_lateness={capture_lateness:.0f}s"
            )
            failures.append(message)
            print(f"::error::{message}")
            if scheduled_at.time() == time(14, 0):
                break
            continue
        print(
            f"WP completed-bar iteration {scheduled_at:%H:%M} started: "
            f"{started_at:%Y-%m-%d %H:%M:%S}; "
            f"capture_lateness={capture_lateness:.0f}s"
        )
        try:
            slot = scheduled_at.strftime("%H:%M")
            if slot == "14:00":
                run_once(slot)
            elif slot == "14:05":
                if not _has_fixed_signal_session(trade_date):
                    raise RuntimeError(
                        "immutable 14:00 signal is absent; "
                        "14:05 settlement is forbidden"
                    )
                run_once(settlement_slot=slot)
            else:
                run_once()
        except Exception as error:
            message = f"{scheduled_at:%H:%M} failed: {error}"
            failures.append(message)
            print(f"::error::{message}")
            if slot == "14:00":
                break

    print(f"WP session completed: {now_cn():%Y-%m-%d %H:%M:%S}")
    if failures:
        raise RuntimeError("WP session completed with slot failures: " + "; ".join(failures))


def main() -> None:
    mode = os.environ.get("WP_RUN_MODE", "once").strip().lower()
    if mode in {"session", "exact"}:
        run_session()
        return
    if mode == "auto":
        current = now_cn()
        if current.time() <= time(14, 0):
            run_session()
        else:
            run_once_if_due()
        return
    run_once_if_due()


if __name__ == "__main__":
    main()
