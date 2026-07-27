from __future__ import annotations

import os
import subprocess
import sys
import time as time_module
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import tushare as ts

try:
    from build_direct_rank_input import build_direct_rank_input
    from build_wp_v3_live_input import build_live_input
except ModuleNotFoundError:  # pragma: no cover - package import in tests
    from scripts.build_direct_rank_input import build_direct_rank_input
    from scripts.build_wp_v3_live_input import build_live_input


CN_TZ = ZoneInfo("Asia/Shanghai")
INTERVAL_SECONDS = int(os.environ.get("WP_SESSION_INTERVAL_SECONDS", "300"))
SCHEDULE_GRACE_SECONDS = int(os.environ.get("WP_SCHEDULE_GRACE_SECONDS", "120"))
PREP_START = time(14, 15)
RUN_START = time(14, 20)
RUN_END = time(15, 0)
MAX_SLOT_LATENESS_SECONDS = int(
    os.environ.get("WP_MAX_SLOT_LATENESS_SECONDS", "420")
)
LIVE_COMMIT_PATHS = [
    "outputs/html_reports/latest.html",
    "outputs/html_reports/latest.md",
    "outputs/csv/wp_top50.csv",
    "outputs/csv/wp_full_rank.csv",
    "outputs/csv/wp_model_debug.csv",
    "outputs/csv/wp_buy_plan.csv",
    "outputs/csv/wp_buy_decision.csv",
    "outputs/csv/wp_tail_observation.csv",
    "outputs/csv/wp_buy_plan_validation.csv",
    "outputs/csv/wp_tail_sampling.csv",
    "outputs/csv/wp_t1_forecast.csv",
    "outputs/csv/wp_decision_support.csv",
    "outputs/csv/wp_strategy_ledger.csv",
    "outputs/csv/wp_legacy_history_audit.csv",
    "outputs/csv/wp_t1_exit_guidance.csv",
    "outputs/csv/wp_v3_live_predictions.csv",
    "outputs/json/latest.json",
    "outputs/json/wp_buy_plan.json",
    "outputs/json/wp_buy_plan_validation.json",
    "outputs/json/wp_tail_sampling.json",
    "outputs/json/wp_tail_observation.json",
    "outputs/json/wp_t1_forecast.json",
    "outputs/json/wp_decision_support.json",
    "outputs/json/wp_strategy_ledger.json",
    "outputs/json/wp_legacy_history_audit.json",
    "outputs/json/wp_t1_exit_guidance.json",
    "outputs/json/wp_v3_candidate_ledger.json",
    "outputs/json/wp_model_registry_v3.json",
    "outputs/json/wp_manifest.json",
    "outputs/json/wp_data_healthcheck.json",
    "data/cache/wp_latest_rank_input.csv",
    "data/direct/latest/wp_latest_rank_input.csv",
    "data/direct/latest/wp_market_regime_input.csv",
    "data/direct/latest/wp_manifest.json",
    "data/v3/latest/wp_v3_live_features.csv",
    "data/v3/latest/wp_v3_live_manifest.json",
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
    end = datetime.combine(today, RUN_END, CN_TZ) + timedelta(seconds=SCHEDULE_GRACE_SECONDS)
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


def output_commit_paths(mode: str) -> list[str]:
    paths = list(LIVE_COMMIT_PATHS)
    latest_archive = _latest_file("outputs/html_reports/archive/*/*.html")
    latest_log = _latest_file("logs/wp_*.log")
    latest_tail_snapshot = _latest_file("outputs/snapshots/*/*_tail.csv")
    latest_decision_snapshot = _latest_file("outputs/snapshots/*/*_decision.json")
    latest_exit_snapshot = _latest_file("outputs/snapshots/*/*_exit.csv")
    if latest_archive:
        paths.append(latest_archive)
    if latest_log:
        paths.append(latest_log)
    for snapshot in (latest_tail_snapshot, latest_decision_snapshot, latest_exit_snapshot):
        if snapshot:
            paths.append(snapshot)
    if mode == "backtest":
        paths.extend(
            [
                "outputs/backtests",
                "outputs/html_reports/backtest_latest.html",
                "outputs/json/wp_backtest_latest.json",
            ]
        )
    return paths


def run_once() -> None:
    env = os.environ.copy()
    if not env.get("WP_MODE", "").strip():
        env["WP_MODE"] = "live"
    if env.get("GITHUB_EVENT_NAME", "").strip() == "push":
        # A push is either a code change or an explicit self-heal trigger.
        # Rebuild even when the market-data hash is unchanged so corrected
        # logic and report rendering cannot remain hidden behind old outputs.
        env["WP_FORCE_REBUILD"] = "true"
    mode = env["WP_MODE"].strip().lower()
    engine = env.get("WP_ENGINE_VERSION", "v3").strip().lower()
    env["WP_ENGINE_VERSION"] = engine
    direct_enabled = env.get("WP_DIRECT_SOURCE_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
    explicit_source = env.get("WP_SOURCE_CSV", "").strip()
    if mode == "live" and engine == "v3":
        current = now_cn()
        live_path = Path("data/v3/latest/wp_v3_live_features.csv")
        if time(14, 20) <= current.time() <= time(14, 50):
            source_path, source_manifest = build_live_input(root=Path.cwd(), env=env)
            env["WP_V3_SOURCE_CSV"] = source_path.as_posix()
            env["WP_EXPECTED_TRADE_DATE"] = str(source_manifest["trade_date"])
            env["WP_V3_SIGNAL_SLOT"] = str(source_manifest["signal_slot"])
            print(
                "WP V3 causal source ready: "
                f"{source_path} slot={source_manifest['signal_slot']}"
            )
        elif current.time() > time(14, 50) and live_path.exists():
            env["WP_V3_SOURCE_CSV"] = live_path.as_posix()
            print("WP V3 uses the frozen 14:50 feature snapshot for close-state rendering.")
        else:
            print("::warning::WP V3 live source is absent after the signal window.")
    elif mode == "live" and direct_enabled and not explicit_source:
        direct = build_direct_rank_input(root=Path.cwd(), env=env)
        env["WP_DIRECT_ATTEMPTED"] = str(direct.attempted).lower()
        env["WP_DIRECT_ERROR"] = direct.error
        if direct.ok:
            env["WP_SOURCE_CSV"] = direct.source_path
            env["WP_EXPECTED_TRADE_DATE"] = direct.source_trade_date
            env["WP_SOURCE_MODE"] = "direct_tushare"
            env["WP_SOURCE_REPOSITORY"] = "njedu2023-prog/a-share-top3-data"
            print(f"WP direct Tushare source ready: {direct.source_path}")
        else:
            env["WP_SOURCE_MODE"] = "upstream_fallback"
            env["WP_SOURCE_REPOSITORY"] = "njedu2023-prog/a-share-top3-data"
            print(f"::warning::WP direct source unavailable; use upstream fallback: {direct.error}")
    elif explicit_source:
        env.setdefault("WP_SOURCE_MODE", "explicit_source")
        env.setdefault("WP_DIRECT_ATTEMPTED", "false")
    else:
        env.setdefault("WP_SOURCE_MODE", "not_applicable")
        env.setdefault("WP_DIRECT_ATTEMPTED", "false")
    manifest_path = Path("outputs/json/wp_manifest.json")
    manifest_before = manifest_path.read_bytes() if manifest_path.exists() else None
    subprocess.run([sys.executable, "-m", "wp.main"], check=True, env=env)
    manifest_after = manifest_path.read_bytes() if manifest_path.exists() else None
    if manifest_before == manifest_after:
        print("Skip GitHub output commit: WP manifest is unchanged.")
        return
    commit_paths = output_commit_paths(mode)
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

    start_dt, end_dt = window
    if current < start_dt:
        wait_seconds = max(0.0, (start_dt - current).total_seconds())
        print(f"Wait until WP session start: {start_dt:%Y-%m-%d %H:%M:%S}, wait={wait_seconds:.0f}s")
        time_module.sleep(wait_seconds)

    failures: list[str] = []
    schedule = [
        datetime.combine(current.date(), time(14, minute), CN_TZ)
        for minute in (20, 25, 30, 35, 40, 45, 50, 55)
    ] + [datetime.combine(current.date(), time(15, 0), CN_TZ)]
    for scheduled_at in schedule:
        current = now_cn()
        if current < scheduled_at:
            wait_seconds = (scheduled_at - current).total_seconds()
            print(
                f"Wait for anchored WP slot {scheduled_at:%H:%M}; "
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
            f"WP anchored iteration {scheduled_at:%H:%M} started: "
            f"{started_at:%Y-%m-%d %H:%M:%S}; lateness={lateness:.0f}s"
        )
        try:
            run_once()
        except Exception as error:
            message = f"{scheduled_at:%H:%M} failed: {error}"
            failures.append(message)
            print(f"::error::{message}")

    print(f"WP session completed: {now_cn():%Y-%m-%d %H:%M:%S}")
    if failures:
        raise RuntimeError("WP session completed with slot failures: " + "; ".join(failures))


def main() -> None:
    mode = os.environ.get("WP_MODE", "").strip().lower()
    event_name = os.environ.get("GITHUB_EVENT_NAME", "").strip()
    # check_upstream_revision.py is the single gate for push-triggered runs.
    # Reapplying the market-window gate here used to discard monitor/self-heal
    # trigger commits after the workflow had already approved them.
    if event_name in {"workflow_dispatch", "push"} or mode == "backtest":
        run_once()
        return
    if os.environ.get("WP_RUN_MODE", "once").strip().lower() == "session":
        run_session()
        return
    run_once_if_due()


if __name__ == "__main__":
    main()
