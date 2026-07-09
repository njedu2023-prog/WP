from __future__ import annotations

import os
import subprocess
import sys
import time as time_module
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import tushare as ts


CN_TZ = ZoneInfo("Asia/Shanghai")
INTERVAL_SECONDS = int(os.environ.get("WP_SESSION_INTERVAL_SECONDS", "600"))
PREP_START = time(9, 0)
RUN_START = time(9, 28)
LUNCH_START = time(11, 38)
LUNCH_END = time(12, 58)
RUN_END = time(15, 12)
MORNING_START = time(9, 25)
MORNING_END = time(11, 35)
AFTERNOON_START = time(12, 55)
AFTERNOON_END = time(15, 10)


def now_cn() -> datetime:
    return datetime.now(CN_TZ)


def today_window(now: datetime) -> tuple[datetime, datetime, datetime, datetime] | None:
    today = now.date()
    prep_dt = datetime.combine(today, PREP_START, CN_TZ)
    start_dt = datetime.combine(today, RUN_START, CN_TZ)
    lunch_start_dt = datetime.combine(today, LUNCH_START, CN_TZ)
    lunch_end_dt = datetime.combine(today, LUNCH_END, CN_TZ)
    end_dt = datetime.combine(today, RUN_END, CN_TZ)
    if prep_dt <= now <= end_dt:
        return start_dt, lunch_start_dt, lunch_end_dt, end_dt
    return None


def in_run_window(now: datetime) -> bool:
    current = now.time()
    return MORNING_START <= current <= MORNING_END or AFTERNOON_START <= current <= AFTERNOON_END


def is_trade_day(token: str, day: str) -> bool:
    ts.set_token(token)
    pro = ts.pro_api()
    cal = pro.trade_cal(exchange="SSE", start_date=day, end_date=day)
    return bool(len(cal) and int(cal.iloc[0].get("is_open", 0)) == 1)


def run_once() -> None:
    env = os.environ.copy()
    if not env.get("WP_MODE", "").strip():
        env["WP_MODE"] = "live"
    subprocess.run([sys.executable, "-m", "wp.main"], check=True, env=env)
    subprocess.run(
        [sys.executable, "scripts/github_commit_paths.py", "Update WP outputs", "outputs", "logs", "data/cache"],
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

    start_dt, lunch_start_dt, lunch_end_dt, end_dt = window
    if current < start_dt:
        wait_seconds = max(0.0, (start_dt - current).total_seconds())
        print(f"Wait until WP session start: {start_dt:%Y-%m-%d %H:%M:%S}, wait={wait_seconds:.0f}s")
        time_module.sleep(wait_seconds)

    while now_cn() <= end_dt:
        iteration_start = now_cn()
        if lunch_start_dt <= iteration_start < lunch_end_dt:
            sleep_seconds = max(0.0, (lunch_end_dt - iteration_start).total_seconds())
            print(f"Pause during A-share lunch break until {lunch_end_dt:%Y-%m-%d %H:%M:%S}, sleep={sleep_seconds:.0f}s")
            time_module.sleep(sleep_seconds)
            continue
        print(f"WP iteration started: {iteration_start:%Y-%m-%d %H:%M:%S}")
        run_once()
        next_at = iteration_start + timedelta(seconds=INTERVAL_SECONDS)
        current = now_cn()
        next_boundary = lunch_start_dt if current < lunch_start_dt < next_at else end_dt
        sleep_seconds = min((next_at - current).total_seconds(), (next_boundary - current).total_seconds())
        if sleep_seconds <= 0:
            continue
        print(f"Next WP iteration at {next_at:%Y-%m-%d %H:%M:%S}, sleep={sleep_seconds:.0f}s")
        time_module.sleep(sleep_seconds)

    print(f"WP session completed: {now_cn():%Y-%m-%d %H:%M:%S}")


def main() -> None:
    mode = os.environ.get("WP_MODE", "").strip().lower()
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch" or mode == "backtest":
        run_once()
        return
    if os.environ.get("WP_RUN_MODE", "once").strip().lower() == "session":
        run_session()
        return
    run_once_if_due()


if __name__ == "__main__":
    main()
