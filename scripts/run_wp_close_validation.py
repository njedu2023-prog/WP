from __future__ import annotations

import os
import subprocess
import sys
import time as time_module
from datetime import datetime, time
from pathlib import Path

try:
    from run_wp_session import is_trade_day, now_cn
except ModuleNotFoundError:  # pragma: no cover - package import in tests
    from scripts.run_wp_session import is_trade_day, now_cn


CLOSE_COMMIT_PATHS = [
    "outputs/html_reports/latest.html",
    "outputs/csv/wp_buy_plan_validation.csv",
    "outputs/json/wp_buy_plan_validation.json",
    "outputs/json/wp_strategy_ledger.json",
    "outputs/json/wp_manifest.json",
    "outputs/json/wp_v3_candidate_ledger.json",
    "outputs/json/wp_model_registry_v3.json",
    "outputs/json/wp_v40_backtest_202605_202607.json",
    "outputs/csv/wp_v40_backtest_qualified_202605_202607.csv",
    "outputs/csv/wp_v40_backtest_observations_202605_202607.csv",
]


def _latest_close_archive() -> str | None:
    matches = [path for path in Path.cwd().glob("outputs/html_reports/archive/*/*_close.html") if path.is_file()]
    return max(matches, key=lambda path: path.stat().st_mtime).as_posix() if matches else None


def run_once() -> int:
    tracked_paths = [
        Path("outputs/json/wp_v3_candidate_ledger.json"),
        Path("outputs/json/wp_model_registry_v3.json"),
        Path("outputs/json/wp_v40_backtest_202605_202607.json"),
    ]
    before = tuple(path.read_bytes() if path.exists() else b"" for path in tracked_paths)
    env = os.environ.copy()
    subprocess.run([sys.executable, "-m", "wp.close_validation"], check=True, env=env)
    after = tuple(path.read_bytes() if path.exists() else b"" for path in tracked_paths)
    payload_path = Path("outputs/json/wp_buy_plan_validation.json")
    payload = {}
    if payload_path.exists():
        import json

        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    pending = int(
        summary.get(
            "pending_due_count",
            summary.get("pending_count", 0),
        )
    )
    if before == after:
        print(f"No V40 close-truth state change; pending_due={pending}.")
        return pending
    commit_paths = list(CLOSE_COMMIT_PATHS)
    archive = _latest_close_archive()
    if archive:
        commit_paths.append(archive)
    subprocess.run(
        [sys.executable, "scripts/github_commit_paths.py", "Validate WP V40 next-day close", *commit_paths],
        check=True,
        env=env,
    )
    print(f"V40 close-truth state committed; pending_due={pending}.")
    return pending


def main() -> None:
    current = now_cn()
    trade_date = current.strftime("%Y%m%d")
    token = os.environ.get("TUSHARE_TOKEN", "").strip()

    if current.weekday() >= 5:
        print(f"Skip WP close validation on weekend: {trade_date}.")
        return
    if token and not is_trade_day(token, trade_date):
        print(f"Skip WP close validation: {trade_date} is not an A-share trading day.")
        return
    if not token:
        print("WP close validation calendar fallback: TUSHARE_TOKEN is not configured.")

    print(f"WP close validation started: {current:%Y-%m-%d %H:%M:%S}")
    if os.environ.get("WP_CLOSE_RUN_MODE", "once").strip().lower() != "session":
        run_once()
        return

    interval = int(os.environ.get("WP_CLOSE_INTERVAL_SECONDS", "300"))
    end = datetime.combine(current.date(), time(16, 10), current.tzinfo)
    if current > end:
        pending = run_once()
        if pending == 0:
            print(
                "WP close validation completed in delayed-start mode: "
                "no due records remain."
            )
            return
        raise SystemExit(
            "WP close validation delayed-start run completed with "
            f"{pending} due record(s) still pending."
        )

    while now_cn() <= end:
        pending = run_once()
        if pending == 0:
            print("WP close validation completed: no due records remain.")
            return
        wait = min(interval, max(0, int((end - now_cn()).total_seconds())))
        if wait <= 0:
            break
        print(f"Close truth not ready; retry in {wait}s.")
        time_module.sleep(wait)
    raise SystemExit("WP close validation timed out with due records still pending.")


if __name__ == "__main__":
    main()
