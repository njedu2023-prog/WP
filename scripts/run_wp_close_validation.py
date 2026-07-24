from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from run_wp_session import is_trade_day, now_cn


CLOSE_COMMIT_PATHS = [
    "outputs/html_reports/latest.html",
    "outputs/csv/wp_buy_plan_validation.csv",
    "outputs/csv/wp_tail_sampling.csv",
    "outputs/json/latest.json",
    "outputs/json/wp_buy_plan_validation.json",
    "outputs/json/wp_tail_sampling.json",
    "outputs/json/wp_manifest.json",
    "outputs/json/wp_data_healthcheck.json",
]


def _latest_close_archive() -> str | None:
    matches = [path for path in Path.cwd().glob("outputs/html_reports/archive/*/*_close.html") if path.is_file()]
    return max(matches, key=lambda path: path.stat().st_mtime).as_posix() if matches else None


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
    env = os.environ.copy()
    subprocess.run([sys.executable, "-m", "wp.close_validation"], check=True, env=env)
    commit_paths = list(CLOSE_COMMIT_PATHS)
    archive = _latest_close_archive()
    if archive:
        commit_paths.append(archive)
    subprocess.run(
        [sys.executable, "scripts/github_commit_paths.py", "Validate WP next-day close", *commit_paths],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()
