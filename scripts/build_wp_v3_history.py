from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import tushare as ts

from wp.v3.contracts import load_v3_config
from wp.v3.history import TushareHistoryClient, build_three_year_panel


ROOT = Path(__file__).resolve().parents[1]
CN_TZ = ZoneInfo("Asia/Shanghai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the three-year point-in-time WP V4 causal feature panel."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "wp_v3.yml"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "wp_v3_history"))
    parser.add_argument("--cache-dir", default=str(ROOT / ".cache" / "wp_v3_tushare"))
    parser.add_argument("--start-date", help="YYYYMMDD or auto")
    parser.add_argument("--end-date", help="YYYYMMDD or auto")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for the three-year causal rebuild")
    config = load_v3_config(args.config)
    pro = ts.pro_api(token)
    start_date, end_date = _resolve_history_range(
        pro,
        args.start_date,
        args.end_date,
        default_start=config.history.start_date,
        default_end=config.history.end_date,
    )
    if (
        start_date != config.history.start_date
        or end_date != config.history.end_date
    ):
        raw = Path(args.config).read_text(encoding="utf-8")
        import yaml

        payload = yaml.safe_load(raw)
        payload["history"]["start_date"] = start_date
        payload["history"]["end_date"] = end_date
        temporary = Path(args.output_dir) / "_resolved_wp_v3.yml"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
        config = load_v3_config(temporary)

    client = TushareHistoryClient(
        pro,
        args.cache_dir,
        page_size=config.history.tushare_page_size,
        requests_per_minute=config.history.tushare_requests_per_minute,
    )
    manifest = build_three_year_panel(
        client,
        config,
        args.output_dir,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _resolve_history_range(
    pro,
    requested_start: str | None,
    requested_end: str | None,
    *,
    default_start: str,
    default_end: str,
) -> tuple[str, str]:
    start_value = (requested_start or default_start).strip().lower()
    end_value = (requested_end or default_end).strip().lower()
    if start_value != "auto" and end_value != "auto":
        return start_value, end_value

    current = datetime.now(CN_TZ)
    latest_possible = current.date()
    if current.time() < time(16, 0):
        latest_possible -= timedelta(days=1)
    search_start = latest_possible - timedelta(days=20)
    calendar = pro.trade_cal(
        exchange="SSE",
        start_date=search_start.strftime("%Y%m%d"),
        end_date=latest_possible.strftime("%Y%m%d"),
        is_open="1",
        fields="cal_date,is_open",
    )
    open_dates = sorted(
        calendar.loc[
            calendar["is_open"].astype(str).eq("1"),
            "cal_date",
        ].astype(str)
    )
    if not open_dates:
        raise RuntimeError("cannot resolve the latest completed A-share trade date")
    end_date = open_dates[-1] if end_value == "auto" else end_value
    if start_value != "auto":
        return start_value, end_date

    anchor = (
        pd.Timestamp(datetime.strptime(end_date, "%Y%m%d"))
        - pd.DateOffset(years=3)
    ).strftime("%Y%m%d")
    start_calendar = pro.trade_cal(
        exchange="SSE",
        start_date=anchor,
        end_date=(
            datetime.strptime(anchor, "%Y%m%d") + timedelta(days=14)
        ).strftime("%Y%m%d"),
        is_open="1",
        fields="cal_date,is_open",
    )
    start_dates = sorted(
        start_calendar.loc[
            start_calendar["is_open"].astype(str).eq("1"),
            "cal_date",
        ].astype(str)
    )
    if not start_dates:
        raise RuntimeError("cannot resolve the first trade date in the three-year window")
    return start_dates[0], end_date


if __name__ == "__main__":
    raise SystemExit(main())
