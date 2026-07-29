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
from wp.v3.io import atomic_write_text


ROOT = Path(__file__).resolve().parents[1]
CN_TZ = ZoneInfo("Asia/Shanghai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the causal WP research panel and its complete three-year "
            "out-of-sample evaluation contract."
        )
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "wp_v3.yml"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "wp_v3_history"))
    parser.add_argument("--cache-dir", default=str(ROOT / ".cache" / "wp_v3_tushare"))
    parser.add_argument("--start-date", help="YYYYMMDD or auto")
    parser.add_argument("--end-date", help="YYYYMMDD or auto")
    parser.add_argument("--evaluation-start-date", help="YYYYMMDD or auto")
    parser.add_argument("--evaluation-end-date", help="YYYYMMDD or auto")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for the causal research rebuild")
    config = load_v3_config(args.config)
    pro = ts.pro_api(token)
    (
        start_date,
        end_date,
        evaluation_start_date,
        evaluation_end_date,
    ) = _resolve_history_range(
        pro,
        args.start_date,
        args.end_date,
        args.evaluation_start_date,
        args.evaluation_end_date,
        default_start=config.history.start_date,
        default_end=config.history.end_date,
        default_evaluation_start=config.history.evaluation_start_date,
        default_evaluation_end=config.history.evaluation_end_date,
    )
    raw = Path(args.config).read_text(encoding="utf-8")
    import yaml

    payload = yaml.safe_load(raw)
    payload["history"]["start_date"] = start_date
    payload["history"]["end_date"] = end_date
    payload["history"]["evaluation_start_date"] = evaluation_start_date
    payload["history"]["evaluation_end_date"] = evaluation_end_date
    resolved_config = Path(args.output_dir) / "wp_v9_resolved_config.yml"
    resolved_config.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        resolved_config,
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
    )
    config = load_v3_config(resolved_config)

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
    requested_evaluation_start: str | None,
    requested_evaluation_end: str | None,
    *,
    default_start: str,
    default_end: str,
    default_evaluation_start: str,
    default_evaluation_end: str,
) -> tuple[str, str, str, str]:
    start_value = (requested_start or default_start).strip().lower()
    end_value = (requested_end or default_end).strip().lower()
    evaluation_start_value = (
        requested_evaluation_start or default_evaluation_start
    ).strip().lower()
    evaluation_end_value = (
        requested_evaluation_end or default_evaluation_end
    ).strip().lower()
    if all(
        value != "auto"
        for value in (
            start_value,
            end_value,
            evaluation_start_value,
            evaluation_end_value,
        )
    ):
        return (
            start_value,
            end_value,
            evaluation_start_value,
            evaluation_end_value,
        )

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
    evaluation_end_date = (
        end_date if evaluation_end_value == "auto" else evaluation_end_value
    )
    start_date = (
        _first_open_date_on_or_after(
            pro,
            (
                pd.Timestamp(datetime.strptime(end_date, "%Y%m%d"))
                - pd.DateOffset(years=5)
            ).strftime("%Y%m%d"),
        )
        if start_value == "auto"
        else start_value
    )
    evaluation_start_date = (
        _first_open_date_on_or_after(
            pro,
            (
                pd.Timestamp(
                    datetime.strptime(evaluation_end_date, "%Y%m%d")
                )
                - pd.DateOffset(years=3)
            ).strftime("%Y%m%d"),
        )
        if evaluation_start_value == "auto"
        else evaluation_start_value
    )
    return start_date, end_date, evaluation_start_date, evaluation_end_date


def _first_open_date_on_or_after(pro, anchor: str) -> str:
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
        raise RuntimeError(f"cannot resolve the first trade date on or after {anchor}")
    return start_dates[0]


if __name__ == "__main__":
    raise SystemExit(main())
