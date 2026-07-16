from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .buy_validation import update_buy_plan_validation
from .calendar import now_cn
from .main import ROOT, load_backtest_summaries
from .report_html import render_html
from .tail_profit_model import TAIL_PROFIT_MODEL_VERSION
from .utils import ensure_dir, write_json


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(
            path,
            keep_default_na=False,
            dtype={"ts_code": str, "trade_date": str, "plan_trade_date": str, "target_trade_date": str},
        )
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def run_close_validation(
    output_root: Path | None = None,
    current: datetime | None = None,
) -> dict:
    output_root = output_root or ROOT / "outputs"
    current = current or now_cn()
    update_time = current.strftime("%Y-%m-%d %H:%M:%S")

    health_path = output_root / "json" / "wp_data_healthcheck.json"
    health = _read_json(health_path)
    health["buy_model_version"] = str(health.get("buy_model_version") or TAIL_PROFIT_MODEL_VERSION)
    health["wp_run_time"] = update_time
    health["report_revision"] = update_time
    health["validation_updated_at"] = update_time

    # An empty buy plan prevents the close job from changing the locked tail snapshot.
    validation_result = update_buy_plan_validation(pd.DataFrame(), health, output_root, current)
    top50 = _read_csv(output_root / "csv" / "wp_top50.csv")
    full_rank = _read_csv(output_root / "csv" / "wp_full_rank.csv")
    buy_plan = _read_csv(output_root / "csv" / "wp_buy_plan.csv")
    backtests = load_backtest_summaries(output_root)

    latest_path = output_root / "json" / "latest.json"
    latest_payload = _read_json(latest_path)
    latest_payload.update(
        {
            "generated_at": update_time,
            "wp_run_time": update_time,
            "validation_updated_at": update_time,
            "buy_plan_validation": validation_result.table.to_dict(orient="records"),
            "buy_plan_validation_summary": validation_result.summary,
        }
    )
    write_json(latest_path, latest_payload)

    validation_path = output_root / "json" / "wp_buy_plan_validation.json"
    validation_payload = _read_json(validation_path)
    validation_payload.update(
        {
            "generated_at": update_time,
            "wp_run_time": update_time,
            "validation_updated_at": update_time,
            "market_data_time": health.get("market_data_time", validation_payload.get("market_data_time", "")),
            "summary": validation_result.summary,
            "records": validation_result.table.to_dict(orient="records"),
        }
    )
    write_json(validation_path, validation_payload)

    manifest_path = output_root / "json" / "wp_manifest.json"
    manifest = _read_json(manifest_path)
    manifest.update(
        {
            "latest_update": update_time,
            "wp_run_time": update_time,
            "report_revision": update_time,
            "validation_updated_at": update_time,
            "validation_summary": validation_result.summary,
        }
    )
    write_json(manifest_path, manifest)
    write_json(health_path, health)

    latest_html = output_root / "html_reports" / "latest.html"
    archive_dir = ensure_dir(output_root / "html_reports" / "archive" / current.strftime("%Y%m%d"))
    archive_html = archive_dir / f"{current.strftime('%H%M')}_close.html"
    render_html(
        top50,
        full_rank,
        health,
        latest_html,
        buy_plan=buy_plan,
        validation=validation_result.table,
        validation_summary=validation_result.summary,
        backtests=backtests,
    )
    render_html(
        top50,
        full_rank,
        health,
        archive_html,
        buy_plan=buy_plan,
        validation=validation_result.table,
        validation_summary=validation_result.summary,
        backtests=backtests,
    )
    return validation_result.summary


if __name__ == "__main__":
    run_close_validation()
