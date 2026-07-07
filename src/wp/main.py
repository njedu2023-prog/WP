from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .calendar import now_cn
from .candidate_filter import filter_candidates
from .data_loader import read_rank_input
from .feature_engineering import add_feature_scores
from .ranking import rank_candidates
from .report_html import render_html
from .report_md import render_markdown
from .scoring_model import add_scores
from .utils import ensure_dir, load_yaml, write_json
from .validation import assert_top50_rules, build_healthcheck


ROOT = Path(__file__).resolve().parents[2]


def setup_logging(update_key: str) -> Path:
    log_path = ROOT / "logs" / f"wp_{update_key}.log"
    ensure_dir(log_path.parent)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    return log_path


def run() -> dict:
    current = now_cn()
    update_time = current.strftime("%Y-%m-%d %H:%M:%S")
    update_key = current.strftime("%Y%m%d_%H%M")
    log_path = setup_logging(update_key)
    config = load_yaml(ROOT / "config" / "wp_config.yml")
    output_root = ROOT / "outputs"
    mode = os.environ.get("WP_MODE", "").strip().lower()
    backtest_start = os.environ.get("WP_BACKTEST_START", "").strip()
    backtest_end = os.environ.get("WP_BACKTEST_END", "").strip()
    if mode == "backtest" or (backtest_start and backtest_end):
        start_date = backtest_start or current.strftime("%Y%m%d")
        end_date = backtest_end or start_date
        logging.info("WP backtest started, start=%s end=%s", start_date, end_date)
        result = run_backtest(start_date, end_date, output_root, top_n=int(config.get("top_n", 50)))
        write_json(
            output_root / "json" / "wp_manifest.json",
            {"latest_update": update_time, "mode": "backtest", "backtest": result.summary},
        )
        logging.info("WP backtest completed: %s", result.summary)
        return result.summary

    cache_path = ROOT / "data" / "cache" / "wp_latest_rank_input.csv"
    source = os.environ.get("WP_SOURCE_CSV", "").strip() or config.get("input_url", "")
    logging.info("WP run started, source=%s", source)
    load_result = read_rank_input(source, cache_path=cache_path)
    raw = load_result.frame
    expected_trade_date = current.strftime("%Y%m%d")
    candidates = filter_candidates(
        raw,
        min_pct_chg=float(config.get("min_pct_chg", 6.0)),
        min_amount=float(config.get("min_amount", 100000000)),
        exclude_st=bool(config.get("exclude_st", True)),
        exclude_suspended=bool(config.get("exclude_suspended", True)),
        exclude_new_stock_days=int(config.get("exclude_new_stock_days", 10)),
    )
    ranked_input = add_scores(add_feature_scores(candidates))
    top50, full_rank = rank_candidates(ranked_input, update_time, top_n=int(config.get("top_n", 50)))
    health = build_healthcheck(
        raw,
        candidates,
        top50,
        load_result.ok,
        load_result.error,
        load_result.fallback_used,
        update_time,
        expected_trade_date,
        load_result.metadata,
    )
    if health["status"] == "数据日期过期" and os.environ.get("WP_ALLOW_STALE_DATA", "").strip() != "1":
        logging.error("Stale WP data: data_trade_date=%s expected=%s", health.get("data_trade_date"), expected_trade_date)
        top50 = top50.iloc[0:0].copy()
        full_rank = full_rank.iloc[0:0].copy()
        ranked_input = ranked_input.iloc[0:0].copy()
        health["candidate_count"] = 0
        health["top50_count"] = 0
    rule_errors = assert_top50_rules(top50)
    if rule_errors:
        health["status"] = "规则自检失败"
        health["rule_errors"] = rule_errors
        logging.error("Top50 rule errors: %s", rule_errors)

    ensure_dir(output_root / "csv")
    ensure_dir(output_root / "json")
    archive_dir = output_root / "html_reports" / "archive" / current.strftime("%Y%m%d")
    ensure_dir(archive_dir)
    top50.to_csv(output_root / "csv" / "wp_top50.csv", index=False, encoding="utf-8-sig")
    full_rank.to_csv(output_root / "csv" / "wp_full_rank.csv", index=False, encoding="utf-8-sig")
    ranked_input.to_csv(output_root / "csv" / "wp_model_debug.csv", index=False, encoding="utf-8-sig")
    latest_html = output_root / "html_reports" / "latest.html"
    archive_html = archive_dir / f"{current.strftime('%H%M')}.html"
    preserve_latest = (not load_result.ok) and (not load_result.fallback_used) and latest_html.exists()
    health["preserved_latest_html"] = bool(preserve_latest)
    latest_payload = {"generated_at": update_time, "health": health, "top50": top50.to_dict(orient="records")}
    write_json(output_root / "json" / "latest.json", latest_payload)
    write_json(
        output_root / "json" / "wp_manifest.json",
        {
            "latest_update": update_time,
            "top50_count": len(top50),
            "health_status": health["status"],
            "preserved_latest_html": bool(preserve_latest),
        },
    )
    write_json(output_root / "json" / "wp_data_healthcheck.json", health)
    if preserve_latest:
        logging.error("Source failed; preserving existing latest.html. error=%s", load_result.error)
    else:
        render_html(top50, full_rank, health, latest_html)
    render_html(top50, full_rank, health, archive_html)
    render_markdown(top50, output_root / "html_reports" / "latest.md")
    logging.info(
        "WP run completed: raw=%s candidates=%s top50=%s missing_fields=%s fallback=%s outputs=%s log=%s",
        len(raw),
        len(candidates),
        len(top50),
        ",".join(health.get("missing_fields", [])) or "none",
        health.get("fallback_used"),
        output_root,
        log_path,
    )
    return health


if __name__ == "__main__":
    run()
