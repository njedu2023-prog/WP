from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

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
    cache_path = ROOT / "data" / "cache" / "wp_latest_rank_input.csv"
    source = config.get("input_url", "")
    logging.info("WP run started, source=%s", source)
    load_result = read_rank_input(source, cache_path=cache_path)
    raw = load_result.frame
    candidates = filter_candidates(
        raw,
        min_pct_chg=float(config.get("min_pct_chg", 6.0)),
        min_amount=float(config.get("min_amount", 100000000)),
        exclude_st=bool(config.get("exclude_st", True)),
    )
    ranked_input = add_scores(add_feature_scores(candidates))
    top50, full_rank = rank_candidates(ranked_input, update_time, top_n=int(config.get("top_n", 50)))
    health = build_healthcheck(raw, candidates, top50, load_result.ok, load_result.error, load_result.fallback_used, update_time)
    rule_errors = assert_top50_rules(top50)
    if rule_errors:
        health["status"] = "规则自检失败"
        health["rule_errors"] = rule_errors
        logging.error("Top50 rule errors: %s", rule_errors)

    ensure_dir(output_root / "csv")
    ensure_dir(output_root / "json")
    ensure_dir(output_root / "html_reports" / "archive" / current.strftime("%Y%m%d"))
    top50.to_csv(output_root / "csv" / "wp_top50.csv", index=False, encoding="utf-8-sig")
    full_rank.to_csv(output_root / "csv" / "wp_full_rank.csv", index=False, encoding="utf-8-sig")
    ranked_input.to_csv(output_root / "csv" / "wp_model_debug.csv", index=False, encoding="utf-8-sig")
    latest_payload = {"generated_at": update_time, "health": health, "top50": top50.to_dict(orient="records")}
    write_json(output_root / "json" / "latest.json", latest_payload)
    write_json(output_root / "json" / "wp_manifest.json", {"latest_update": update_time, "top50_count": len(top50), "health_status": health["status"]})
    write_json(output_root / "json" / "wp_data_healthcheck.json", health)
    render_html(top50, full_rank, health, output_root / "html_reports" / "latest.html")
    render_html(top50, full_rank, health, output_root / "html_reports" / "archive" / current.strftime("%Y%m%d") / f"{current.strftime('%H%M')}.html")
    render_markdown(top50, output_root / "html_reports" / "latest.md")
    logging.info("WP run completed: raw=%s candidates=%s top50=%s log=%s", len(raw), len(candidates), len(top50), log_path)
    return health


if __name__ == "__main__":
    run()
