from __future__ import annotations

import json
import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .buy_decision import build_buy_decision
from .buy_validation import update_buy_plan_validation
from .calendar import now_cn
from .candidate_filter import enrich_basic_fields, filter_candidates, flag_limitup
from .data_loader import read_rank_input
from .feature_engineering import add_feature_scores
from .decision_support import build_decision_support
from .exit_guidance import build_exit_guidance
from .market_regime import MARKET_REGIME_MODEL_VERSION, assess_market_regime
from .ranking import build_ranked_pool, rank_candidates
from .report_html import render_html
from .report_md import render_markdown
from .scoring_model import MODEL_VERSION, add_scores
from .tail_profit_model import TAIL_PROFIT_MODEL_VERSION, add_tail_profit_scores
from .tail_observation import update_tail_observation
from .tail_sampling import update_tail_sampling
from .tail_window import accepts_new_tail_primary, tail_window_phase
from .t1_forecast import FORECAST_COLUMNS, T1_FORECAST_MODEL_VERSION, build_t1_forecasts
from .utils import ensure_dir, load_yaml, write_json
from .validation import assert_top50_rules, build_healthcheck, resolve_market_data_time


ROOT = Path(__file__).resolve().parents[2]


def _read_existing_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def source_data_hash(frame: pd.DataFrame) -> str:
    """Build a stable fingerprint for the complete upstream input table."""
    digest = hashlib.sha256()
    digest.update("\x1f".join(str(column) for column in frame.columns).encode("utf-8"))
    if not frame.empty:
        row_hashes = pd.util.hash_pandas_object(frame, index=True, categorize=True)
        digest.update(row_hashes.to_numpy().tobytes())
    return digest.hexdigest()


def should_rebuild_live_report(
    frame: pd.DataFrame,
    source_metadata: dict | None,
    existing_manifest: dict,
    *,
    force: bool = False,
) -> tuple[bool, str, str]:
    market_data_time = resolve_market_data_time(frame, source_metadata or {}, "")
    data_hash = source_data_hash(frame)
    if force or not existing_manifest:
        return True, market_data_time, data_hash

    existing_hash = str(existing_manifest.get("source_data_hash") or "").strip()
    if existing_hash:
        return existing_hash != data_hash, market_data_time, data_hash

    existing_time = str(
        existing_manifest.get("data_revision")
        or existing_manifest.get("market_data_time")
        or ""
    ).strip()
    if market_data_time and existing_time:
        return market_data_time != existing_time, market_data_time, data_hash
    return True, market_data_time, data_hash


def load_backtest_summaries(output_root: Path) -> list[dict]:
    summaries: list[dict] = []
    for path in sorted((output_root / "backtests").glob("*/summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("model_version") == MODEL_VERSION
            and payload.get("buy_model_version") == TAIL_PROFIT_MODEL_VERSION
        ):
            summaries.append(payload)
    summaries.sort(key=lambda item: (str(item.get("start_date", "")), str(item.get("end_date", ""))))
    return [
        summary
        for summary in summaries
        if not any(
            str(other.get("start_date", "")) <= str(summary.get("start_date", ""))
            and str(other.get("end_date", "")) >= str(summary.get("end_date", ""))
            and (
                str(other.get("start_date", "")) < str(summary.get("start_date", ""))
                or str(other.get("end_date", "")) > str(summary.get("end_date", ""))
            )
            for other in summaries
        )
    ]


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


def _trade_date_from_frame(frame: pd.DataFrame, fallback: str = "") -> str:
    if frame is None or frame.empty or "trade_date" not in frame.columns:
        return str(fallback or "").replace("-", "")
    values = (
        frame["trade_date"]
        .fillna("")
        .astype(str)
        .str.replace("-", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
    )
    valid = values[values.str.fullmatch(r"\d{8}")]
    return str(valid.max()) if not valid.empty else str(fallback or "").replace("-", "")


def _read_validation_history(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path,
            keep_default_na=False,
            dtype={"plan_trade_date": str, "target_trade_date": str, "ts_code": str},
        )
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _read_market_context(path: Path, fallback: pd.DataFrame, trade_date: str) -> tuple[pd.DataFrame, str]:
    if path.is_file():
        try:
            frame = pd.read_csv(path, keep_default_na=False, dtype={"ts_code": str, "trade_date": str})
            required = {"trade_date", "update_time", "ts_code", "name", "price", "open", "high", "low", "pre_close", "pct_chg", "amount"}
            if required.issubset(frame.columns) and len(frame) >= 200:
                if "trade_date" in frame.columns:
                    dates = (
                        frame["trade_date"]
                        .fillna("")
                        .astype(str)
                        .str.replace("-", "", regex=False)
                        .str.replace(r"\.0$", "", regex=True)
                    )
                    valid_dates = dates[dates.ne("")]
                    if not valid_dates.empty and not valid_dates.eq(trade_date).all():
                        raise ValueError("market context trade date mismatch")
                return frame, path.as_posix()
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
            pass
    return fallback.copy(), "candidate_input_fallback"


def _archive_decision_snapshots(
    output_root: Path,
    ranked_input: pd.DataFrame,
    decision_table: pd.DataFrame,
    exit_guidance: pd.DataFrame,
    decision_summary: dict,
    health: dict,
) -> list[str]:
    market_data_time = str(health.get("market_data_time") or "")
    if not accepts_new_tail_primary(market_data_time):
        return []
    parsed = pd.to_datetime(market_data_time, errors="coerce")
    trade_date = str(health.get("data_trade_date") or "").replace("-", "")
    if len(trade_date) != 8:
        return []
    snapshot_dir = ensure_dir(output_root / "snapshots" / trade_date)
    slot = parsed.strftime("%H%M")
    candidate_path = snapshot_dir / f"{slot}_tail.csv"
    decision_path = snapshot_dir / f"{slot}_decision.json"
    exit_path = snapshot_dir / f"{slot}_exit.csv"
    ranked_input.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    write_json(
        decision_path,
        {
            "market_data_time": health.get("market_data_time", ""),
            "trade_date": trade_date,
            "decision": decision_summary,
            "records": decision_table.to_dict(orient="records"),
            "manual_execution_only": True,
            "order_routing_enabled": False,
        },
    )
    paths = [candidate_path.as_posix(), decision_path.as_posix()]
    if exit_guidance is not None and not exit_guidance.empty:
        exit_guidance.to_csv(exit_path, index=False, encoding="utf-8-sig")
        paths.append(exit_path.as_posix())
    return paths


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
    manifest_path = output_root / "json" / "wp_manifest.json"
    existing_manifest = _read_existing_manifest(manifest_path)
    if not load_result.ok and existing_manifest:
        logging.error("Skip WP rebuild: upstream unavailable and no usable cache. error=%s", load_result.error)
        return {
            "status": "source_unavailable",
            "market_data_time": str(existing_manifest.get("market_data_time") or ""),
            "error": load_result.error,
        }
    force_rebuild = os.environ.get("WP_FORCE_REBUILD", "").strip().lower() in {"1", "true", "yes"}
    should_rebuild, source_market_time, input_hash = should_rebuild_live_report(
        raw,
        load_result.metadata,
        existing_manifest,
        force=force_rebuild,
    )
    if not should_rebuild:
        logging.info(
            "Skip WP rebuild: upstream data unchanged, market_data_time=%s source_data_hash=%s",
            source_market_time or "unknown",
            input_hash,
        )
        return {
            "status": "no_new_data",
            "market_data_time": source_market_time,
            "source_data_hash": input_hash,
        }
    expected_trade_date = (
        os.environ.get("WP_EXPECTED_TRADE_DATE", "").strip()
        or current.strftime("%Y%m%d")
    )
    ensure_dir(output_root / "csv")
    ensure_dir(output_root / "json")
    validation_history_path = output_root / "csv" / "wp_buy_plan_validation.csv"
    historical_primaries = _read_validation_history(validation_history_path)
    forecast_trade_date = _trade_date_from_frame(raw, expected_trade_date)
    market_context_path = Path(
        os.environ.get("WP_MARKET_CONTEXT_CSV", "").strip()
        or ROOT / "data" / "direct" / "latest" / "wp_market_regime_input.csv"
    )
    market_context, market_context_source = _read_market_context(
        market_context_path,
        raw,
        forecast_trade_date,
    )
    candidates = filter_candidates(
        raw,
        min_pct_chg=float(config.get("min_pct_chg", 8.0)),
        min_amount=float(config.get("min_amount", 100000000)),
        exclude_st=bool(config.get("exclude_st", True)),
        exclude_suspended=bool(config.get("exclude_suspended", True)),
        exclude_new_stock_days=int(config.get("exclude_new_stock_days", 10)),
        max_limit_up_pct=float(config.get("max_limit_up_pct", 10.0)),
    )
    scored_input = add_tail_profit_scores(add_scores(add_feature_scores(candidates)), config)
    forecast_result = build_t1_forecasts(
        scored_input,
        historical_primaries,
        output_root,
        forecast_trade_date,
        config,
    )
    ranked_input = forecast_result.table
    top_n = int(config.get("top_n", 50))
    top50, full_rank = rank_candidates(ranked_input, update_time, top_n=top_n)
    buy_pool = build_ranked_pool(ranked_input, full_rank, top_n)
    buy_decision = build_buy_decision(buy_pool, config)
    buy_plan = buy_decision.buy_plan
    buy_decision_table = buy_decision.decision_table
    market_regime = assess_market_regime(market_context, candidates, config)
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
    health["model_version"] = MODEL_VERSION
    health["buy_model_version"] = TAIL_PROFIT_MODEL_VERSION
    health["forecast_model_version"] = T1_FORECAST_MODEL_VERSION
    health["market_regime_model_version"] = MARKET_REGIME_MODEL_VERSION
    health["market_regime"] = market_regime
    health["forecast_summary"] = forecast_result.summary
    health["market_context_count"] = int(len(market_context))
    health["market_context_source"] = market_context_source
    health["source_data_hash"] = input_hash
    health["report_revision"] = update_time
    health["source_mode"] = (
        os.environ.get("WP_SOURCE_MODE", "").strip()
        or health.get("source_mode")
        or "upstream_repository"
    )
    health["source_repository"] = (
        os.environ.get("WP_SOURCE_REPOSITORY", "").strip()
        or health.get("source_repository")
        or "njedu2023-prog/a-share-top3-data"
    )
    health["direct_attempted"] = os.environ.get("WP_DIRECT_ATTEMPTED", "").strip().lower() in {
        "1", "true", "yes"
    }
    health["direct_error"] = os.environ.get("WP_DIRECT_ERROR", "").strip()
    health["direct_fallback_used"] = health["source_mode"] == "upstream_fallback"
    if health["status"] == "数据日期过期" and os.environ.get("WP_ALLOW_STALE_DATA", "").strip() != "1":
        logging.error("Stale WP data: data_trade_date=%s expected=%s", health.get("data_trade_date"), expected_trade_date)
        top50 = top50.iloc[0:0].copy()
        full_rank = full_rank.iloc[0:0].copy()
        ranked_input = ranked_input.iloc[0:0].copy()
        health["candidate_count"] = 0
        health["top50_count"] = 0
        buy_plan = buy_plan.iloc[0:0].copy()
        buy_decision_table = buy_decision_table.iloc[0:0].copy()
        buy_decision.summary["buy_count"] = 0
    rule_errors = assert_top50_rules(top50)
    if rule_errors:
        health["status"] = "规则自检失败"
        health["rule_errors"] = rule_errors
        logging.error("Top50 rule errors: %s", rule_errors)

    if health.get("status") not in {"ok", "无符合条件股票"}:
        market_regime = {
            **market_regime,
            "state": "数据不足",
            "manual_action": "建议空仓",
            "reason": f"数据健康状态为{health.get('status')}，禁止生成新开仓建议",
            "manual_decision_support_only": True,
        }
        health["market_regime"] = market_regime

    tail_phase = tail_window_phase(str(health.get("market_data_time") or ""))
    health["tail_window_state"] = tail_phase
    if not accepts_new_tail_primary(str(health.get("market_data_time") or "")):
        buy_plan = buy_plan.iloc[0:0].copy()
        buy_decision_table = buy_decision_table.iloc[0:0].copy()
        buy_decision.summary["buy_count"] = 0
        buy_decision.summary["tail_window_state"] = tail_phase

    market_universe = flag_limitup(enrich_basic_fields(market_context))
    tail_observation_result = update_tail_observation(
        ranked_input,
        buy_plan,
        market_universe,
        health,
        output_root / "csv" / "wp_tail_observation.csv",
        historical_primaries=historical_primaries,
        max_limit_up_pct=float(config.get("max_limit_up_pct", 10.0)),
    )
    tail_observation = tail_observation_result.table
    health["tail_observation_count"] = int(tail_observation_result.summary.get("count", 0))
    health["tail_observation_active_count"] = int(tail_observation_result.summary.get("active_count", 0))
    health["tail_observation_sealed_count"] = int(tail_observation_result.summary.get("sealed_count", 0))
    health["tail_observation_review_count"] = int(tail_observation_result.summary.get("review_count", 0))
    health["tail_observation_state"] = str(tail_observation_result.summary.get("status", ""))
    decision_support = build_decision_support(
        tail_observation,
        market_regime,
        str(health.get("market_data_time") or ""),
        config,
    )
    health["decision_support_action"] = decision_support.summary.get("action", "")
    health["manual_execution_only"] = True
    health["order_routing_enabled"] = False
    archive_dir = output_root / "html_reports" / "archive" / current.strftime("%Y%m%d")
    ensure_dir(archive_dir)
    validation_result = update_buy_plan_validation(buy_plan, health, output_root, current)
    sampling_result = update_tail_sampling(
        validation_result.table,
        health,
        output_root / "csv" / "wp_tail_sampling.csv",
    )
    validation_result.summary["sampling_days"] = sampling_result.summary["days"]
    validation_result.summary["sampling_missing_days"] = sampling_result.summary["missing_day_count"]
    health["tail_sampling_missing_days"] = sampling_result.summary["missing_day_count"]
    exit_guidance = build_exit_guidance(
        validation_result.table,
        market_universe,
        str(health.get("data_trade_date") or forecast_trade_date),
        str(health.get("market_data_time") or ""),
        config,
    )
    snapshot_paths = _archive_decision_snapshots(
        output_root,
        ranked_input,
        decision_support.table,
        exit_guidance.table,
        decision_support.summary,
        health,
    )
    health["decision_snapshot_paths"] = snapshot_paths
    backtest_summaries = load_backtest_summaries(output_root)
    top50.to_csv(output_root / "csv" / "wp_top50.csv", index=False, encoding="utf-8-sig")
    full_rank.to_csv(output_root / "csv" / "wp_full_rank.csv", index=False, encoding="utf-8-sig")
    ranked_input.to_csv(output_root / "csv" / "wp_model_debug.csv", index=False, encoding="utf-8-sig")
    forecast_columns = [
        column
        for column in ["ts_code", "name", "price", "pct_chg", "sector_name", "tail_profit_score", *FORECAST_COLUMNS]
        if column in ranked_input.columns
    ]
    ranked_input[forecast_columns].to_csv(output_root / "csv" / "wp_t1_forecast.csv", index=False, encoding="utf-8-sig")
    buy_plan.to_csv(output_root / "csv" / "wp_buy_plan.csv", index=False, encoding="utf-8-sig")
    buy_decision_table.to_csv(output_root / "csv" / "wp_buy_decision.csv", index=False, encoding="utf-8-sig")
    decision_support.table.to_csv(output_root / "csv" / "wp_decision_support.csv", index=False, encoding="utf-8-sig")
    exit_guidance.table.to_csv(output_root / "csv" / "wp_t1_exit_guidance.csv", index=False, encoding="utf-8-sig")
    latest_html = output_root / "html_reports" / "latest.html"
    archive_html = archive_dir / f"{current.strftime('%H%M')}.html"
    preserve_latest = (not load_result.ok) and (not load_result.fallback_used) and latest_html.exists()
    health["preserved_latest_html"] = bool(preserve_latest)
    health["buy_plan_count"] = int(len(buy_plan))
    latest_payload = {
        "generated_at": update_time,
        "market_data_time": health.get("market_data_time", ""),
        "source_data_hash": input_hash,
        "wp_run_time": update_time,
        "health": health,
        "buy_plan_validation": validation_result.table.to_dict(orient="records"),
        "buy_plan_validation_summary": validation_result.summary,
        "tail_sampling": sampling_result.table.to_dict(orient="records"),
        "buy_plan": buy_plan.to_dict(orient="records"),
        "tail_observation": tail_observation.to_dict(orient="records"),
        "tail_observation_summary": tail_observation_result.summary,
        "market_regime": market_regime,
        "t1_forecast_summary": forecast_result.summary,
        "decision_support": decision_support.summary,
        "decision_support_records": decision_support.table.to_dict(orient="records"),
        "t1_exit_guidance_summary": exit_guidance.summary,
        "t1_exit_guidance": exit_guidance.table.to_dict(orient="records"),
        "top50": top50.to_dict(orient="records"),
        "backtests": backtest_summaries,
        "buy_model_version": TAIL_PROFIT_MODEL_VERSION,
    }
    write_json(output_root / "json" / "latest.json", latest_payload)
    write_json(
        output_root / "json" / "wp_buy_plan.json",
        {
            "generated_at": update_time,
            "market_data_time": health.get("market_data_time", ""),
            "source_data_hash": input_hash,
            "wp_run_time": update_time,
            "summary": buy_decision.summary,
            "buy_model_version": TAIL_PROFIT_MODEL_VERSION,
            "buy_plan": buy_plan.to_dict(orient="records"),
            "tail_observation": tail_observation.to_dict(orient="records"),
            "tail_observation_summary": tail_observation_result.summary,
            "market_regime": market_regime,
            "t1_forecast_summary": forecast_result.summary,
            "decision_support": decision_support.summary,
            "decision_support_records": decision_support.table.to_dict(orient="records"),
        },
    )
    write_json(
        output_root / "json" / "wp_tail_observation.json",
        {
            "generated_at": update_time,
            "market_data_time": health.get("market_data_time", ""),
            "source_data_hash": input_hash,
            "summary": tail_observation_result.summary,
            "records": tail_observation.to_dict(orient="records"),
        },
    )
    write_json(
        output_root / "json" / "wp_t1_forecast.json",
        {
            "generated_at": update_time,
            "market_data_time": health.get("market_data_time", ""),
            "summary": forecast_result.summary,
            "records": ranked_input[forecast_columns].to_dict(orient="records"),
        },
    )
    write_json(
        output_root / "json" / "wp_decision_support.json",
        {
            "generated_at": update_time,
            "market_data_time": health.get("market_data_time", ""),
            "market_regime": market_regime,
            "summary": decision_support.summary,
            "records": decision_support.table.to_dict(orient="records"),
            "manual_execution_only": True,
            "order_routing_enabled": False,
        },
    )
    write_json(
        output_root / "json" / "wp_t1_exit_guidance.json",
        {
            "generated_at": update_time,
            "market_data_time": health.get("market_data_time", ""),
            "summary": exit_guidance.summary,
            "records": exit_guidance.table.to_dict(orient="records"),
            "manual_execution_only": True,
            "order_routing_enabled": False,
        },
    )
    write_json(
        output_root / "json" / "wp_buy_plan_validation.json",
        {
            "generated_at": update_time,
            "market_data_time": health.get("market_data_time", ""),
            "source_data_hash": input_hash,
            "wp_run_time": update_time,
            "summary": validation_result.summary,
            "records": validation_result.table.to_dict(orient="records"),
        },
    )
    write_json(
        output_root / "json" / "wp_tail_sampling.json",
        {
            "generated_at": update_time,
            "market_data_time": health.get("market_data_time", ""),
            "summary": sampling_result.summary,
            "records": sampling_result.table.to_dict(orient="records"),
        },
    )
    write_json(
        manifest_path,
        {
            "latest_update": update_time,
            "market_data_time": health.get("market_data_time", ""),
            "data_revision": health.get("market_data_time", ""),
            "source_trade_date": health.get("source_trade_date", health.get("data_trade_date", "")),
            "source_generated_at": health.get("source_generated_at", ""),
            "source_scheduled_slot": health.get("source_scheduled_slot", ""),
            "source_mode": health.get("source_mode", ""),
            "source_repository": health.get("source_repository", ""),
            "source_processor_revision": health.get("source_processor_revision", ""),
            "direct_attempted": health.get("direct_attempted", False),
            "direct_fallback_used": health.get("direct_fallback_used", False),
            "direct_error": health.get("direct_error", ""),
            "source_data_hash": input_hash,
            "wp_run_time": update_time,
            "report_revision": update_time,
            "top50_count": len(top50),
            "buy_plan_count": len(buy_plan),
            "tail_observation_count": len(tail_observation),
            "tail_observation_summary": tail_observation_result.summary,
            "validation_summary": validation_result.summary,
            "forecast_summary": forecast_result.summary,
            "market_regime": market_regime,
            "decision_support": decision_support.summary,
            "exit_guidance_summary": exit_guidance.summary,
            "manual_execution_only": True,
            "order_routing_enabled": False,
            "model_version": MODEL_VERSION,
            "buy_model_version": TAIL_PROFIT_MODEL_VERSION,
            "backtest_window_count": len(backtest_summaries),
            "health_status": health["status"],
            "preserved_latest_html": bool(preserve_latest),
        },
    )
    write_json(output_root / "json" / "wp_data_healthcheck.json", health)
    if preserve_latest:
        logging.error("Source failed; preserving existing latest.html. error=%s", load_result.error)
    else:
        render_html(top50, full_rank, health, latest_html, buy_plan=buy_plan, observation_pool=tail_observation, validation=validation_result.table, validation_summary=validation_result.summary, backtests=backtest_summaries, decision_support=decision_support.summary, market_regime=market_regime, t1_forecasts=decision_support.table, exit_guidance=exit_guidance.table)
    render_html(top50, full_rank, health, archive_html, buy_plan=buy_plan, observation_pool=tail_observation, validation=validation_result.table, validation_summary=validation_result.summary, backtests=backtest_summaries, decision_support=decision_support.summary, market_regime=market_regime, t1_forecasts=decision_support.table, exit_guidance=exit_guidance.table)
    render_markdown(top50, output_root / "html_reports" / "latest.md", buy_plan=buy_plan, observation_pool=tail_observation, decision_support=decision_support.summary, exit_guidance=exit_guidance.table)
    logging.info(
        "WP run completed: raw=%s candidates=%s top50=%s buy_plan=%s tail_observation=%s missing_fields=%s fallback=%s outputs=%s log=%s",
        len(raw),
        len(candidates),
        len(top50),
        len(buy_plan),
        len(tail_observation),
        ",".join(health.get("missing_fields", [])) or "none",
        health.get("fallback_used"),
        output_root,
        log_path,
    )
    return health


if __name__ == "__main__":
    run()
