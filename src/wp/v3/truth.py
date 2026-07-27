from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tushare as ts

from wp.calendar import now_cn
from wp.utils import write_json

from .contracts import V3Config, load_v3_config
from .dashboard import render_v3_dashboard
from .ledger import assert_ledger_invariants, load_shadow_ledger, save_shadow_ledger
from .registry import (
    load_registry,
    refresh_shadow_metrics,
    save_registry,
)


ROOT = Path(__file__).resolve().parents[3]


def run_v3_close_validation(
    output_root: Path | None = None,
    current: datetime | None = None,
) -> dict[str, Any]:
    output = output_root or ROOT / "outputs"
    current = current or now_cn()
    today = current.strftime("%Y%m%d")
    config = load_v3_config(ROOT / "config" / "wp_v3.yml")
    ledger_path = output / "json" / "wp_v3_candidate_ledger.json"
    registry_path = output / "json" / "wp_model_registry_v3.json"
    ledger = load_shadow_ledger(ledger_path)
    ledger_before = json.dumps(ledger, ensure_ascii=False, sort_keys=True)
    due = [
        candidate
        for session in ledger.get("sessions", [])
        for candidate in session.get("candidates", [])
        if str(candidate.get("target_trade_date") or "") <= today
        and str(candidate.get("target_trade_date") or "").isdigit()
        and candidate.get("truth_status") != "verified"
    ]
    if due:
        token = os.environ.get("TUSHARE_TOKEN", "").strip()
        if not token:
            raise RuntimeError("TUSHARE_TOKEN is required for V3 close-truth validation")
        pro = ts.pro_api(token)
        truth_by_date = {}
        truth_errors = {}
        truth_dates = {
            str(candidate[date_field])
            for candidate in due
            for date_field in ("trade_date", "target_trade_date")
            if str(candidate.get(date_field) or "").isdigit()
        }
        for target_date in sorted(truth_dates):
            try:
                truth_by_date[target_date] = _fetch_close_truth(pro, target_date)
            except RuntimeError as error:
                truth_errors[target_date] = str(error)
        for candidate in due:
            target_date = str(candidate["target_trade_date"])
            entry_date = str(candidate["trade_date"])
            if target_date in truth_by_date and entry_date in truth_by_date:
                _verify_candidate(
                    candidate,
                    truth_by_date[target_date],
                    config,
                    entry_truth=truth_by_date[entry_date],
                )
            elif target_date in truth_errors:
                candidate["truth_error"] = truth_errors[target_date]
            elif entry_date in truth_errors:
                candidate["truth_error"] = truth_errors[entry_date]
    assert_ledger_invariants(ledger, config)
    if json.dumps(ledger, ensure_ascii=False, sort_keys=True) != ledger_before:
        save_shadow_ledger(ledger, ledger_path)

    registry = load_registry(registry_path)
    registry_before = json.dumps(registry, ensure_ascii=False, sort_keys=True)
    fingerprints = {
        str(value)
        for value in (
            registry.get("active_model_fingerprint"),
            registry.get("shadow_model_fingerprint"),
        )
        if value
    }
    promotion_results = {}
    for fingerprint in fingerprints:
        if any(
            model.get("fingerprint") == fingerprint
            for model in registry.get("models", [])
        ):
            decision = refresh_shadow_metrics(registry, fingerprint, ledger, config)
            promotion_results[fingerprint] = {
                "eligible": decision.eligible,
                "checks": decision.checks,
                "reason": decision.reason,
            }
    if json.dumps(registry, ensure_ascii=False, sort_keys=True) != registry_before:
        save_registry(registry, registry_path)

    validation = _validation_frame(ledger)
    pending_mask = (
        validation.get("truth_status", pd.Series("", index=validation.index))
        != "verified"
    )
    target_dates = validation.get(
        "target_trade_date",
        pd.Series("", index=validation.index),
    ).astype(str)
    due_pending_mask = (
        pending_mask
        & target_dates.str.fullmatch(r"\d{8}", na=False)
        & target_dates.le(today)
    )
    validation.to_csv(
        output / "csv" / "wp_buy_plan_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest_path = output / "json" / "wp_manifest.json"
    manifest = _read_json(manifest_path)
    update_time = current.strftime("%Y-%m-%d %H:%M:%S")
    manifest.update(
        {
            "latest_update": update_time,
            "report_revision": update_time,
            "validation_updated_at": update_time,
            "session_phase": "CLOSED",
            "buy_plan_count": 0,
            "pending_truth_count": int(
                pending_mask.sum()
            ),
            "pending_due_truth_count": int(due_pending_mask.sum()),
            "verified_candidate_count": int(
                (validation.get("truth_status", pd.Series(dtype=str)) == "verified").sum()
            ),
        }
    )
    write_json(manifest_path, manifest)
    write_json(
        output / "json" / "wp_buy_plan_validation.json",
        {
            "generated_at": update_time,
            "schema_version": "wp_v3_candidate_truth_1",
            "summary": {
                "candidate_count": int(len(validation)),
                "verified_count": int(
                    (validation.get("truth_status", pd.Series(dtype=str)) == "verified").sum()
                ),
                "pending_count": int(
                    pending_mask.sum()
                ),
                "pending_due_count": int(due_pending_mask.sum()),
                "promotion": promotion_results,
            },
            "records": validation.to_dict(orient="records"),
        },
    )
    write_json(
        output / "json" / "wp_strategy_ledger.json",
        {
            "generated_at": update_time,
            "schema_version": "wp_strategy_ledger_v3_bridge",
            "summary": {
                "candidate_semantics": "model_candidates_not_user_fills",
                "verified_count": int(
                    (validation.get("truth_status", pd.Series(dtype=str)) == "verified").sum()
                ),
            },
            "sessions": ledger.get("sessions", []),
        },
    )
    predictions = _read_csv(output / "csv" / "wp_v3_live_predictions.csv")
    render_v3_dashboard(
        output / "html_reports" / "latest.html",
        manifest=manifest,
        predictions=predictions,
        ledger=ledger,
        registry=registry,
        config=config,
        replay=_read_json(output / "json" / "wp_v3_historical_replay.json"),
    )
    archive = (
        output
        / "html_reports"
        / "archive"
        / today
        / f"{current.strftime('%H%M%S')}_close.html"
    )
    render_v3_dashboard(
        archive,
        manifest=manifest,
        predictions=predictions,
        ledger=ledger,
        registry=registry,
        config=config,
        replay=_read_json(output / "json" / "wp_v3_historical_replay.json"),
    )
    newly_verified = sum(
        candidate.get("truth_status") == "verified" for candidate in due
    )
    return {
        "verified_count": int(
            (validation.get("truth_status", pd.Series(dtype=str)) == "verified").sum()
        ),
        "pending_count": int(
            due_pending_mask.sum()
        ),
        "pending_total_count": int(pending_mask.sum()),
        "newly_verified": int(newly_verified),
    }


def _fetch_close_truth(pro: Any, trade_date: str) -> pd.DataFrame:
    daily = pro.daily(
        trade_date=trade_date,
        fields="ts_code,trade_date,open,high,low,close,vol,amount",
    )
    limits = pro.stk_limit(
        trade_date=trade_date,
        fields="trade_date,ts_code,up_limit,down_limit",
    )
    adjustments = pro.adj_factor(
        trade_date=trade_date,
        fields="ts_code,trade_date,adj_factor",
    )
    if len(daily) <= 1_000 or len(limits) <= 1_000 or len(adjustments) <= 1_000:
        raise RuntimeError(
            f"close truth for {trade_date} is incomplete: daily={len(daily)} "
            f"limits={len(limits)} adj={len(adjustments)}"
        )
    truth = (
        adjustments.merge(
            daily,
            on=["trade_date", "ts_code"],
            how="left",
        )
        .merge(limits, on=["trade_date", "ts_code"], how="left")
    )
    for column in ("open", "high", "low", "close", "vol", "down_limit", "adj_factor"):
        truth[column] = pd.to_numeric(truth[column], errors="coerce")
    return truth.drop_duplicates("ts_code").set_index("ts_code")


def _verify_candidate(
    candidate: dict[str, Any],
    truth: pd.DataFrame,
    config: V3Config,
    *,
    entry_truth: pd.DataFrame | None = None,
) -> None:
    code = str(candidate["ts_code"])
    if code not in truth.index:
        return
    row = truth.loc[code]
    signal_price = float(candidate["first_signal_price"])
    entry_price = signal_price * (1 + config.execution.entry_slippage_bps / 10_000.0)
    entry_adj_factor = _positive_float(candidate.get("entry_adj_factor"))
    if entry_truth is not None:
        if code not in entry_truth.index:
            candidate["truth_error"] = "missing_entry_adjustment_truth"
            return
        entry_adj_factor = _positive_float(
            entry_truth.loc[code].get("adj_factor")
        )
    target_adj_factor = _positive_float(row.get("adj_factor"))
    if not entry_adj_factor or not target_adj_factor:
        candidate["truth_error"] = "missing_adjustment_factor"
        return
    close = _positive_float(row.get("close"))
    volume = _positive_float(row.get("vol"))
    down_limit = _positive_float(row.get("down_limit"))
    exit_fillable = bool(
        close
        and volume
        and (
            down_limit is None
            or close > down_limit * 1.0001
        )
    )
    total_return_close = (
        close * target_adj_factor / entry_adj_factor if close else None
    )
    gross = (
        (total_return_close / entry_price - 1.0) * 100.0
        if total_return_close
        else None
    )
    net = (
        gross - config.execution.round_trip_cost_bps / 100.0
        if gross is not None
        else config.execution.non_fill_penalty_pct
    )
    if not exit_fillable:
        net = min(net, config.execution.non_fill_penalty_pct)
    candidate.update(
        {
            "entry_price": entry_price,
            "entry_adj_factor_truth": entry_adj_factor,
            "entry_slippage_bps": config.execution.entry_slippage_bps,
            "round_trip_cost_bps": config.execution.round_trip_cost_bps,
            "baseline_all_in_cost_bps": (
                config.execution.baseline_all_in_cost_bps
            ),
            "t1_close": close,
            "t1_adj_factor": target_adj_factor,
            "t1_total_return_close": total_return_close,
            "exit_fillable": exit_fillable,
            "gross_return_pct": gross,
            "net_return_pct": net,
            "net_positive": bool(exit_fillable and net > 0),
            "truth_status": "verified",
            "truth_verified_at": now_cn().isoformat(),
            "truth_contract": "immutable_first_signal_price_to_T+1_close_after_costs",
            "corporate_action_adjustment": "adj_factor_total_return",
            "non_fill_penalty_pct": (
                config.execution.non_fill_penalty_pct
                if not exit_fillable
                else None
            ),
        }
    )


def _validation_frame(ledger: dict[str, Any]) -> pd.DataFrame:
    records = []
    for session in ledger.get("sessions", []):
        for candidate in session.get("candidates", []):
            row = dict(candidate)
            row["plan_trade_date"] = row.get("trade_date")
            records.append(row)
    return pd.DataFrame(records)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None
