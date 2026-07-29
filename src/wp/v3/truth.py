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
from .io import atomic_write_csv
from .ledger import (
    assert_ledger_invariants,
    load_shadow_ledger,
    save_shadow_ledger,
    settle_entry_benchmarks,
)
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
        _backfill_missing_entry_benchmarks(
            pro,
            due,
            truth_by_date=truth_by_date,
            config=config,
        )
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
    atomic_write_csv(
        validation,
        output / "csv" / "wp_buy_plan_validation.csv",
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
    for column in (
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "up_limit",
        "down_limit",
        "adj_factor",
    ):
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
    uses_v6_entry = (
        candidate.get("entry_contract")
        == config.execution.entry_price_contract
    )
    if uses_v6_entry:
        benchmark_status = str(
            candidate.get("entry_benchmark_status") or "PENDING"
        )
        if benchmark_status == "PENDING":
            candidate["truth_error"] = "missing_entry_benchmark"
            return
        entry_price = _positive_float(candidate.get("entry_price"))
        entry_fillable = bool(candidate.get("entry_fillable"))
        if benchmark_status == "SETTLED" and not entry_price:
            candidate["truth_error"] = "invalid_entry_benchmark"
            return
    else:
        signal_price = float(candidate["first_signal_price"])
        entry_price = signal_price * (
            1 + config.execution.entry_slippage_bps / 10_000.0
        )
        entry_fillable = True
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
        if total_return_close and entry_price
        else None
    )
    if not entry_fillable:
        # No position was established, so the realized contract return is cash.
        # The miss still counts as a negative class for target_net_positive and
        # remains visible in the entry-fill calibration statistics.
        net = 0.0
        execution_status = "ENTRY_NOT_FILLED"
    elif not exit_fillable:
        net = config.execution.non_fill_penalty_pct
        execution_status = "EXIT_NOT_FILLED"
    else:
        net = gross - config.execution.round_trip_cost_bps / 100.0
        execution_status = "ROUND_TRIP_FILLED"
    candidate.update(
        {
            "entry_price": entry_price,
            "entry_fillable": entry_fillable,
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
            "execution_status": execution_status,
            "gross_return_pct": gross,
            "net_return_pct": net,
            "net_positive": bool(entry_fillable and exit_fillable and net > 0),
            "truth_status": "verified",
            "truth_verified_at": now_cn().isoformat(),
            "truth_contract": (
                "immutable_next_5m_entry_to_T+1_close_after_costs"
                if uses_v6_entry
                else "legacy_first_signal_price_to_T+1_close_after_costs"
            ),
            "corporate_action_adjustment": "adj_factor_total_return",
            "non_fill_penalty_pct": (
                config.execution.non_fill_penalty_pct
                if entry_fillable and not exit_fillable
                else None
            ),
        }
    )
    candidate.pop("truth_error", None)


def _backfill_missing_entry_benchmarks(
    pro: Any,
    candidates: list[dict[str, Any]],
    *,
    truth_by_date: dict[str, pd.DataFrame],
    config: V3Config,
) -> None:
    for candidate in candidates:
        if (
            candidate.get("entry_contract")
            != config.execution.entry_price_contract
            or str(candidate.get("entry_benchmark_status") or "PENDING")
            != "PENDING"
        ):
            continue
        trade_date = str(candidate.get("trade_date") or "")
        settlement_slot = str(candidate.get("entry_benchmark_slot") or "")
        code = str(candidate.get("ts_code") or "")
        if not (
            len(trade_date) == 8
            and settlement_slot
            and code
            and trade_date in truth_by_date
        ):
            candidate["truth_error"] = "entry_benchmark_contract_incomplete"
            continue
        try:
            raw = pro.query(
                "stk_mins",
                ts_code=code,
                start_date=(
                    f"{trade_date[:4]}-{trade_date[4:6]}-"
                    f"{trade_date[6:]} 14:15:00"
                ),
                end_date=(
                    f"{trade_date[:4]}-{trade_date[4:6]}-"
                    f"{trade_date[6:]} 15:00:00"
                ),
                freq="5min",
                fields="ts_code,trade_time,open,high,low,close,vol,amount",
            )
        except Exception as error:  # provider failures must remain visible
            candidate["truth_error"] = (
                f"entry_benchmark_query_failed:{type(error).__name__}"
            )
            continue
        frame = pd.DataFrame(raw).copy()
        if not frame.empty:
            frame["trade_time"] = pd.to_datetime(
                frame["trade_time"],
                errors="coerce",
            )
            frame = frame.loc[
                frame["trade_time"].dt.strftime("%H:%M").eq(settlement_slot)
            ].copy()
        if frame.empty:
            settlement = pd.DataFrame(columns=["ts_code"])
        else:
            frame = frame.sort_values("trade_time", kind="stable").tail(1)
            entry_truth = truth_by_date[trade_date]
            up_limit = (
                _positive_float(entry_truth.loc[code].get("up_limit"))
                if code in entry_truth.index
                else None
            )
            settlement = pd.DataFrame(
                [
                    {
                        "ts_code": code,
                        "entry_benchmark_slot": settlement_slot,
                        "entry_benchmark_price": frame.iloc[0].get("close"),
                        "entry_benchmark_amount": frame.iloc[0].get("amount"),
                        "entry_benchmark_bar_time": frame.iloc[0].get(
                            "trade_time"
                        ),
                        "data_age_seconds": 0.0,
                        "up_limit": up_limit,
                    }
                ]
            )
        settle_entry_benchmarks(
            {
                "sessions": [
                    {
                        "trade_date": trade_date,
                        "candidates": [candidate],
                    }
                ]
            },
            settlement,
            trade_date=trade_date,
            settlement_slot=settlement_slot,
            config=config,
            settled_at=now_cn().isoformat(),
        )
        candidate["entry_benchmark_source"] = "reconstructed_stk_mins"


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
