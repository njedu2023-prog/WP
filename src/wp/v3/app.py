from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from wp.calendar import now_cn
from wp.utils import ensure_dir, write_json

from .contracts import V3Config, load_v3_config, session_phase
from .cohorts import attach_cohort_labels, select_live_cohorts
from .dashboard import render_v3_dashboard
from .evidence import archive_signal_evidence
from .io import atomic_write_csv
from .ledger import (
    assert_ledger_invariants,
    freeze_shadow_session,
    load_shadow_ledger,
    record_shadow_slot,
    save_shadow_ledger,
    session_records,
    settle_entry_benchmarks,
)
from .live import inference_manifest, run_live_inference
from .registry import load_registry


ROOT = Path(__file__).resolve().parents[3]


def run_v3() -> dict[str, Any]:
    current = now_cn()
    output = ROOT / "outputs"
    ensure_dir(output / "json")
    ensure_dir(output / "csv")
    ensure_dir(output / "html_reports")
    config = load_v3_config(ROOT / "config" / "wp_v3.yml")
    source_path = Path(
        os.environ.get("WP_V3_SOURCE_CSV", "").strip()
        or ROOT / "data" / "v3" / "latest" / "wp_v3_live_features.csv"
    )
    source_manifest_path = source_path.with_name("wp_v3_live_manifest.json")
    if not source_path.exists() or not source_manifest_path.exists():
        return _write_not_ready(
            output,
            config,
            current,
            f"V3 causal live input is missing: {source_path}",
        )
    frame = pd.read_csv(source_path, keep_default_na=False, dtype={"ts_code": str})
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    market_time = pd.to_datetime(source_manifest.get("market_data_time"), errors="coerce")
    decision_time = _decision_time(source_manifest, fallback=current)
    frame = _refresh_data_age(
        frame,
        current=decision_time,
        market_time=market_time,
    )
    runtime_data_age = _data_age_seconds(
        frame,
        current=current,
        market_time=market_time,
    )
    trade_date = str(source_manifest.get("trade_date") or current.strftime("%Y%m%d"))
    signal_slot = str(source_manifest.get("signal_slot") or "")
    phase = session_phase(current, config)
    source_signal_authorized = _source_signal_authorized(
        source_manifest,
        config,
    )
    source_recovery_authorized = _source_recovery_authorized(
        source_manifest,
        config,
    )
    source_evidence_authorized = (
        source_signal_authorized or source_recovery_authorized
    )
    record_signal = (
        source_evidence_authorized
        and (
            phase in {"SIGNAL", "NO_NEW_SIGNAL"}
            or source_recovery_authorized
        )
    )
    live_display_allowed = (
        source_signal_authorized
        and trade_date == current.strftime("%Y%m%d")
        and phase in {"SIGNAL", "NO_NEW_SIGNAL", "FROZEN"}
    )
    registry_path = ROOT / "outputs" / "json" / "wp_model_registry_v3.json"
    registry = load_registry(registry_path)
    model_path = _resolve_model_path(registry)
    explicit_model_path = os.environ.get("WP_V3_MODEL_PATH", "").strip()
    if explicit_model_path:
        model_path = Path(explicit_model_path)
    inference = run_live_inference(
        frame,
        config,
        model_path=model_path,
        registry_path=registry_path,
    )
    cohort_selection = select_live_cohorts(inference.predictions, config)
    predictions = attach_cohort_labels(
        inference.predictions,
        cohort_selection,
    )
    inference_summary = inference_manifest(inference)
    evidence_manifest: dict[str, Any] = {}
    if source_evidence_authorized and signal_slot in config.strategy.signal_slots:
        evidence_features = _evidence_feature_universe(frame, predictions)
        evidence_manifest = archive_signal_evidence(
            output,
            features=evidence_features,
            predictions=predictions,
            source_manifest=source_manifest,
            inference_manifest=inference_summary,
            config=config,
        )
    ledger_path = output / "json" / "wp_v3_candidate_ledger.json"
    ledger = load_shadow_ledger(ledger_path)
    settlement_summary: dict[str, Any] = {}
    if source_evidence_authorized and signal_slot in config.strategy.signal_slots:
        settle_entry_benchmarks(
            ledger,
            frame,
            trade_date=trade_date,
            settlement_slot=signal_slot,
            config=config,
            settled_at=str(
                source_manifest.get("capture_completed_at")
                or source_manifest.get("market_data_time")
                or current.isoformat()
            ),
        )
    settlement_frame, settlement_manifest = _load_entry_settlement()
    if not settlement_frame.empty or settlement_manifest:
        settlement_trade_date = str(
            settlement_manifest.get("trade_date") or trade_date
        )
        settlement_slot = str(
            settlement_manifest.get("settlement_slot") or ""
        )
        if settlement_trade_date != trade_date:
            raise ValueError(
                "entry settlement trade date differs from live source"
            )
        settle_entry_benchmarks(
            ledger,
            settlement_frame,
            trade_date=trade_date,
            settlement_slot=settlement_slot,
            config=config,
            settled_at=str(
                settlement_manifest.get("capture_completed_at")
                or current.isoformat()
            ),
        )
        settlement_summary = {
            "entry_settlement_slot": settlement_slot,
            "entry_settlement_requested_symbols": settlement_manifest.get(
                "requested_symbols"
            ),
            "entry_settlement_observed_symbols": settlement_manifest.get(
                "observed_symbols"
            ),
            "entry_settlement_fresh_symbols": settlement_manifest.get(
                "fresh_symbols"
            ),
        }
    if (
        record_signal
        and signal_slot in config.strategy.signal_slots
        and inference.model_fingerprint
    ):
        record_shadow_slot(
            ledger,
            predictions,
            trade_date=trade_date,
            signal_slot=signal_slot,
            config=config,
            observed_at=str(
                source_manifest.get("recovered_at")
                or source_manifest.get("market_data_time")
                or current.isoformat()
            ),
            evidence_tier=str(
                source_manifest.get("evidence_tier")
                or "PROSPECTIVE_LIVE"
            ),
            prospective_eligible=bool(
                source_manifest.get("prospective_eligible", True)
            ),
            recovery_reason=source_manifest.get("recovery_reason"),
        )
    if phase in {"FROZEN", "CLOSED"} and inference.model_fingerprint:
        freeze_shadow_session(
            ledger,
            trade_date=trade_date,
            config=config,
            model_fingerprint=inference.model_fingerprint,
            policy_fingerprint=inference.policy_fingerprint,
        )
    assert_ledger_invariants(ledger, config)
    save_shadow_ledger(ledger, ledger_path)
    replay = _read_json(output / "json" / "wp_v3_historical_replay.json")
    legacy_audit = _read_json(
        output / "json" / "wp_legacy_history_audit.json"
    )
    current_session = next(
        (
            session
            for session in ledger.get("sessions", [])
            if str(session.get("trade_date")) == trade_date
        ),
        {},
    )
    missing_slots = list(current_session.get("missing_slots", []))
    pending_entry_benchmarks = sum(
        str(candidate.get("entry_benchmark_status") or "PENDING")
        == "PENDING"
        and candidate.get("entry_contract")
        == config.execution.entry_price_contract
        for candidate in session_records(current_session)
    )
    integrity_status = str(
        current_session.get("integrity_status") or "COLLECTING"
    )

    qualified_live = cohort_selection.qualified.copy()
    observations_live = cohort_selection.observations.copy()
    if not live_display_allowed:
        qualified_live = qualified_live.iloc[0:0].copy()
        observations_live = observations_live.iloc[0:0].copy()
    qualified_live["production_authorized"] = inference.formal_authorization
    qualified_live["manual_execution_only"] = True
    observations_live["production_authorized"] = False
    observations_live["manual_execution_only"] = True
    recorded_qualified_count = (
        int(current_session.get("qualified_count", len(qualified_live)))
        if current_session
        else int(len(qualified_live))
    )
    recorded_observation_count = (
        int(current_session.get("observation_count", len(observations_live)))
        if current_session
        else int(len(observations_live))
    )
    data_age = pd.to_numeric(runtime_data_age, errors="coerce")
    finite_data_age = data_age.loc[data_age.notna() & data_age.ge(0)]
    update_time = current.strftime("%Y-%m-%d %H:%M:%S")
    manifest = {
        "schema_version": "wp_manifest_v3",
        "latest_update": update_time,
        "report_revision": update_time,
        "source_trade_date": trade_date,
        "target_trade_date": source_manifest.get("target_trade_date"),
        "signal_slot": signal_slot,
        "source_scheduled_slot": signal_slot,
        "market_data_time": source_manifest.get("market_data_time"),
        "source_mode": (
            "direct_tushare_v41_same_day_recovery"
            if source_recovery_authorized
            else "direct_tushare_v41"
        ),
        "source_repository": "njedu2023-prog/WP",
        "session_phase": phase,
        "signal_capture_started_at": source_manifest.get("capture_started_at"),
        "signal_capture_completed_at": source_manifest.get(
            "capture_completed_at"
        ),
        "signal_source_authorized": source_signal_authorized,
        "signal_recovery_authorized": source_recovery_authorized,
        "signal_evidence_authorized": source_evidence_authorized,
        "signal_evidence_tier": source_manifest.get(
            "evidence_tier",
            "PROSPECTIVE_LIVE",
        ),
        "prospective_eligible": source_manifest.get(
            "prospective_eligible",
            True,
        ),
        "recovery_reason": source_manifest.get("recovery_reason"),
        "recovered_at": source_manifest.get("recovered_at"),
        "live_display_allowed": live_display_allowed,
        "buy_plan_count": int(len(qualified_live)),
        "qualified_count": recorded_qualified_count,
        "observation_count": recorded_observation_count,
        "observation_target_count": config.strategy.observation_count,
        "observation_selection_status": (
            cohort_selection.observation_selection_status
        ),
        "observation_shortfall_reason": (
            cohort_selection.observation_shortfall_reason
        ),
        "shadow_qualified_count": int(
            predictions.get(
                "passes_policy",
                pd.Series(False, index=predictions.index),
            ).sum()
        ),
        "live_universe_count": int(len(frame)),
        "eligible_universe_count": int(frame.get("execution_eligible", pd.Series(dtype=bool)).sum()),
        "market_data_p95_age_seconds": (
            float(finite_data_age.quantile(0.95))
            if not finite_data_age.empty
            else None
        ),
        "latest_bar_slot": source_manifest.get("latest_bar_slot"),
        "expected_symbols": source_manifest.get("expected_symbols"),
        "fresh_row_count": source_manifest.get("fresh_row_count"),
        "open_universe_coverage": source_manifest.get("open_universe_coverage"),
        "tail_universe_coverage": source_manifest.get("tail_universe_coverage"),
        "health_status": (
            "session_integrity_fault"
            if phase in {"FROZEN", "CLOSED"}
            and (
                missing_slots
                or pending_entry_benchmarks
                or (
                    current_session.get("cohort_contract_version")
                    == "dual_cohort_v1"
                    and int(current_session.get("observation_count") or 0)
                    != config.strategy.observation_count
                )
            )
            else "ok"
        ),
        "session_integrity_status": integrity_status,
        "covered_slots": list(current_session.get("covered_slots", [])),
        "missing_slots": missing_slots,
        "pending_entry_benchmark_count": pending_entry_benchmarks,
        "manual_execution_only": True,
        "order_routing_enabled": False,
        "retrospective_evidence_start": (
            config.evidence.retrospective_start_date
        ),
        "retrospective_evidence_end": (
            config.evidence.retrospective_end_date
        ),
        "live_shadow_start_date": config.evidence.live_shadow_start_date,
        "cohort_statistics_separate": (
            config.evidence.keep_cohort_statistics_separate
        ),
        "signal_evidence_digest": evidence_manifest.get("evidence_digest"),
        "signal_evidence_path": (
            (
                Path("outputs")
                / "audit"
                / trade_date[:4]
                / trade_date
                / signal_slot.replace(":", "")
                / "manifest.json"
            ).as_posix()
            if evidence_manifest
            else None
        ),
        **settlement_summary,
        **inference_summary,
    }
    predictions_path = output / "csv" / "wp_v3_live_predictions.csv"
    atomic_write_csv(predictions, predictions_path)
    atomic_write_csv(
        qualified_live,
        output / "csv" / "wp_buy_plan.csv",
    )
    write_json(output / "json" / "wp_manifest.json", manifest)
    write_json(
        output / "json" / "wp_decision_support.json",
        {
            "generated_at": update_time,
            "market_data_time": manifest["market_data_time"],
            "summary": manifest,
            "records": qualified_live.to_dict(orient="records"),
            "qualified_records": qualified_live.to_dict(orient="records"),
            "observation_records": observations_live.to_dict(
                orient="records"
            ),
            "cohort_contract": {
                "decision_time": config.strategy.signal_slots[0],
                "qualified_count_rule": "all_fixed_gate_passes",
                "observation_count": config.strategy.observation_count,
                "statistics_separate": True,
            },
            "manual_execution_only": True,
            "order_routing_enabled": False,
        },
    )
    write_json(
        output / "json" / "wp_strategy_ledger.json",
        {
            "generated_at": update_time,
            "schema_version": "wp_strategy_ledger_v4_bridge",
            "summary": {
                "state": inference.state,
                "formal_authorization": inference.formal_authorization,
                "candidate_semantics": "model_candidates_not_user_fills",
                "qualified_statistics_scope": "official_strategy",
                "observation_statistics_scope": "research_only",
                "live_shadow_start_date": (
                    config.evidence.live_shadow_start_date
                ),
            },
            "sessions": ledger.get("sessions", []),
        },
    )
    retrospective = _read_json(
        output / "json" / "wp_v41_backtest_202605_202607.json"
    )
    research_seed = _read_json(
        ROOT / "config" / "wp_v15_frozen_shadow_candidate.json"
    )
    report_path = output / "html_reports" / "latest.html"
    render_v3_dashboard(
        report_path,
        manifest=manifest,
        predictions=predictions,
        ledger=ledger,
        registry=registry,
        config=config,
        replay=replay,
        legacy_audit=legacy_audit,
        retrospective=retrospective,
        research_seed=research_seed,
    )
    archive = (
        output
        / "html_reports"
        / "archive"
        / trade_date
        / f"{current.strftime('%H%M%S')}_v3.html"
    )
    render_v3_dashboard(
        archive,
        manifest=manifest,
        predictions=predictions,
        ledger=ledger,
        registry=registry,
        config=config,
        replay=replay,
        legacy_audit=legacy_audit,
        retrospective=retrospective,
        research_seed=research_seed,
    )
    return manifest


def _evidence_feature_universe(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    if "ts_code" not in features or "ts_code" not in predictions:
        raise ValueError("signal evidence requires ts_code in both frames")
    feature_codes = features["ts_code"].astype(str)
    prediction_codes = predictions["ts_code"].astype(str)
    if feature_codes.duplicated().any() or prediction_codes.duplicated().any():
        raise ValueError("signal evidence requires one row per stock")
    unknown = sorted(set(prediction_codes) - set(feature_codes))
    if unknown:
        raise ValueError(
            "prediction universe contains stocks absent from source features: "
            + ",".join(unknown[:5])
        )
    return features.loc[feature_codes.isin(set(prediction_codes))].copy()


def _refresh_data_age(
    frame: pd.DataFrame,
    *,
    current: datetime,
    market_time: pd.Timestamp | Any,
) -> pd.DataFrame:
    result = frame.copy()
    result["data_age_seconds"] = _data_age_seconds(
        result,
        current=current,
        market_time=market_time,
    )
    return result


def _data_age_seconds(
    frame: pd.DataFrame,
    *,
    current: datetime,
    market_time: pd.Timestamp | Any,
) -> pd.Series:
    current_naive = current.replace(tzinfo=None)
    if "slot_bar_time" in frame:
        bar_time = pd.to_datetime(frame["slot_bar_time"], errors="coerce")
        return (
            current_naive - bar_time
        ).dt.total_seconds().clip(lower=0)
    if pd.notna(market_time):
        age = max(
            0.0,
            (
                current_naive
                - pd.Timestamp(market_time).to_pydatetime().replace(tzinfo=None)
            ).total_seconds(),
        )
        return pd.Series(age, index=frame.index, dtype=float)
    return pd.Series(float("inf"), index=frame.index, dtype=float)


def _decision_time(
    source_manifest: dict[str, Any],
    *,
    fallback: datetime,
) -> datetime:
    for field in (
        "decision_reference_time",
        "capture_completed_at",
        "capture_started_at",
        "market_data_time",
    ):
        value = source_manifest.get(field)
        if not value:
            continue
        try:
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("Asia/Shanghai")
            else:
                timestamp = timestamp.tz_convert("Asia/Shanghai")
            return timestamp.to_pydatetime()
        except (TypeError, ValueError):
            continue
    return fallback


def _source_signal_authorized(
    source_manifest: dict[str, Any],
    config: V3Config,
) -> bool:
    trade_date = str(source_manifest.get("trade_date") or "")
    signal_slot = str(source_manifest.get("signal_slot") or "")
    capture_started_at = source_manifest.get("capture_started_at")
    market_data_time = source_manifest.get("market_data_time")
    if (
        len(trade_date) != 8
        or signal_slot not in config.strategy.signal_slots
        or not capture_started_at
        or not market_data_time
    ):
        return False
    try:
        capture = pd.Timestamp(capture_started_at)
        if capture.tzinfo is None:
            capture = capture.tz_localize(config.strategy.timezone)
        else:
            capture = capture.tz_convert(config.strategy.timezone)
        scheduled = pd.Timestamp(
            f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} "
            f"{signal_slot}:00",
            tz=config.strategy.timezone,
        )
        market_time = pd.Timestamp(market_data_time)
        if market_time.tzinfo is None:
            market_time = market_time.tz_localize(config.strategy.timezone)
        else:
            market_time = market_time.tz_convert(config.strategy.timezone)
    except (TypeError, ValueError):
        return False
    return bool(
        scheduled
        <= capture
        <= scheduled
        + pd.Timedelta(
            config.execution.max_market_data_age_seconds,
            unit="s",
        )
        and scheduled
        <= market_time
        < scheduled + pd.Timedelta(60, unit="s")
    )


def _source_recovery_authorized(
    source_manifest: dict[str, Any],
    config: V3Config,
) -> bool:
    if (
        source_manifest.get("capture_contract")
        != "retrospective_same_day_rt_min_daily"
        or source_manifest.get("evidence_tier")
        != "RECOVERED_SAME_DAY"
        or source_manifest.get("prospective_eligible") is not False
        or source_manifest.get("source_api") != "rt_min_daily"
    ):
        return False
    trade_date = str(source_manifest.get("trade_date") or "")
    signal_slot = str(source_manifest.get("signal_slot") or "")
    market_data_time = source_manifest.get("market_data_time")
    recovered_at = source_manifest.get("recovered_at")
    decision_reference_time = source_manifest.get(
        "decision_reference_time"
    )
    if (
        len(trade_date) != 8
        or signal_slot not in config.strategy.signal_slots
        or not market_data_time
        or not recovered_at
        or not decision_reference_time
    ):
        return False
    try:
        scheduled = pd.Timestamp(
            f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} "
            f"{signal_slot}:00",
            tz=config.strategy.timezone,
        )
        market_time = _localized_timestamp(
            market_data_time,
            config.strategy.timezone,
        )
        recovered = _localized_timestamp(
            recovered_at,
            config.strategy.timezone,
        )
        decision_reference = _localized_timestamp(
            decision_reference_time,
            config.strategy.timezone,
        )
        row_count = int(source_manifest.get("row_count") or 0)
        fresh_count = int(source_manifest.get("fresh_row_count") or 0)
        open_coverage = float(
            source_manifest.get("open_universe_coverage") or 0.0
        )
        tail_coverage = float(
            source_manifest.get("tail_universe_coverage") or 0.0
        )
    except (TypeError, ValueError):
        return False
    return bool(
        scheduled
        <= market_time
        < scheduled + pd.Timedelta(1, unit="min")
        and scheduled
        <= decision_reference
        <= scheduled
        + pd.Timedelta(
            config.execution.max_market_data_age_seconds,
            unit="s",
        )
        and recovered.strftime("%Y%m%d") == trade_date
        and str(source_manifest.get("latest_bar_slot") or "")
        == signal_slot
        and row_count >= 1_000
        and fresh_count >= 1_000
        and open_coverage
        >= config.history.minimum_minute_universe_coverage
        and tail_coverage
        >= config.history.minimum_minute_universe_coverage
    )


def _localized_timestamp(value: Any, timezone: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone)
    return timestamp.tz_convert(timezone)


def _write_not_ready(
    output: Path,
    config: V3Config,
    current: datetime,
    error: str,
) -> dict[str, Any]:
    phase = session_phase(current, config)
    registry = load_registry(output / "json" / "wp_model_registry_v3.json")
    model_record = _resolve_model_record(registry)
    model_path = _resolve_model_path(registry)
    model_ready = bool(model_record and model_path.exists())
    state, health_status, message = _missing_input_state(
        phase=phase,
        model_status=(
            _live_state_from_registry(str(model_record.get("status") or ""))
            if model_ready
            else None
        ),
        error=error,
    )
    manifest = {
        "schema_version": "wp_manifest_v3",
        "latest_update": current.strftime("%Y-%m-%d %H:%M:%S"),
        "report_revision": current.strftime("%Y-%m-%d %H:%M:%S"),
        "source_trade_date": current.strftime("%Y%m%d"),
        "market_data_time": None,
        "session_phase": phase,
        "buy_plan_count": 0,
        "shadow_qualified_count": 0,
        "live_display_allowed": False,
        "health_status": health_status,
        "v3_state": state,
        "v3_model_version": (
            model_record.get("model_version") if model_ready else None
        ),
        "v3_model_fingerprint": (
            model_record.get("fingerprint") if model_ready else None
        ),
        "v3_policy_fingerprint": (
            model_record.get("policy_fingerprint") if model_ready else None
        ),
        "v3_formal_authorization": state == "PRODUCTION",
        "v3_message": message,
    }
    ledger = load_shadow_ledger(output / "json" / "wp_v3_candidate_ledger.json")
    replay = _read_json(output / "json" / "wp_v3_historical_replay.json")
    legacy_audit = _read_json(
        output / "json" / "wp_legacy_history_audit.json"
    )
    retrospective = _read_json(
        output / "json" / "wp_v41_backtest_202605_202607.json"
    )
    research_seed = _read_json(
        ROOT / "config" / "wp_v15_frozen_shadow_candidate.json"
    )
    write_json(output / "json" / "wp_manifest.json", manifest)
    render_v3_dashboard(
        output / "html_reports" / "latest.html",
        manifest=manifest,
        predictions=pd.DataFrame(),
        ledger=ledger,
        registry=registry,
        config=config,
        replay=replay,
        legacy_audit=legacy_audit,
        retrospective=retrospective,
        research_seed=research_seed,
    )
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_entry_settlement() -> tuple[pd.DataFrame, dict[str, Any]]:
    csv_value = os.environ.get("WP_V3_ENTRY_SETTLEMENT_CSV", "").strip()
    manifest_value = os.environ.get(
        "WP_V3_ENTRY_SETTLEMENT_MANIFEST",
        "",
    ).strip()
    if not csv_value and not manifest_value:
        return pd.DataFrame(), {}
    csv_path = Path(csv_value)
    manifest_path = (
        Path(manifest_value)
        if manifest_value
        else csv_path.with_name("wp_v3_entry_settlement_manifest.json")
    )
    if not csv_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("entry settlement payload is incomplete")
    try:
        frame = pd.read_csv(
            csv_path,
            keep_default_na=False,
            dtype={"ts_code": str},
        )
    except pd.errors.EmptyDataError:
        frame = pd.DataFrame()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return frame, manifest


def _resolve_model_record(registry: dict[str, Any]) -> dict[str, Any]:
    fingerprint = (
        registry.get("active_model_fingerprint")
        or registry.get("shadow_model_fingerprint")
    )
    return next(
        (
            model
            for model in registry.get("models", [])
            if model.get("fingerprint") == fingerprint
        ),
        {},
    )


def _resolve_model_path(registry: dict[str, Any]) -> Path:
    record = _resolve_model_record(registry)
    if not record or not record.get("artifact_path"):
        return ROOT / "artifacts" / "wp_v3_research" / "model_not_ready.joblib"
    return ROOT / str(record["artifact_path"])


def _live_state_from_registry(status: str) -> str:
    return {
        "PROMOTED": "PRODUCTION",
        "SHADOW": "SHADOW",
        "SHADOW_OBSERVATION": "SHADOW_OBSERVATION",
    }.get(status, "MODEL_NOT_DESIGNATED")


def _missing_input_state(
    *,
    phase: str,
    model_status: str | None,
    error: str,
) -> tuple[str, str, str]:
    if model_status is None:
        return (
            "MODEL_NOT_READY",
            "model_not_ready",
            "V41 可部署模型尚未发布，合格候选保持关闭；不会用旧模型冒充。",
        )
    if phase == "PRE_SIGNAL":
        return (
            model_status,
            "ok",
            "模型已就绪，等待 14:00 固定时点生成当日双队列决策。",
        )
    if phase == "CLOSED":
        return (
            model_status,
            "ok",
            "交易窗口已关闭；没有盘后补造候选，历史台账继续等待真值。",
        )
    return model_status, "v3_input_not_ready", error
