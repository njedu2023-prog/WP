from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import (
    CN_TZ,
    DEFAULT_SIGNAL_SLOTS,
    V3Config,
    entry_benchmark_slot,
)
from .features import FEATURE_COLUMNS
from .io import atomic_write_json


def empty_shadow_ledger() -> dict[str, Any]:
    return {
        "schema_version": "wp_candidate_ledger_v3",
        "generated_at": None,
        "sessions": [],
    }


def load_shadow_ledger(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return empty_shadow_ledger()
    ledger = json.loads(target.read_text(encoding="utf-8"))
    if ledger.get("schema_version") != "wp_candidate_ledger_v3":
        raise ValueError("unsupported candidate ledger schema")
    for session in ledger.get("sessions", []):
        if session.get("frozen"):
            expected = session.setdefault(
                "expected_slots",
                list(DEFAULT_SIGNAL_SLOTS),
            )
            missing = session.setdefault(
                "missing_slots",
                [
                    slot
                    for slot in expected
                    if slot not in session.get("covered_slots", [])
                ],
            )
            session.setdefault(
                "integrity_status",
                "COMPLETE" if not missing else "INCOMPLETE",
            )
        else:
            session.setdefault("integrity_status", "COLLECTING")
    return ledger


def save_shadow_ledger(ledger: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ledger["generated_at"] = datetime.now(CN_TZ).isoformat()
    atomic_write_json(target, ledger)


def record_shadow_slot(
    ledger: dict[str, Any],
    predictions: pd.DataFrame,
    *,
    trade_date: str,
    signal_slot: str,
    config: V3Config,
    observed_at: str | None = None,
) -> dict[str, Any]:
    if signal_slot not in config.strategy.signal_slots:
        raise ValueError(f"slot {signal_slot} is outside the immutable signal window")
    session = _session(ledger, trade_date)
    if session.get("frozen"):
        raise ValueError(f"candidate ledger for {trade_date} is already frozen")

    observed_at = observed_at or datetime.now(CN_TZ).isoformat()
    _bind_session_model(session, predictions)
    slots = session.setdefault("covered_slots", [])
    if signal_slot in slots:
        return ledger
    if signal_slot not in slots:
        slots.append(signal_slot)
        slots.sort(key=config.strategy.signal_slots.index)

    passed = predictions.loc[predictions["passes_policy"].fillna(False)].copy()
    candidates = session.setdefault("candidates", [])
    by_code = {str(candidate["ts_code"]): candidate for candidate in candidates}
    for row in passed.to_dict(orient="records"):
        code = str(row["ts_code"])
        existing = by_code.get(code)
        if existing is None:
            candidate = {
                "trade_date": trade_date,
                "target_trade_date": row.get("target_trade_date"),
                "ts_code": code,
                "name": row.get("name", ""),
                "status": (
                    row.get("candidate_state")
                    if row.get("candidate_state") in {"QUALIFIED", "SHADOW_QUALIFIED"}
                    else "SHADOW_QUALIFIED"
                ),
                "first_signal_time": signal_slot,
                "first_signal_price": _float(row.get("signal_price")),
                "entry_adj_factor": _float(row.get("adj_factor")),
                "entry_adj_factor_observed": _float(row.get("adj_factor")),
                "entry_benchmark_slot": entry_benchmark_slot(signal_slot, config),
                "entry_benchmark_status": "PENDING",
                "entry_order_notional": config.execution.reference_order_notional,
                "entry_price": None,
                "entry_fillable": None,
                "last_signal_time": signal_slot,
                "last_signal_price": _float(row.get("signal_price")),
                "appearance_count": 1,
                "p_net_positive": _float(row.get("p_net_positive")),
                "p_net_positive_lower": _float(row.get("p_net_positive_lower")),
                "expected_net_return_pct": _float(row.get("expected_net_return_pct")),
                "downside_q10_pct": _float(row.get("downside_q10_pct")),
                "ranking_score": _float(row.get("ranking_score")),
                "selection_score": _float(row.get("selection_score")),
                "selection_rank_pct": _float(row.get("selection_rank_pct")),
                "selection_rank_spread": _float(
                    row.get("selection_rank_spread")
                ),
                "model_version": row.get("model_version"),
                "model_fingerprint": row.get("model_fingerprint"),
                "policy_fingerprint": row.get("policy_fingerprint"),
                "feature_version": config.model.feature_version,
                "entry_contract": config.execution.entry_price_contract,
                "exit_contract": config.strategy.exit_contract,
                "exit_order_contract": config.execution.exit_order_contract,
                "entry_slippage_bps": config.execution.entry_slippage_bps,
                "round_trip_cost_bps": config.execution.round_trip_cost_bps,
                "baseline_all_in_cost_bps": (
                    config.execution.baseline_all_in_cost_bps
                ),
                "first_signal_market_data_time": row.get("market_data_time"),
                "first_signal_bar_time": row.get("slot_bar_time"),
                "first_signal_features": {
                    feature: _float(row.get(feature))
                    for feature in FEATURE_COLUMNS
                },
                "qualification_evidence": {
                    field: _json_scalar(row.get(field))
                    for field in (
                        "p_net_positive",
                        "p_net_positive_lower",
                        "probability_model_spread",
                        "expected_net_return_pct",
                        "downside_q10_pct",
                        "ranking_score",
                        "selection_score",
                        "selection_rank_pct",
                        "selection_rank_spread",
                        "selection_evidence_candidate_events",
                        "selection_evidence_candidate_days",
                        "selection_evidence_win_rate",
                        "selection_evidence_mean_net_return_pct",
                        "calibration_bin_count",
                        "calibration_bin_days",
                        "calibration_bin_win_rate",
                        "calibration_bin_wilson_lower",
                        "calibration_bin_clustered_lower",
                        "data_age_seconds",
                        "execution_eligible",
                        "passes_probability",
                        "passes_probability_lower",
                        "passes_expected_return",
                        "passes_downside",
                        "passes_selection_rank",
                        "passes_sample",
                        "passes_empirical_lower",
                        "passes_stability",
                        "passes_freshness",
                    )
                },
                "observed_at": observed_at,
                "truth_status": "pending",
            }
            candidates.append(candidate)
            by_code[code] = candidate
        else:
            _assert_immutable(existing, row, signal_slot)
            existing["last_signal_time"] = signal_slot
            existing["last_signal_price"] = _float(row.get("signal_price"))
            existing["appearance_count"] = int(existing.get("appearance_count", 1)) + 1
            existing["last_observed_at"] = observed_at
    session["status"] = "COLLECTING"
    session["integrity_status"] = "COLLECTING"
    session["candidate_count"] = len(candidates)
    return ledger


def settle_entry_benchmarks(
    ledger: dict[str, Any],
    settlement: pd.DataFrame,
    *,
    trade_date: str,
    settlement_slot: str,
    config: V3Config,
    settled_at: str | None = None,
) -> dict[str, Any]:
    session = next(
        (
            item
            for item in ledger.get("sessions", [])
            if str(item.get("trade_date")) == str(trade_date)
        ),
        None,
    )
    if session is None:
        return ledger
    due = [
        candidate
        for candidate in session.get("candidates", [])
        if _requires_entry_benchmark(candidate, config)
        and str(candidate.get("entry_benchmark_slot") or "")
        == str(settlement_slot)
    ]
    if not due:
        return ledger

    frame = settlement.copy()
    if not frame.empty:
        if "ts_code" not in frame:
            raise ValueError("entry settlement frame is missing ts_code")
        frame["ts_code"] = frame["ts_code"].astype(str)
        if frame["ts_code"].duplicated().any():
            raise ValueError("entry settlement frame contains duplicate stocks")
        frame = frame.set_index("ts_code", drop=False)
    settled_at = settled_at or datetime.now(CN_TZ).isoformat()
    for candidate in due:
        code = str(candidate.get("ts_code"))
        existing_status = str(
            candidate.get("entry_benchmark_status") or "PENDING"
        )
        if existing_status != "PENDING":
            _assert_settlement_immutable(
                candidate,
                frame.loc[code] if not frame.empty and code in frame.index else None,
                config=config,
            )
            continue
        row = frame.loc[code] if not frame.empty and code in frame.index else None
        if row is None:
            candidate.update(
                {
                    "entry_benchmark_status": "NON_FILL",
                    "entry_fillable": False,
                    "entry_non_fill_reason": "missing_exact_benchmark_bar",
                    "entry_benchmark_settled_at": settled_at,
                }
            )
            continue
        benchmark_price = _float(
            row.get("entry_benchmark_price", row.get("slot_close"))
        )
        benchmark_amount = _float(
            row.get("entry_benchmark_amount", row.get("slot_amount"))
        )
        benchmark_time = row.get(
            "entry_benchmark_bar_time",
            row.get("slot_bar_time"),
        )
        bar_slot = _bar_slot(
            row.get("entry_benchmark_slot") or benchmark_time
        )
        up_limit = _float(row.get("up_limit"))
        distance_to_up_limit = _float(
            row.get("entry_benchmark_distance_to_up_limit_pct")
        )
        if distance_to_up_limit is None and benchmark_price and up_limit:
            distance_to_up_limit = (up_limit / benchmark_price - 1.0) * 100.0
        data_age = _float(row.get("data_age_seconds"))
        reasons: list[str] = []
        if benchmark_price is None or benchmark_price <= 0:
            reasons.append("invalid_benchmark_price")
        if bar_slot != settlement_slot:
            reasons.append("wrong_benchmark_bar")
        if benchmark_amount is None or benchmark_amount < config.execution.min_slot_amount:
            reasons.append("insufficient_slot_amount")
        if (
            benchmark_amount is None
            or benchmark_amount * config.execution.max_entry_pct_of_slot_amount
            < config.execution.reference_order_notional
        ):
            reasons.append("insufficient_order_capacity")
        if (
            distance_to_up_limit is None
            or distance_to_up_limit
            < config.execution.min_distance_to_up_limit_pct
        ):
            reasons.append("too_close_to_up_limit")
        if (
            data_age is not None
            and not -60 <= data_age <= config.execution.max_market_data_age_seconds
        ):
            reasons.append("stale_benchmark_bar")
        fillable = not reasons
        entry_price = (
            benchmark_price
            * (1.0 + config.execution.entry_slippage_bps / 10_000.0)
            if benchmark_price
            else None
        )
        candidate.update(
            {
                "entry_benchmark_status": (
                    "SETTLED" if fillable else "NON_FILL"
                ),
                "entry_benchmark_price": benchmark_price,
                "entry_benchmark_amount": benchmark_amount,
                "entry_benchmark_bar_time": (
                    str(benchmark_time) if benchmark_time is not None else None
                ),
                "entry_benchmark_data_age_seconds": data_age,
                "entry_benchmark_distance_to_up_limit_pct": (
                    distance_to_up_limit
                ),
                "entry_price": entry_price,
                "entry_fillable": fillable,
                "entry_non_fill_reason": "|".join(reasons) if reasons else None,
                "entry_benchmark_settled_at": settled_at,
            }
        )
    return ledger


def freeze_shadow_session(
    ledger: dict[str, Any],
    *,
    trade_date: str,
    config: V3Config,
    frozen_at: str | None = None,
    model_fingerprint: str | None = None,
    policy_fingerprint: str | None = None,
) -> dict[str, Any]:
    session = _session(ledger, trade_date)
    if session.get("frozen"):
        existing_fingerprint = session.get("model_fingerprint")
        if (
            model_fingerprint
            and existing_fingerprint
            and existing_fingerprint != model_fingerprint
        ):
            raise ValueError(
                f"cannot change frozen session model: "
                f"{existing_fingerprint} -> {model_fingerprint}"
            )
        if model_fingerprint and not existing_fingerprint:
            session["model_fingerprint"] = model_fingerprint
        existing_policy = session.get("policy_fingerprint")
        if policy_fingerprint and existing_policy and existing_policy != policy_fingerprint:
            raise ValueError(
                f"cannot change frozen session policy: "
                f"{existing_policy} -> {policy_fingerprint}"
            )
        if policy_fingerprint and not existing_policy:
            session["policy_fingerprint"] = policy_fingerprint
        return ledger
    if model_fingerprint:
        session["model_fingerprint"] = model_fingerprint
    if policy_fingerprint:
        session["policy_fingerprint"] = policy_fingerprint
    session["frozen"] = True
    session["frozen_at"] = frozen_at or datetime.now(CN_TZ).isoformat()
    session["status"] = "FROZEN" if session.get("candidates") else "NO_SIGNAL"
    session["candidate_count"] = len(session.get("candidates", []))
    session["expected_slots"] = list(config.strategy.signal_slots)
    session["missing_slots"] = [
        slot for slot in config.strategy.signal_slots if slot not in session.get("covered_slots", [])
    ]
    pending_entries = _pending_entry_candidates(session, config)
    session["pending_entry_benchmark_count"] = len(pending_entries)
    session["integrity_status"] = (
        "COMPLETE"
        if not session["missing_slots"] and not pending_entries
        else (
            "INCOMPLETE_ENTRY"
            if not session["missing_slots"]
            else "INCOMPLETE"
        )
    )
    return ledger


def assert_ledger_invariants(ledger: dict[str, Any], config: V3Config) -> None:
    for session in ledger.get("sessions", []):
        if session.get("frozen"):
            expected = list(config.strategy.signal_slots)
            missing = [
                slot
                for slot in expected
                if slot not in session.get("covered_slots", [])
            ]
            if session.get("expected_slots") != expected:
                raise ValueError(
                    f"frozen session {session.get('trade_date')} has wrong expected slots"
                )
            if session.get("missing_slots") != missing:
                raise ValueError(
                    f"frozen session {session.get('trade_date')} has inconsistent missing slots"
                )
            pending_entries = _pending_entry_candidates(session, config)
            expected_integrity = (
                "COMPLETE"
                if not missing and not pending_entries
                else ("INCOMPLETE_ENTRY" if not missing else "INCOMPLETE")
            )
            if session.get("integrity_status") != expected_integrity:
                raise ValueError(
                    f"frozen session {session.get('trade_date')} has inconsistent integrity"
                )
        seen: set[str] = set()
        for candidate in session.get("candidates", []):
            code = str(candidate.get("ts_code"))
            if code in seen:
                raise ValueError(f"duplicate shadow candidate {session.get('trade_date')} {code}")
            seen.add(code)
            if candidate.get("first_signal_time") not in config.strategy.signal_slots:
                raise ValueError(f"invalid first signal time for {code}")
            if _float(candidate.get("first_signal_price")) is None:
                raise ValueError(f"missing immutable first signal price for {code}")
            if candidate.get("exit_contract") != "T+1_close":
                raise ValueError(f"invalid exit contract for {code}")
            if _requires_entry_benchmark(candidate, config):
                expected_benchmark = entry_benchmark_slot(
                    str(candidate.get("first_signal_time")),
                    config,
                )
                if candidate.get("entry_benchmark_slot") != expected_benchmark:
                    raise ValueError(f"invalid entry benchmark slot for {code}")
                benchmark_status = str(
                    candidate.get("entry_benchmark_status") or "PENDING"
                )
                if benchmark_status not in {"PENDING", "SETTLED", "NON_FILL"}:
                    raise ValueError(f"invalid entry benchmark status for {code}")
                if benchmark_status == "SETTLED":
                    if _float(candidate.get("entry_price")) is None:
                        raise ValueError(f"missing immutable entry price for {code}")
                    if candidate.get("entry_fillable") is not True:
                        raise ValueError(f"settled entry is not fillable for {code}")
                if benchmark_status == "NON_FILL" and candidate.get(
                    "entry_fillable"
                ) is not False:
                    raise ValueError(f"non-fill entry marked fillable for {code}")
            if (
                session.get("policy_fingerprint")
                and candidate.get("policy_fingerprint")
                and session.get("policy_fingerprint")
                != candidate.get("policy_fingerprint")
            ):
                raise ValueError(f"candidate policy changed within session for {code}")


def _session(ledger: dict[str, Any], trade_date: str) -> dict[str, Any]:
    for session in ledger.setdefault("sessions", []):
        if str(session.get("trade_date")) == str(trade_date):
            return session
    session = {
        "trade_date": str(trade_date),
        "status": "COLLECTING",
        "frozen": False,
        "covered_slots": [],
        "candidate_count": 0,
        "candidates": [],
        "integrity_status": "COLLECTING",
    }
    ledger["sessions"].append(session)
    ledger["sessions"].sort(key=lambda item: str(item.get("trade_date")))
    return session


def _assert_immutable(existing: dict[str, Any], row: dict[str, Any], slot: str) -> None:
    if existing.get("model_fingerprint") != row.get("model_fingerprint"):
        raise ValueError(
            f"model changed within session for {row.get('ts_code')}: "
            f"{existing.get('model_fingerprint')} -> {row.get('model_fingerprint')}"
        )
    if slot < str(existing.get("first_signal_time")):
        raise ValueError(f"cannot insert an earlier signal after first crossing for {row.get('ts_code')}")


def _pending_entry_candidates(
    session: dict[str, Any],
    config: V3Config,
) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in session.get("candidates", [])
        if _requires_entry_benchmark(candidate, config)
        and str(candidate.get("entry_benchmark_status") or "PENDING")
        == "PENDING"
    ]


def _requires_entry_benchmark(
    candidate: dict[str, Any],
    config: V3Config,
) -> bool:
    return (
        candidate.get("entry_contract")
        == config.execution.entry_price_contract
    )


def _bar_slot(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    try:
        return pd.Timestamp(text).strftime("%H:%M")
    except (TypeError, ValueError):
        return text[:5]


def _assert_settlement_immutable(
    candidate: dict[str, Any],
    row: pd.Series | None,
    *,
    config: V3Config,
) -> None:
    if row is None:
        return
    incoming = _float(row.get("entry_benchmark_price", row.get("slot_close")))
    existing = _float(candidate.get("entry_benchmark_price"))
    if (
        incoming is not None
        and existing is not None
        and abs(incoming - existing) > 1e-9
    ):
        raise ValueError(
            f"cannot rewrite immutable entry benchmark for "
            f"{candidate.get('ts_code')}: {existing} -> {incoming}"
        )


def _bind_session_model(session: dict[str, Any], predictions: pd.DataFrame) -> None:
    fingerprints = {
        str(value)
        for value in predictions.get(
            "model_fingerprint",
            pd.Series(dtype=str),
        ).dropna()
        if str(value)
    }
    policies = {
        str(value)
        for value in predictions.get(
            "policy_fingerprint",
            pd.Series(dtype=str),
        ).dropna()
        if str(value)
    }
    if len(fingerprints) > 1 or len(policies) > 1:
        raise ValueError("one tail session cannot mix model or policy fingerprints")
    fingerprint = next(iter(fingerprints), None)
    policy = next(iter(policies), None)
    if (
        fingerprint
        and session.get("model_fingerprint")
        and session["model_fingerprint"] != fingerprint
    ):
        raise ValueError("model changed within the same tail session")
    if (
        policy
        and session.get("policy_fingerprint")
        and session["policy_fingerprint"] != policy
    ):
        raise ValueError("policy changed within the same tail session")
    if fingerprint:
        session["model_fingerprint"] = fingerprint
    if policy:
        session["policy_fingerprint"] = policy


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if pd.notna(parsed) else None
    except (TypeError, ValueError):
        return None


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
