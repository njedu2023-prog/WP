from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import V3Config
from .statistics import day_clustered_intervals, wilson_interval


@dataclass(frozen=True)
class PromotionDecision:
    eligible: bool
    checks: dict[str, bool]
    reason: str


def empty_registry() -> dict[str, Any]:
    return {
        "schema_version": "wp_model_registry_v3",
        "generated_at": None,
        "active_model_fingerprint": None,
        "active_policy_fingerprint": None,
        "shadow_model_fingerprint": None,
        "shadow_policy_fingerprint": None,
        "models": [],
    }


def load_registry(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return empty_registry()
    raw = json.loads(target.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "wp_model_registry_v3":
        raise ValueError("unsupported WP model registry schema")
    raw.setdefault("active_model_fingerprint", None)
    raw.setdefault("active_policy_fingerprint", None)
    raw.setdefault("shadow_model_fingerprint", None)
    raw.setdefault("shadow_policy_fingerprint", None)
    raw.setdefault("models", [])
    if not raw["shadow_model_fingerprint"]:
        raw["shadow_model_fingerprint"] = next(
            (
                model.get("fingerprint")
                for model in reversed(raw["models"])
                if model.get("status") in {"SHADOW", "SHADOW_OBSERVATION"}
            ),
            None,
        )
    for model in raw["models"]:
        model.setdefault("policy_fingerprint", model.get("fingerprint"))
    if not raw["shadow_policy_fingerprint"] and raw["shadow_model_fingerprint"]:
        record = model_record(raw, str(raw["shadow_model_fingerprint"]))
        raw["shadow_policy_fingerprint"] = (
            record.get("policy_fingerprint") if record else None
        )
    if not raw["active_policy_fingerprint"] and raw["active_model_fingerprint"]:
        record = model_record(raw, str(raw["active_model_fingerprint"]))
        raw["active_policy_fingerprint"] = (
            record.get("policy_fingerprint") if record else None
        )
    return raw


def save_registry(registry: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    registry["generated_at"] = datetime.now(timezone.utc).isoformat()
    target.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def register_research_model(
    registry: dict[str, Any],
    *,
    metadata: dict[str, Any],
    backtest: dict[str, Any],
    artifact_path: str,
) -> dict[str, Any]:
    fingerprint = str(metadata["fingerprint"])
    policy = str(metadata.get("policy_fingerprint") or fingerprint)
    existing = model_record(registry, fingerprint)
    backtest_passed = bool(
        backtest.get("backtest_gate", {}).get("passed", False)
    )
    current_shadow = model_record(
        registry,
        str(registry.get("shadow_model_fingerprint") or ""),
    )
    current_shadow_passed = bool(
        current_shadow
        and current_shadow.get("backtest", {})
        .get("backtest_gate", {})
        .get("passed", False)
    )

    if registry.get("active_policy_fingerprint") == policy and backtest_passed:
        status = "PROMOTED"
        previous = registry.get("active_model_fingerprint")
        _supersede(registry, previous, "SUPERSEDED_PRODUCTION")
        registry["active_model_fingerprint"] = fingerprint
    elif registry.get("active_policy_fingerprint") == policy:
        status = "RESEARCH"
    elif backtest_passed:
        status = "SHADOW"
        previous = registry.get("shadow_model_fingerprint")
        _supersede(registry, previous, "SUPERSEDED_SHADOW")
        registry["shadow_policy_fingerprint"] = policy
        registry["shadow_model_fingerprint"] = fingerprint
    elif current_shadow_passed:
        # A rejected challenger may be retained for research, but it must not
        # displace a shadow policy that already passed the historical gate.
        status = "RESEARCH"
    else:
        # Observation is not approval. It lets the latest frozen policy collect
        # genuinely future sessions while every promotion gate remains closed.
        status = "SHADOW_OBSERVATION"
        previous = registry.get("shadow_model_fingerprint")
        _supersede(registry, previous, "SUPERSEDED_OBSERVATION")
        registry["shadow_policy_fingerprint"] = policy
        registry["shadow_model_fingerprint"] = fingerprint

    record = {
        "model_version": metadata["model_version"],
        "fingerprint": fingerprint,
        "policy_fingerprint": policy,
        "training_data_digest": metadata.get("training_data_digest"),
        "feature_version": metadata["feature_version"],
        "trained_at": metadata["trained_at"],
        "train_start": metadata["train_start"],
        "train_end": metadata["train_end"],
        "calibration_start": metadata.get("calibration_start"),
        "calibration_end": metadata.get("calibration_end"),
        "calibration_fit_end": metadata.get("calibration_fit_end"),
        "evidence_start": metadata.get("evidence_start"),
        "selection_evidence": metadata.get("selection_evidence", {}),
        "artifact_path": artifact_path,
        "status": status,
        "backtest": backtest,
        "shadow": _empty_shadow(),
        "promotion": {
            "eligible": False,
            "checks": {},
            "reason": "shadow_not_started",
        },
    }
    if existing is not None:
        record["shadow"] = existing.get("shadow", record["shadow"])
        record["promotion"] = existing.get("promotion", record["promotion"])
        if (
            existing.get("status") == "PROMOTED"
            and registry.get("active_model_fingerprint") == fingerprint
            and backtest_passed
        ):
            record["status"] = existing["status"]
    same_policy = [
        model
        for model in registry.get("models", [])
        if model.get("policy_fingerprint") == policy
    ]
    if same_policy and not existing:
        latest_shadow = max(
            same_policy,
            key=lambda item: str(item.get("trained_at") or ""),
        ).get("shadow")
        if latest_shadow:
            record["shadow"] = latest_shadow

    registry["models"] = [
        model
        for model in registry.get("models", [])
        if model.get("fingerprint") != fingerprint
    ] + [record]
    return registry


def model_record(registry: dict[str, Any], fingerprint: str) -> dict[str, Any] | None:
    return next(
        (
            model
            for model in registry.get("models", [])
            if model.get("fingerprint") == fingerprint
        ),
        None,
    )


def evaluate_promotion(record: dict[str, Any], config: V3Config) -> PromotionDecision:
    backtest = record.get("backtest", {})
    shadow = record.get("shadow", {})
    promotion = config.promotion
    checks = {
        "backtest_gate": bool(backtest.get("backtest_gate", {}).get("passed", False)),
        "shadow_trading_days": int(shadow.get("trading_days", 0))
        >= promotion.minimum_shadow_trading_days,
        "shadow_candidate_days": int(shadow.get("candidate_days", 0))
        >= promotion.minimum_shadow_candidate_days,
        "shadow_candidates": int(shadow.get("verified_candidates", 0))
        >= promotion.minimum_shadow_candidates,
        "shadow_win_rate": _number(shadow.get("win_rate"), 0.0)
        >= promotion.minimum_oos_win_rate,
        "shadow_win_rate_lower": _number(
            shadow.get("win_rate_wilson_lower"),
            0.0,
        )
        >= promotion.minimum_oos_win_rate_lower,
        "shadow_clustered_win_rate_lower": _number(
            shadow.get("win_rate_day_clustered_lower"),
            0.0,
        )
        >= promotion.minimum_clustered_win_rate_lower,
        "shadow_mean_net_return": _number(
            shadow.get("mean_net_return_pct"),
            -999.0,
        )
        >= promotion.minimum_mean_net_return_pct,
        "shadow_clustered_mean_return_lower": _number(
            shadow.get("mean_net_return_day_clustered_lower_pct"),
            -999.0,
        )
        >= promotion.minimum_clustered_mean_return_lower_pct,
        "shadow_median_net_return": _number(
            shadow.get("median_net_return_pct"),
            -999.0,
        )
        >= promotion.minimum_median_net_return_pct,
        "shadow_profit_factor": _number(shadow.get("profit_factor"), 0.0)
        >= promotion.minimum_profit_factor,
        "shadow_ece": _number(shadow.get("ece"), 999.0)
        <= promotion.maximum_ece,
        "shadow_50bps_stress": (
            not promotion.require_50bps_stress_nonnegative
            or bool(shadow.get("stress_50bps_positive_total_return", False))
        ),
    }
    eligible = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    reason = "all_promotion_gates_passed" if eligible else "failed:" + ",".join(failed)
    return PromotionDecision(eligible=eligible, checks=checks, reason=reason)


def apply_promotion_decision(
    registry: dict[str, Any],
    fingerprint: str,
    config: V3Config,
    *,
    authorize: bool = False,
) -> PromotionDecision:
    record = model_record(registry, fingerprint)
    if record is None:
        raise KeyError(f"model fingerprint not found: {fingerprint}")
    decision = evaluate_promotion(record, config)
    previous = record.get("promotion", {})
    payload = {
        "eligible": decision.eligible,
        "checks": decision.checks,
        "reason": decision.reason,
    }
    if any(previous.get(key) != value for key, value in payload.items()):
        payload["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    elif previous.get("evaluated_at"):
        payload["evaluated_at"] = previous["evaluated_at"]
    record["promotion"] = payload
    if decision.eligible and authorize:
        old_active = registry.get("active_model_fingerprint")
        if old_active and old_active != fingerprint:
            _supersede(registry, old_active, "SUPERSEDED_PRODUCTION")
        record["status"] = "PROMOTED"
        registry["active_model_fingerprint"] = fingerprint
        registry["active_policy_fingerprint"] = record.get("policy_fingerprint")
        if registry.get("shadow_policy_fingerprint") == record.get(
            "policy_fingerprint"
        ):
            registry["shadow_model_fingerprint"] = None
            registry["shadow_policy_fingerprint"] = None
    return decision


def is_model_promoted(registry: dict[str, Any], fingerprint: str) -> bool:
    record = model_record(registry, fingerprint)
    return bool(
        record
        and record.get("status") == "PROMOTED"
        and registry.get("active_model_fingerprint") == fingerprint
        and registry.get("active_policy_fingerprint")
        == record.get("policy_fingerprint")
    )


def refresh_shadow_metrics(
    registry: dict[str, Any],
    fingerprint: str,
    ledger: dict[str, Any],
    config: V3Config,
) -> PromotionDecision:
    requested = model_record(registry, fingerprint)
    if requested is None:
        raise KeyError(f"model fingerprint not found: {fingerprint}")
    policy = str(requested.get("policy_fingerprint") or fingerprint)
    policy_models = {
        str(model.get("fingerprint"))
        for model in registry.get("models", [])
        if str(model.get("policy_fingerprint") or model.get("fingerprint")) == policy
    }
    sessions = [
        session
        for session in ledger.get("sessions", [])
        if session.get("frozen")
        and not session.get("missing_slots")
        and _session_belongs_to_policy(session, policy, policy_models)
    ]
    candidates = [
        candidate
        for session in sessions
        for candidate in session.get("candidates", [])
        if _candidate_belongs_to_policy(candidate, policy, policy_models)
    ]
    verified = [
        candidate
        for candidate in candidates
        if candidate.get("truth_status") == "verified"
        and candidate.get("net_return_pct") is not None
    ]
    returns = np.asarray(
        [float(candidate["net_return_pct"]) for candidate in verified],
        dtype=float,
    )
    evidence = pd.DataFrame(
        {
            "trade_date": [
                str(candidate.get("trade_date") or "") for candidate in verified
            ],
            "net_return_pct": returns,
        }
    )
    clustered = day_clustered_intervals(evidence)
    probabilities = np.asarray(
        [
            float(candidate["p_net_positive"])
            for candidate in verified
            if candidate.get("p_net_positive") is not None
        ],
        dtype=float,
    )
    probability_targets = np.asarray(
        [
            int(float(candidate["net_return_pct"]) > 0)
            for candidate in verified
            if candidate.get("p_net_positive") is not None
        ],
        dtype=int,
    )
    wins = int(np.sum(returns > 0))
    lower, upper = wilson_interval(wins, len(returns))
    profits = float(returns[returns > 0].sum()) if len(returns) else 0.0
    losses = float(-returns[returns < 0].sum()) if len(returns) else 0.0
    stress_50 = returns - (
        50.0 - config.execution.baseline_all_in_cost_bps
    ) / 100.0
    shadow = {
        "policy_fingerprint": policy,
        "started_trade_date": min(
            (str(session.get("trade_date")) for session in sessions),
            default=None,
        ),
        "trading_days": len(
            {str(session.get("trade_date")) for session in sessions}
        ),
        "candidate_days": len(
            {
                str(candidate.get("trade_date"))
                for candidate in candidates
                if candidate.get("trade_date")
            }
        ),
        "candidates": len(candidates),
        "verified_candidates": len(returns),
        "wins": wins,
        "win_rate": float(np.mean(returns > 0)) if len(returns) else None,
        "win_rate_wilson_lower": lower if len(returns) else None,
        "win_rate_wilson_upper": upper if len(returns) else None,
        "win_rate_day_clustered_lower": (
            clustered.win_rate_lower if len(returns) else None
        ),
        "win_rate_day_clustered_upper": (
            clustered.win_rate_upper if len(returns) else None
        ),
        "mean_net_return_pct": float(np.mean(returns)) if len(returns) else None,
        "mean_net_return_day_clustered_lower_pct": (
            clustered.mean_return_lower_pct if len(returns) else None
        ),
        "mean_net_return_day_clustered_upper_pct": (
            clustered.mean_return_upper_pct if len(returns) else None
        ),
        "median_net_return_pct": (
            float(np.median(returns)) if len(returns) else None
        ),
        "profit_factor": (
            profits / losses
            if losses > 0
            else (999.0 if profits > 0 else None)
        ),
        "ece": (
            _ece(probability_targets, probabilities)
            if len(probabilities)
            else None
        ),
        "stress_50bps_positive_total_return": (
            bool(stress_50.sum() > 0) if len(stress_50) else False
        ),
    }
    for record in registry.get("models", []):
        if str(record.get("policy_fingerprint") or record.get("fingerprint")) == policy:
            record["shadow"] = dict(shadow)

    target_fingerprint = _current_policy_model(
        registry,
        policy,
        fallback=fingerprint,
    )
    return apply_promotion_decision(
        registry,
        target_fingerprint,
        config,
        authorize=config.promotion.auto_promote_when_all_gates_pass,
    )


def _empty_shadow() -> dict[str, Any]:
    return {
        "policy_fingerprint": None,
        "started_trade_date": None,
        "trading_days": 0,
        "candidate_days": 0,
        "candidates": 0,
        "verified_candidates": 0,
        "wins": 0,
        "win_rate": None,
        "win_rate_wilson_lower": None,
        "win_rate_wilson_upper": None,
        "win_rate_day_clustered_lower": None,
        "win_rate_day_clustered_upper": None,
        "mean_net_return_pct": None,
        "mean_net_return_day_clustered_lower_pct": None,
        "mean_net_return_day_clustered_upper_pct": None,
        "median_net_return_pct": None,
        "profit_factor": None,
        "ece": None,
        "stress_50bps_positive_total_return": False,
    }


def _candidate_belongs_to_policy(
    candidate: dict[str, Any],
    policy: str,
    model_fingerprints: set[str],
) -> bool:
    candidate_policy = candidate.get("policy_fingerprint")
    if candidate_policy:
        return str(candidate_policy) == policy
    return str(candidate.get("model_fingerprint")) in model_fingerprints


def _session_belongs_to_policy(
    session: dict[str, Any],
    policy: str,
    model_fingerprints: set[str],
) -> bool:
    session_policy = session.get("policy_fingerprint")
    if session_policy:
        return str(session_policy) == policy
    if str(session.get("model_fingerprint")) in model_fingerprints:
        return True
    return any(
        _candidate_belongs_to_policy(candidate, policy, model_fingerprints)
        for candidate in session.get("candidates", [])
    )


def _current_policy_model(
    registry: dict[str, Any],
    policy: str,
    *,
    fallback: str,
) -> str:
    if registry.get("active_policy_fingerprint") == policy:
        return str(registry.get("active_model_fingerprint") or fallback)
    if registry.get("shadow_policy_fingerprint") == policy:
        return str(registry.get("shadow_model_fingerprint") or fallback)
    return fallback


def _supersede(
    registry: dict[str, Any],
    fingerprint: Any,
    status: str,
) -> None:
    if not fingerprint:
        return
    record = model_record(registry, str(fingerprint))
    if record is not None:
        record["status"] = status


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ece(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    total = 0.0
    for left, right in zip(
        np.linspace(0.0, 1.0, bins + 1)[:-1],
        np.linspace(0.0, 1.0, bins + 1)[1:],
        strict=False,
    ):
        mask = (probability >= left) & (probability <= right)
        if not mask.any():
            continue
        total += float(mask.mean()) * abs(
            float(target[mask].mean()) - float(probability[mask].mean())
        )
    return total
