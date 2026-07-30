from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import V3Config
from .forward_risk import (
    DISCOVERY_END_DATE,
    DISCOVERY_FOLD,
    FORWARD_VALIDATION_FOLDS,
    SAFE_RISK_RANK_MAX,
    frozen_meta_policy,
)


FROZEN_SCHEMA_VERSION = "wp_frozen_shadow_candidate_v1"
FROZEN_CANDIDATE_ID = "wp_v15_safe_half_forward_20260730"
FROZEN_STATUS = "FROZEN_RESEARCH_SHADOW_CANDIDATE"
FROZEN_SHADOW_STATUS = "NOT_STARTED_REQUIRES_DEPLOYABLE_BUNDLE"

EXPECTED_PROVENANCE = {
    "source_commit_sha": "b90fe6f124bd5cca6cdb4ca690ee07885c8df1cd",
    "v15_forward_validation_run_id": 30552732652,
    "v15_forward_validation_job_id": 90905243772,
    "artifact_id": 8763553503,
    "artifact_zip_sha256": (
        "33b01745d34b41417aac7bd4260a5805519e3ca21730bf3fce590205fb9bbc07"
    ),
    "v15_scored_frontier_sha256": (
        "f6f124ba8eff19df615ccd8c1d21917e14cd8f98bc1f53b30c9cb356b97007be"
    ),
}

EXPECTED_EVIDENCE = {
    "status": "positive_forward_direction_unconfirmed",
    "events": 55,
    "trade_days": 29,
    "wins": 31,
    "win_rate": 0.5636363636363636,
    "mean_net_return_pct": 0.45503570559970735,
    "mean_net_return_day_clustered_lower_pct": -0.14942755908786864,
    "profit_factor": 1.4203663398588322,
    "stress_50bps_mean_net_return_pct": 0.30503570559970716,
}


class FrozenCandidateError(ValueError):
    pass


def load_frozen_candidate(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_frozen_candidate(payload)
    return payload


def freeze_manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(manifest))
    integrity = payload.get("integrity")
    if not isinstance(integrity, dict):
        raise FrozenCandidateError("manifest integrity block is missing")
    integrity.pop("freeze_manifest_sha256", None)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def verify_frozen_candidate(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    _expect(manifest, "schema_version", FROZEN_SCHEMA_VERSION, errors)
    _expect(manifest, "candidate_id", FROZEN_CANDIDATE_ID, errors)
    _expect(manifest, "status", FROZEN_STATUS, errors)
    _expect(manifest, "research_only", True, errors)
    _expect(manifest, "production_model_changed", False, errors)
    _expect(manifest, "production_authorized", False, errors)
    _expect(manifest, "decision.confirmed_by_user", True, errors)

    _expect(
        manifest,
        "candidate_spec.entry_policy",
        frozen_meta_policy().as_dict(),
        errors,
    )
    _expect(
        manifest,
        "candidate_spec.exit_failure_risk_gate.maximum",
        SAFE_RISK_RANK_MAX,
        errors,
    )
    _expect(
        manifest,
        "candidate_spec.exit_failure_risk_gate.order",
        "risk_gate_before_daily_top_three_selection",
        errors,
    )
    _expect(
        manifest,
        "causality_contract.discovery_fold",
        DISCOVERY_FOLD,
        errors,
    )
    _expect(
        manifest,
        "causality_contract.discovery_end_trade_date",
        DISCOVERY_END_DATE,
        errors,
    )
    _expect(
        manifest,
        "causality_contract.forward_validation_folds",
        list(FORWARD_VALIDATION_FOLDS),
        errors,
    )
    _expect(
        manifest,
        "causality_contract.threshold_or_policy_retuning_on_forward_folds",
        False,
        errors,
    )
    _expect(
        manifest,
        "causality_contract.future_information_allowed",
        False,
        errors,
    )

    for key, expected in EXPECTED_PROVENANCE.items():
        path = {
            "source_commit_sha": "provenance.source_commit_sha",
            "v15_forward_validation_run_id": (
                "provenance.runs.v15_forward_validation_run_id"
            ),
            "v15_forward_validation_job_id": (
                "provenance.runs.v15_forward_validation_job_id"
            ),
            "artifact_id": "provenance.artifact.artifact_id",
            "artifact_zip_sha256": "provenance.artifact.zip_sha256",
            "v15_scored_frontier_sha256": (
                "provenance.source_digests.v15_scored_frontier_sha256"
            ),
        }[key]
        _expect(manifest, path, expected, errors)

    _expect(
        manifest,
        "evidence_snapshot.status",
        EXPECTED_EVIDENCE["status"],
        errors,
    )
    for key in (
        "events",
        "trade_days",
        "wins",
        "win_rate",
        "mean_net_return_pct",
        "mean_net_return_day_clustered_lower_pct",
        "profit_factor",
        "stress_50bps_mean_net_return_pct",
    ):
        _expect(
            manifest,
            f"evidence_snapshot.challenger.{key}",
            EXPECTED_EVIDENCE[key],
            errors,
        )
    _expect(
        manifest,
        "evidence_snapshot.gates.all_gates_passed",
        False,
        errors,
    )
    _expect(
        manifest,
        "evidence_snapshot.gates.minimum_250_events",
        False,
        errors,
    )
    _expect(
        manifest,
        "evidence_snapshot.gates.minimum_60_trade_days",
        False,
        errors,
    )
    _expect(
        manifest,
        "evidence_snapshot.gates.positive_day_clustered_mean_lower",
        False,
        errors,
    )
    _expect(
        manifest,
        "evidence_snapshot.gates.day_clustered_win_lower_at_least_52pct",
        False,
        errors,
    )

    _expect(
        manifest,
        "shadow_contract.status",
        FROZEN_SHADOW_STATUS,
        errors,
    )
    _expect(manifest, "shadow_contract.minimum_trading_days", 150, errors)
    _expect(manifest, "shadow_contract.minimum_candidate_days", 50, errors)
    _expect(
        manifest,
        "shadow_contract.minimum_verified_candidates",
        250,
        errors,
    )
    _expect(
        manifest,
        "shadow_contract.clock_inheritance_allowed",
        False,
        errors,
    )
    _expect(manifest, "shadow_contract.started_trade_date", None, errors)
    _expect(
        manifest,
        "shadow_contract.production_promotion_allowed",
        False,
        errors,
    )

    stored_fingerprint = _read(
        manifest,
        "integrity.freeze_manifest_sha256",
    )
    actual_fingerprint = freeze_manifest_fingerprint(manifest)
    if stored_fingerprint != actual_fingerprint:
        errors.append(
            "integrity.freeze_manifest_sha256: "
            f"{stored_fingerprint!r} != {actual_fingerprint!r}"
        )
    if errors:
        raise FrozenCandidateError(
            "frozen V15 candidate verification failed:\n- "
            + "\n- ".join(errors)
        )
    return {
        "verified": True,
        "candidate_id": FROZEN_CANDIDATE_ID,
        "status": FROZEN_STATUS,
        "freeze_manifest_sha256": actual_fingerprint,
        "production_authorized": False,
        "shadow_status": FROZEN_SHADOW_STATUS,
    }


def verify_runtime_contract(
    manifest: Mapping[str, Any],
    config: V3Config,
) -> dict[str, Any]:
    verify_frozen_candidate(manifest)
    errors: list[str] = []
    expected_execution = {
        "entry_price_contract": config.execution.entry_price_contract,
        "entry_delay_minutes": config.execution.entry_delay_minutes,
        "entry_execution_deadline": (
            config.execution.entry_execution_deadline
        ),
        "entry_slippage_bps": config.execution.entry_slippage_bps,
        "round_trip_cost_bps": config.execution.round_trip_cost_bps,
        "baseline_all_in_cost_bps": (
            config.execution.baseline_all_in_cost_bps
        ),
        "stress_cost_bps": list(config.execution.stress_cost_bps),
        "exit_order_contract": config.execution.exit_order_contract,
        "non_fill_penalty_pct": config.execution.non_fill_penalty_pct,
    }
    for key, actual in expected_execution.items():
        expected = _read(manifest, f"execution_contract.{key}")
        if actual != expected:
            errors.append(
                f"runtime execution.{key}: {actual!r} != {expected!r}"
            )
    if config.strategy.exit_contract != _read(
        manifest,
        "execution_contract.exit_contract",
    ):
        errors.append("runtime strategy.exit_contract changed")
    if config.strategy.board_scope != _read(
        manifest,
        "execution_contract.board_scope",
    ):
        errors.append("runtime strategy.board_scope changed")
    if config.promotion.mode != "shadow":
        errors.append(
            f"runtime promotion.mode is {config.promotion.mode!r}, not 'shadow'"
        )

    minimums = {
        "minimum_shadow_trading_days": (
            "minimum_trading_days",
            config.promotion.minimum_shadow_trading_days,
        ),
        "minimum_shadow_candidate_days": (
            "minimum_candidate_days",
            config.promotion.minimum_shadow_candidate_days,
        ),
        "minimum_shadow_candidates": (
            "minimum_verified_candidates",
            config.promotion.minimum_shadow_candidates,
        ),
    }
    for runtime_name, (manifest_name, actual) in minimums.items():
        frozen_floor = int(
            _read(manifest, f"shadow_contract.{manifest_name}")
        )
        if actual < frozen_floor:
            errors.append(
                f"runtime promotion.{runtime_name} weakened: "
                f"{actual} < {frozen_floor}"
            )
    if errors:
        raise FrozenCandidateError(
            "runtime contract is incompatible with frozen V15 candidate:\n- "
            + "\n- ".join(errors)
        )
    return {
        "runtime_contract_verified": True,
        "promotion_mode": config.promotion.mode,
        "minimum_shadow_trading_days": (
            config.promotion.minimum_shadow_trading_days
        ),
        "minimum_shadow_candidate_days": (
            config.promotion.minimum_shadow_candidate_days
        ),
        "minimum_shadow_candidates": (
            config.promotion.minimum_shadow_candidates
        ),
    }


def _read(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _Missing(path)
        value = value[part]
    return value


def _expect(
    mapping: Mapping[str, Any],
    path: str,
    expected: Any,
    errors: list[str],
) -> None:
    actual = _read(mapping, path)
    if actual != expected:
        errors.append(f"{path}: {actual!r} != {expected!r}")


class _Missing:
    def __init__(self, path: str) -> None:
        self.path = path

    def __repr__(self) -> str:
        return f"<missing {self.path}>"
