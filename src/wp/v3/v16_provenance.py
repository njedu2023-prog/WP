from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import joblib

from .io import file_sha256
from .model import MODEL_SCHEMA_VERSION, ModelBundle, load_bundle


BASE_MODEL_CONTRACT_SCHEMA = "wp_v16_base_model_contract_1"


def bind_base_model(
    registry_path: str | Path,
    *,
    expected_fingerprint: str,
    repository_root: str | Path,
) -> tuple[bytes, ModelBundle, dict[str, Any]]:
    expected = str(expected_fingerprint).strip()
    if not expected:
        raise ValueError("expected base-model fingerprint is required")
    root = Path(repository_root).resolve()
    registry = _resolve_inside(root, registry_path)
    raw = json.loads(registry.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "wp_model_registry_v3":
        raise ValueError("unsupported WP base-model registry schema")
    matches = [
        record
        for record in raw.get("models", [])
        if str(record.get("fingerprint") or "") == expected
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one base-model registry record for "
            f"{expected}; found {len(matches)}"
        )
    record = matches[0]
    artifact_relative = str(record.get("artifact_path") or "").strip()
    if not artifact_relative:
        raise RuntimeError("base-model registry record has no artifact path")
    artifact = _resolve_inside(root, artifact_relative)
    if not artifact.is_file():
        raise FileNotFoundError(
            f"base-model artifact does not exist: {artifact_relative}"
        )
    artifact_bytes = artifact.read_bytes()
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    bundle = load_bundle(artifact)
    if bundle.fingerprint != expected:
        raise RuntimeError(
            "base-model artifact fingerprint mismatch: "
            f"{bundle.fingerprint} != {expected}"
        )
    record_policy = str(record.get("policy_fingerprint") or "")
    if record_policy and bundle.policy_fingerprint != record_policy:
        raise RuntimeError(
            "base-model policy fingerprint mismatch: "
            f"{bundle.policy_fingerprint} != {record_policy}"
        )
    contract = {
        "schema_version": BASE_MODEL_CONTRACT_SCHEMA,
        "binding": "embedded_exact_model_artifact",
        "model_fingerprint": bundle.fingerprint,
        "policy_fingerprint": bundle.policy_fingerprint,
        "model_version": bundle.model_version,
        "model_schema_version": bundle.schema_version,
        "registry_status": record.get("status"),
        "registry_generated_at": raw.get("generated_at"),
        "registry_path": _relative_to(root, registry),
        "registry_sha256": file_sha256(registry),
        "artifact_path": _relative_to(root, artifact),
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": len(artifact_bytes),
        "historical_oos_role": (
            "not_used; historical V16 scores consume immutable V9 "
            "walk-forward OOS predictions"
        ),
        "future_shadow_role": (
            "required first-stage model for generating live causal V9 scores"
        ),
    }
    return artifact_bytes, bundle, contract


def verify_embedded_base_model(
    payload: dict[str, Any],
) -> tuple[ModelBundle, dict[str, Any]]:
    contract = payload.get("base_model_contract")
    if not isinstance(contract, dict):
        raise TypeError("V16 bundle has no base-model contract")
    if contract.get("schema_version") != BASE_MODEL_CONTRACT_SCHEMA:
        raise ValueError("unsupported V16 base-model contract schema")
    artifact = payload.get("base_model_artifact")
    if not isinstance(artifact, bytes) or not artifact:
        raise TypeError("V16 bundle has no embedded base-model artifact")
    actual_sha256 = hashlib.sha256(artifact).hexdigest()
    expected_sha256 = str(contract.get("artifact_sha256") or "")
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "embedded base-model digest mismatch: "
            f"{actual_sha256} != {expected_sha256}"
        )
    bundle = joblib.load(io.BytesIO(artifact))
    if not isinstance(bundle, ModelBundle):
        raise TypeError("embedded artifact is not a WP V9 ModelBundle")
    if bundle.schema_version != MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported embedded WP V9 model schema")
    expected_model = str(contract.get("model_fingerprint") or "")
    if bundle.fingerprint != expected_model:
        raise RuntimeError(
            "embedded base-model fingerprint mismatch: "
            f"{bundle.fingerprint} != {expected_model}"
        )
    expected_policy = str(contract.get("policy_fingerprint") or "")
    if bundle.policy_fingerprint != expected_policy:
        raise RuntimeError(
            "embedded base-model policy mismatch: "
            f"{bundle.policy_fingerprint} != {expected_policy}"
        )
    return bundle, contract


def _resolve_inside(root: Path, value: str | Path) -> Path:
    raw = Path(value)
    resolved = (raw if raw.is_absolute() else root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"path escapes repository root: {value}"
        ) from error
    return resolved


def _relative_to(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()
