from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pytest

from wp.v3 import v16_provenance


class DummyBundle:
    schema_version = "wp_v9_hurdle_net_return_bundle_1"
    fingerprint = "model-123"
    policy_fingerprint = "policy-123"
    model_version = "wpv9-test"


def write_registry(root: Path, artifact: Path) -> Path:
    registry = root / "outputs/json/wp_model_registry_v3.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "schema_version": "wp_model_registry_v3",
                "generated_at": "2026-07-30T00:00:00Z",
                "models": [
                    {
                        "fingerprint": "model-123",
                        "policy_fingerprint": "policy-123",
                        "artifact_path": artifact.relative_to(root).as_posix(),
                        "status": "SHADOW_OBSERVATION",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return registry


def test_base_model_binding_is_exact_and_self_contained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifacts/model.joblib"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"frozen-model-bytes")
    registry = write_registry(tmp_path, artifact)
    monkeypatch.setattr(
        v16_provenance,
        "load_bundle",
        lambda _: DummyBundle(),
    )

    embedded, bundle, contract = v16_provenance.bind_base_model(
        registry,
        expected_fingerprint="model-123",
        repository_root=tmp_path,
    )

    assert embedded == b"frozen-model-bytes"
    assert bundle.fingerprint == "model-123"
    assert contract["artifact_sha256"] == hashlib.sha256(embedded).hexdigest()
    assert contract["binding"] == "embedded_exact_model_artifact"


def test_base_model_binding_rejects_a_different_fingerprint(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifacts/model.joblib"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"model")
    registry = write_registry(tmp_path, artifact)

    with pytest.raises(RuntimeError, match="found 0"):
        v16_provenance.bind_base_model(
            registry,
            expected_fingerprint="different",
            repository_root=tmp_path,
        )


def test_embedded_base_model_verifier_rejects_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v16_provenance, "ModelBundle", DummyBundle)
    artifact = joblib.dumps(DummyBundle()) if hasattr(joblib, "dumps") else None
    if artifact is None:
        import io

        stream = io.BytesIO()
        joblib.dump(DummyBundle(), stream)
        artifact = stream.getvalue()
    payload = {
        "base_model_contract": {
            "schema_version": (
                v16_provenance.BASE_MODEL_CONTRACT_SCHEMA
            ),
            "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
            "model_fingerprint": "model-123",
            "policy_fingerprint": "policy-123",
        },
        "base_model_artifact": artifact + b"tampered",
    }

    with pytest.raises(RuntimeError, match="digest mismatch"):
        v16_provenance.verify_embedded_base_model(payload)
