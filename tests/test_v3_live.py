from types import SimpleNamespace

import pandas as pd

from wp.v3.contracts import V3Config
from wp.v3.live import run_live_inference


def test_live_inference_fails_closed_on_policy_mismatch(monkeypatch, tmp_path):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"model")
    bundle = SimpleNamespace(
        policy_fingerprint="stale-policy",
        feature_version=V3Config().model.feature_version,
        model_version="stale-model",
        fingerprint="stale-fingerprint",
    )
    monkeypatch.setattr("wp.v3.live.load_bundle", lambda path: bundle)

    result = run_live_inference(
        pd.DataFrame([{"ts_code": "600001.SH"}]),
        V3Config(),
        model_path=artifact,
        registry_path=tmp_path / "registry.json",
    )

    assert result.state == "POLICY_MISMATCH"
    assert result.formal_authorization is False
    assert result.predictions["passes_policy"].tolist() == [False]
    assert result.predictions["candidate_state"].tolist() == ["POLICY_MISMATCH"]


def test_live_inference_fails_closed_on_invalid_artifact(monkeypatch, tmp_path):
    artifact = tmp_path / "old-model.joblib"
    artifact.write_bytes(b"old")
    monkeypatch.setattr(
        "wp.v3.live.load_bundle",
        lambda path: (_ for _ in ()).throw(ValueError("old schema")),
    )

    result = run_live_inference(
        pd.DataFrame([{"ts_code": "600001.SH"}]),
        V3Config(),
        model_path=artifact,
        registry_path=tmp_path / "registry.json",
    )

    assert result.state == "MODEL_ARTIFACT_INVALID"
    assert result.formal_authorization is False
    assert result.predictions["passes_policy"].tolist() == [False]
    assert result.predictions["candidate_state"].tolist() == [
        "MODEL_ARTIFACT_INVALID"
    ]
