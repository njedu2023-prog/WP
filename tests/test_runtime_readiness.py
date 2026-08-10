from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.check_wp_runtime_readiness import runtime_readiness
from wp.v3.contracts import load_v3_config, policy_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def _runtime_root(tmp_path: Path) -> tuple[Path, SimpleNamespace]:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    shutil.copy(ROOT / "config" / "wp_v3.yml", config_dir / "wp_v3.yml")
    config = load_v3_config(config_dir / "wp_v3.yml")
    contract = policy_fingerprint(config)
    artifact = tmp_path / "artifacts" / "v41.joblib"
    artifact.parent.mkdir()
    artifact.write_bytes(b"bundle")
    bundle = SimpleNamespace(
        fingerprint="v41-fingerprint",
        strategy_id=config.strategy.strategy_id,
        model_version="wpv41-20260731-v41finger",
        feature_version=config.model.feature_version,
        contract_fingerprint=contract,
        policy_fingerprint="v41-policy",
    )
    registry_dir = tmp_path / "outputs" / "json"
    registry_dir.mkdir(parents=True)
    (registry_dir / "wp_model_registry_v3.json").write_text(
        json.dumps(
            {
                "schema_version": "wp_model_registry_v3",
                "active_model_fingerprint": None,
                "active_policy_fingerprint": None,
                "shadow_model_fingerprint": bundle.fingerprint,
                "shadow_policy_fingerprint": bundle.policy_fingerprint,
                "models": [
                    {
                        "fingerprint": bundle.fingerprint,
                        "policy_fingerprint": bundle.policy_fingerprint,
                        "artifact_path": "artifacts/v41.joblib",
                        "status": "SHADOW_OBSERVATION",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path, bundle


def test_runtime_readiness_accepts_exact_v41_shadow_bundle(tmp_path):
    root, bundle = _runtime_root(tmp_path)

    result = runtime_readiness(
        root=root,
        bundle_loader=lambda path: bundle,
    )

    assert result["status"] == "ready"
    assert result["signal_slot"] == "14:00"
    assert result["market_data_cutoff_slot"] == "13:55"
    assert result["decision_publish_deadline"] == "14:00"
    assert result["entry_settlement_slot"] == "14:05"
    assert result["model_fingerprint"] == bundle.fingerprint


def test_runtime_readiness_rejects_legacy_contract(tmp_path):
    root, bundle = _runtime_root(tmp_path)
    bundle.contract_fingerprint = "legacy-1430-contract"

    with pytest.raises(
        RuntimeError,
        match="contract_fingerprint",
    ):
        runtime_readiness(
            root=root,
            bundle_loader=lambda path: bundle,
        )
