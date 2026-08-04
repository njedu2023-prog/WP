from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from wp.v3.contracts import policy_fingerprint, load_v3_config
from wp.v3.registry import load_registry, model_record
from wp.v3.v40_model import V40ModelBundle, load_v40_bundle


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_STATUSES = {"PROMOTED", "SHADOW", "SHADOW_OBSERVATION"}


def runtime_readiness(
    *,
    root: Path = ROOT,
    bundle_loader: Callable[[str | Path], V40ModelBundle] = load_v40_bundle,
) -> dict[str, Any]:
    config = load_v3_config(root / "config" / "wp_v3.yml")
    registry_path = root / "outputs" / "json" / "wp_model_registry_v3.json"
    registry = load_registry(registry_path)
    fingerprint = str(
        registry.get("active_model_fingerprint")
        or registry.get("shadow_model_fingerprint")
        or ""
    )
    if not fingerprint:
        raise RuntimeError("model registry has no active or shadow model")
    record = model_record(registry, fingerprint)
    if not record:
        raise RuntimeError(f"designated model is absent from registry: {fingerprint}")

    artifact_value = str(record.get("artifact_path") or "").strip()
    if not artifact_value:
        raise RuntimeError("designated model has no artifact_path")
    artifact_path = root / artifact_value
    if not artifact_path.exists():
        raise RuntimeError(f"designated model artifact is missing: {artifact_value}")

    bundle = bundle_loader(artifact_path)
    expected_contract = policy_fingerprint(config)
    checks = {
        "fingerprint": str(bundle.fingerprint) == fingerprint,
        "strategy_id": str(bundle.strategy_id) == config.strategy.strategy_id,
        "model_version": str(bundle.model_version).startswith("wpv41-"),
        "feature_version": (
            str(bundle.feature_version) == config.model.feature_version
        ),
        "contract_fingerprint": (
            str(bundle.contract_fingerprint) == expected_contract
        ),
        "policy_fingerprint": (
            str(bundle.policy_fingerprint)
            == str(record.get("policy_fingerprint") or "")
        ),
        "registry_status": str(record.get("status") or "") in ALLOWED_STATUSES,
    }
    if registry.get("active_model_fingerprint") == fingerprint:
        checks["registry_designation"] = (
            registry.get("active_policy_fingerprint")
            == bundle.policy_fingerprint
        )
    else:
        checks["registry_designation"] = (
            registry.get("shadow_model_fingerprint") == fingerprint
            and registry.get("shadow_policy_fingerprint")
            == bundle.policy_fingerprint
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "designated runtime model failed: " + ",".join(failed)
        )
    return {
        "status": "ready",
        "strategy_id": config.strategy.strategy_id,
        "signal_slot": config.strategy.signal_slots[0],
        "entry_settlement_slot": config.execution.entry_execution_deadline,
        "model_version": bundle.model_version,
        "model_fingerprint": bundle.fingerprint,
        "policy_fingerprint": bundle.policy_fingerprint,
        "contract_fingerprint": bundle.contract_fingerprint,
        "registry_status": record["status"],
        "artifact_path": artifact_value,
        "checks": checks,
    }


def main() -> int:
    try:
        result = runtime_readiness()
    except Exception as error:
        print(f"::error::WP V41 runtime is not ready: {error}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
