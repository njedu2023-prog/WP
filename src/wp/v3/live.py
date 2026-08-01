from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import V3Config, policy_fingerprint
from .dataset import execution_eligibility
from .features import enrich_feature_frame
from .model import ModelBundle, load_bundle, predict_bundle
from .registry import is_model_promoted, load_registry, model_record
from .v40_model import (
    V40ModelBundle,
    load_v40_bundle,
    predict_v40_bundle,
)


@dataclass(frozen=True)
class LiveInference:
    state: str
    model_version: str | None
    model_fingerprint: str | None
    policy_fingerprint: str | None
    formal_authorization: bool
    predictions: pd.DataFrame
    message: str


def run_live_inference(
    frame: pd.DataFrame,
    config: V3Config,
    *,
    model_path: str | Path,
    registry_path: str | Path,
) -> LiveInference:
    artifact = Path(model_path)
    if not artifact.exists():
        empty = frame.copy()
        empty["passes_policy"] = False
        empty["candidate_state"] = "MODEL_NOT_READY"
        return LiveInference(
            state="MODEL_NOT_READY",
            model_version=None,
            model_fingerprint=None,
            policy_fingerprint=None,
            formal_authorization=False,
            predictions=empty,
            message="V9 has no trained artifact; no candidate can be authorized.",
        )

    try:
        bundle = _load_runtime_bundle(artifact)
    except Exception as error:
        rejected = frame.copy()
        rejected["passes_policy"] = False
        rejected["candidate_state"] = "MODEL_ARTIFACT_INVALID"
        return LiveInference(
            state="MODEL_ARTIFACT_INVALID",
            model_version=None,
            model_fingerprint=None,
            policy_fingerprint=None,
            formal_authorization=False,
            predictions=rejected,
            message=(
                "Model artifact failed the V9 schema or integrity check; "
                f"all candidates are rejected ({type(error).__name__})."
            ),
        )
    expected_policy = policy_fingerprint(config)
    if (
        getattr(bundle, "contract_fingerprint", bundle.policy_fingerprint)
        != expected_policy
        or bundle.feature_version != config.model.feature_version
    ):
        rejected = frame.copy()
        rejected["passes_policy"] = False
        rejected["candidate_state"] = "POLICY_MISMATCH"
        rejected["model_version"] = bundle.model_version
        rejected["model_fingerprint"] = bundle.fingerprint
        rejected["policy_fingerprint"] = bundle.policy_fingerprint
        return LiveInference(
            state="POLICY_MISMATCH",
            model_version=bundle.model_version,
            model_fingerprint=bundle.fingerprint,
            policy_fingerprint=bundle.policy_fingerprint,
            formal_authorization=False,
            predictions=rejected,
            message=(
                "Model artifact does not match the current immutable policy or "
                "feature contract; all candidates are rejected."
            ),
        )
    registry = load_registry(registry_path)
    promoted = is_model_promoted(registry, bundle.fingerprint)
    record = model_record(registry, bundle.fingerprint) or {}
    observed = (
        registry.get("shadow_model_fingerprint") == bundle.fingerprint
        and registry.get("shadow_policy_fingerprint") == bundle.policy_fingerprint
    )
    if not promoted and not observed:
        rejected = frame.copy()
        rejected["passes_policy"] = False
        rejected["candidate_state"] = "MODEL_NOT_DESIGNATED"
        rejected["model_version"] = bundle.model_version
        rejected["model_fingerprint"] = bundle.fingerprint
        rejected["policy_fingerprint"] = bundle.policy_fingerprint
        return LiveInference(
            state="MODEL_NOT_DESIGNATED",
            model_version=bundle.model_version,
            model_fingerprint=bundle.fingerprint,
            policy_fingerprint=bundle.policy_fingerprint,
            formal_authorization=False,
            predictions=rejected,
            message="Model is registered for research but is not the designated live observer.",
        )
    features = enrich_feature_frame(frame)
    features["execution_eligible"] = execution_eligibility(features, config)
    predictions = (
        predict_v40_bundle(bundle, features, config=config)
        if isinstance(bundle, V40ModelBundle)
        else predict_bundle(bundle, features, config=config)
    )
    predictions["candidate_state"] = "REJECTED"
    predictions.loc[predictions["passes_policy"], "candidate_state"] = (
        "QUALIFIED" if promoted else "SHADOW_QUALIFIED"
    )
    backtest_passed = bool(
        record.get("backtest", {}).get("backtest_gate", {}).get("passed", False)
    )
    state = (
        "PRODUCTION"
        if promoted
        else ("SHADOW" if backtest_passed else "SHADOW_OBSERVATION")
    )
    return LiveInference(
        state=state,
        model_version=bundle.model_version,
        model_fingerprint=bundle.fingerprint,
        policy_fingerprint=bundle.policy_fingerprint,
        formal_authorization=promoted,
        predictions=predictions,
        message=(
            "Model is production-authorized."
            if promoted
            else (
                "Model passed the historical gate and remains in the mandatory "
                "150-trading-day shadow period."
                if backtest_passed
                else "Research model is collecting forward observation only; "
                "its historical gate failed and it is not authorized for trading."
            )
        ),
    )


def _load_runtime_bundle(path: str | Path) -> ModelBundle | V40ModelBundle:
    try:
        return load_v40_bundle(path)
    except Exception as v40_error:
        try:
            return load_bundle(path)
        except Exception:
            raise v40_error


def inference_manifest(inference: LiveInference) -> dict[str, Any]:
    passed = inference.predictions.get("passes_policy", pd.Series(dtype=bool)).fillna(False)
    return {
        "v3_state": inference.state,
        "v3_model_version": inference.model_version,
        "v3_model_fingerprint": inference.model_fingerprint,
        "v3_policy_fingerprint": inference.policy_fingerprint,
        "v3_formal_authorization": inference.formal_authorization,
        "v3_qualified_count": int(passed.sum()),
        "v3_message": inference.message,
    }
