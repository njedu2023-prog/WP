from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .contracts import V3Config, policy_fingerprint
from .exit_risk import ExitFailureRiskBundle, fit_exit_failure_risk
from .io import stable_frame_digest
from .meta_alpha import (
    MetaAlphaBundle,
    fit_meta_alpha,
    prune_candidate_universe,
)
from .model import ModelBundle, predict_bundle
from .v40 import (
    V41Policy,
    attach_v40_policy_gates,
)


V40_MODEL_SCHEMA_VERSION = "wp_v41_fixed_1400_bundle_1"
META_TRAIN_DAYS = 126
META_CALIBRATION_DAYS = 21
# Exit failures are rare. Keep one full trading year so every fitted risk
# model clears the shared 5,000-row minimum without weakening that standard.
RISK_TRAIN_DAYS = 252
RISK_CALIBRATION_DAYS = 42
PURGE_DAYS = 2


@dataclass
class V40ModelBundle:
    schema_version: str
    strategy_id: str
    contract_fingerprint: str
    policy_fingerprint: str
    model_version: str
    feature_version: str
    trained_at: str
    train_start: str
    train_end: str
    calibration_start: str
    calibration_end: str
    training_rows: int
    training_data_digest: str
    top_per_base_score: int
    policy: V41Policy
    base_bundle: ModelBundle
    meta_bundle: MetaAlphaBundle
    exit_risk_bundle: ExitFailureRiskBundle
    fingerprint: str


def train_v40_bundle(
    *,
    base_bundle: ModelBundle,
    oos_frontier: pd.DataFrame,
    config: V3Config,
    top_per_base_score: int = 12,
    model_version: str | None = None,
) -> V40ModelBundle:
    frame = oos_frontier.copy()
    frame["trade_date"] = (
        frame["trade_date"].astype(str).str.replace("-", "", regex=False)
    )
    labelled = _boolean(
        frame.get(
            "label_available",
            pd.Series(False, index=frame.index),
        )
    )
    execution = _boolean(
        frame.get(
            "execution_eligible",
            pd.Series(False, index=frame.index),
        )
    )
    usable = frame.loc[labelled & execution].copy()
    dates = sorted(usable["trade_date"].unique())
    meta_segments = _segments(
        dates,
        train_days=META_TRAIN_DAYS,
        calibration_days=META_CALIBRATION_DAYS,
    )
    risk_segments = _segments(
        dates,
        train_days=RISK_TRAIN_DAYS,
        calibration_days=RISK_CALIBRATION_DAYS,
    )
    meta_train_dates, meta_calibration_dates = meta_segments
    risk_train_dates, risk_calibration_dates = risk_segments
    meta_train = usable.loc[
        usable["trade_date"].isin(meta_train_dates)
    ].copy()
    meta_calibration = usable.loc[
        usable["trade_date"].isin(meta_calibration_dates)
    ].copy()
    risk_source = usable.loc[
        _boolean(
            usable.get(
                "entry_fillable",
                pd.Series(False, index=usable.index),
            )
        )
    ].copy()
    risk_train = risk_source.loc[
        risk_source["trade_date"].isin(risk_train_dates)
    ].copy()
    risk_calibration = risk_source.loc[
        risk_source["trade_date"].isin(risk_calibration_dates)
    ].copy()
    meta_bundle = fit_meta_alpha(
        meta_train,
        meta_calibration,
        random_seed=config.model.random_seed + 40_100,
    )
    exit_risk_bundle = fit_exit_failure_risk(
        risk_train,
        risk_calibration,
        random_seed=config.model.random_seed + 40_200,
    )
    policy = V41Policy(
        observation_count=config.strategy.observation_count
    )
    contract = policy_fingerprint(config)
    training_columns = [
        column
        for column in (
            "trade_date",
            "signal_slot",
            "ts_code",
            "fold",
            "target_net_positive",
            "net_return_pct",
            "entry_fillable",
            "exit_fillable",
            "meta_p_positive",
            "risk_p_exit_failure",
        )
        if column in usable
    ]
    training_digest = stable_frame_digest(usable.loc[:, training_columns])
    payload = {
        "schema_version": V40_MODEL_SCHEMA_VERSION,
        "strategy_id": config.strategy.strategy_id,
        "contract_fingerprint": contract,
        "feature_version": config.model.feature_version,
        "base_model_fingerprint": base_bundle.fingerprint,
        "policy": policy.as_dict(),
        "training_data_digest": training_digest,
        "meta_train_start": meta_train_dates[0],
        "meta_train_end": meta_train_dates[-1],
        "risk_train_start": risk_train_dates[0],
        "risk_train_end": risk_train_dates[-1],
        "calibration_end": max(
            meta_calibration_dates[-1],
            risk_calibration_dates[-1],
        ),
        "top_per_base_score": top_per_base_score,
    }
    learned_policy = _digest(payload)
    fingerprint = _digest(
        {
            **payload,
            "policy_fingerprint": learned_policy,
        }
    )
    train_start = min(meta_train_dates[0], risk_train_dates[0])
    train_end = max(meta_train_dates[-1], risk_train_dates[-1])
    calibration_start = min(
        meta_calibration_dates[0],
        risk_calibration_dates[0],
    )
    calibration_end = max(
        meta_calibration_dates[-1],
        risk_calibration_dates[-1],
    )
    return V40ModelBundle(
        schema_version=V40_MODEL_SCHEMA_VERSION,
        strategy_id=config.strategy.strategy_id,
        contract_fingerprint=contract,
        policy_fingerprint=learned_policy,
        model_version=(
            model_version
            or f"wpv41-{calibration_end}-{fingerprint[:8]}"
        ),
        feature_version=config.model.feature_version,
        trained_at=datetime.now(timezone.utc).isoformat(),
        train_start=train_start,
        train_end=train_end,
        calibration_start=calibration_start,
        calibration_end=calibration_end,
        training_rows=int(len(usable)),
        training_data_digest=training_digest,
        top_per_base_score=top_per_base_score,
        policy=policy,
        base_bundle=base_bundle,
        meta_bundle=meta_bundle,
        exit_risk_bundle=exit_risk_bundle,
        fingerprint=fingerprint,
    )


def predict_v40_bundle(
    bundle: V40ModelBundle,
    frame: pd.DataFrame,
    *,
    config: V3Config,
) -> pd.DataFrame:
    base_scored = predict_bundle(bundle.base_bundle, frame, config=None)
    frontier = prune_candidate_universe(
        base_scored,
        top_per_score=bundle.top_per_base_score,
        require_label=False,
    )
    if frontier.empty:
        return frontier
    meta_scored = bundle.meta_bundle.predict(frontier)
    risk_scored = bundle.exit_risk_bundle.predict(meta_scored)
    fixed = risk_scored.loc[
        risk_scored["signal_slot"].astype(str).eq(
            bundle.policy.signal_slot
        )
    ].copy()
    fixed = attach_v40_policy_gates(fixed, bundle.policy)
    fresh = pd.to_numeric(
        fixed.get(
            "data_age_seconds",
            pd.Series(float("inf"), index=fixed.index),
        ),
        errors="coerce",
    ).between(
        -60,
        config.execution.max_market_data_age_seconds,
        inclusive="both",
    )
    fixed["passes_freshness"] = fresh.fillna(False)
    fixed["passes_policy"] &= fixed["passes_freshness"]
    fixed["passes_entry_fill_probability"] = fixed[
        "passes_round_trip_fill"
    ]
    fixed["passes_exit_fill_probability"] = fixed[
        "passes_round_trip_fill"
    ]
    fixed["passes_round_trip_fill_probability"] = fixed[
        "passes_round_trip_fill"
    ]
    fixed["passes_probability"] = fixed["passes_meta_probability"]
    fixed["passes_probability_lower"] = fixed[
        "passes_meta_probability"
    ]
    fixed["passes_conditional_probability"] = fixed[
        "passes_meta_probability"
    ]
    fixed["passes_severe_loss"] = (
        fixed["passes_meta_severe_loss"]
        & fixed["passes_exit_failure_rank"]
    )
    fixed["passes_selection_rank"] = fixed["passes_meta_rank"]
    fixed["passes_expected_utility"] = fixed[
        "passes_meta_expected_return"
    ]
    fixed["passes_expected_utility_lower"] = fixed[
        "passes_meta_expected_return"
    ]
    fixed["passes_downside"] = True
    fixed["passes_stability"] = True
    fixed["passes_prior_oos_evidence"] = True
    fixed["p_net_positive_lower"] = fixed["meta_p_positive"]
    fixed["expected_utility_lower_pct"] = fixed[
        "meta_expected_net_return_pct"
    ]
    fixed["selection_score"] = fixed["meta_score"]
    fixed["model_version"] = bundle.model_version
    fixed["model_fingerprint"] = bundle.fingerprint
    fixed["policy_fingerprint"] = bundle.policy_fingerprint
    fixed["candidate_policy_id"] = bundle.policy.policy_id
    fixed["candidate_policy_authorized"] = False
    fixed["candidate_state"] = "REJECTED"
    fixed.loc[fixed["passes_policy"], "candidate_state"] = (
        "SHADOW_QUALIFIED"
    )
    return fixed.sort_values(
        ["meta_score", "ts_code"],
        ascending=[False, True],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def save_v40_bundle(bundle: V40ModelBundle, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(bundle, temporary, compress=3)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_v40_bundle(path: str | Path) -> V40ModelBundle:
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, V40ModelBundle):
        raise TypeError("model artifact is not a WP V41 bundle")
    if bundle.schema_version != V40_MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported WP V41 model bundle schema")
    return bundle


def v40_bundle_metadata(bundle: V40ModelBundle) -> dict[str, Any]:
    return {
        "schema_version": bundle.schema_version,
        "strategy_id": bundle.strategy_id,
        "contract_fingerprint": bundle.contract_fingerprint,
        "policy_fingerprint": bundle.policy_fingerprint,
        "model_version": bundle.model_version,
        "feature_version": bundle.feature_version,
        "trained_at": bundle.trained_at,
        "train_start": bundle.train_start,
        "train_end": bundle.train_end,
        "calibration_start": bundle.calibration_start,
        "calibration_end": bundle.calibration_end,
        "training_rows": bundle.training_rows,
        "training_data_digest": bundle.training_data_digest,
        "top_per_base_score": bundle.top_per_base_score,
        "policy": bundle.policy.as_dict(),
        "base_model_fingerprint": bundle.base_bundle.fingerprint,
        "meta_feature_columns": list(bundle.meta_bundle.feature_columns),
        "exit_risk_feature_columns": list(
            bundle.exit_risk_bundle.feature_columns
        ),
        "fingerprint": bundle.fingerprint,
    }


def _segments(
    dates: list[str],
    *,
    train_days: int,
    calibration_days: int,
) -> tuple[list[str], list[str]]:
    needed = train_days + PURGE_DAYS + calibration_days
    if len(dates) < needed:
        raise ValueError(
            f"V41 model requires {needed} labelled dates; received {len(dates)}"
        )
    selected = dates[-needed:]
    train = selected[:train_days]
    calibration = selected[train_days + PURGE_DAYS :]
    if len(calibration) != calibration_days:
        raise AssertionError("invalid V41 calibration segment")
    return train, calibration


def _boolean(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y", "qualified", "pass"})
    )


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:20]
