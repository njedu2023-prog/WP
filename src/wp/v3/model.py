from __future__ import annotations

import gc
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .contracts import V3Config, policy_fingerprint
from .features import (
    FEATURE_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    enrich_feature_frame,
    feature_matrix,
)
from .policy import (
    CandidatePolicy,
    PolicySelection,
    candidate_policy_diagnostics,
    no_signal_policy,
)
from .statistics import clustered_binary_lower


TARGET_RANK_COLUMN = "_target_net_return_rank"
MODEL_SCHEMA_VERSION = "wp_v8_all_in_net_return_bundle_1"


@dataclass
class BetaCalibrator:
    model: LogisticRegression | None = None
    constant: float | None = None

    def fit(
        self,
        probability: np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "BetaCalibrator":
        probability = np.asarray(probability, dtype=float)
        target = np.asarray(target, dtype=int)
        if len(target) == 0:
            raise ValueError("calibration target is empty")
        if len(np.unique(target)) < 2:
            self.constant = float(np.average(target, weights=sample_weight))
            self.model = None
            return self
        self.model = LogisticRegression(
            C=0.5,
            max_iter=2_000,
            solver="lbfgs",
        )
        self.model.fit(
            _beta_features(probability),
            target,
            sample_weight=sample_weight,
        )
        self.constant = None
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        probability = np.asarray(probability, dtype=float)
        if self.model is None:
            if self.constant is None:
                return np.clip(probability, 1e-4, 1 - 1e-4)
            return np.full(len(probability), self.constant, dtype=float)
        return self.model.predict_proba(_beta_features(probability))[:, 1]


@dataclass
class ModelMember:
    name: str
    window_days: int
    train_start: str
    train_end: str
    entry_fill_classifier: Any
    exit_fill_classifier: Any
    net_positive_classifier: Any
    cross_section_classifier: Any
    severe_loss_classifier: Any
    ranker: Any
    mean_regressor: Any
    downside_regressor: Any


@dataclass
class ConstantBinaryClassifier:
    probability: float

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probability = np.full(len(features), self.probability, dtype=float)
        return np.column_stack((1.0 - probability, probability))


@dataclass
class ModelBundle:
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
    calibration_fit_end: str
    evidence_start: str
    training_rows: int
    eligible_fit_rows: int
    calibration_rows: int
    eligible_calibration_rows: int
    evidence_rows: int
    training_data_digest: str
    positive_rate: float
    feature_columns: tuple[str, ...]
    market_feature_columns: tuple[str, ...]
    members: list[ModelMember]
    entry_fill_calibrator: BetaCalibrator
    exit_fill_calibrator: BetaCalibrator
    positive_calibrator: BetaCalibrator
    cross_section_calibrator: BetaCalibrator
    severe_loss_calibrator: BetaCalibrator
    calibration_table: list[dict[str, float | int]]
    candidate_policy: CandidatePolicy
    policy_selection: dict[str, Any]
    selection_evidence: dict[str, Any]
    max_market_data_age_seconds: int
    exit_non_fill_penalty_pct: float
    fingerprint: str


def train_bundle(
    panel: pd.DataFrame,
    config: V3Config,
    *,
    allow_below_minimum: bool = False,
    model_version: str | None = None,
    policy_selection: PolicySelection | None = None,
) -> ModelBundle:
    eligible = panel.loc[
        panel["label_available"].fillna(False)
        & panel["execution_eligible"].fillna(False)
        & pd.to_numeric(panel["target_net_positive"], errors="coerce").notna()
    ].copy()
    eligible["trade_date"] = eligible["trade_date"].astype(str)
    eligible.sort_values(
        ["trade_date", "signal_slot", "ts_code"],
        kind="stable",
        inplace=True,
    )
    eligible.reset_index(drop=True, inplace=True)
    eligible = _attach_full_universe_rank_target(eligible, copy=False)
    eligible = _ensure_multitask_targets(eligible, config, copy=False)
    eligible_rows = int(len(eligible))
    if not allow_below_minimum and len(eligible) < config.model.min_train_rows:
        raise ValueError(
            f"V8 training requires {config.model.min_train_rows:,} eligible "
            f"labelled rows; received {len(eligible):,}"
        )

    unique_dates = np.array(sorted(eligible["trade_date"].unique()))
    minimum_dates = (
        config.model.minimum_train_days
        + config.model.calibration_days
        + config.model.purge_days
    )
    if not allow_below_minimum and len(unique_dates) < minimum_dates:
        raise ValueError(
            f"V8 temporal training requires at least {minimum_dates} dates; "
            f"received {len(unique_dates)}"
        )
    if allow_below_minimum and len(unique_dates) < (
        config.model.calibration_days + config.model.purge_days + 20
    ):
        raise ValueError("insufficient temporal depth for V8 calibration")

    calibration_dates = unique_dates[-config.model.calibration_days :]
    fit_end = len(unique_dates) - config.model.calibration_days - config.model.purge_days
    fit_dates = unique_dates[:fit_end]
    if len(fit_dates) < 20:
        raise ValueError("V8 fit period is too short")
    development_dates = fit_dates[-min(max(config.model.ensemble_windows_days), len(fit_dates)) :]
    sampled_development = _deterministic_training_sample(
        eligible,
        rows_per_slot=config.model.max_training_rows_per_slot,
        allowed_dates=development_dates,
    )
    calibration = _deterministic_training_sample(
        eligible,
        rows_per_slot=config.model.max_training_rows_per_slot,
        allowed_dates=calibration_dates,
    )
    eligible_calibration_rows = int(
        eligible["trade_date"].isin(calibration_dates).sum()
    )
    sampled_development = enrich_feature_frame(sampled_development, copy=False)
    calibration = enrich_feature_frame(calibration, copy=False)
    print(
        f"[wp-v8] calibration fit={fit_dates[0]}..{fit_dates[-1]} "
        f"rows={len(sampled_development):,} "
        f"calibration={calibration_dates[0]}..{calibration_dates[-1]} "
        f"rows={len(calibration):,}/{eligible_calibration_rows:,}",
        flush=True,
    )
    development_members = _fit_members(
        sampled_development,
        development_dates,
        config,
        seed_offset=0,
        allow_single=True,
        calibration_only=True,
    )
    calibration_raw = _raw_score_frame(calibration, development_members)
    calibration_weight = _group_temporal_weights(
        calibration,
        half_life_days=config.model.temporal_half_life_days,
    )
    entry_mask = pd.to_numeric(
        calibration["target_entry_fillable"],
        errors="coerce",
    ).notna()
    exit_mask = pd.to_numeric(
        calibration["target_exit_fillable"],
        errors="coerce",
    ).notna()
    positive_mask = pd.to_numeric(
        calibration["target_net_positive"],
        errors="coerce",
    ).notna()
    cross_mask = pd.to_numeric(
        calibration["target_cross_section_top"],
        errors="coerce",
    ).notna()
    severe_mask = pd.to_numeric(
        calibration["target_severe_loss"],
        errors="coerce",
    ).notna()
    entry_fill_calibrator = BetaCalibrator().fit(
        calibration_raw.loc[entry_mask, "p_entry_fill_raw"].to_numpy(),
        calibration.loc[entry_mask, "target_entry_fillable"].astype(int).to_numpy(),
        calibration_weight[entry_mask.to_numpy()],
    )
    exit_fill_calibrator = BetaCalibrator().fit(
        calibration_raw.loc[exit_mask, "p_exit_fill_raw"].to_numpy(),
        calibration.loc[exit_mask, "target_exit_fillable"].astype(int).to_numpy(),
        calibration_weight[exit_mask.to_numpy()],
    )
    positive_calibrator = BetaCalibrator().fit(
        calibration_raw.loc[
            positive_mask,
            "p_net_positive_raw",
        ].to_numpy(),
        calibration.loc[
            positive_mask,
            "target_net_positive",
        ].astype(int).to_numpy(),
        calibration_weight[positive_mask.to_numpy()],
    )
    cross_calibrator = BetaCalibrator().fit(
        calibration_raw.loc[
            cross_mask,
            "p_cross_section_top_raw",
        ].to_numpy(),
        calibration.loc[
            cross_mask,
            "target_cross_section_top",
        ].astype(int).to_numpy(),
        calibration_weight[cross_mask.to_numpy()],
    )
    severe_calibrator = BetaCalibrator().fit(
        calibration_raw.loc[
            severe_mask,
            "p_severe_loss_raw",
        ].to_numpy(),
        calibration.loc[
            severe_mask,
            "target_severe_loss",
        ].astype(int).to_numpy(),
        calibration_weight[severe_mask.to_numpy()],
    )
    calibration_table = _build_calibration_table(
        positive_calibrator.predict(
            calibration_raw["p_net_positive_raw"].to_numpy()
        ),
        calibration["target_net_positive"].astype(int).to_numpy(),
        calibration["trade_date"].astype(str).to_numpy(),
        seed=config.model.random_seed,
    )
    calibration_rows = int(len(calibration))

    # Calibration models are only needed to freeze the mapping. Releasing them
    # before the seven-task execution ensemble prevents late-fold memory peaks.
    del calibration
    del calibration_raw
    del calibration_weight
    del development_members
    del sampled_development
    gc.collect()

    # After every calibration object is frozen on strictly earlier predictions,
    # refit the execution ensemble through the latest labelled date. The outer
    # walk-forward purge still separates this refit from each test fold.
    sampled_refit = _deterministic_training_sample(
        eligible,
        rows_per_slot=config.model.max_training_rows_per_slot,
    )
    sampled_refit = enrich_feature_frame(sampled_refit, copy=False)
    training_rows = int(len(sampled_refit))
    positive_rate = float(
        sampled_refit["target_net_positive"].astype(int).mean()
    )
    training_digest = _training_data_digest(sampled_refit)
    del eligible
    gc.collect()
    execution_members = _fit_members(
        sampled_refit,
        unique_dates,
        config,
        seed_offset=50_000,
        allow_single=False,
        calibration_only=False,
    )
    del sampled_refit
    gc.collect()
    candidate_policy = (
        policy_selection.policy
        if policy_selection is not None
        else no_signal_policy("nested_oos_policy_not_attached")
    )
    selection_payload = (
        policy_selection.as_dict() if policy_selection is not None else {}
    )
    contract = policy_fingerprint(config)
    learned_policy_fingerprint = _learned_policy_fingerprint(
        contract,
        candidate_policy,
    )
    version = model_version or f"wpv8-{unique_dates[-1]}-{eligible_rows}"
    evidence = (
        dict(policy_selection.confirmation)
        if policy_selection is not None
        else {}
    )
    if evidence and "period_start" not in evidence:
        evidence["period_start"] = selection_payload.get("confirmation", {}).get(
            "period_start"
        )
    metadata = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "strategy_id": config.strategy.strategy_id,
        "contract_fingerprint": contract,
        "policy_fingerprint": learned_policy_fingerprint,
        "model_version": version,
        "feature_version": config.model.feature_version,
        "train_start": str(unique_dates[0]),
        "train_end": str(unique_dates[-1]),
        "calibration_start": str(calibration_dates[0]),
        "calibration_end": str(calibration_dates[-1]),
        "calibration_fit_end": str(fit_dates[-1]),
        "training_rows": training_rows,
        "eligible_fit_rows": eligible_rows,
        "calibration_rows": calibration_rows,
        "eligible_calibration_rows": eligible_calibration_rows,
        "training_data_digest": training_digest,
        "members": [member.name for member in execution_members],
        "features": FEATURE_COLUMNS,
        "market_features_in_main_model": MARKET_FEATURE_COLUMNS,
        "candidate_policy": asdict(candidate_policy),
        "policy_selection": selection_payload,
    }
    fingerprint = _digest(metadata)
    return ModelBundle(
        schema_version=MODEL_SCHEMA_VERSION,
        strategy_id=config.strategy.strategy_id,
        contract_fingerprint=contract,
        policy_fingerprint=learned_policy_fingerprint,
        model_version=version,
        feature_version=config.model.feature_version,
        trained_at=datetime.now(timezone.utc).isoformat(),
        train_start=str(unique_dates[0]),
        train_end=str(unique_dates[-1]),
        calibration_start=str(calibration_dates[0]),
        calibration_end=str(calibration_dates[-1]),
        calibration_fit_end=str(fit_dates[-1]),
        evidence_start=(
            str(selection_payload.get("confirmation", {}).get("period_start") or "")
        ),
        training_rows=training_rows,
        eligible_fit_rows=eligible_rows,
        calibration_rows=calibration_rows,
        eligible_calibration_rows=eligible_calibration_rows,
        evidence_rows=int(evidence.get("events", 0) or 0),
        training_data_digest=training_digest,
        positive_rate=positive_rate,
        feature_columns=FEATURE_COLUMNS,
        market_feature_columns=MARKET_FEATURE_COLUMNS,
        members=execution_members,
        entry_fill_calibrator=entry_fill_calibrator,
        exit_fill_calibrator=exit_fill_calibrator,
        positive_calibrator=positive_calibrator,
        cross_section_calibrator=cross_calibrator,
        severe_loss_calibrator=severe_calibrator,
        calibration_table=calibration_table,
        candidate_policy=candidate_policy,
        policy_selection=selection_payload,
        selection_evidence=evidence,
        max_market_data_age_seconds=config.execution.max_market_data_age_seconds,
        exit_non_fill_penalty_pct=config.execution.non_fill_penalty_pct,
        fingerprint=fingerprint,
    )


def attach_policy(
    bundle: ModelBundle,
    selection: PolicySelection,
) -> ModelBundle:
    learned = _learned_policy_fingerprint(
        bundle.contract_fingerprint,
        selection.policy,
    )
    payload = {
        "model_fingerprint": bundle.fingerprint,
        "policy_fingerprint": learned,
        "policy": asdict(selection.policy),
        "selection": selection.as_dict(),
    }
    return replace(
        bundle,
        policy_fingerprint=learned,
        candidate_policy=selection.policy,
        policy_selection=selection.as_dict(),
        selection_evidence=dict(selection.confirmation),
        evidence_start=str(
            selection.confirmation.get("period_start") or bundle.evidence_start
        ),
        evidence_rows=int(selection.confirmation.get("events", 0) or 0),
        fingerprint=_digest(payload),
    )


def predict_bundle(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    *,
    config: V3Config | None = None,
) -> pd.DataFrame:
    result = _score_frame(bundle, frame)
    result["model_version"] = bundle.model_version
    result["model_fingerprint"] = bundle.fingerprint
    result["policy_fingerprint"] = bundle.policy_fingerprint
    result["candidate_policy_id"] = bundle.candidate_policy.policy_id
    result["candidate_policy_authorized"] = bundle.candidate_policy.authorized
    evidence = bundle.selection_evidence
    result["selection_evidence_candidate_events"] = int(
        evidence.get("events", 0) or 0
    )
    result["selection_evidence_candidate_days"] = int(
        evidence.get("trade_days", 0) or 0
    )
    result["selection_evidence_win_rate"] = float(
        evidence.get("win_rate", 0.0) or 0.0
    )
    result["selection_evidence_mean_net_return_pct"] = float(
        evidence.get("mean_net_return_pct", 0.0) or 0.0
    )
    if config is None:
        result["passes_policy"] = False
        result["passes_prior_oos_evidence"] = False
        result["rejection_reasons"] = "runtime_contract_missing"
    else:
        diagnostics = candidate_policy_diagnostics(
            result,
            bundle.candidate_policy,
            config,
        )
        for column in diagnostics:
            result[column] = diagnostics[column]
    return result


def save_bundle(bundle: ModelBundle, path: str | Path) -> None:
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


def load_bundle(path: str | Path) -> ModelBundle:
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, ModelBundle):
        raise TypeError("model artifact is not a WP V8 ModelBundle")
    if bundle.schema_version != MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported WP model bundle schema")
    return bundle


def bundle_metadata(bundle: ModelBundle) -> dict[str, Any]:
    data = asdict(bundle)
    for key in (
        "members",
        "entry_fill_calibrator",
        "exit_fill_calibrator",
        "positive_calibrator",
        "cross_section_calibrator",
        "severe_loss_calibrator",
    ):
        data.pop(key, None)
    data["ensemble_members"] = [
        {
            "name": member.name,
            "window_days": member.window_days,
            "train_start": member.train_start,
            "train_end": member.train_end,
        }
        for member in bundle.members
    ]
    return data


def _fit_members(
    frame: pd.DataFrame,
    dates: np.ndarray,
    config: V3Config,
    *,
    seed_offset: int,
    allow_single: bool,
    calibration_only: bool,
) -> list[ModelMember]:
    members: list[ModelMember] = []
    fitted_lengths: set[int] = set()
    for window in config.model.ensemble_windows_days:
        member_dates = dates[-min(int(window), len(dates)) :]
        if len(member_dates) in fitted_lengths:
            continue
        member_frame = frame.loc[
            frame["trade_date"].astype(str).isin(member_dates)
        ].copy()
        if len(member_frame) < 100 or member_frame["target_net_positive"].nunique() < 2:
            continue
        member_frame = member_frame.sort_values(
            ["trade_date", "signal_slot", "ts_code"],
            kind="stable",
        )
        exit_frame = member_frame.loc[
            pd.to_numeric(
                member_frame["target_exit_fillable"],
                errors="coerce",
            ).notna()
        ].copy()
        all_in_frame = member_frame.loc[
            pd.to_numeric(
                member_frame["target_net_positive"],
                errors="coerce",
            ).notna()
            & pd.to_numeric(
                member_frame["net_return_pct"],
                errors="coerce",
            ).notna()
            & pd.to_numeric(
                member_frame[TARGET_RANK_COLUMN],
                errors="coerce",
            ).notna()
        ].copy()
        if len(exit_frame) < 100 or len(all_in_frame) < 100:
            continue
        features = feature_matrix(member_frame)
        exit_features = feature_matrix(exit_frame)
        all_in_features = feature_matrix(all_in_frame)
        weights = _group_temporal_weights(
            member_frame,
            half_life_days=config.model.temporal_half_life_days,
        )
        exit_weights = _group_temporal_weights(
            exit_frame,
            half_life_days=config.model.temporal_half_life_days,
        )
        all_in_weights = _group_temporal_weights(
            all_in_frame,
            half_life_days=config.model.temporal_half_life_days,
        )
        rank_percentile, rank_groups = _ranking_target_and_groups(
            all_in_frame
        )
        rank_relevance = np.clip(
            np.floor(rank_percentile * 5.0),
            0,
            4,
        ).astype(int)
        seed = config.model.random_seed + seed_offset + int(window)
        print(
            f"[wp-v8] fit member window={len(member_dates)}d "
            f"rows={len(member_frame):,} exit={len(exit_frame):,} "
            f"all_in={len(all_in_frame):,}",
            flush=True,
        )
        members.append(
            ModelMember(
                name=f"all_in_net_return_{len(member_dates)}d",
                window_days=int(len(member_dates)),
                train_start=str(member_dates[0]),
                train_end=str(member_dates[-1]),
                entry_fill_classifier=_fit_classifier(
                    features,
                    member_frame["target_entry_fillable"].astype(int).to_numpy(),
                    weights,
                    seed,
                ),
                exit_fill_classifier=_fit_classifier(
                    exit_features,
                    exit_frame["target_exit_fillable"].astype(int).to_numpy(),
                    exit_weights,
                    seed + 1_000,
                ),
                net_positive_classifier=_fit_classifier(
                    all_in_features,
                    all_in_frame["target_net_positive"].astype(int).to_numpy(),
                    all_in_weights,
                    seed + 2_000,
                ),
                cross_section_classifier=_fit_classifier(
                    all_in_features,
                    all_in_frame[
                        "target_cross_section_top"
                    ].astype(int).to_numpy(),
                    all_in_weights,
                    seed + 3_000,
                ),
                severe_loss_classifier=_fit_classifier(
                    all_in_features,
                    all_in_frame["target_severe_loss"].astype(int).to_numpy(),
                    all_in_weights,
                    seed + 4_000,
                ),
                ranker=(
                    None
                    if calibration_only
                    else _fit_ranker(
                        all_in_features,
                        rank_relevance,
                        rank_groups,
                        all_in_weights,
                        seed + 5_000,
                    )
                ),
                mean_regressor=(
                    None
                    if calibration_only
                    else _fit_regressor(
                        all_in_features,
                        pd.to_numeric(
                            all_in_frame["net_return_pct"],
                            errors="coerce",
                        ).clip(-15.0, 15.0).to_numpy(),
                        all_in_weights,
                        seed + 6_000,
                        objective="regression_l2",
                    )
                ),
                downside_regressor=(
                    None
                    if calibration_only
                    else _fit_regressor(
                        all_in_features,
                        pd.to_numeric(
                            all_in_frame["net_return_pct"],
                            errors="coerce",
                        ).clip(-15.0, 15.0).to_numpy(),
                        all_in_weights,
                        seed + 7_000,
                        objective="quantile",
                        alpha=0.10,
                    )
                ),
            )
        )
        del features
        del exit_features
        del all_in_features
        del exit_frame
        del all_in_frame
        del member_frame
        del exit_weights
        del all_in_weights
        del rank_groups
        del rank_percentile
        del rank_relevance
        del weights
        gc.collect()
        fitted_lengths.add(len(member_dates))
    required = 1 if allow_single else 2
    if len(members) < required:
        raise ValueError(
            f"V8 temporal ensemble requires at least {required} trained members"
        )
    return members


def _raw_score_frame(
    frame: pd.DataFrame,
    members: list[ModelMember],
) -> pd.DataFrame:
    features = feature_matrix(frame)
    entry_members = np.column_stack(
        [
            _predict_probability(member.entry_fill_classifier, features)
            for member in members
        ]
    )
    exit_members = np.column_stack(
        [
            _predict_probability(member.exit_fill_classifier, features)
            for member in members
        ]
    )
    net_positive_members = np.column_stack(
        [
            _predict_probability(
                member.net_positive_classifier,
                features,
            )
            for member in members
        ]
    )
    cross_members = np.column_stack(
        [
            _predict_probability(member.cross_section_classifier, features)
            for member in members
        ]
    )
    severe_members = np.column_stack(
        [
            _predict_probability(
                member.severe_loss_classifier,
                features,
            )
            for member in members
        ]
    )
    return pd.DataFrame(
        {
            "p_entry_fill_raw": entry_members.mean(axis=1),
            "p_exit_fill_raw": exit_members.mean(axis=1),
            "p_net_positive_raw": net_positive_members.mean(axis=1),
            "p_cross_section_top_raw": cross_members.mean(axis=1),
            "p_severe_loss_raw": severe_members.mean(axis=1),
        },
        index=frame.index,
    )


def _score_frame(bundle: ModelBundle, frame: pd.DataFrame) -> pd.DataFrame:
    frame = enrich_feature_frame(frame)
    features = feature_matrix(frame)
    members = bundle.members
    entry_members_raw = np.column_stack(
        [
            _predict_probability(member.entry_fill_classifier, features)
            for member in members
        ]
    )
    exit_members_raw = np.column_stack(
        [
            _predict_probability(member.exit_fill_classifier, features)
            for member in members
        ]
    )
    net_positive_members_raw = np.column_stack(
        [
            _predict_probability(
                member.net_positive_classifier,
                features,
            )
            for member in members
        ]
    )
    cross_members_raw = np.column_stack(
        [
            _predict_probability(member.cross_section_classifier, features)
            for member in members
        ]
    )
    severe_members_raw = np.column_stack(
        [
            _predict_probability(
                member.severe_loss_classifier,
                features,
            )
            for member in members
        ]
    )
    entry_members = np.column_stack(
        [
            bundle.entry_fill_calibrator.predict(entry_members_raw[:, index])
            for index in range(entry_members_raw.shape[1])
        ]
    )
    exit_members = np.column_stack(
        [
            bundle.exit_fill_calibrator.predict(exit_members_raw[:, index])
            for index in range(exit_members_raw.shape[1])
        ]
    )
    positive_members = np.column_stack(
        [
            bundle.positive_calibrator.predict(
                net_positive_members_raw[:, index]
            )
            for index in range(net_positive_members_raw.shape[1])
        ]
    )
    cross_members = np.column_stack(
        [
            bundle.cross_section_calibrator.predict(cross_members_raw[:, index])
            for index in range(cross_members_raw.shape[1])
        ]
    )
    severe_members = np.column_stack(
        [
            bundle.severe_loss_calibrator.predict(severe_members_raw[:, index])
            for index in range(severe_members_raw.shape[1])
        ]
    )
    expected_members = np.column_stack(
        [
            np.asarray(member.mean_regressor.predict(features), dtype=float)
            for member in members
        ]
    )
    downside_members = np.column_stack(
        [
            np.asarray(member.downside_regressor.predict(features), dtype=float)
            for member in members
        ]
    )
    rank_members_raw = np.column_stack(
        [
            np.asarray(member.ranker.predict(features), dtype=float)
            for member in members
        ]
    )
    rank_members = np.column_stack(
        [
            _group_percentile_rank(rank_members_raw[:, index], frame)
            for index in range(rank_members_raw.shape[1])
        ]
    )

    entry_probability = bundle.entry_fill_calibrator.predict(
        entry_members_raw.mean(axis=1)
    )
    exit_probability = bundle.exit_fill_calibrator.predict(
        exit_members_raw.mean(axis=1)
    )
    probability = bundle.positive_calibrator.predict(
        net_positive_members_raw.mean(axis=1)
    )
    round_trip_members = entry_members * exit_members
    round_trip_probability = entry_probability * exit_probability
    conditional_probability = np.clip(
        probability / np.maximum(round_trip_probability, 1e-4),
        0.0,
        1.0,
    )
    cross_probability = bundle.cross_section_calibrator.predict(
        cross_members_raw.mean(axis=1)
    )
    severe_probability = bundle.severe_loss_calibrator.predict(
        severe_members_raw.mean(axis=1)
    )
    expected_utility = expected_members.mean(axis=1)
    expected_utility_lower = np.quantile(
        expected_members,
        0.10,
        axis=1,
    )
    all_in_downside = downside_members.mean(axis=1)
    conditional_expected = expected_utility / np.maximum(
        entry_probability,
        1e-4,
    )
    rank_score = rank_members.mean(axis=1)
    positive_rank = _group_percentile_rank(probability, frame)
    fill_rank = _group_percentile_rank(round_trip_probability, frame)
    cross_rank = _group_percentile_rank(cross_probability, frame)
    severe_safe_rank = _group_percentile_rank(1.0 - severe_probability, frame)
    expected_rank = _group_percentile_rank(expected_utility, frame)
    downside_rank = _group_percentile_rank(all_in_downside, frame)
    selection_score = (
        0.30 * expected_rank
        + 0.25 * rank_score
        + 0.20 * positive_rank
        + 0.10 * fill_rank
        + 0.05 * cross_rank
        + 0.05 * downside_rank
        + 0.05 * severe_safe_rank
    )

    result = frame.copy()
    result["p_net_positive_raw"] = net_positive_members_raw.mean(axis=1)
    result["p_net_positive_component_product"] = (
        round_trip_probability * conditional_probability
    )
    result["p_entry_fill"] = entry_probability
    result["p_exit_fill_given_entry"] = exit_probability
    result["p_round_trip_fill"] = round_trip_probability
    result["p_round_trip_fill_lower"] = np.quantile(
        round_trip_members,
        0.10,
        axis=1,
    )
    result["p_conditional_net_positive"] = conditional_probability
    result["p_net_positive"] = probability
    result["p_net_positive_lower"] = np.quantile(
        positive_members,
        0.10,
        axis=1,
    )
    result["p_cross_section_top"] = cross_probability
    result["p_severe_loss"] = severe_probability
    result["probability_model_spread"] = positive_members.std(axis=1)
    result["fill_probability_model_spread"] = round_trip_members.std(axis=1)
    result["conditional_expected_net_return_pct"] = conditional_expected
    result["expected_utility_pct"] = expected_utility
    result["expected_utility_lower_pct"] = expected_utility_lower
    result["expected_return_model_spread"] = expected_members.std(axis=1)
    result["downside_q10_pct"] = all_in_downside
    result["ranking_score"] = rank_score
    result["selection_score"] = selection_score
    result["selection_rank_pct"] = _group_percentile_rank(
        selection_score,
        frame,
    )
    result["selection_rank_spread"] = rank_members.std(axis=1)
    return result


def _fit_classifier(
    features: pd.DataFrame,
    target: np.ndarray,
    weights: np.ndarray,
    seed: int,
    *,
    compact: bool = False,
) -> Any:
    classes, counts = np.unique(np.asarray(target, dtype=int), return_counts=True)
    if len(classes) < 2 or int(counts.min()) < 2:
        probability = float(np.average(target, weights=weights))
        return ConstantBinaryClassifier(
            probability=float(np.clip(probability, 1e-4, 1 - 1e-4))
        )
    lightgbm = _load_lightgbm()
    if lightgbm is not None:
        model = lightgbm.LGBMClassifier(
            objective="binary",
            n_estimators=96 if compact else 160,
            learning_rate=0.035 if compact else 0.04,
            num_leaves=7 if compact else 23,
            max_depth=3 if compact else -1,
            min_child_samples=max(20, _minimum_leaf(len(features))),
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=5.0,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        return model.fit(features, target, sample_weight=weights)
    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.04,
                    max_iter=120 if compact else 180,
                    max_leaf_nodes=7 if compact else 23,
                    min_samples_leaf=_minimum_leaf(len(features)),
                    l2_regularization=5.0,
                    max_bins=127,
                    early_stopping=True,
                    validation_fraction=0.12,
                    random_state=seed,
                ),
            ),
        ]
    )
    return model.fit(features, target, model__sample_weight=weights)


def _fit_ranker(
    features: pd.DataFrame,
    relevance: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> Any:
    if int(np.asarray(groups, dtype=int).sum()) != len(features):
        raise ValueError("ranking groups do not cover the feature rows")
    lightgbm = _load_lightgbm()
    if lightgbm is not None:
        model = lightgbm.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            eval_at=(1, 3, 5),
            label_gain=(0, 1, 3, 7, 15),
            n_estimators=160,
            learning_rate=0.04,
            num_leaves=23,
            min_child_samples=max(20, _minimum_leaf(len(features))),
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=5.0,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        return model.fit(
            features,
            relevance,
            group=groups,
            sample_weight=weights,
        )
    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.04,
                    max_iter=180,
                    max_leaf_nodes=23,
                    min_samples_leaf=_minimum_leaf(len(features)),
                    l2_regularization=5.0,
                    max_bins=127,
                    early_stopping=True,
                    validation_fraction=0.12,
                    random_state=seed,
                ),
            ),
        ]
    )
    return model.fit(features, relevance, model__sample_weight=weights)


def _fit_regressor(
    features: pd.DataFrame,
    target: np.ndarray,
    weights: np.ndarray,
    seed: int,
    *,
    objective: str,
    alpha: float | None = None,
) -> Any:
    lightgbm = _load_lightgbm()
    if lightgbm is not None:
        kwargs: dict[str, Any] = {}
        if alpha is not None:
            kwargs["alpha"] = alpha
        model = lightgbm.LGBMRegressor(
            objective=objective,
            n_estimators=144,
            learning_rate=0.04,
            num_leaves=23,
            min_child_samples=max(20, _minimum_leaf(len(features))),
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=5.0,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            **kwargs,
        )
        return model.fit(features, target, sample_weight=weights)
    loss = {
        "regression_l1": "absolute_error",
        "regression_l2": "squared_error",
    }.get(objective, objective)
    kwargs = {"loss": loss}
    if alpha is not None:
        kwargs["quantile"] = alpha
    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.04,
                    max_iter=240,
                    max_leaf_nodes=23,
                    min_samples_leaf=_minimum_leaf(len(features)),
                    l2_regularization=5.0,
                    max_bins=127,
                    early_stopping=True,
                    validation_fraction=0.12,
                    random_state=seed,
                    **kwargs,
                ),
            ),
        ]
    )
    return model.fit(features, target, model__sample_weight=weights)


def _ranking_target_and_groups(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    keys = ["trade_date", "signal_slot"]
    if TARGET_RANK_COLUMN in frame:
        percentile = pd.to_numeric(frame[TARGET_RANK_COLUMN], errors="coerce")
    else:
        percentile = (
            pd.to_numeric(
                frame["net_return_pct"],
                errors="coerce",
            )
            .groupby([frame[column] for column in keys], sort=False)
            .rank(method="average", pct=True)
        )
    groups = (
        frame.groupby(keys, sort=False, observed=True)
        .size()
        .to_numpy(dtype=int)
    )
    if int(groups.sum()) != len(frame):
        raise RuntimeError("ranking groups do not cover the training rows")
    return percentile.fillna(0.0).to_numpy(dtype=float), groups


def _attach_full_universe_rank_target(
    frame: pd.DataFrame,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    result = frame.copy() if copy else frame
    result[TARGET_RANK_COLUMN] = (
        pd.to_numeric(
            result["net_return_pct"],
            errors="coerce",
        )
        .groupby(
            [result["trade_date"], result["signal_slot"]],
            sort=False,
        )
        .rank(method="average", pct=True)
    )
    return result


def _ensure_multitask_targets(
    frame: pd.DataFrame,
    config: V3Config,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    result = frame.copy() if copy else frame
    all_in_returns = pd.to_numeric(result["net_return_pct"], errors="coerce")
    conditional_returns = pd.to_numeric(
        result["conditional_net_return_pct"],
        errors="coerce",
    )
    round_trip = (
        pd.to_numeric(
            result.get("target_entry_fillable"),
            errors="coerce",
        ).eq(1)
        & pd.to_numeric(
            result.get("target_exit_fillable"),
            errors="coerce",
        ).eq(1)
    )
    if "target_conditional_net_positive" not in result:
        result["target_conditional_net_positive"] = np.where(
            round_trip,
            conditional_returns.gt(0).astype("int8"),
            np.nan,
        )
    if "target_conditional_severe_loss" not in result:
        result["target_conditional_severe_loss"] = np.where(
            round_trip,
            conditional_returns.le(
                config.model.severe_loss_threshold_pct
            ).astype("int8"),
            np.nan,
        )
    if "target_severe_loss" not in result:
        result["target_severe_loss"] = (
            all_in_returns
            .le(config.model.severe_loss_threshold_pct)
            .astype("int8")
        )
    rank = pd.to_numeric(
        result[TARGET_RANK_COLUMN],
        errors="coerce",
    )
    result["target_cross_section_top"] = rank.ge(
        1.0 - config.model.cross_section_top_fraction
    ).astype("Int8")
    return result


def _deterministic_training_sample(
    frame: pd.DataFrame,
    *,
    rows_per_slot: int,
    allowed_dates: np.ndarray | list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    identity = frame.loc[:, ["trade_date", "signal_slot", "ts_code"]].copy()
    identity["_sample_hash"] = pd.util.hash_pandas_object(
        identity,
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64)
    identity["_row_position"] = np.arange(len(frame), dtype=np.int64)
    if allowed_dates is not None:
        allowed = {str(value) for value in allowed_dates}
        identity = identity.loc[identity["trade_date"].astype(str).isin(allowed)]
    identity.sort_values(
        ["trade_date", "signal_slot", "_sample_hash", "ts_code"],
        kind="stable",
        inplace=True,
    )
    positions = (
        identity.groupby(
            ["trade_date", "signal_slot"],
            sort=False,
            group_keys=False,
        )
        .head(rows_per_slot)
        ["_row_position"]
        .to_numpy(dtype=np.int64)
    )
    sampled = frame.iloc[positions].copy()
    return sampled.sort_values(
        ["trade_date", "signal_slot", "ts_code"],
        kind="stable",
    ).reset_index(drop=True)


def _group_temporal_weights(
    frame: pd.DataFrame,
    *,
    half_life_days: int,
) -> np.ndarray:
    group_size = frame.groupby(
        ["trade_date", "signal_slot"],
        sort=False,
    )["ts_code"].transform("size").to_numpy(dtype=float)
    temporal = _temporal_weights(frame, half_life_days=half_life_days)
    weights = temporal / np.maximum(group_size, 1.0)
    return weights / max(float(np.mean(weights)), 1e-12)


def _temporal_weights(
    frame: pd.DataFrame,
    *,
    half_life_days: int,
) -> np.ndarray:
    dates = np.array(sorted(frame["trade_date"].astype(str).unique()))
    age = {date: len(dates) - 1 - index for index, date in enumerate(dates)}
    values = frame["trade_date"].astype(str).map(age).to_numpy(dtype=float)
    weights = np.exp(-np.log(2.0) * values / max(half_life_days, 1))
    return weights / max(float(np.mean(weights)), 1e-12)


def _predict_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)


def _group_percentile_rank(
    values: np.ndarray,
    frame: pd.DataFrame,
) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float), index=frame.index)
    group_columns = [
        column
        for column in ("trade_date", "signal_slot")
        if column in frame
    ]
    if group_columns:
        ranked = series.groupby(
            [frame[column] for column in group_columns],
            sort=False,
        ).rank(method="average", pct=True)
    else:
        ranked = series.rank(method="average", pct=True)
    return ranked.fillna(0.0).to_numpy(dtype=float)


def _build_calibration_table(
    probability: np.ndarray,
    target: np.ndarray,
    trade_dates: np.ndarray,
    *,
    seed: int,
    bins: int = 10,
) -> list[dict[str, float | int]]:
    table: list[dict[str, float | int]] = []
    for index in range(bins):
        left = index / bins
        right = (index + 1) / bins
        mask = (probability >= left) & (
            probability < right if index < bins - 1 else probability <= right
        )
        count = int(mask.sum())
        wins = int(target[mask].sum()) if count else 0
        clustered_days, clustered_lower = clustered_binary_lower(
            target[mask],
            trade_dates[mask],
            seed=seed + index,
        )
        table.append(
            {
                "left": left,
                "right": right,
                "count": count,
                "days": clustered_days,
                "win_rate": wins / count if count else 0.0,
                "wilson_lower": _wilson_lower(wins, count),
                "clustered_lower": clustered_lower,
            }
        )
    return table


def _training_data_digest(frame: pd.DataFrame) -> str:
    identity_columns = [
        "trade_date",
        "signal_slot",
        "ts_code",
        "signal_price",
        "target_entry_fillable",
        "target_exit_fillable",
        "target_net_positive",
        "target_conditional_net_positive",
        "target_cross_section_top",
        "target_severe_loss",
        "target_conditional_severe_loss",
        "conditional_net_return_pct",
        "net_return_pct",
    ]
    digest = hashlib.sha256()
    chunk_rows = 25_000
    for start in range(0, len(frame), chunk_rows):
        chunk = frame.iloc[start : start + chunk_rows]
        digest_frame = pd.concat(
            [
                chunk.reindex(columns=identity_columns).reset_index(drop=True),
                feature_matrix(chunk).reset_index(drop=True),
            ],
            axis=1,
        )
        row_hashes = pd.util.hash_pandas_object(
            digest_frame,
            index=False,
            categorize=True,
        ).to_numpy(dtype=np.uint64)
        digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _learned_policy_fingerprint(
    contract_fingerprint: str,
    policy: CandidatePolicy,
) -> str:
    return _digest(
        {
            "contract_fingerprint": contract_fingerprint,
            "candidate_policy": asdict(policy),
        }
    )


def _load_lightgbm():
    try:
        import lightgbm

        return lightgbm
    except (ImportError, OSError):
        return None


def _minimum_leaf(rows: int) -> int:
    return max(20, min(150, rows // 750))


def _beta_features(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-5, 1 - 1e-5)
    return np.column_stack((np.log(clipped), -np.log1p(-clipped)))


def _wilson_lower(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = proportion + z**2 / (2 * total)
    margin = z * np.sqrt(
        (proportion * (1 - proportion) + z**2 / (4 * total)) / total
    )
    return float((centre - margin) / denominator)


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:20]
