from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .contracts import V3Config, policy_fingerprint
from .dataset import first_crossing_candidates
from .features import FEATURE_COLUMNS, feature_matrix
from .statistics import clustered_binary_lower, day_clustered_intervals


TARGET_RANK_COLUMN = "_target_net_return_rank"


@dataclass
class PlattCalibrator:
    model: LogisticRegression | None = None
    constant: float | None = None

    def fit(self, probability: np.ndarray, target: np.ndarray) -> "PlattCalibrator":
        probability = np.asarray(probability, dtype=float)
        target = np.asarray(target, dtype=int)
        if len(np.unique(target)) < 2:
            self.constant = float(np.mean(target))
            self.model = None
            return self
        self.model = LogisticRegression(C=1.0, max_iter=1_000, solver="lbfgs")
        self.model.fit(_logit(probability).reshape(-1, 1), target)
        self.constant = None
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        probability = np.asarray(probability, dtype=float)
        if self.model is None:
            if self.constant is None:
                return np.clip(probability, 1e-4, 1 - 1e-4)
            return np.full(len(probability), self.constant, dtype=float)
        return self.model.predict_proba(_logit(probability).reshape(-1, 1))[:, 1]


@dataclass
class ClassifierMember:
    name: str
    window_days: int
    train_start: str
    train_end: str
    classifier: Pipeline
    ranker: Pipeline

    def predict_probability(self, features: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(features)[:, 1]

    def predict_rank_score(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.ranker.predict(features), dtype=float)


@dataclass
class ModelBundle:
    strategy_id: str
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
    evidence_rows: int
    training_data_digest: str
    positive_rate: float
    feature_columns: tuple[str, ...]
    members: list[ClassifierMember]
    calibrator: PlattCalibrator
    mean_regressor: Pipeline
    downside_regressor: Pipeline
    calibration_table: list[dict[str, float | int]]
    selection_evidence: dict[str, Any]
    probability_threshold: float
    probability_lower_threshold: float
    min_expected_net_return_pct: float
    min_downside_q10_pct: float
    minimum_selection_rank_percentile: float
    min_calibration_bin_samples: int
    min_calibration_bin_days: int
    min_calibration_bin_wilson_lower: float
    min_calibration_bin_clustered_lower: float
    max_probability_model_spread: float
    max_selection_rank_spread: float
    max_market_data_age_seconds: int
    fingerprint: str


def train_bundle(
    panel: pd.DataFrame,
    config: V3Config,
    *,
    allow_below_minimum: bool = False,
    model_version: str | None = None,
) -> ModelBundle:
    eligible = panel.loc[
        panel["label_available"].fillna(False)
        & panel["execution_eligible"].fillna(False)
        & pd.to_numeric(panel["target_net_positive"], errors="coerce").notna()
    ].copy()
    eligible["trade_date"] = eligible["trade_date"].astype(str)
    eligible = eligible.sort_values(
        ["trade_date", "signal_slot", "ts_code"],
        kind="stable",
    )
    eligible = _attach_full_universe_rank_target(eligible)
    training_data_digest = _training_data_digest(eligible)
    contract_fingerprint = policy_fingerprint(config)
    if not allow_below_minimum and len(eligible) < config.model.min_train_rows:
        raise ValueError(
            f"V4 training requires {config.model.min_train_rows:,} eligible labelled rows; "
            f"received {len(eligible):,}"
        )

    unique_dates = np.array(sorted(eligible["trade_date"].unique()))
    minimum_dates = config.model.calibration_days + config.model.purge_days + 20
    if len(unique_dates) < minimum_dates:
        raise ValueError(
            f"insufficient temporal depth: {len(unique_dates)} dates, "
            f"need at least {minimum_dates}"
        )
    calibration_dates = unique_dates[-config.model.calibration_days :]
    calibration_split = max(1, len(calibration_dates) // 2)
    calibration_fit_dates = calibration_dates[:calibration_split]
    evidence_dates = calibration_dates[calibration_split:]
    if len(evidence_dates) < 1:
        raise ValueError("calibration contract requires a distinct policy-evidence period")

    fit_end_index = (
        len(unique_dates)
        - config.model.calibration_days
        - config.model.purge_days
    )
    fit_dates = unique_dates[:fit_end_index]
    if not allow_below_minimum and len(fit_dates) < config.model.minimum_train_days:
        raise ValueError(
            f"training fit window has {len(fit_dates)} trade days; "
            f"requires {config.model.minimum_train_days}"
        )
    fit = eligible.loc[eligible["trade_date"].isin(fit_dates)].copy()
    calibration_fit = eligible.loc[
        eligible["trade_date"].isin(calibration_fit_dates)
    ].copy()
    evidence = eligible.loc[eligible["trade_date"].isin(evidence_dates)].copy()
    sampled_fit = _deterministic_training_sample(
        fit,
        rows_per_slot=config.model.max_training_rows_per_slot,
    )
    y_fit = sampled_fit["target_net_positive"].astype(int).to_numpy()
    y_calibration = calibration_fit["target_net_positive"].astype(int).to_numpy()
    if len(np.unique(y_fit)) < 2:
        raise ValueError("training sample contains only one target class")

    members: list[ClassifierMember] = []
    minimum_member_rows = min(2_000, max(100, config.model.min_train_rows // 20))
    fitted_window_lengths: set[int] = set()
    for window_days in config.model.ensemble_windows_days:
        member_dates = fit_dates[-min(window_days, len(fit_dates)) :]
        if len(member_dates) in fitted_window_lengths:
            continue
        member_frame = sampled_fit.loc[
            sampled_fit["trade_date"].isin(member_dates)
        ].copy()
        if (
            len(member_frame) < minimum_member_rows
            or member_frame["target_net_positive"].nunique() < 2
        ):
            continue
        member_frame = member_frame.sort_values(
            ["trade_date", "signal_slot", "ts_code"],
            kind="stable",
        )
        x_member = feature_matrix(member_frame)
        y_member = member_frame["target_net_positive"].astype(int).to_numpy()
        rank_target, rank_groups = _ranking_target_and_groups(member_frame)
        members.append(
            ClassifierMember(
                name=f"tail_rank_{len(member_dates)}d",
                window_days=int(len(member_dates)),
                train_start=str(member_dates[0]),
                train_end=str(member_dates[-1]),
                classifier=_fit_classifier(
                    x_member,
                    y_member,
                    config.model.random_seed + int(window_days),
                ),
                ranker=_fit_ranker(
                    x_member,
                    rank_target,
                    rank_groups,
                    config.model.random_seed + 10_000 + int(window_days),
                ),
            )
        )
        fitted_window_lengths.add(len(member_dates))
    if len(members) < 2:
        raise ValueError("temporal ensemble requires at least two trained windows")

    x_calibration = feature_matrix(calibration_fit)
    raw_calibration, _ = _member_probability_predictions(members, x_calibration)
    calibrator = PlattCalibrator().fit(raw_calibration, y_calibration)

    x_fit = feature_matrix(sampled_fit)
    net_return = pd.to_numeric(
        sampled_fit["net_return_pct"],
        errors="coerce",
    ).fillna(-10.0).clip(-15.0, 15.0)
    mean_regressor = _fit_regressor(
        x_fit,
        net_return.to_numpy(),
        config.model.random_seed + 101,
        objective="absolute_error",
    )
    downside_regressor = _fit_regressor(
        x_fit,
        net_return.to_numpy(),
        config.model.random_seed + 202,
        objective="quantile",
        alpha=0.10,
    )

    evidence_prediction = _score_frame(
        evidence,
        members=members,
        calibrator=calibrator,
        mean_regressor=mean_regressor,
        downside_regressor=downside_regressor,
    )
    calibration_table = _build_calibration_table(
        pd.to_numeric(
            evidence_prediction["p_net_positive"],
            errors="coerce",
        ).to_numpy(),
        evidence_prediction["target_net_positive"].astype(int).to_numpy(),
        evidence_prediction["trade_date"].astype(str).to_numpy(),
        seed=config.model.random_seed,
    )
    selection_evidence = _build_selection_evidence(
        evidence_prediction,
        config,
    )

    version = model_version or f"wpv4-{fit_dates[-1]}-{len(eligible)}"
    metadata = {
        "strategy_id": config.strategy.strategy_id,
        "policy_fingerprint": contract_fingerprint,
        "model_version": version,
        "feature_version": config.model.feature_version,
        "train_start": str(fit_dates[0]),
        "train_end": str(fit_dates[-1]),
        "calibration_start": str(calibration_dates[0]),
        "calibration_end": str(calibration_dates[-1]),
        "calibration_fit_end": str(calibration_fit_dates[-1]),
        "evidence_start": str(evidence_dates[0]),
        "rows": int(len(sampled_fit)),
        "eligible_fit_rows": int(len(fit)),
        "calibration_rows": int(len(calibration_fit)),
        "evidence_rows": int(len(evidence)),
        "training_data_digest": training_data_digest,
        "members": [member.name for member in members],
        "features": FEATURE_COLUMNS,
        "selection_evidence": selection_evidence,
    }
    fingerprint = hashlib.sha256(
        json.dumps(metadata, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    return ModelBundle(
        strategy_id=config.strategy.strategy_id,
        policy_fingerprint=contract_fingerprint,
        model_version=version,
        feature_version=config.model.feature_version,
        trained_at=datetime.now(timezone.utc).isoformat(),
        train_start=str(fit_dates[0]),
        train_end=str(fit_dates[-1]),
        calibration_start=str(calibration_dates[0]),
        calibration_end=str(calibration_dates[-1]),
        calibration_fit_end=str(calibration_fit_dates[-1]),
        evidence_start=str(evidence_dates[0]),
        training_rows=int(len(sampled_fit)),
        eligible_fit_rows=int(len(fit)),
        calibration_rows=int(len(calibration_fit)),
        evidence_rows=int(len(evidence)),
        training_data_digest=training_data_digest,
        positive_rate=float(np.mean(y_fit)),
        feature_columns=FEATURE_COLUMNS,
        members=members,
        calibrator=calibrator,
        mean_regressor=mean_regressor,
        downside_regressor=downside_regressor,
        calibration_table=calibration_table,
        selection_evidence=selection_evidence,
        probability_threshold=config.model.probability_threshold,
        probability_lower_threshold=config.model.probability_lower_threshold,
        min_expected_net_return_pct=config.model.min_expected_net_return_pct,
        min_downside_q10_pct=config.model.min_downside_q10_pct,
        minimum_selection_rank_percentile=(
            config.model.minimum_selection_rank_percentile
        ),
        min_calibration_bin_samples=config.model.min_calibration_bin_samples,
        min_calibration_bin_days=config.model.min_calibration_bin_days,
        min_calibration_bin_wilson_lower=(
            config.model.min_calibration_bin_wilson_lower
        ),
        min_calibration_bin_clustered_lower=(
            config.model.min_calibration_bin_clustered_lower
        ),
        max_probability_model_spread=config.model.max_probability_model_spread,
        max_selection_rank_spread=config.model.max_selection_rank_spread,
        max_market_data_age_seconds=config.execution.max_market_data_age_seconds,
        fingerprint=fingerprint,
    )


def predict_bundle(bundle: ModelBundle, frame: pd.DataFrame) -> pd.DataFrame:
    result = _score_frame(
        frame,
        members=bundle.members,
        calibrator=bundle.calibrator,
        mean_regressor=bundle.mean_regressor,
        downside_regressor=bundle.downside_regressor,
    )
    result["model_version"] = bundle.model_version
    result["model_fingerprint"] = bundle.fingerprint
    result["policy_fingerprint"] = bundle.policy_fingerprint
    (
        bin_count,
        bin_days,
        bin_rate,
        bin_lower,
        bin_clustered_lower,
    ) = _calibration_bin_evidence(
        pd.to_numeric(result["p_net_positive"], errors="coerce").to_numpy(),
        bundle.calibration_table,
    )
    result["calibration_bin_count"] = bin_count
    result["calibration_bin_days"] = bin_days
    result["calibration_bin_win_rate"] = bin_rate
    result["calibration_bin_wilson_lower"] = bin_lower
    result["calibration_bin_clustered_lower"] = bin_clustered_lower

    result["passes_probability"] = (
        result["p_net_positive"] >= bundle.probability_threshold
    )
    result["passes_probability_lower"] = (
        result["p_net_positive_lower"] >= bundle.probability_lower_threshold
    )
    result["passes_expected_return"] = (
        result["expected_net_return_pct"] >= bundle.min_expected_net_return_pct
    )
    result["passes_downside"] = (
        result["downside_q10_pct"] >= bundle.min_downside_q10_pct
    )
    result["passes_selection_rank"] = (
        result["selection_rank_pct"]
        >= bundle.minimum_selection_rank_percentile
    )
    result["passes_stability"] = (
        (result["probability_model_spread"] <= bundle.max_probability_model_spread)
        & (
            result["selection_rank_spread"]
            <= bundle.max_selection_rank_spread
        )
    )
    evidence = bundle.selection_evidence
    sample_pass = (
        int(evidence.get("candidate_events", 0))
        >= bundle.min_calibration_bin_samples
        and int(evidence.get("candidate_days", 0))
        >= bundle.min_calibration_bin_days
    )
    empirical_pass = (
        float(evidence.get("win_rate_wilson_lower", 0.0) or 0.0)
        >= bundle.min_calibration_bin_wilson_lower
        and float(evidence.get("win_rate_day_clustered_lower", 0.0) or 0.0)
        >= bundle.min_calibration_bin_clustered_lower
        and float(evidence.get("mean_net_return_pct", -999.0) or -999.0) > 0.0
        and float(
            evidence.get("mean_net_return_day_clustered_lower_pct", -999.0)
            if evidence.get("mean_net_return_day_clustered_lower_pct") is not None
            else -999.0
        )
        >= 0.0
    )
    result["selection_evidence_candidate_events"] = int(
        evidence.get("candidate_events", 0)
    )
    result["selection_evidence_candidate_days"] = int(
        evidence.get("candidate_days", 0)
    )
    result["selection_evidence_win_rate"] = float(
        evidence.get("win_rate", 0.0) or 0.0
    )
    result["selection_evidence_mean_net_return_pct"] = float(
        evidence.get("mean_net_return_pct", 0.0) or 0.0
    )
    result["passes_sample"] = sample_pass
    result["passes_empirical_lower"] = empirical_pass

    data_age = pd.to_numeric(
        result.get("data_age_seconds", pd.Series(0.0, index=result.index)),
        errors="coerce",
    )
    result["passes_freshness"] = data_age.le(bundle.max_market_data_age_seconds)
    execution = result.get(
        "execution_eligible",
        pd.Series(True, index=result.index),
    ).fillna(False)
    result["passes_policy"] = (
        execution.astype(bool)
        & result["passes_probability"]
        & result["passes_probability_lower"]
        & result["passes_expected_return"]
        & result["passes_downside"]
        & result["passes_selection_rank"]
        & result["passes_sample"]
        & result["passes_empirical_lower"]
        & result["passes_stability"]
        & result["passes_freshness"]
    )
    return result


def save_bundle(bundle: ModelBundle, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target, compress=3)


def load_bundle(path: str | Path) -> ModelBundle:
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, ModelBundle):
        raise TypeError("model artifact is not a WP V4 ModelBundle")
    return bundle


def bundle_metadata(bundle: ModelBundle) -> dict[str, Any]:
    data = asdict(bundle)
    for key in (
        "members",
        "calibrator",
        "mean_regressor",
        "downside_regressor",
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


def _score_frame(
    frame: pd.DataFrame,
    *,
    members: list[ClassifierMember],
    calibrator: PlattCalibrator,
    mean_regressor: Pipeline,
    downside_regressor: Pipeline,
) -> pd.DataFrame:
    features = feature_matrix(frame)
    raw_probability, probability_members = _member_probability_predictions(
        members,
        features,
    )
    calibrated = calibrator.predict(raw_probability)
    calibrated_members = np.column_stack(
        [
            calibrator.predict(probability_members[:, index])
            for index in range(probability_members.shape[1])
        ]
    )
    lower = np.quantile(calibrated_members, 0.10, axis=1)
    probability_spread = np.std(calibrated_members, axis=1)
    expected = np.asarray(mean_regressor.predict(features), dtype=float)
    downside = np.asarray(downside_regressor.predict(features), dtype=float)

    raw_rank_members = np.column_stack(
        [member.predict_rank_score(features) for member in members]
    )
    rank_members = np.column_stack(
        [
            _group_percentile_rank(raw_rank_members[:, index], frame)
            for index in range(raw_rank_members.shape[1])
        ]
    )
    rank_score = np.mean(rank_members, axis=1)
    rank_spread = np.std(rank_members, axis=1)
    probability_rank = _group_percentile_rank(calibrated, frame)
    expected_rank = _group_percentile_rank(expected, frame)
    downside_rank = _group_percentile_rank(downside, frame)
    selection_score = (
        0.40 * rank_score
        + 0.25 * expected_rank
        + 0.20 * probability_rank
        + 0.15 * downside_rank
    )
    selection_rank = _group_percentile_rank(selection_score, frame)

    result = frame.copy()
    result["p_net_positive_raw"] = raw_probability
    result["p_net_positive"] = calibrated
    result["p_net_positive_lower"] = lower
    result["probability_model_spread"] = probability_spread
    result["expected_net_return_pct"] = expected
    result["downside_q10_pct"] = downside
    result["ranking_score"] = rank_score
    result["selection_score"] = selection_score
    result["selection_rank_pct"] = selection_rank
    result["selection_rank_spread"] = rank_spread
    return result


def _build_selection_evidence(
    prediction: pd.DataFrame,
    config: V3Config,
) -> dict[str, Any]:
    frame = prediction.copy()
    frame["_evidence_pass"] = (
        frame.get(
            "execution_eligible",
            pd.Series(True, index=frame.index),
        ).fillna(False)
        & pd.to_numeric(frame["p_net_positive"], errors="coerce").ge(
            config.model.probability_threshold
        )
        & pd.to_numeric(frame["p_net_positive_lower"], errors="coerce").ge(
            config.model.probability_lower_threshold
        )
        & pd.to_numeric(frame["expected_net_return_pct"], errors="coerce").ge(
            config.model.min_expected_net_return_pct
        )
        & pd.to_numeric(frame["downside_q10_pct"], errors="coerce").ge(
            config.model.min_downside_q10_pct
        )
        & pd.to_numeric(frame["selection_rank_pct"], errors="coerce").ge(
            config.model.minimum_selection_rank_percentile
        )
        & pd.to_numeric(frame["probability_model_spread"], errors="coerce").le(
            config.model.max_probability_model_spread
        )
        & pd.to_numeric(frame["selection_rank_spread"], errors="coerce").le(
            config.model.max_selection_rank_spread
        )
    )
    candidates = first_crossing_candidates(
        frame,
        config,
        status_column="_evidence_pass",
    )
    returns = pd.to_numeric(
        candidates.get("net_return_pct"),
        errors="coerce",
    ).dropna()
    wins = int(returns.gt(0).sum())
    total = int(len(returns))
    candidate_days = int(
        candidates.get("trade_date", pd.Series(dtype=str)).nunique()
    )
    clustered_days, clustered_lower = clustered_binary_lower(
        returns.gt(0).astype(int).to_numpy(),
        candidates.loc[returns.index, "trade_date"].astype(str).to_numpy()
        if total
        else np.array([], dtype=str),
        seed=config.model.random_seed + 30_000,
    )
    intervals = day_clustered_intervals(candidates)
    profits = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    profit_factor = (
        profits / losses
        if losses > 0
        else (float("inf") if profits > 0 else 0.0)
    )
    return {
        "period_start": (
            str(prediction["trade_date"].astype(str).min())
            if not prediction.empty
            else None
        ),
        "period_end": (
            str(prediction["trade_date"].astype(str).max())
            if not prediction.empty
            else None
        ),
        "slot_rows": int(len(prediction)),
        "candidate_events": total,
        "candidate_days": candidate_days,
        "clustered_days": int(clustered_days),
        "wins": wins,
        "win_rate": wins / total if total else 0.0,
        "win_rate_wilson_lower": _wilson_lower(wins, total),
        "win_rate_day_clustered_lower": clustered_lower,
        "mean_net_return_pct": _finite_or_none(returns.mean()) if total else None,
        "mean_net_return_day_clustered_lower_pct": (
            _finite_or_none(intervals.mean_return_lower_pct)
            if intervals.mean_return_lower_pct is not None
            else None
        ),
        "median_net_return_pct": _finite_or_none(returns.median()) if total else None,
        "profit_factor": (
            float(profit_factor) if np.isfinite(profit_factor) else None
        ),
    }


def _deterministic_training_sample(
    frame: pd.DataFrame,
    *,
    rows_per_slot: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    sampled = frame.copy()
    identity = (
        sampled["trade_date"].astype(str)
        + "|"
        + sampled["signal_slot"].astype(str)
        + "|"
        + sampled["ts_code"].astype(str)
    )
    sampled["_sample_hash"] = pd.util.hash_pandas_object(
        identity,
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64)
    sampled = sampled.sort_values(
        ["trade_date", "signal_slot", "_sample_hash", "ts_code"],
        kind="stable",
    )
    sampled = (
        sampled.groupby(
            ["trade_date", "signal_slot"],
            sort=False,
            group_keys=False,
        )
        .head(rows_per_slot)
        .drop(columns="_sample_hash")
    )
    return sampled.reset_index(drop=True)


def _fit_classifier(
    features: pd.DataFrame,
    target: np.ndarray,
    seed: int,
) -> Pipeline:
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
                    max_iter=260,
                    max_leaf_nodes=31,
                    min_samples_leaf=_minimum_leaf(len(features)),
                    l2_regularization=4.0,
                    max_bins=127,
                    early_stopping=True,
                    validation_fraction=0.12,
                    random_state=seed,
                ),
            ),
        ]
    )
    return model.fit(features, target)


def _fit_ranker(
    features: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> Pipeline:
    if int(np.asarray(groups, dtype=int).sum()) != len(features):
        raise ValueError("ranking groups do not cover the feature rows")
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
                    max_iter=260,
                    max_leaf_nodes=31,
                    min_samples_leaf=_minimum_leaf(len(features)),
                    l2_regularization=4.0,
                    max_bins=127,
                    early_stopping=True,
                    validation_fraction=0.12,
                    random_state=seed,
                ),
            ),
        ]
    )
    return model.fit(features, target)


def _fit_regressor(
    features: pd.DataFrame,
    target: np.ndarray,
    seed: int,
    *,
    objective: str,
    alpha: float | None = None,
) -> Pipeline:
    kwargs: dict[str, Any] = {"loss": objective}
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
                    max_leaf_nodes=31,
                    min_samples_leaf=_minimum_leaf(len(features)),
                    l2_regularization=4.0,
                    max_bins=127,
                    early_stopping=True,
                    validation_fraction=0.12,
                    random_state=seed,
                    **kwargs,
                ),
            ),
        ]
    )
    return model.fit(features, target)


def _ranking_target_and_groups(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    keys = ["trade_date", "signal_slot"]
    if TARGET_RANK_COLUMN in frame:
        percentile = pd.to_numeric(
            frame[TARGET_RANK_COLUMN],
            errors="coerce",
        )
    else:
        percentile = (
            pd.to_numeric(frame["net_return_pct"], errors="coerce")
            .groupby([frame[column] for column in keys], sort=False)
            .rank(method="average", pct=True)
        )
    relevance = percentile.fillna(0.0).to_numpy(dtype=float)
    groups = (
        frame.groupby(keys, sort=False, observed=True)
        .size()
        .to_numpy(dtype=int)
    )
    if int(groups.sum()) != len(frame):
        raise RuntimeError("ranking groups do not cover the training rows")
    return relevance, groups


def _attach_full_universe_rank_target(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result[TARGET_RANK_COLUMN] = (
        pd.to_numeric(result["net_return_pct"], errors="coerce")
        .groupby(
            [result["trade_date"], result["signal_slot"]],
            sort=False,
        )
        .rank(method="average", pct=True)
    )
    return result


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


def _member_probability_predictions(
    members: list[ClassifierMember],
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.column_stack(
        [member.predict_probability(features) for member in members]
    )
    return np.mean(matrix, axis=1), matrix


def _minimum_leaf(rows: int) -> int:
    return max(20, min(120, rows // 500))


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped))


def _build_calibration_table(
    probability: np.ndarray,
    target: np.ndarray,
    trade_dates: np.ndarray,
    *,
    seed: int,
    bins: int = 10,
) -> list[dict[str, float | int]]:
    table = []
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


def _calibration_bin_evidence(
    probability: np.ndarray,
    table: list[dict[str, float | int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = np.zeros(len(probability), dtype=int)
    days = np.zeros(len(probability), dtype=int)
    rate = np.zeros(len(probability), dtype=float)
    lower = np.zeros(len(probability), dtype=float)
    clustered_lower = np.zeros(len(probability), dtype=float)
    for index, item in enumerate(table):
        left, right = float(item["left"]), float(item["right"])
        mask = (probability >= left) & (
            probability < right if index < len(table) - 1 else probability <= right
        )
        count[mask] = int(item["count"])
        days[mask] = int(item.get("days", 0))
        rate[mask] = float(item["win_rate"])
        lower[mask] = float(item["wilson_lower"])
        clustered_lower[mask] = float(item.get("clustered_lower", 0.0))
    return count, days, rate, lower, clustered_lower


def _training_data_digest(frame: pd.DataFrame) -> str:
    identity = frame.reindex(
        columns=[
            "trade_date",
            "signal_slot",
            "ts_code",
            "signal_price",
            "target_net_positive",
            "net_return_pct",
        ]
    )
    digest_frame = pd.concat(
        [
            identity.reset_index(drop=True),
            feature_matrix(frame).reset_index(drop=True),
        ],
        axis=1,
    )
    row_hashes = pd.util.hash_pandas_object(
        digest_frame,
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64)
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


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
    margin = z * sqrt(
        (proportion * (1 - proportion) + z**2 / (4 * total)) / total
    )
    return (centre - margin) / denominator


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None
