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
from sklearn.preprocessing import RobustScaler

from .contracts import V3Config, policy_fingerprint
from .features import FEATURE_COLUMNS, feature_matrix
from .statistics import clustered_binary_lower


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
        logits = _logit(probability).reshape(-1, 1)
        self.model = LogisticRegression(C=1.0, max_iter=1_000, solver="lbfgs")
        self.model.fit(logits, target)
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
    logistic: Pipeline
    gradient_boosting: Pipeline

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        p_logistic = self.logistic.predict_proba(features)[:, 1]
        p_boosting = self.gradient_boosting.predict_proba(features)[:, 1]
        return 0.45 * p_logistic + 0.55 * p_boosting


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
    training_rows: int
    eligible_fit_rows: int
    calibration_rows: int
    training_data_digest: str
    positive_rate: float
    feature_columns: tuple[str, ...]
    members: list[ClassifierMember]
    calibrator: PlattCalibrator
    mean_regressor: Pipeline
    downside_regressor: Pipeline
    calibration_table: list[dict[str, float | int]]
    probability_threshold: float
    probability_lower_threshold: float
    min_expected_net_return_pct: float
    min_downside_q10_pct: float
    min_calibration_bin_samples: int
    min_calibration_bin_days: int
    min_calibration_bin_wilson_lower: float
    min_calibration_bin_clustered_lower: float
    max_probability_model_spread: float
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
    eligible = eligible.sort_values(["trade_date", "signal_slot", "ts_code"], kind="stable")
    training_data_digest = _training_data_digest(eligible)
    contract_fingerprint = policy_fingerprint(config)
    if not allow_below_minimum and len(eligible) < config.model.min_train_rows:
        raise ValueError(
            f"V3 training requires {config.model.min_train_rows:,} eligible labelled rows; "
            f"received {len(eligible):,}"
        )
    unique_dates = np.array(sorted(eligible["trade_date"].unique()))
    minimum_dates = config.model.calibration_days + config.model.purge_days + 20
    if len(unique_dates) < minimum_dates:
        raise ValueError(
            f"insufficient temporal depth: {len(unique_dates)} dates, need at least {minimum_dates}"
        )

    calibration_dates = unique_dates[-config.model.calibration_days :]
    fit_end_index = len(unique_dates) - config.model.calibration_days - config.model.purge_days
    fit_dates = unique_dates[:fit_end_index]
    if (
        not allow_below_minimum
        and len(fit_dates) < config.model.minimum_train_days
    ):
        raise ValueError(
            f"training fit window has {len(fit_dates)} trade days; "
            f"requires {config.model.minimum_train_days}"
        )
    fit = eligible.loc[eligible["trade_date"].isin(fit_dates)].copy()
    calibration = eligible.loc[eligible["trade_date"].isin(calibration_dates)].copy()
    sampled_fit = _deterministic_training_sample(
        fit,
        rows_per_slot=config.model.max_training_rows_per_slot,
    )
    y_fit = sampled_fit["target_net_positive"].astype(int).to_numpy()
    y_calibration = calibration["target_net_positive"].astype(int).to_numpy()
    if len(np.unique(y_fit)) < 2:
        raise ValueError("training sample contains only one target class")

    members: list[ClassifierMember] = []
    minimum_member_rows = min(500, max(200, config.model.min_train_rows // 20))
    fitted_window_lengths: set[int] = set()
    for window_days in config.model.ensemble_windows_days:
        member_dates = fit_dates[-min(window_days, len(fit_dates)) :]
        if len(member_dates) in fitted_window_lengths:
            continue
        member_frame = sampled_fit.loc[
            sampled_fit["trade_date"].isin(member_dates)
        ]
        if (
            len(member_frame) < minimum_member_rows
            or member_frame["target_net_positive"].nunique() < 2
        ):
            continue
        x_member = feature_matrix(member_frame)
        y_member = member_frame["target_net_positive"].astype(int).to_numpy()
        members.append(
            ClassifierMember(
                name=f"temporal_{len(member_dates)}d",
                window_days=int(len(member_dates)),
                train_start=str(member_dates[0]),
                train_end=str(member_dates[-1]),
                logistic=_fit_logistic(x_member, y_member, config.model.random_seed),
                gradient_boosting=_fit_classifier(
                    x_member,
                    y_member,
                    config.model.random_seed + int(window_days),
                ),
            )
        )
        fitted_window_lengths.add(len(member_dates))
    if len(members) < 2:
        raise ValueError(
            "temporal ensemble requires at least two distinct trained windows"
        )

    x_calibration = feature_matrix(calibration)
    raw_calibration, _ = _member_predictions(members, x_calibration)
    calibrator = PlattCalibrator().fit(raw_calibration, y_calibration)
    calibrated_probability = calibrator.predict(raw_calibration)
    calibration_table = _build_calibration_table(
        calibrated_probability,
        y_calibration,
        calibration["trade_date"].astype(str).to_numpy(),
        seed=config.model.random_seed,
    )

    x_fit = feature_matrix(sampled_fit)
    net_return = pd.to_numeric(
        sampled_fit["net_return_pct"],
        errors="coerce",
    ).fillna(-10.0)
    mean_regressor = _fit_regressor(
        x_fit,
        net_return.to_numpy(),
        config.model.random_seed + 101,
        loss="squared_error",
    )
    downside_regressor = _fit_regressor(
        x_fit,
        net_return.to_numpy(),
        config.model.random_seed + 202,
        loss="quantile",
        quantile=0.10,
    )

    version = model_version or f"wpv3-{fit_dates[-1]}-{len(eligible)}"
    metadata = {
        "strategy_id": config.strategy.strategy_id,
        "policy_fingerprint": contract_fingerprint,
        "model_version": version,
        "feature_version": config.model.feature_version,
        "train_start": str(fit_dates[0]),
        "train_end": str(fit_dates[-1]),
        "calibration_start": str(calibration_dates[0]),
        "calibration_end": str(calibration_dates[-1]),
        "rows": int(len(sampled_fit)),
        "eligible_fit_rows": int(len(fit)),
        "calibration_rows": int(len(calibration)),
        "training_data_digest": training_data_digest,
        "members": [member.name for member in members],
        "features": FEATURE_COLUMNS,
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
        training_rows=int(len(sampled_fit)),
        eligible_fit_rows=int(len(fit)),
        calibration_rows=int(len(calibration)),
        training_data_digest=training_data_digest,
        positive_rate=float(np.mean(y_fit)),
        feature_columns=FEATURE_COLUMNS,
        members=members,
        calibrator=calibrator,
        mean_regressor=mean_regressor,
        downside_regressor=downside_regressor,
        calibration_table=calibration_table,
        probability_threshold=config.model.probability_threshold,
        probability_lower_threshold=config.model.probability_lower_threshold,
        min_expected_net_return_pct=config.model.min_expected_net_return_pct,
        min_downside_q10_pct=config.model.min_downside_q10_pct,
        min_calibration_bin_samples=config.model.min_calibration_bin_samples,
        min_calibration_bin_days=config.model.min_calibration_bin_days,
        min_calibration_bin_wilson_lower=config.model.min_calibration_bin_wilson_lower,
        min_calibration_bin_clustered_lower=(
            config.model.min_calibration_bin_clustered_lower
        ),
        max_probability_model_spread=config.model.max_probability_model_spread,
        max_market_data_age_seconds=config.execution.max_market_data_age_seconds,
        fingerprint=fingerprint,
    )


def predict_bundle(bundle: ModelBundle, frame: pd.DataFrame) -> pd.DataFrame:
    features = feature_matrix(frame)
    raw_probability, member_matrix = _member_predictions(bundle.members, features)
    calibrated = bundle.calibrator.predict(raw_probability)
    calibrated_members = np.column_stack(
        [bundle.calibrator.predict(member_matrix[:, index]) for index in range(member_matrix.shape[1])]
    )
    lower = np.quantile(calibrated_members, 0.10, axis=1)
    spread = np.std(calibrated_members, axis=1)
    expected = bundle.mean_regressor.predict(features)
    downside = bundle.downside_regressor.predict(features)

    result = frame.copy()
    result["model_version"] = bundle.model_version
    result["model_fingerprint"] = bundle.fingerprint
    result["policy_fingerprint"] = bundle.policy_fingerprint
    result["p_net_positive_raw"] = raw_probability
    result["p_net_positive"] = calibrated
    result["p_net_positive_lower"] = lower
    result["probability_model_spread"] = spread
    (
        bin_count,
        bin_days,
        bin_rate,
        bin_lower,
        bin_clustered_lower,
    ) = _calibration_bin_evidence(
        calibrated,
        bundle.calibration_table,
    )
    result["calibration_bin_count"] = bin_count
    result["calibration_bin_days"] = bin_days
    result["calibration_bin_win_rate"] = bin_rate
    result["calibration_bin_wilson_lower"] = bin_lower
    result["calibration_bin_clustered_lower"] = bin_clustered_lower
    result["expected_net_return_pct"] = expected
    result["downside_q10_pct"] = downside
    result["passes_probability"] = calibrated >= bundle.probability_threshold
    result["passes_probability_lower"] = lower >= bundle.probability_lower_threshold
    result["passes_expected_return"] = expected >= bundle.min_expected_net_return_pct
    result["passes_downside"] = downside >= bundle.min_downside_q10_pct
    result["passes_sample"] = (
        (bin_count >= bundle.min_calibration_bin_samples)
        & (bin_days >= bundle.min_calibration_bin_days)
    )
    result["passes_empirical_lower"] = (
        bin_lower >= bundle.min_calibration_bin_wilson_lower
    ) & (
        bin_clustered_lower >= bundle.min_calibration_bin_clustered_lower
    )
    result["passes_stability"] = spread <= bundle.max_probability_model_spread
    data_age = pd.to_numeric(
        result.get("data_age_seconds", pd.Series(0.0, index=result.index)),
        errors="coerce",
    )
    result["passes_freshness"] = data_age.le(bundle.max_market_data_age_seconds)
    execution = result.get("execution_eligible", pd.Series(True, index=result.index)).fillna(False)
    result["passes_policy"] = (
        execution.astype(bool)
        & result["passes_probability"]
        & result["passes_probability_lower"]
        & result["passes_expected_return"]
        & result["passes_downside"]
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
        raise TypeError("model artifact is not a WP V3 ModelBundle")
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


def _fit_logistic(features: pd.DataFrame, target: np.ndarray, seed: int) -> Pipeline:
    pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
            (
                "model",
                LogisticRegression(
                    C=0.25,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return pipeline.fit(features, target)


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


def _fit_classifier(features: pd.DataFrame, target: np.ndarray, seed: int) -> Pipeline:
    minimum_leaf = max(25, min(100, len(features) // 200))
    pipeline = Pipeline(
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
                    learning_rate=0.045,
                    max_iter=220,
                    max_leaf_nodes=15,
                    min_samples_leaf=minimum_leaf,
                    l2_regularization=2.0,
                    early_stopping=True,
                    validation_fraction=0.12,
                    random_state=seed,
                ),
            ),
        ]
    )
    return pipeline.fit(features, target)


def _fit_regressor(
    features: pd.DataFrame,
    target: np.ndarray,
    seed: int,
    *,
    loss: str,
    quantile: float | None = None,
) -> Pipeline:
    kwargs: dict[str, Any] = {"loss": loss}
    if quantile is not None:
        kwargs["quantile"] = quantile
    minimum_leaf = max(25, min(100, len(features) // 200))
    pipeline = Pipeline(
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
                    learning_rate=0.045,
                    max_iter=180,
                    max_leaf_nodes=15,
                    min_samples_leaf=minimum_leaf,
                    l2_regularization=2.0,
                    early_stopping=True,
                    validation_fraction=0.12,
                    random_state=seed,
                    **kwargs,
                ),
            ),
        ]
    )
    return pipeline.fit(features, target)


def _member_predictions(
    members: list[ClassifierMember],
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.column_stack([member.predict(features) for member in members])
    return np.mean(matrix, axis=1), matrix


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


def _wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = proportion + z**2 / (2 * total)
    margin = z * sqrt((proportion * (1 - proportion) + z**2 / (4 * total)) / total)
    return (centre - margin) / denominator
