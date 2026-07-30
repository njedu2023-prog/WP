from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, feature_matrix, slot_to_minute
from .meta_alpha import ProbabilityCalibrator


@dataclass
class ExitFailureRiskBundle:
    tree: HistGradientBoostingClassifier
    linear: Pipeline
    calibrator: ProbabilityCalibrator
    failure_weight: float
    feature_columns: tuple[str, ...]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        scored = frame.copy()
        features = _exit_risk_feature_matrix(scored).loc[
            :,
            self.feature_columns,
        ]
        raw = _blended_failure_probability(
            self.tree,
            self.linear,
            features,
        )
        scored["risk_p_exit_failure"] = np.clip(
            self.calibrator.predict(raw),
            0.0001,
            0.9999,
        )
        scored["risk_p_exit_safe"] = 1.0 - scored["risk_p_exit_failure"]
        scored["risk_failure_rank_pct"] = scored.groupby(
            ["trade_date", "signal_slot"],
            sort=False,
        )["risk_p_exit_failure"].rank(
            method="average",
            pct=True,
            ascending=True,
        )
        return scored


def fit_exit_failure_risk(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    random_seed: int,
) -> ExitFailureRiskBundle:
    if len(train) < 5_000 or len(calibration) < 500:
        raise ValueError("insufficient rows for exit-failure risk model")
    train_target = exit_failure_target(train)
    calibration_target = exit_failure_target(calibration)
    if train_target.nunique() < 2 or calibration_target.nunique() < 2:
        raise ValueError("exit-failure risk windows require both classes")
    train_failures = int(train_target.sum())
    calibration_failures = int(calibration_target.sum())
    if train_failures < 20 or calibration_failures < 5:
        raise ValueError(
            "exit-failure risk windows contain too few observed failures"
        )

    train_feature_matrix = _exit_risk_feature_matrix(train)
    feature_columns = tuple(
        column
        for column in FEATURE_COLUMNS
        if train_feature_matrix[column].notna().sum() >= 100
        and train_feature_matrix[column].nunique(dropna=True) >= 2
    )
    if not feature_columns:
        raise ValueError("exit-failure risk window has no usable features")
    train_features = train_feature_matrix.loc[:, feature_columns]
    calibration_features = _exit_risk_feature_matrix(calibration).loc[
        :,
        feature_columns,
    ]
    train_day_weight = day_equal_weights(train)
    calibration_weight = day_equal_weights(calibration)
    failure_weight = float(
        np.clip(
            np.sqrt(
                max(len(train_target) - train_failures, 1)
                / max(train_failures, 1)
            ),
            4.0,
            16.0,
        )
    )
    train_target_values = train_target.to_numpy(dtype=int)
    fit_weight = train_day_weight * np.where(
        train_target_values == 1,
        failure_weight,
        1.0,
    )

    tree = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=12.0,
        random_state=random_seed,
    )
    tree.fit(
        train_features,
        train_target_values,
        sample_weight=fit_weight,
    )
    linear = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.10,
                    max_iter=2_000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    linear.fit(
        train_features,
        train_target_values,
        model__sample_weight=fit_weight,
    )
    raw_calibration = _blended_failure_probability(
        tree,
        linear,
        calibration_features,
    )
    calibrator = ProbabilityCalibrator().fit(
        raw_calibration,
        calibration_target.to_numpy(dtype=int),
        calibration_weight,
    )
    return ExitFailureRiskBundle(
        tree=tree,
        linear=linear,
        calibrator=calibrator,
        failure_weight=failure_weight,
        feature_columns=feature_columns,
    )


def exit_failure_target(frame: pd.DataFrame) -> pd.Series:
    if "exit_fillable" not in frame:
        raise ValueError("exit_fillable is required for exit-risk training")
    values = frame["exit_fillable"]
    if values.dtype == bool:
        fillable = values.fillna(False)
    else:
        fillable = values.astype(str).str.strip().str.lower().isin(
            {"1", "true", "yes", "y"}
        )
    return (~fillable).astype("int8")


def day_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    dates = frame["trade_date"].astype(str)
    counts = dates.groupby(dates, sort=False).transform("size")
    weights = 1.0 / pd.to_numeric(counts, errors="coerce").clip(lower=1.0)
    values = weights.to_numpy(dtype=float)
    return values * (len(values) / values.sum())


def _blended_failure_probability(
    tree: HistGradientBoostingClassifier,
    linear: Pipeline,
    features: pd.DataFrame,
) -> np.ndarray:
    return (
        0.70 * tree.predict_proba(features)[:, 1]
        + 0.30 * linear.predict_proba(features)[:, 1]
    )


def _exit_risk_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame
    if "slot_minute" not in frame and "signal_slot" in frame:
        prepared = frame.copy()
        prepared["slot_minute"] = slot_to_minute(prepared["signal_slot"])
    return feature_matrix(prepared)


def feature_contract() -> tuple[str, ...]:
    return FEATURE_COLUMNS
