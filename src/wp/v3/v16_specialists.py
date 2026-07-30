from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .features import enrich_feature_frame
from .meta_alpha import (
    IDENTITY_COLUMNS,
    ProbabilityCalibrator,
    attach_meta_context,
)


COMMON_FEATURES = (
    "slot_minute",
    "ret_from_prev_close_pct",
    "ret_from_open_pct",
    "ret_5m_pct",
    "ret_10m_pct",
    "ret_20m_pct",
    "tail_directional_efficiency",
    "tail_trend_slope_pct",
    "tail_trend_r2",
    "tail_max_drawdown_pct",
    "tail_rebound_from_low_pct",
    "tail_amount_acceleration",
    "tail_volume_price_confirmation",
    "distance_to_up_limit_pct",
    "distance_to_down_limit_pct",
    "relative_market_return_pct",
    "relative_industry_return_pct",
    "tail_relative_market_pct",
    "tail_relative_industry_pct",
    "market_regime_strength",
    "industry_regime_strength",
    "relative_strength_alignment",
    "tail_trend_quality",
    "tail_reversal_pressure",
    "tail_breakout_pressure",
    "risk_adjusted_tail_return",
    "return_cs_rank",
    "tail_return_cs_rank",
    "tail_efficiency_cs_rank",
    "slot_amount_ratio_cs_rank",
    "tail_amount_ratio_cs_rank",
    "volatility_cs_rank",
    "up_limit_distance_cs_rank",
    "p_entry_fill",
    "p_exit_fill_given_entry",
    "p_round_trip_fill_lower",
    "p_net_positive",
    "p_net_positive_lower",
    "p_conditional_net_positive",
    "p_severe_loss",
    "selection_score",
    "selection_rank_pct",
    "expected_utility_pct",
    "expected_utility_lower_pct",
    "downside_q10_pct",
    "probability_model_spread",
    "fill_probability_model_spread",
    "selection_rank_spread",
    "expected_return_model_spread",
    "context_breadth_positive",
    "context_breadth_above_2pct",
    "context_breadth_above_5pct",
    "context_return_dispersion_pct",
    "context_probability_mean",
    "context_probability_dispersion",
    "context_return_change_from_1420_pct",
    "return_context_relative_pct",
    "return_context_zscore",
    "return_rank_pct",
    "probability_rank_pct",
    "utility_rank_pct",
)


@dataclass(frozen=True)
class SpecialistSpec:
    expert_id: str
    description: str
    predicate: Callable[[pd.DataFrame], pd.Series]
    extra_features: tuple[str, ...] = ()

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*COMMON_FEATURES, *self.extra_features)))


@dataclass
class SpecialistBundle:
    spec: SpecialistSpec
    positive_tree: HistGradientBoostingClassifier
    positive_linear: Pipeline
    severe_tree: HistGradientBoostingClassifier | None
    severe_constant: float | None
    return_tree: HistGradientBoostingRegressor
    positive_calibrator: ProbabilityCalibrator
    severe_calibrator: ProbabilityCalibrator
    return_calibrator: Ridge
    feature_columns: tuple[str, ...]
    fit_rows: int
    calibration_rows: int

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = prepare_specialist_frame(frame)
        eligible = self.spec.predicate(prepared).fillna(False).astype(bool)
        scored = prepared.loc[eligible].copy()
        if scored.empty:
            return _empty_prediction_frame()
        features = feature_matrix(scored, self.feature_columns)
        tree_probability = self.positive_tree.predict_proba(features)[:, 1]
        linear_probability = self.positive_linear.predict_proba(features)[:, 1]
        raw_positive = 0.70 * tree_probability + 0.30 * linear_probability
        if self.severe_tree is None:
            raw_severe = np.full(
                len(scored),
                float(self.severe_constant or 0.0),
                dtype=float,
            )
        else:
            raw_severe = self.severe_tree.predict_proba(features)[:, 1]
        raw_return = self.return_tree.predict(features)
        scored["expert_id"] = self.spec.expert_id
        scored["expert_p_positive"] = np.clip(
            self.positive_calibrator.predict(raw_positive),
            0.001,
            0.999,
        )
        scored["expert_p_severe"] = np.clip(
            self.severe_calibrator.predict(raw_severe),
            0.001,
            0.999,
        )
        scored["expert_expected_net_return_pct"] = (
            self.return_calibrator.predict(
                np.asarray(raw_return, dtype=float).reshape(-1, 1)
            )
        )
        scored["expert_score"] = (
            scored["expert_expected_net_return_pct"]
            + 1.25 * (scored["expert_p_positive"] - 0.50)
            - 1.50 * scored["expert_p_severe"]
            - 0.50
            * (
                1.0
                - _numeric(scored, "p_round_trip_fill_lower").clip(0.0, 1.0)
            )
        )
        keep = [
            *IDENTITY_COLUMNS,
            "expert_id",
            "expert_p_positive",
            "expert_p_severe",
            "expert_expected_net_return_pct",
            "expert_score",
        ]
        return scored.loc[:, keep].reset_index(drop=True)


def specialist_specs() -> tuple[SpecialistSpec, ...]:
    return (
        SpecialistSpec(
            "early_structure",
            "14:20-14:35 early tail structure",
            _early_structure,
            (
                "tail_return_from_1400_pct",
                "tail_range_since_1400_pct",
                "tail_close_position_since_1400",
            ),
        ),
        SpecialistSpec(
            "late_confirmation",
            "14:40-14:50 late confirmation",
            _late_confirmation,
            (
                "tail_latest_amount_share",
                "tail_amount_concentration",
                "tail_range_since_1400_pct",
            ),
        ),
        SpecialistSpec(
            "market_industry_leader",
            "positive stock strength versus both market and industry",
            _market_industry_leader,
            (
                "industry_return_pct",
                "industry_breadth",
                "industry_tail_return_pct",
                "industry_tail_breadth",
                "market_return_pct",
                "market_breadth",
            ),
        ),
        SpecialistSpec(
            "trend_persistence",
            "positive and efficient 14:00-tail trend",
            _trend_persistence,
            (
                "tail_up_bar_share",
                "tail_down_bar_share",
                "tail_realized_volatility_pct",
                "tail_mean_abs_return_pct",
            ),
        ),
        SpecialistSpec(
            "pullback_recovery",
            "intraday pullback followed by causal recovery",
            _pullback_recovery,
            (
                "bar_lower_wick_pct",
                "tail_close_position_10m",
                "tail_reversal_pressure",
                "prev_5d_return_pct",
                "prev_20d_drawdown_pct",
            ),
        ),
        SpecialistSpec(
            "liquidity_breakout",
            "price move confirmed by accelerating executable liquidity",
            _liquidity_breakout,
            (
                "slot_amount_log",
                "slot_amount_ratio_20d",
                "tail_cumulative_amount_ratio_20d",
                "tail_latest_amount_share",
                "tail_amount_concentration",
                "liquidity_acceleration",
            ),
        ),
    )


def prepare_specialist_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = enrich_feature_frame(frame)
    return attach_meta_context(prepared)


def fit_specialist(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    spec: SpecialistSpec,
    *,
    random_seed: int,
    minimum_train_rows: int = 1_500,
    minimum_calibration_rows: int = 300,
) -> SpecialistBundle:
    prepared_train = prepare_specialist_frame(train)
    prepared_calibration = prepare_specialist_frame(calibration)
    prepared_train = _labeled_rows(prepared_train)
    prepared_calibration = _labeled_rows(prepared_calibration)
    train_mask = spec.predicate(prepared_train).fillna(False).astype(bool)
    calibration_mask = (
        spec.predicate(prepared_calibration).fillna(False).astype(bool)
    )
    fit = prepared_train.loc[train_mask].copy()
    calibrate = prepared_calibration.loc[calibration_mask].copy()
    if len(fit) < minimum_train_rows:
        raise ValueError(
            f"{spec.expert_id} has only {len(fit)} train rows; "
            f"requires {minimum_train_rows}"
        )
    if len(calibrate) < minimum_calibration_rows:
        raise ValueError(
            f"{spec.expert_id} has only {len(calibrate)} calibration rows; "
            f"requires {minimum_calibration_rows}"
        )
    features = active_feature_columns(fit, calibrate, spec.feature_columns)
    x_fit = feature_matrix(fit, features)
    x_calibrate = feature_matrix(calibrate, features)
    y_positive = _binary_target(fit, "target_net_positive")
    y_positive_calibrate = _binary_target(
        calibrate,
        "target_net_positive",
    )
    if y_positive.nunique() < 2 or y_positive_calibrate.nunique() < 2:
        raise ValueError(f"{spec.expert_id} positive target lacks both classes")
    fit_weight = day_temporal_weights(fit)
    calibration_weight = day_temporal_weights(calibrate)

    positive_tree = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=120,
        l2_regularization=10.0,
        random_state=random_seed,
    )
    positive_tree.fit(x_fit, y_positive, sample_weight=fit_weight)
    positive_linear = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.20,
                    max_iter=600,
                    class_weight="balanced",
                    random_state=random_seed + 1,
                ),
            ),
        ]
    )
    positive_linear.fit(
        x_fit,
        y_positive,
        model__sample_weight=fit_weight,
    )
    calibration_raw_positive = (
        0.70 * positive_tree.predict_proba(x_calibrate)[:, 1]
        + 0.30 * positive_linear.predict_proba(x_calibrate)[:, 1]
    )
    positive_calibrator = ProbabilityCalibrator().fit(
        calibration_raw_positive,
        y_positive_calibrate.to_numpy(dtype=int),
        calibration_weight,
    )

    severe_target = _binary_target(fit, "target_severe_loss")
    severe_calibration_target = _binary_target(
        calibrate,
        "target_severe_loss",
    )
    severe_tree: HistGradientBoostingClassifier | None = None
    severe_constant: float | None = None
    if severe_target.nunique() < 2:
        severe_constant = float(
            np.average(severe_target, weights=fit_weight)
        )
        severe_calibration_raw = np.full(
            len(calibrate),
            severe_constant,
            dtype=float,
        )
    else:
        severe_tree = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=150,
            max_leaf_nodes=11,
            min_samples_leaf=140,
            l2_regularization=12.0,
            random_state=random_seed + 2,
        )
        severe_tree.fit(
            x_fit,
            severe_target,
            sample_weight=fit_weight,
        )
        severe_calibration_raw = severe_tree.predict_proba(x_calibrate)[:, 1]
    severe_calibrator = ProbabilityCalibrator().fit(
        severe_calibration_raw,
        severe_calibration_target.to_numpy(dtype=int),
        calibration_weight,
    )

    return_target = _numeric(fit, "net_return_pct").fillna(0.0)
    return_calibration_target = _numeric(
        calibrate,
        "net_return_pct",
    ).fillna(0.0)
    return_tree = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.04,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=120,
        l2_regularization=10.0,
        random_state=random_seed + 3,
    )
    return_tree.fit(
        x_fit,
        return_target,
        sample_weight=fit_weight,
    )
    raw_calibration_return = return_tree.predict(x_calibrate)
    return_calibrator = Ridge(alpha=20.0)
    return_calibrator.fit(
        raw_calibration_return.reshape(-1, 1),
        return_calibration_target,
        sample_weight=calibration_weight,
    )
    return SpecialistBundle(
        spec=spec,
        positive_tree=positive_tree,
        positive_linear=positive_linear,
        severe_tree=severe_tree,
        severe_constant=severe_constant,
        return_tree=return_tree,
        positive_calibrator=positive_calibrator,
        severe_calibrator=severe_calibrator,
        return_calibrator=return_calibrator,
        feature_columns=features,
        fit_rows=int(len(fit)),
        calibration_rows=int(len(calibrate)),
    )


def fit_and_score_specialists(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    *,
    random_seed: int,
    specs: tuple[SpecialistSpec, ...] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    bundles, audit = fit_specialists(
        train,
        calibration,
        random_seed=random_seed,
        specs=specs,
    )
    return score_specialists(test, bundles), audit


def fit_specialists(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    random_seed: int,
    specs: tuple[SpecialistSpec, ...] | None = None,
) -> tuple[list[SpecialistBundle], list[dict[str, Any]]]:
    definitions = specs or specialist_specs()
    bundles: list[SpecialistBundle] = []
    audit: list[dict[str, Any]] = []
    for offset, spec in enumerate(definitions):
        try:
            bundle = fit_specialist(
                train,
                calibration,
                spec,
                random_seed=random_seed + offset * 101,
            )
        except ValueError as error:
            audit.append(
                {
                    "expert_id": spec.expert_id,
                    "scored": False,
                    "reason": str(error),
                }
            )
            continue
        bundles.append(bundle)
        audit.append(
            {
                "expert_id": spec.expert_id,
                "scored": True,
                "description": spec.description,
                "fit_rows": bundle.fit_rows,
                "calibration_rows": bundle.calibration_rows,
                "feature_count": len(bundle.feature_columns),
            }
        )
    return bundles, audit


def score_specialists(
    source: pd.DataFrame,
    bundles: list[SpecialistBundle],
) -> pd.DataFrame:
    predictions = [bundle.predict(source) for bundle in bundles]
    long = (
        pd.concat(predictions, ignore_index=True)
        if predictions
        else _empty_prediction_frame()
    )
    return aggregate_expert_predictions(source, long)


def aggregate_expert_predictions(
    source: pd.DataFrame,
    long_predictions: pd.DataFrame,
) -> pd.DataFrame:
    result = source.copy()
    for column in IDENTITY_COLUMNS:
        result[column] = result[column].astype(str)
    if long_predictions.empty:
        for column in (
            "expert_count",
            "expert_p_positive",
            "expert_p_positive_lower",
            "expert_probability_spread",
            "expert_expected_net_return_pct",
            "expert_expected_return_lower_pct",
            "expert_p_severe",
            "expert_score",
        ):
            result[column] = np.nan
        result["expert_count"] = 0
        return result
    long = long_predictions.copy()
    for column in IDENTITY_COLUMNS:
        long[column] = long[column].astype(str)
    if long.duplicated([*IDENTITY_COLUMNS, "expert_id"], keep=False).any():
        raise ValueError("duplicate expert predictions for an identity")
    grouped = long.groupby(list(IDENTITY_COLUMNS), sort=False)
    aggregate = grouped.agg(
        expert_count=("expert_id", "nunique"),
        expert_p_positive=("expert_p_positive", "median"),
        expert_p_positive_lower=("expert_p_positive", "min"),
        expert_probability_min=("expert_p_positive", "min"),
        expert_probability_max=("expert_p_positive", "max"),
        expert_expected_net_return_pct=(
            "expert_expected_net_return_pct",
            "median",
        ),
        expert_expected_return_min_pct=(
            "expert_expected_net_return_pct",
            "min",
        ),
        expert_expected_return_std_pct=(
            "expert_expected_net_return_pct",
            "std",
        ),
        expert_p_severe=("expert_p_severe", "max"),
        expert_score=("expert_score", "median"),
        expert_ids=(
            "expert_id",
            lambda values: ",".join(sorted(set(values.astype(str)))),
        ),
    ).reset_index()
    aggregate["expert_probability_spread"] = (
        aggregate["expert_probability_max"]
        - aggregate["expert_probability_min"]
    )
    aggregate["expert_expected_return_lower_pct"] = (
        aggregate["expert_expected_return_min_pct"]
        - 0.50
        * aggregate["expert_expected_return_std_pct"].fillna(0.0)
    )
    return result.merge(
        aggregate,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
    ).assign(
        expert_count=lambda frame: pd.to_numeric(
            frame["expert_count"],
            errors="coerce",
        ).fillna(0).astype(int)
    )


def active_feature_columns(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    candidates: tuple[str, ...],
) -> tuple[str, ...]:
    active = []
    for column in candidates:
        if column not in train or column not in calibration:
            continue
        train_values = pd.to_numeric(train[column], errors="coerce")
        calibration_values = pd.to_numeric(
            calibration[column],
            errors="coerce",
        )
        if train_values.notna().mean() < 0.50:
            continue
        if calibration_values.notna().mean() < 0.50:
            continue
        if train_values.nunique(dropna=True) < 2:
            continue
        active.append(column)
    if len(active) < 12:
        raise ValueError(
            f"specialist has only {len(active)} usable causal features"
        )
    return tuple(active)


def feature_matrix(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    values = frame.reindex(columns=columns).copy()
    for column in columns:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan).astype("float32")


def day_temporal_weights(
    frame: pd.DataFrame,
    *,
    half_life_days: float = 252.0,
) -> np.ndarray:
    dates = frame["trade_date"].astype(str)
    ordered_dates = sorted(dates.unique())
    age = {
        date: len(ordered_dates) - 1 - index
        for index, date in enumerate(ordered_dates)
    }
    counts = dates.value_counts()
    day_equal = dates.map(lambda value: 1.0 / counts[value]).to_numpy(float)
    temporal = dates.map(
        lambda value: 0.5 ** (age[value] / half_life_days)
    ).to_numpy(float)
    weights = day_equal * temporal
    return weights / max(float(np.mean(weights)), 1e-12)


def _binary_target(frame: pd.DataFrame, column: str) -> pd.Series:
    values = _numeric(frame, column)
    if values.isna().any():
        raise ValueError(f"{column} contains missing labels")
    return values.astype(int).clip(0, 1)


def _labeled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = (
        "net_return_pct",
        "target_net_positive",
        "target_severe_loss",
    )
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"specialist labels missing columns: {missing}")
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for column in required:
        mask &= _numeric(frame, column).notna()
    return frame.loc[mask].copy()


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _early_structure(frame: pd.DataFrame) -> pd.Series:
    return frame["signal_slot"].astype(str).isin(
        {"14:20", "14:25", "14:30", "14:35"}
    )


def _late_confirmation(frame: pd.DataFrame) -> pd.Series:
    return frame["signal_slot"].astype(str).isin(
        {"14:40", "14:45", "14:50"}
    )


def _market_industry_leader(frame: pd.DataFrame) -> pd.Series:
    return (
        _numeric(frame, "relative_market_return_pct").gt(0)
        & _numeric(frame, "relative_industry_return_pct").gt(0)
        & _numeric(frame, "return_cs_rank").ge(0.60)
    )


def _trend_persistence(frame: pd.DataFrame) -> pd.Series:
    return (
        _numeric(frame, "tail_trend_slope_pct").gt(0)
        & _numeric(frame, "tail_directional_efficiency").ge(0.15)
        & _numeric(frame, "tail_close_position_since_1400").ge(0.55)
    )


def _pullback_recovery(frame: pd.DataFrame) -> pd.Series:
    return (
        _numeric(frame, "tail_max_drawdown_pct").le(-0.50)
        & _numeric(frame, "tail_rebound_from_low_pct").ge(0.50)
        & _numeric(frame, "ret_5m_pct").gt(0)
    )


def _liquidity_breakout(frame: pd.DataFrame) -> pd.Series:
    return (
        _numeric(frame, "tail_amount_acceleration").gt(0)
        & _numeric(frame, "tail_volume_price_confirmation").gt(0)
        & _numeric(frame, "slot_amount_ratio_cs_rank").ge(0.60)
    )


def _empty_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            *IDENTITY_COLUMNS,
            "expert_id",
            "expert_p_positive",
            "expert_p_severe",
            "expert_expected_net_return_pct",
            "expert_score",
        ]
    )
