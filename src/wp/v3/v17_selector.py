from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .meta_alpha import (
    IDENTITY_COLUMNS,
    ProbabilityCalibrator,
    attach_meta_context,
)
from .v16_policy import (
    benjamini_hochberg,
    policy_metrics,
)


SELECTOR_FEATURES = (
    "slot_minute",
    "ret_from_prev_close_pct",
    "ret_abs_pct",
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
    "probability_uncertainty",
    "utility_uncertainty_pct",
    "context_return_mean_pct",
    "context_return_dispersion_pct",
    "context_breadth_positive",
    "context_breadth_above_2pct",
    "context_breadth_above_5pct",
    "context_breadth_above_7pct",
    "context_probability_mean",
    "context_probability_dispersion",
    "context_return_change_from_1420_pct",
    "return_context_relative_pct",
    "return_context_zscore",
    "return_rank_pct",
    "probability_rank_pct",
    "utility_rank_pct",
    "selection_rank_recomputed_pct",
    "severe_quality_rank_pct",
)


@dataclass
class SelectorBundle:
    positive_tree: HistGradientBoostingClassifier
    positive_linear: Pipeline
    return_location: HistGradientBoostingRegressor
    return_lower: HistGradientBoostingRegressor
    probability_calibrator: ProbabilityCalibrator
    return_location_adjustment: float
    return_lower_adjustment: float
    feature_columns: tuple[str, ...]
    train_rows: int
    calibration_rows: int

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        scored = prepare_selector_frame(frame)
        features = feature_matrix(scored, self.feature_columns)
        tree_probability = self.positive_tree.predict_proba(features)[:, 1]
        linear_probability = self.positive_linear.predict_proba(features)[:, 1]
        raw_probability = (
            0.70 * tree_probability + 0.30 * linear_probability
        )
        calibrated_probability = np.clip(
            self.probability_calibrator.predict(raw_probability),
            0.001,
            0.999,
        )
        spread = np.abs(tree_probability - linear_probability)
        scored["selector_p_positive"] = calibrated_probability
        scored["selector_probability_spread"] = spread
        scored["selector_p_positive_lower"] = np.clip(
            calibrated_probability - 0.50 * spread - 0.02,
            0.001,
            0.999,
        )
        scored["selector_expected_net_return_pct"] = (
            self.return_location.predict(features)
            + self.return_location_adjustment
        )
        scored["selector_return_q25_pct"] = (
            self.return_lower.predict(features)
            + self.return_lower_adjustment
        )
        severe = _numeric(scored, "p_severe_loss").clip(0.0, 1.0)
        fill = _numeric(
            scored,
            "p_round_trip_fill_lower",
        ).clip(0.0, 1.0)
        scored["selector_score"] = (
            scored["selector_expected_net_return_pct"]
            + 1.25 * (scored["selector_p_positive"] - 0.50)
            + 0.25 * scored["selector_return_q25_pct"]
            - 1.50 * severe
            - 0.50 * (1.0 - fill)
        )
        scored["selector_score_rank_pct"] = scored.groupby(
            ["trade_date", "signal_slot"],
            sort=False,
        )["selector_score"].rank(method="average", pct=True)
        return scored


@dataclass(frozen=True)
class SelectorPolicy:
    probability_lower_min: float
    expected_return_min_pct: float
    score_rank_min: float
    max_candidates_per_day: int
    return_q25_min_pct: float = -1.00
    severe_loss_max: float = 0.35
    round_trip_fill_min: float = 0.95
    probability_spread_max: float = 0.20

    @property
    def policy_id(self) -> str:
        return (
            f"p{self.probability_lower_min:.2f}-"
            f"e{self.expected_return_min_pct:.2f}-"
            f"q{self.return_q25_min_pct:.2f}-"
            f"s{self.severe_loss_max:.2f}-"
            f"f{self.round_trip_fill_min:.2f}-"
            f"d{self.probability_spread_max:.2f}-"
            f"r{self.score_rank_min:.2f}-"
            f"k{self.max_candidates_per_day}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "probability_lower_min": self.probability_lower_min,
            "expected_return_min_pct": self.expected_return_min_pct,
            "return_q25_min_pct": self.return_q25_min_pct,
            "severe_loss_max": self.severe_loss_max,
            "round_trip_fill_min": self.round_trip_fill_min,
            "probability_spread_max": self.probability_spread_max,
            "score_rank_min": self.score_rank_min,
            "max_candidates_per_day": self.max_candidates_per_day,
        }


@dataclass(frozen=True)
class SelectorPolicySelection:
    policy: SelectorPolicy | None
    design: dict[str, Any]
    confirmation: dict[str, Any]
    policies_evaluated: int
    design_passed: int
    confirmation_passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.as_dict() if self.policy else None,
            "design": self.design,
            "confirmation": self.confirmation,
            "search": {
                "policies_evaluated": self.policies_evaluated,
                "design_passed": self.design_passed,
                "confirmation_passed": self.confirmation_passed,
            },
        }


def selector_policy_grid() -> tuple[SelectorPolicy, ...]:
    return tuple(
        SelectorPolicy(
            probability_lower_min=probability,
            expected_return_min_pct=expected,
            score_rank_min=rank,
            max_candidates_per_day=maximum,
        )
        for probability, expected, rank, maximum in product(
            (0.52, 0.55, 0.58),
            (0.00, 0.15),
            (0.90, 0.95),
            (2, 3),
        )
    )


def prepare_selector_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = attach_meta_context(frame)
    required = set(IDENTITY_COLUMNS)
    missing = sorted(required - set(prepared.columns))
    if missing:
        raise ValueError(f"selector frame missing columns: {missing}")
    return prepared


def fit_selector(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    random_seed: int,
    minimum_train_rows: int = 2_000,
    minimum_calibration_rows: int = 500,
) -> SelectorBundle:
    prepared_train = _labeled_rows(prepare_selector_frame(train))
    prepared_calibration = _labeled_rows(
        prepare_selector_frame(calibration)
    )
    if len(prepared_train) < minimum_train_rows:
        raise ValueError(
            f"selector has only {len(prepared_train)} train rows; "
            f"requires {minimum_train_rows}"
        )
    if len(prepared_calibration) < minimum_calibration_rows:
        raise ValueError(
            f"selector has only {len(prepared_calibration)} calibration rows; "
            f"requires {minimum_calibration_rows}"
        )
    features = active_feature_columns(
        prepared_train,
        prepared_calibration,
    )
    x_train = feature_matrix(prepared_train, features)
    x_calibration = feature_matrix(prepared_calibration, features)
    y_positive = _numeric(
        prepared_train,
        "target_net_positive",
    ).astype(int)
    y_positive_calibration = _numeric(
        prepared_calibration,
        "target_net_positive",
    ).astype(int)
    if y_positive.nunique() < 2 or y_positive_calibration.nunique() < 2:
        raise ValueError("selector positive target lacks both classes")
    train_weight = day_temporal_weights(prepared_train)
    calibration_weight = day_temporal_weights(prepared_calibration)
    min_leaf = max(40, min(160, len(prepared_train) // 80))

    positive_tree = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=180,
        max_leaf_nodes=11,
        min_samples_leaf=min_leaf,
        l2_regularization=15.0,
        random_state=random_seed,
    )
    positive_tree.fit(
        x_train,
        y_positive,
        sample_weight=train_weight,
    )
    positive_linear = Pipeline(
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
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=random_seed + 1,
                ),
            ),
        ]
    )
    positive_linear.fit(
        x_train,
        y_positive,
        model__sample_weight=train_weight,
    )
    calibration_probability = (
        0.70 * positive_tree.predict_proba(x_calibration)[:, 1]
        + 0.30 * positive_linear.predict_proba(x_calibration)[:, 1]
    )
    probability_calibrator = ProbabilityCalibrator().fit(
        calibration_probability,
        y_positive_calibration.to_numpy(dtype=int),
        calibration_weight,
    )

    y_return = _numeric(prepared_train, "net_return_pct")
    y_return_calibration = _numeric(
        prepared_calibration,
        "net_return_pct",
    )
    return_location = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.035,
        max_iter=180,
        max_leaf_nodes=11,
        min_samples_leaf=min_leaf,
        l2_regularization=15.0,
        random_state=random_seed + 2,
    )
    return_location.fit(
        x_train,
        y_return,
        sample_weight=train_weight,
    )
    return_lower = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.25,
        learning_rate=0.035,
        max_iter=180,
        max_leaf_nodes=11,
        min_samples_leaf=min_leaf,
        l2_regularization=15.0,
        random_state=random_seed + 3,
    )
    return_lower.fit(
        x_train,
        y_return,
        sample_weight=train_weight,
    )
    location_residual = (
        y_return_calibration.to_numpy(float)
        - return_location.predict(x_calibration)
    )
    lower_residual = (
        y_return_calibration.to_numpy(float)
        - return_lower.predict(x_calibration)
    )
    return SelectorBundle(
        positive_tree=positive_tree,
        positive_linear=positive_linear,
        return_location=return_location,
        return_lower=return_lower,
        probability_calibrator=probability_calibrator,
        return_location_adjustment=_weighted_quantile(
            location_residual,
            calibration_weight,
            0.50,
        ),
        return_lower_adjustment=_weighted_quantile(
            lower_residual,
            calibration_weight,
            0.25,
        ),
        feature_columns=features,
        train_rows=int(len(prepared_train)),
        calibration_rows=int(len(prepared_calibration)),
    )


def apply_selector_policy(
    frame: pd.DataFrame,
    policy: SelectorPolicy,
) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "selector_p_positive_lower",
        "selector_expected_net_return_pct",
        "selector_return_q25_pct",
        "selector_probability_spread",
        "selector_score",
        "selector_score_rank_pct",
        "p_severe_loss",
        "p_round_trip_fill_lower",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"selector policy frame missing columns: {missing}")
    qualified = frame.loc[
        _numeric(frame, "selector_p_positive_lower").ge(
            policy.probability_lower_min
        )
        & _numeric(frame, "selector_expected_net_return_pct").ge(
            policy.expected_return_min_pct
        )
        & _numeric(frame, "selector_return_q25_pct").ge(
            policy.return_q25_min_pct
        )
        & _numeric(frame, "p_severe_loss").le(policy.severe_loss_max)
        & _numeric(frame, "p_round_trip_fill_lower").ge(
            policy.round_trip_fill_min
        )
        & _numeric(frame, "selector_probability_spread").le(
            policy.probability_spread_max
        )
        & _numeric(frame, "selector_score_rank_pct").ge(
            policy.score_rank_min
        )
    ].copy()
    if qualified.empty:
        qualified["selector_policy_id"] = policy.policy_id
        return qualified
    qualified["_slot_minute"] = _slot_minute(qualified["signal_slot"])
    qualified.sort_values(
        ["trade_date", "ts_code", "_slot_minute", "selector_score"],
        ascending=[True, True, True, False],
        kind="stable",
        inplace=True,
    )
    first_signal = qualified.drop_duplicates(
        ["trade_date", "ts_code"],
        keep="first",
    ).copy()
    first_signal.sort_values(
        ["trade_date", "selector_score", "_slot_minute", "ts_code"],
        ascending=[True, False, True, True],
        kind="stable",
        inplace=True,
    )
    within_day = first_signal.groupby("trade_date", sort=False).cumcount()
    selected = first_signal.loc[
        within_day.lt(policy.max_candidates_per_day)
    ].drop(columns="_slot_minute")
    selected["selector_policy_id"] = policy.policy_id
    return selected.reset_index(drop=True)


def select_selector_policy(
    design: pd.DataFrame,
    confirmation: pd.DataFrame,
    *,
    design_total_days: int,
    confirmation_total_days: int,
    policies: Iterable[SelectorPolicy] | None = None,
    seed: int,
    bootstrap_samples: int = 800,
) -> SelectorPolicySelection:
    grid = tuple(policies or selector_policy_grid())
    rows: list[dict[str, Any]] = []
    for offset, policy in enumerate(grid):
        metrics = policy_metrics(
            apply_selector_policy(design, policy),
            total_days=design_total_days,
            seed=seed + offset,
            bootstrap_samples=bootstrap_samples,
        )
        rows.append({**policy.as_dict(), **metrics})
    q_values = benjamini_hochberg(
        float(row["mean_return_p_value"]) for row in rows
    )
    for row, q_value in zip(rows, q_values, strict=True):
        row["mean_return_q_value"] = q_value
        row["design_gate_passed"] = selector_design_gate(row)
    passed = [row for row in rows if row["design_gate_passed"]]
    if not passed:
        return SelectorPolicySelection(
            policy=None,
            design={
                "reason": "no_design_policy_passed_predeclared_gates",
                "policies": rows,
            },
            confirmation={"reason": "not_run"},
            policies_evaluated=len(rows),
            design_passed=0,
            confirmation_passed=False,
        )
    champion_row = max(
        passed,
        key=lambda row: (
            float(row["clustered_mean_lower_pct"]),
            float(row["win_rate_wilson_lower"]),
            -abs(float(row["candidate_day_rate"]) - 0.30),
            float(row["mean_net_return_pct"]),
        ),
    )
    champion = next(
        policy
        for policy in grid
        if policy.policy_id == champion_row["policy_id"]
    )
    confirmation_metrics = policy_metrics(
        apply_selector_policy(confirmation, champion),
        total_days=confirmation_total_days,
        seed=seed + len(grid) + 1,
        bootstrap_samples=max(1_000, bootstrap_samples),
    )
    confirmed = selector_confirmation_gate(confirmation_metrics)
    if not confirmed:
        return SelectorPolicySelection(
            policy=None,
            design=champion_row,
            confirmation={
                **confirmation_metrics,
                "reason": "frozen_design_champion_failed_confirmation",
            },
            policies_evaluated=len(rows),
            design_passed=len(passed),
            confirmation_passed=False,
        )
    return SelectorPolicySelection(
        policy=champion,
        design=champion_row,
        confirmation={
            **confirmation_metrics,
            "reason": "frozen_design_champion_passed_once",
        },
        policies_evaluated=len(rows),
        design_passed=len(passed),
        confirmation_passed=True,
    )


def selector_design_gate(metrics: dict[str, Any]) -> bool:
    rate = float(metrics["candidate_day_rate"])
    return bool(
        int(metrics["events"]) >= 40
        and int(metrics["candidate_days"]) >= 20
        and 0.15 <= rate <= 0.60
        and float(metrics["win_rate"]) >= 0.52
        and float(metrics["mean_net_return_pct"]) > 0.0
        and float(metrics["profit_factor"]) >= 1.05
        and float(metrics["stress_50bps_mean_net_return_pct"]) >= 0.0
        and float(metrics["mean_return_q_value"]) <= 0.10
    )


def selector_confirmation_gate(metrics: dict[str, Any]) -> bool:
    return bool(
        int(metrics["events"]) >= 20
        and int(metrics["candidate_days"]) >= 10
        and float(metrics["candidate_day_rate"]) >= 0.12
        and float(metrics["win_rate"]) >= 0.50
        and float(metrics["mean_net_return_pct"]) > 0.0
        and float(metrics["profit_factor"]) >= 1.0
        and float(metrics["stress_50bps_mean_net_return_pct"]) >= 0.0
    )


def active_feature_columns(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
) -> tuple[str, ...]:
    active = []
    for column in SELECTOR_FEATURES:
        if column not in train or column not in calibration:
            continue
        train_values = _numeric(train, column)
        calibration_values = _numeric(calibration, column)
        if train_values.notna().mean() < 0.50:
            continue
        if calibration_values.notna().mean() < 0.50:
            continue
        if train_values.nunique(dropna=True) < 2:
            continue
        active.append(column)
    if len(active) < 12:
        raise ValueError(
            f"selector has only {len(active)} usable causal features"
        )
    return tuple(active)


def feature_matrix(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    matrix = frame.reindex(columns=columns).copy()
    for column in columns:
        matrix[column] = _numeric(matrix, column)
    return matrix.replace([np.inf, -np.inf], np.nan).astype("float32")


def day_temporal_weights(
    frame: pd.DataFrame,
    *,
    half_life_days: float = 252.0,
) -> np.ndarray:
    dates = frame["trade_date"].astype(str)
    ordered = sorted(dates.unique())
    age = {
        date: len(ordered) - 1 - index
        for index, date in enumerate(ordered)
    }
    counts = dates.value_counts()
    day_equal = dates.map(lambda value: 1.0 / counts[value]).to_numpy(float)
    temporal = dates.map(
        lambda value: 0.5 ** (age[value] / half_life_days)
    ).to_numpy(float)
    weights = day_equal * temporal
    return weights / max(float(np.mean(weights)), 1e-12)


def _labeled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    net = _numeric(frame, "net_return_pct")
    positive = _numeric(frame, "target_net_positive")
    return frame.loc[net.notna() & positive.notna()].copy()


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    clean = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not clean.any():
        return 0.0
    ordered = np.argsort(values[clean])
    sorted_values = values[clean][ordered]
    sorted_weights = weights[clean][ordered]
    cumulative = np.cumsum(sorted_weights)
    cutoff = float(quantile) * float(cumulative[-1])
    index = min(
        int(np.searchsorted(cumulative, cutoff, side="left")),
        len(sorted_values) - 1,
    )
    return float(sorted_values[index])


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _slot_minute(values: pd.Series) -> pd.Series:
    parsed = values.astype(str).str.extract(
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2})"
    )
    return (
        pd.to_numeric(parsed["hour"], errors="coerce") * 60
        + pd.to_numeric(parsed["minute"], errors="coerce")
    )
