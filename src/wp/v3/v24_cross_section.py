from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .meta_alpha import IDENTITY_COLUMNS, ProbabilityCalibrator
from .v17_selector import active_feature_columns, day_temporal_weights
from .v23_microstructure import (
    MicrostructureGateBundle,
    SOURCE_PRIOR_FEATURES,
    V23_FEATURE_COLUMNS,
    feature_matrix,
    labeled_complete_rows,
    rolling_microstructure_segments,
)
from .v24_data import V24_DERIVED_SOURCE_FEATURE_COLUMNS


SCHEMA_VERSION = "wp_v24_cross_section_microstructure_1"
MODEL_TRAIN_DAYS = 252
MODEL_CALIBRATION_DAYS = 42
MODEL_PURGE_DAYS = 2
MINIMUM_TRAIN_ROWS = 4_000
MINIMUM_CALIBRATION_ROWS = 800

FIXED_TARGET_CANDIDATE_DAY_RATE = 0.25
FIXED_MAX_CANDIDATES_PER_DAY = 3
ROUND_TRIP_FILL_MIN = 0.95
SOURCE_SEVERE_LOSS_MAX = 0.45
PROBABILITY_SPREAD_MAX = 0.40
RETURN_SPREAD_MAX_PCT = 5.0
MAX_DATA_AGE_SECONDS = 420.0

MODEL_FEATURES = tuple(
    dict.fromkeys(
        (
            *V23_FEATURE_COLUMNS,
            *SOURCE_PRIOR_FEATURES,
            *V24_DERIVED_SOURCE_FEATURE_COLUMNS,
        )
    )
)


@dataclass(frozen=True)
class CrossSectionPolicySpec:
    target_candidate_day_rate: float = FIXED_TARGET_CANDIDATE_DAY_RATE
    max_candidates_per_day: int = FIXED_MAX_CANDIDATES_PER_DAY

    @property
    def policy_id(self) -> str:
        return (
            f"v24-rate{self.target_candidate_day_rate:.2f}-"
            f"k{self.max_candidates_per_day}-top5"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "target_candidate_day_rate": self.target_candidate_day_rate,
            "max_candidates_per_day": self.max_candidates_per_day,
            "round_trip_fill_min": ROUND_TRIP_FILL_MIN,
            "source_severe_loss_max": SOURCE_SEVERE_LOSS_MAX,
            "probability_spread_max": PROBABILITY_SPREAD_MAX,
            "return_spread_max_pct": RETURN_SPREAD_MAX_PCT,
            "max_data_age_seconds": MAX_DATA_AGE_SECONDS,
            "hard_probability_thresholds": False,
            "uncertainty_treatment": "soft_score_penalty",
        }


@dataclass(frozen=True)
class FrozenCrossSectionPolicy:
    spec: CrossSectionPolicySpec
    score_threshold: float
    calibration_start: str
    calibration_end: str
    calibration_days: int
    eligible_days: int

    @property
    def policy_id(self) -> str:
        return self.spec.policy_id

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.spec.as_dict(),
            "score_threshold": self.score_threshold,
            "calibration_start": self.calibration_start,
            "calibration_end": self.calibration_end,
            "calibration_days": self.calibration_days,
            "eligible_days": self.eligible_days,
        }


def fit_cross_section_gate(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    random_seed: int,
    minimum_train_rows: int = MINIMUM_TRAIN_ROWS,
    minimum_calibration_rows: int = MINIMUM_CALIBRATION_ROWS,
) -> MicrostructureGateBundle:
    prepared_train = labeled_complete_rows(train)
    prepared_calibration = labeled_complete_rows(calibration)
    if len(prepared_train) < minimum_train_rows:
        raise ValueError(
            f"V24 has {len(prepared_train)} train rows; "
            f"requires {minimum_train_rows}"
        )
    if len(prepared_calibration) < minimum_calibration_rows:
        raise ValueError(
            f"V24 has {len(prepared_calibration)} calibration rows; "
            f"requires {minimum_calibration_rows}"
        )
    features = active_feature_columns(
        prepared_train,
        prepared_calibration,
        candidates=MODEL_FEATURES,
    )
    if len(features) < 30:
        raise ValueError(f"V24 has only {len(features)} active features")
    x_train = feature_matrix(prepared_train, features)
    x_calibration = feature_matrix(prepared_calibration, features)
    net_train = _numeric(prepared_train, "net_return_pct")
    net_calibration = _numeric(prepared_calibration, "net_return_pct")
    targets_train = {
        "positive": net_train.gt(0.0).astype(int),
        "margin": net_train.gt(0.50).astype(int),
        "severe": net_train.le(-2.0).astype(int),
    }
    targets_calibration = {
        "positive": net_calibration.gt(0.0).astype(int),
        "margin": net_calibration.gt(0.50).astype(int),
        "severe": net_calibration.le(-2.0).astype(int),
    }
    for name, target in targets_train.items():
        if target.nunique() < 2:
            raise ValueError(f"V24 {name} target lacks both classes")

    train_weight = stock_day_equalized_temporal_weights(prepared_train)
    calibration_weight = stock_day_equalized_temporal_weights(
        prepared_calibration
    )
    min_leaf = max(60, min(180, len(prepared_train) // 35))
    trees: dict[str, HistGradientBoostingClassifier] = {}
    linears: dict[str, Pipeline] = {}
    calibrators: dict[str, ProbabilityCalibrator] = {}
    for index, name in enumerate(("positive", "margin", "severe")):
        target = targets_train[name]
        tree = HistGradientBoostingClassifier(
            learning_rate=0.025,
            max_iter=220,
            max_leaf_nodes=9,
            min_samples_leaf=min_leaf,
            l2_regularization=40.0,
            random_state=random_seed + index * 101,
        )
        tree.fit(
            x_train,
            target,
            sample_weight=_balanced_weights(target, train_weight),
        )
        linear = _linear_classifier(random_seed + index * 101 + 1)
        linear.fit(
            x_train,
            target,
            model__sample_weight=train_weight,
        )
        raw = _blend(
            tree.predict_proba(x_calibration)[:, 1],
            linear.predict_proba(x_calibration)[:, 1],
        )
        calibrator = ProbabilityCalibrator().fit(
            raw,
            targets_calibration[name].to_numpy(dtype=int),
            calibration_weight,
        )
        trees[name] = tree
        linears[name] = linear
        calibrators[name] = calibrator

    return_tree = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.025,
        max_iter=220,
        max_leaf_nodes=9,
        min_samples_leaf=min_leaf,
        l2_regularization=40.0,
        random_state=random_seed + 404,
    )
    clipped_train = net_train.clip(-10.0, 10.0)
    return_tree.fit(
        x_train,
        clipped_train,
        sample_weight=train_weight,
    )
    return_linear = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=40.0)),
        ]
    )
    return_linear.fit(
        x_train,
        clipped_train,
        model__sample_weight=train_weight,
    )
    calibration_return_raw = _blend(
        return_tree.predict(x_calibration),
        return_linear.predict(x_calibration),
    )
    return_calibrator = Ridge(alpha=15.0)
    return_calibrator.fit(
        np.asarray(calibration_return_raw).reshape(-1, 1),
        net_calibration.clip(-10.0, 10.0),
        sample_weight=calibration_weight,
    )
    calibration_expected = return_calibrator.predict(
        np.asarray(calibration_return_raw).reshape(-1, 1)
    )
    residual = np.maximum(
        calibration_expected - net_calibration.to_numpy(dtype=float),
        0.0,
    )
    downside_residual = _weighted_quantile(
        residual,
        calibration_weight,
        0.70,
    )
    return MicrostructureGateBundle(
        positive_tree=trees["positive"],
        positive_linear=linears["positive"],
        margin_tree=trees["margin"],
        margin_linear=linears["margin"],
        severe_tree=trees["severe"],
        severe_linear=linears["severe"],
        return_tree=return_tree,
        return_linear=return_linear,
        positive_calibrator=calibrators["positive"],
        margin_calibrator=calibrators["margin"],
        severe_calibrator=calibrators["severe"],
        return_calibrator=return_calibrator,
        return_downside_residual_pct=float(downside_residual),
        feature_columns=features,
        train_rows=int(len(prepared_train)),
        calibration_rows=int(len(prepared_calibration)),
    )


def add_cross_section_score(scored: pd.DataFrame) -> pd.DataFrame:
    result = scored.copy()
    expected = _numeric(
        result,
        "v23_expected_net_return_pct",
    ).clip(-4.0, 4.0)
    probability_uncertainty = (
        _numeric(result, "v23_positive_model_spread")
        + 0.50 * _numeric(result, "v23_margin_model_spread")
        + 0.50 * _numeric(result, "v23_severe_model_spread")
    )
    return_uncertainty = _numeric(
        result,
        "v23_expected_return_model_spread_pct",
    ).clip(0.0, 6.0)
    result["v24_cross_section_score"] = (
        0.50 * expected
        + 1.25 * (_numeric(result, "v23_p_positive") - 0.50)
        + 0.75 * (_numeric(result, "v23_p_margin") - 0.35)
        - 1.00 * _numeric(result, "v23_p_severe_loss")
        - 0.20 * probability_uncertainty
        - 0.05 * return_uncertainty
    )
    return result


def calibrate_cross_section_policy(
    scored_calibration: pd.DataFrame,
    *,
    calibration_dates: Iterable[str],
    spec: CrossSectionPolicySpec | None = None,
) -> FrozenCrossSectionPolicy:
    frozen_spec = spec or CrossSectionPolicySpec()
    dates = sorted(set(map(str, calibration_dates)))
    if not dates:
        raise ValueError("V24 policy calibration has no dates")
    scored = add_cross_section_score(scored_calibration)
    eligible = policy_eligible_rows(scored)
    daily_max = (
        eligible.groupby("trade_date", sort=False)["v24_cross_section_score"]
        .max()
        .sort_values(ascending=False)
    )
    target_days = max(
        1,
        int(np.ceil(frozen_spec.target_candidate_day_rate * len(dates))),
    )
    threshold = (
        float("inf")
        if daily_max.empty
        else float(daily_max.iloc[min(target_days, len(daily_max)) - 1])
    )
    return FrozenCrossSectionPolicy(
        spec=frozen_spec,
        score_threshold=threshold,
        calibration_start=dates[0],
        calibration_end=dates[-1],
        calibration_days=len(dates),
        eligible_days=int(len(daily_max)),
    )


def apply_cross_section_policy(
    scored: pd.DataFrame,
    policy: FrozenCrossSectionPolicy,
) -> pd.DataFrame:
    prepared = add_cross_section_score(scored)
    eligible = policy_eligible_rows(prepared)
    qualified = eligible.loc[
        _numeric(eligible, "v24_cross_section_score").ge(
            policy.score_threshold
        )
    ].copy()
    if qualified.empty:
        qualified["v24_policy_id"] = policy.policy_id
        return qualified
    qualified["_slot_absolute"] = _slot_absolute(qualified["signal_slot"])
    qualified.sort_values(
        [
            "trade_date",
            "_slot_absolute",
            "v24_cross_section_score",
            "ts_code",
        ],
        ascending=[True, True, False, True],
        kind="stable",
        inplace=True,
    )
    first_signal = qualified.drop_duplicates(
        ["trade_date", "ts_code"],
        keep="first",
    ).copy()
    within_day = first_signal.groupby("trade_date", sort=False).cumcount()
    selected = first_signal.loc[
        within_day.lt(policy.spec.max_candidates_per_day)
    ].drop(columns="_slot_absolute")
    selected["v24_policy_id"] = policy.policy_id
    selected["v24_score_threshold"] = policy.score_threshold
    return selected.reset_index(drop=True)


def policy_eligible_rows(scored: pd.DataFrame) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "v23_point_in_time_complete",
        "v23_positive_model_spread",
        "v23_margin_model_spread",
        "v23_severe_model_spread",
        "v23_expected_return_model_spread_pct",
        "v24_cross_section_score",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"V24 policy frame missing columns: {missing}")
    age = _numeric(scored, "data_age_seconds")
    fresh = age.isna() | age.le(MAX_DATA_AGE_SECONDS)
    legal_slot = scored["signal_slot"].astype(str).str.replace(
        ":",
        "",
        regex=False,
    ).between("1420", "1450")
    eligible = (
        _boolean(scored, "v23_point_in_time_complete")
        & _numeric(scored, "p_round_trip_fill_lower").ge(
            ROUND_TRIP_FILL_MIN
        )
        & _numeric(scored, "p_severe_loss").le(SOURCE_SEVERE_LOSS_MAX)
        & _numeric(scored, "v23_positive_model_spread").le(
            PROBABILITY_SPREAD_MAX
        )
        & _numeric(scored, "v23_margin_model_spread").le(
            PROBABILITY_SPREAD_MAX
        )
        & _numeric(scored, "v23_severe_model_spread").le(
            PROBABILITY_SPREAD_MAX
        )
        & _numeric(scored, "v23_expected_return_model_spread_pct").le(
            RETURN_SPREAD_MAX_PCT
        )
        & fresh
        & legal_slot
    )
    return scored.loc[eligible].copy()


def stock_day_equalized_temporal_weights(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    temporal = np.asarray(day_temporal_weights(frame), dtype=float)
    counts = (
        frame.groupby(["trade_date", "ts_code"], sort=False)["ts_code"]
        .transform("size")
        .to_numpy(dtype=float)
    )
    weights = temporal / np.maximum(counts, 1.0)
    mean = float(np.mean(weights))
    return weights / mean if mean > 0.0 else np.ones(len(frame), dtype=float)


def v24_research_readiness(
    metrics: dict[str, Any],
    *,
    yearly: list[dict[str, Any]],
    temporal_integrity: bool,
    source_integrity: bool,
    data_integrity: bool,
) -> dict[str, Any]:
    active_years = [
        row for row in yearly if int(row.get("events", 0)) > 0
    ]
    positive_years = sum(
        float(row.get("mean_net_return_pct") or -999.0) > 0.0
        for row in active_years
    )
    worst_year = min(
        (
            float(row.get("mean_net_return_pct") or -999.0)
            for row in active_years
        ),
        default=-999.0,
    )
    minimum_year_events = min(
        (int(row.get("events", 0)) for row in active_years),
        default=0,
    )
    gates = {
        "minimum_nested_oos_candidates": int(metrics.get("events", 0)) >= 120,
        "minimum_nested_oos_candidate_days": (
            int(metrics.get("candidate_days", 0)) >= 80
        ),
        "practical_candidate_day_rate": (
            0.12 <= float(metrics.get("candidate_day_rate", 0.0)) <= 0.35
        ),
        "minimum_win_rate": float(metrics.get("win_rate", 0.0)) >= 0.55,
        "minimum_wilson_lower": (
            float(metrics.get("win_rate_wilson_lower", 0.0)) >= 0.50
        ),
        "minimum_clustered_win_rate_lower": (
            float(metrics.get("clustered_win_rate_lower", 0.0)) >= 0.48
        ),
        "minimum_margin_hit_rate": (
            float(metrics.get("margin_hit_rate", 0.0)) >= 0.40
        ),
        "maximum_tail_loss_rate": (
            float(metrics.get("tail_loss_rate", 1.0)) <= 0.15
        ),
        "minimum_mean_net_return_pct": (
            float(metrics.get("mean_net_return_pct") or -999.0) >= 0.20
        ),
        "clustered_mean_lower_positive": (
            float(metrics.get("clustered_mean_lower_pct") or -999.0) > 0.0
        ),
        "minimum_profit_factor": (
            float(metrics.get("profit_factor") or 0.0) >= 1.20
        ),
        "real_50bps_stress_nonnegative": (
            float(
                metrics.get("stress_50bps_mean_net_return_pct") or -999.0
            )
            >= 0.0
        ),
        "return_p10_above_minus_3pct": (
            float(metrics.get("return_p10_pct") or -999.0) >= -3.0
        ),
        "minimum_three_active_calendar_years": len(active_years) >= 3,
        "minimum_twenty_candidates_each_active_year": (
            minimum_year_events >= 20
        ),
        "minimum_three_positive_calendar_years": positive_years >= 3,
        "worst_calendar_year_above_minus_0_10pct": worst_year >= -0.10,
        "temporal_integrity": bool(temporal_integrity),
        "source_integrity": bool(source_integrity),
        "data_integrity": bool(data_integrity),
    }
    passed = all(gates.values())
    return {
        "all_historical_gates_passed": passed,
        "gates": gates,
        "failed_gates": [
            name for name, gate_passed in gates.items() if not gate_passed
        ],
        "production_authorized": False,
        "future_shadow_days_required": 150,
        "future_shadow_min_candidates": 60,
        "future_shadow_min_candidate_days": 40,
        "reason": (
            "historical_screen_passed_future_shadow_still_required"
            if passed
            else "historical_evidence_insufficient"
        ),
    }


def validate_feature_contract(features: tuple[str, ...]) -> bool:
    invalid = sorted(set(features) - set(MODEL_FEATURES))
    contaminated = [
        feature
        for feature in features
        if any(
            token in feature.lower()
            for token in (
                "target",
                "truth",
                "future",
                "gross_return",
                "net_return",
                "t1_",
                "exit_price",
            )
        )
    ]
    if len(features) < 30 or invalid or contaminated:
        raise RuntimeError(
            "V24 feature contract violated: "
            f"count={len(features)} invalid={invalid} "
            f"contaminated={contaminated}"
        )
    return True


def validate_selected_contract(
    selected: pd.DataFrame,
    policy: FrozenCrossSectionPolicy | None,
) -> None:
    if selected.empty:
        return
    if selected.duplicated(["trade_date", "ts_code"], keep=False).any():
        raise RuntimeError("V24 selected output rewrote a first signal")
    maximum = (
        policy.spec.max_candidates_per_day
        if policy is not None
        else FIXED_MAX_CANDIDATES_PER_DAY
    )
    if int(selected.groupby("trade_date").size().max()) > maximum:
        raise RuntimeError("V24 selected output exceeds fixed daily maximum")
    slot = selected["signal_slot"].astype(str).str.replace(
        ":",
        "",
        regex=False,
    )
    if not slot.between("1420", "1450").all():
        raise RuntimeError("V24 selected output contains an illegal slot")
    if not selected["v23_point_in_time_complete"].fillna(False).all():
        raise RuntimeError("V24 selected output contains incomplete data")


def rolling_cross_section_segments(
    prior_dates: Iterable[str],
    *,
    reserve_final_purge: bool = True,
) -> tuple[list[str], list[str]] | None:
    return rolling_microstructure_segments(
        prior_dates,
        reserve_final_purge=reserve_final_purge,
    )


def _linear_classifier(random_seed: int) -> Pipeline:
    return Pipeline(
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
                    C=0.04,
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=random_seed,
                ),
            ),
        ]
    )


def _balanced_weights(
    target: pd.Series,
    base_weight: np.ndarray,
) -> np.ndarray:
    values = target.to_numpy(dtype=int)
    counts = np.bincount(values, minlength=2).astype(float)
    class_weight = np.asarray(
        [len(values) / (2.0 * max(count, 1.0)) for count in counts],
        dtype=float,
    )
    return np.asarray(base_weight, dtype=float) * class_weight[values]


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(sorted_weights)
    if not len(sorted_values) or cumulative[-1] <= 0.0:
        return 0.0
    cutoff = float(quantile) * cumulative[-1]
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _blend(tree: np.ndarray, linear: np.ndarray) -> np.ndarray:
    return 0.70 * np.asarray(tree, dtype=float) + 0.30 * np.asarray(
        linear,
        dtype=float,
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(False)
    return values.fillna(False).astype(bool)


def _slot_absolute(values: pd.Series) -> pd.Series:
    parsed = values.astype(str).str.extract(
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2})"
    )
    return (
        pd.to_numeric(parsed["hour"], errors="coerce") * 60
        + pd.to_numeric(parsed["minute"], errors="coerce")
    )
