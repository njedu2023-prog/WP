from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import MARKET_FEATURE_COLUMNS, enrich_feature_frame
from .meta_alpha import IDENTITY_COLUMNS, ProbabilityCalibrator
from .v17_selector import (
    active_feature_columns,
    day_temporal_weights,
    feature_matrix,
)
from .v21_margin import add_economic_metrics


SCHEMA_VERSION = "wp_v22_market_license_1"
GATE_TRAIN_DAYS = 126
GATE_CALIBRATION_DAYS = 42
GATE_PURGE_DAYS = 2
FIXED_TARGET_CANDIDATE_DAY_RATE = 0.12
FIXED_MAX_CANDIDATES_PER_DAY = 2

LICENSE_MODEL_SPREAD_MAX = 0.25
ROUND_TRIP_FILL_MIN = 0.95
SOURCE_SEVERE_LOSS_MAX = 0.40
SOURCE_MODEL_SPREAD_MAX = 0.30
MAX_DATA_AGE_SECONDS = 420.0

MARKET_AGGREGATE_FEATURES = (
    "v22_eligible_count",
    "v22_return_mean_pct",
    "v22_return_median_pct",
    "v22_return_dispersion_pct",
    "v22_breadth_positive",
    "v22_breadth_above_2pct",
    "v22_breadth_above_5pct",
    "v22_breadth_above_7pct",
    "v22_probability_mean",
    "v22_probability_dispersion",
    "v22_utility_mean_pct",
    "v22_score_median",
    "v22_score_std",
    "v22_score_q90",
    "v22_score_top_margin",
    "v22_severe_loss_median",
    "v22_fill_probability_median",
)
MARKET_LICENSE_FEATURES = (
    *(f"v22_market_{column}" for column in MARKET_FEATURE_COLUMNS),
    *MARKET_AGGREGATE_FEATURES,
)


@dataclass
class MarketLicenseBundle:
    positive_tree: HistGradientBoostingClassifier
    positive_linear: Pipeline
    probability_calibrator: ProbabilityCalibrator
    feature_columns: tuple[str, ...]
    train_rows: int
    calibration_rows: int

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        scored = frame.copy()
        features = feature_matrix(scored, self.feature_columns)
        tree_probability = self.positive_tree.predict_proba(features)[:, 1]
        linear_probability = self.positive_linear.predict_proba(features)[:, 1]
        raw_probability = (
            0.70 * tree_probability + 0.30 * linear_probability
        )
        probability = np.clip(
            self.probability_calibrator.predict(raw_probability),
            0.001,
            0.999,
        )
        spread = np.abs(tree_probability - linear_probability)
        scored["v22_license_probability"] = probability
        scored["v22_license_model_spread"] = spread
        scored["v22_license_probability_lower"] = np.clip(
            probability - 0.50 * spread - 0.02,
            0.001,
            0.999,
        )
        return scored


@dataclass(frozen=True)
class MarketLicensePolicySpec:
    target_candidate_day_rate: float = FIXED_TARGET_CANDIDATE_DAY_RATE
    max_candidates_per_day: int = FIXED_MAX_CANDIDATES_PER_DAY
    license_model_spread_max: float = LICENSE_MODEL_SPREAD_MAX

    @property
    def policy_id(self) -> str:
        return (
            f"market-license-rate{self.target_candidate_day_rate:.2f}-"
            f"spread{self.license_model_spread_max:.2f}-"
            f"k{self.max_candidates_per_day}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "target_candidate_day_rate": self.target_candidate_day_rate,
            "max_candidates_per_day": self.max_candidates_per_day,
            "license_model_spread_max": self.license_model_spread_max,
            "round_trip_fill_min": ROUND_TRIP_FILL_MIN,
            "source_severe_loss_max": SOURCE_SEVERE_LOSS_MAX,
            "source_model_spread_max": SOURCE_MODEL_SPREAD_MAX,
            "max_data_age_seconds": MAX_DATA_AGE_SECONDS,
        }


@dataclass(frozen=True)
class FrozenMarketLicensePolicy:
    spec: MarketLicensePolicySpec
    license_probability_lower_threshold: float
    threshold_calibration_start: str
    threshold_calibration_end: str
    threshold_calibration_days: int
    threshold_eligible_days: int

    @property
    def policy_id(self) -> str:
        return self.spec.policy_id

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.spec.as_dict(),
            "license_probability_lower_threshold": (
                self.license_probability_lower_threshold
            ),
            "threshold_calibration_start": self.threshold_calibration_start,
            "threshold_calibration_end": self.threshold_calibration_end,
            "threshold_calibration_days": self.threshold_calibration_days,
            "threshold_eligible_days": self.threshold_eligible_days,
        }


def build_market_slot_leaders(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "selection_score",
        "p_net_positive_lower",
        "expected_utility_lower_pct",
        "p_severe_loss",
        "p_round_trip_fill_lower",
        "probability_model_spread",
        "expected_return_model_spread",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V22 source frame missing columns: {missing}")
    if frame.empty:
        return frame.copy()

    prepared = enrich_feature_frame(frame).reset_index(drop=True)
    eligible = prepared.loc[_source_eligible(prepared)].copy()
    if eligible.empty:
        return prepared.head(0).copy()
    keys = ["trade_date", "signal_slot"]
    eligible["v22_stock_score"] = _numeric(
        eligible,
        "selection_score",
    )
    eligible.sort_values(
        [
            *keys,
            "v22_stock_score",
            "p_net_positive_lower",
            "ts_code",
        ],
        ascending=[True, True, False, False, True],
        kind="stable",
        inplace=True,
    )
    grouped = eligible.groupby(keys, sort=False)
    eligible["v22_stock_rank_in_slot"] = grouped.cumcount() + 1

    score_group = grouped["v22_stock_score"]
    eligible["v22_eligible_count"] = score_group.transform("size")
    eligible["v22_score_median"] = score_group.transform("median")
    eligible["v22_score_std"] = score_group.transform("std").fillna(0.0)
    eligible["v22_score_q90"] = score_group.transform(
        lambda values: values.quantile(0.90)
    )
    second_scores = (
        eligible.loc[
            eligible["v22_stock_rank_in_slot"].eq(2),
            [*keys, "v22_stock_score"],
        ]
        .set_index(keys)["v22_stock_score"]
    )
    group_index = pd.MultiIndex.from_frame(eligible.loc[:, keys])
    top_score = score_group.transform("max")
    eligible["v22_score_top_margin"] = (
        top_score
        - pd.Series(group_index.map(second_scores), index=eligible.index)
    ).fillna(0.0)

    returns = _numeric(eligible, "ret_from_prev_close_pct")
    eligible["v22_return_mean_pct"] = _group_series(
        eligible,
        returns,
        keys,
        "mean",
    )
    eligible["v22_return_median_pct"] = _group_series(
        eligible,
        returns,
        keys,
        "median",
    )
    eligible["v22_return_dispersion_pct"] = _group_series(
        eligible,
        returns,
        keys,
        "std",
    ).fillna(0.0)
    for suffix, threshold in (
        ("positive", 0.0),
        ("above_2pct", 2.0),
        ("above_5pct", 5.0),
        ("above_7pct", 7.0),
    ):
        eligible[f"v22_breadth_{suffix}"] = _group_series(
            eligible,
            returns.gt(threshold).astype(float),
            keys,
            "mean",
        )
    probability = _numeric(eligible, "p_net_positive_lower")
    eligible["v22_probability_mean"] = _group_series(
        eligible,
        probability,
        keys,
        "mean",
    )
    eligible["v22_probability_dispersion"] = _group_series(
        eligible,
        probability,
        keys,
        "std",
    ).fillna(0.0)
    eligible["v22_utility_mean_pct"] = _group_series(
        eligible,
        _numeric(eligible, "expected_utility_lower_pct"),
        keys,
        "mean",
    )
    eligible["v22_severe_loss_median"] = _group_series(
        eligible,
        _numeric(eligible, "p_severe_loss"),
        keys,
        "median",
    )
    eligible["v22_fill_probability_median"] = _group_series(
        eligible,
        _numeric(eligible, "p_round_trip_fill_lower"),
        keys,
        "median",
    )
    for column in MARKET_FEATURE_COLUMNS:
        eligible[f"v22_market_{column}"] = _group_series(
            eligible,
            _numeric(eligible, column),
            keys,
            "median",
        )

    leaders = eligible.loc[
        eligible["v22_stock_rank_in_slot"].eq(1)
    ].copy()
    return (
        leaders.sort_values(list(IDENTITY_COLUMNS), kind="stable")
        .reset_index(drop=True)
    )


def fit_market_license(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    random_seed: int,
    minimum_train_rows: int = 500,
    minimum_calibration_rows: int = 200,
) -> MarketLicenseBundle:
    prepared_train = _labeled_rows(train)
    prepared_calibration = _labeled_rows(calibration)
    if len(prepared_train) < minimum_train_rows:
        raise ValueError(
            f"V22 has only {len(prepared_train)} train rows; "
            f"requires {minimum_train_rows}"
        )
    if len(prepared_calibration) < minimum_calibration_rows:
        raise ValueError(
            f"V22 has only {len(prepared_calibration)} calibration rows; "
            f"requires {minimum_calibration_rows}"
        )
    features = active_feature_columns(
        prepared_train,
        prepared_calibration,
        candidates=MARKET_LICENSE_FEATURES,
    )
    x_train = feature_matrix(prepared_train, features)
    x_calibration = feature_matrix(prepared_calibration, features)
    train_target = _numeric(
        prepared_train,
        "target_net_positive",
    ).astype(int)
    calibration_target = _numeric(
        prepared_calibration,
        "target_net_positive",
    ).astype(int)
    if train_target.nunique() < 2:
        raise ValueError("V22 license target lacks both classes")

    temporal_weight = day_temporal_weights(prepared_train)
    balanced_weight = _balanced_weights(train_target, temporal_weight)
    min_leaf = max(20, min(80, len(prepared_train) // 40))
    positive_tree = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=180,
        max_leaf_nodes=9,
        min_samples_leaf=min_leaf,
        l2_regularization=20.0,
        random_state=random_seed,
    )
    positive_tree.fit(
        x_train,
        train_target,
        sample_weight=balanced_weight,
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
                    C=0.08,
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=random_seed + 1,
                ),
            ),
        ]
    )
    positive_linear.fit(
        x_train,
        train_target,
        model__sample_weight=temporal_weight,
    )
    calibration_raw = _blend_probability(
        positive_tree,
        positive_linear,
        x_calibration,
    )
    probability_calibrator = ProbabilityCalibrator().fit(
        calibration_raw,
        calibration_target.to_numpy(dtype=int),
        day_temporal_weights(prepared_calibration),
    )
    return MarketLicenseBundle(
        positive_tree=positive_tree,
        positive_linear=positive_linear,
        probability_calibrator=probability_calibrator,
        feature_columns=features,
        train_rows=int(len(prepared_train)),
        calibration_rows=int(len(prepared_calibration)),
    )


def rolling_market_license_segments(
    prior_dates: Iterable[str],
    *,
    reserve_final_purge: bool = True,
) -> tuple[list[str], list[str]] | None:
    dates = sorted(set(map(str, prior_dates)))
    final_purge = GATE_PURGE_DAYS if reserve_final_purge else 0
    needed = (
        GATE_TRAIN_DAYS
        + GATE_PURGE_DAYS
        + GATE_CALIBRATION_DAYS
        + final_purge
    )
    if len(dates) < needed:
        return None
    selected = dates[-needed:]
    train = selected[:GATE_TRAIN_DAYS]
    calibration_start = GATE_TRAIN_DAYS + GATE_PURGE_DAYS
    calibration = selected[
        calibration_start : calibration_start + GATE_CALIBRATION_DAYS
    ]
    return train, calibration


def calibrate_market_license_policy(
    scored_calibration: pd.DataFrame,
    *,
    calibration_dates: Iterable[str],
    spec: MarketLicensePolicySpec | None = None,
) -> FrozenMarketLicensePolicy:
    frozen_spec = spec or MarketLicensePolicySpec()
    dates = sorted(set(map(str, calibration_dates)))
    if not dates:
        raise ValueError("V22 policy calibration has no dates")
    calibration = scored_calibration.loc[
        scored_calibration["trade_date"].astype(str).isin(dates)
    ].copy()
    eligible = _license_eligible(calibration, frozen_spec)
    daily_max = (
        eligible.groupby("trade_date", sort=False)[
            "v22_license_probability_lower"
        ]
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
    return FrozenMarketLicensePolicy(
        spec=frozen_spec,
        license_probability_lower_threshold=threshold,
        threshold_calibration_start=dates[0],
        threshold_calibration_end=dates[-1],
        threshold_calibration_days=len(dates),
        threshold_eligible_days=int(len(daily_max)),
    )


def apply_market_license_policy(
    scored: pd.DataFrame,
    policy: FrozenMarketLicensePolicy,
) -> pd.DataFrame:
    eligible = _license_eligible(scored, policy.spec)
    qualified = eligible.loc[
        _numeric(eligible, "v22_license_probability_lower").ge(
            policy.license_probability_lower_threshold
        )
    ].copy()
    if qualified.empty:
        qualified["v22_policy_id"] = policy.policy_id
        return qualified

    qualified["_v22_slot_absolute"] = _slot_absolute(
        qualified["signal_slot"]
    )
    qualified.sort_values(
        [
            "trade_date",
            "_v22_slot_absolute",
            "v22_license_probability_lower",
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
    ].drop(columns="_v22_slot_absolute")
    selected["v22_policy_id"] = policy.policy_id
    selected["v22_license_probability_lower_threshold"] = (
        policy.license_probability_lower_threshold
    )
    selected["v22_target_candidate_day_rate"] = (
        policy.spec.target_candidate_day_rate
    )
    return selected.reset_index(drop=True)


def v22_research_readiness(
    metrics: dict[str, Any],
    *,
    yearly: list[dict[str, Any]],
    temporal_integrity: bool,
    source_integrity: bool,
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
        "minimum_nested_oos_candidates": int(metrics.get("events", 0)) >= 60,
        "minimum_nested_oos_candidate_days": (
            int(metrics.get("candidate_days", 0)) >= 45
        ),
        "practical_candidate_day_rate": (
            0.08 <= float(metrics.get("candidate_day_rate", 0.0)) <= 0.22
        ),
        "minimum_win_rate": float(metrics.get("win_rate", 0.0)) >= 0.55,
        "minimum_wilson_lower": (
            float(metrics.get("win_rate_wilson_lower", 0.0)) >= 0.48
        ),
        "minimum_clustered_win_rate_lower": (
            float(metrics.get("clustered_win_rate_lower", 0.0)) >= 0.48
        ),
        "minimum_margin_hit_rate": (
            float(metrics.get("margin_hit_rate", 0.0)) >= 0.45
        ),
        "maximum_tail_loss_rate": (
            float(metrics.get("tail_loss_rate", 1.0)) <= 0.18
        ),
        "minimum_mean_net_return_pct": (
            float(metrics.get("mean_net_return_pct") or -999.0) >= 0.25
        ),
        "clustered_mean_lower_positive": (
            float(metrics.get("clustered_mean_lower_pct") or -999.0) > 0.0
        ),
        "minimum_profit_factor": (
            float(metrics.get("profit_factor") or 0.0) >= 1.25
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
        "minimum_ten_candidates_each_active_year": minimum_year_events >= 10,
        "minimum_three_positive_calendar_years": positive_years >= 3,
        "worst_calendar_year_above_minus_0_10pct": worst_year >= -0.10,
        "temporal_integrity": bool(temporal_integrity),
        "source_integrity": bool(source_integrity),
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
        "future_shadow_min_candidates": 30,
        "future_shadow_min_candidate_days": 20,
        "reason": (
            "historical_screen_passed_future_shadow_still_required"
            if passed
            else "historical_evidence_insufficient"
        ),
    }


def add_market_economic_metrics(
    metrics: dict[str, Any],
    selected: pd.DataFrame,
) -> dict[str, Any]:
    return add_economic_metrics(metrics, selected)


def _source_eligible(frame: pd.DataFrame) -> pd.Series:
    age = _numeric(frame, "data_age_seconds")
    fresh = age.isna() | age.le(MAX_DATA_AGE_SECONDS)
    return (
        _numeric(frame, "p_round_trip_fill_lower").ge(ROUND_TRIP_FILL_MIN)
        & _numeric(frame, "p_severe_loss").le(SOURCE_SEVERE_LOSS_MAX)
        & _numeric(frame, "probability_model_spread").le(
            SOURCE_MODEL_SPREAD_MAX
        )
        & _numeric(frame, "expected_return_model_spread").le(
            SOURCE_MODEL_SPREAD_MAX
        )
        & fresh
    )


def _license_eligible(
    scored: pd.DataFrame,
    spec: MarketLicensePolicySpec,
) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "v22_license_probability_lower",
        "v22_license_model_spread",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"V22 policy frame missing columns: {missing}")
    return scored.loc[
        _numeric(scored, "v22_license_model_spread").le(
            spec.license_model_spread_max
        )
    ].copy()


def _group_series(
    frame: pd.DataFrame,
    values: pd.Series,
    keys: list[str],
    operation: str,
) -> pd.Series:
    return values.groupby(
        [frame[column] for column in keys],
        sort=False,
    ).transform(operation)


def _blend_probability(
    tree: HistGradientBoostingClassifier,
    linear: Pipeline,
    features: pd.DataFrame,
) -> np.ndarray:
    return (
        0.70 * tree.predict_proba(features)[:, 1]
        + 0.30 * linear.predict_proba(features)[:, 1]
    )


def _balanced_weights(
    target: pd.Series,
    temporal_weight: np.ndarray,
) -> np.ndarray:
    labels = target.to_numpy(dtype=int)
    positive_rate = float(np.mean(labels))
    if not 0.0 < positive_rate < 1.0:
        return np.asarray(temporal_weight, dtype=float)
    class_weight = np.where(
        labels == 1,
        0.50 / positive_rate,
        0.50 / (1.0 - positive_rate),
    )
    return np.asarray(temporal_weight, dtype=float) * class_weight


def _labeled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    net = _numeric(frame, "net_return_pct")
    positive = _numeric(frame, "target_net_positive")
    return frame.loc[net.notna() & positive.notna()].copy()


def _slot_absolute(values: pd.Series) -> pd.Series:
    parsed = values.astype(str).str.extract(
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2})"
    )
    return (
        pd.to_numeric(parsed["hour"], errors="coerce") * 60
        + pd.to_numeric(parsed["minute"], errors="coerce")
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")
