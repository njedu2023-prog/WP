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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .meta_alpha import IDENTITY_COLUMNS, ProbabilityCalibrator
from .v17_selector import (
    active_feature_columns,
    day_temporal_weights,
    feature_matrix,
    prepare_selector_frame,
)
from .v20_opportunity import V20_FEATURES, build_opportunity_leaders


SCHEMA_VERSION = "wp_v21_economic_margin_1"
DEFAULT_LEADERS_PER_SLOT = 3
GATE_TRAIN_DAYS = 126
GATE_CALIBRATION_DAYS = 42
GATE_PURGE_DAYS = 2

MARGIN_TARGET_PCT = 0.50
TAIL_LOSS_TARGET_PCT = -2.00
FIXED_TARGET_CANDIDATE_DAY_RATE = 0.12
FIXED_MAX_CANDIDATES_PER_DAY = 2

MARGIN_MODEL_SPREAD_MAX = 0.25
TAIL_MODEL_SPREAD_MAX = 0.25
TAIL_PROBABILITY_UPPER_MAX = 0.35
RETURN_Q20_MIN_PCT = -3.00
ROUND_TRIP_FILL_MIN = 0.95
SOURCE_SEVERE_LOSS_MAX = 0.40
MAX_DATA_AGE_SECONDS = 420.0

_V20_TO_V21 = {
    column: column.replace("v20_", "v21_", 1)
    for column in V20_FEATURES
    if column.startswith("v20_")
}
V21_FEATURES = tuple(_V20_TO_V21.get(column, column) for column in V20_FEATURES)


@dataclass
class MarginGateBundle:
    margin_tree: HistGradientBoostingClassifier
    margin_linear: Pipeline
    tail_tree: HistGradientBoostingClassifier
    tail_linear: Pipeline
    return_q20: HistGradientBoostingRegressor
    margin_calibrator: ProbabilityCalibrator
    tail_calibrator: ProbabilityCalibrator
    return_q20_adjustment: float
    feature_columns: tuple[str, ...]
    train_rows: int
    calibration_rows: int

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        scored = prepare_selector_frame(frame)
        features = feature_matrix(scored, self.feature_columns)

        margin_tree_probability = self.margin_tree.predict_proba(features)[:, 1]
        margin_linear_probability = self.margin_linear.predict_proba(features)[
            :, 1
        ]
        margin_raw = (
            0.70 * margin_tree_probability
            + 0.30 * margin_linear_probability
        )
        margin_probability = np.clip(
            self.margin_calibrator.predict(margin_raw),
            0.001,
            0.999,
        )
        margin_spread = np.abs(
            margin_tree_probability - margin_linear_probability
        )

        tail_tree_probability = self.tail_tree.predict_proba(features)[:, 1]
        tail_linear_probability = self.tail_linear.predict_proba(features)[:, 1]
        tail_raw = (
            0.70 * tail_tree_probability + 0.30 * tail_linear_probability
        )
        tail_probability = np.clip(
            self.tail_calibrator.predict(tail_raw),
            0.001,
            0.999,
        )
        tail_spread = np.abs(tail_tree_probability - tail_linear_probability)

        scored["v21_margin_probability"] = margin_probability
        scored["v21_margin_model_spread"] = margin_spread
        scored["v21_margin_probability_lower"] = np.clip(
            margin_probability - 0.50 * margin_spread - 0.02,
            0.001,
            0.999,
        )
        scored["v21_tail_probability"] = tail_probability
        scored["v21_tail_model_spread"] = tail_spread
        scored["v21_tail_probability_upper"] = np.clip(
            tail_probability + 0.50 * tail_spread + 0.02,
            0.001,
            0.999,
        )
        scored["v21_return_q20_pct"] = (
            self.return_q20.predict(features) + self.return_q20_adjustment
        )
        scored["v21_rank_score"] = scored["v21_margin_probability_lower"]
        scored["v21_rank_pct"] = scored.groupby(
            ["trade_date", "signal_slot"],
            sort=False,
        )["v21_rank_score"].rank(method="average", pct=True)
        return scored


@dataclass(frozen=True)
class MarginPolicySpec:
    target_candidate_day_rate: float = FIXED_TARGET_CANDIDATE_DAY_RATE
    max_candidates_per_day: int = FIXED_MAX_CANDIDATES_PER_DAY
    margin_model_spread_max: float = MARGIN_MODEL_SPREAD_MAX
    tail_model_spread_max: float = TAIL_MODEL_SPREAD_MAX
    tail_probability_upper_max: float = TAIL_PROBABILITY_UPPER_MAX
    return_q20_min_pct: float = RETURN_Q20_MIN_PCT
    round_trip_fill_min: float = ROUND_TRIP_FILL_MIN
    source_severe_loss_max: float = SOURCE_SEVERE_LOSS_MAX
    max_data_age_seconds: float = MAX_DATA_AGE_SECONDS

    @property
    def policy_id(self) -> str:
        return (
            f"margin{MARGIN_TARGET_PCT:.2f}-"
            f"rate{self.target_candidate_day_rate:.2f}-"
            f"tail{self.tail_probability_upper_max:.2f}-"
            f"q20{self.return_q20_min_pct:.2f}-"
            f"k{self.max_candidates_per_day}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "margin_target_pct": MARGIN_TARGET_PCT,
            "tail_loss_target_pct": TAIL_LOSS_TARGET_PCT,
            "target_candidate_day_rate": self.target_candidate_day_rate,
            "max_candidates_per_day": self.max_candidates_per_day,
            "margin_model_spread_max": self.margin_model_spread_max,
            "tail_model_spread_max": self.tail_model_spread_max,
            "tail_probability_upper_max": self.tail_probability_upper_max,
            "return_q20_min_pct": self.return_q20_min_pct,
            "round_trip_fill_min": self.round_trip_fill_min,
            "source_severe_loss_max": self.source_severe_loss_max,
            "max_data_age_seconds": self.max_data_age_seconds,
        }


@dataclass(frozen=True)
class FrozenMarginPolicy:
    spec: MarginPolicySpec
    margin_probability_lower_threshold: float
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
            "margin_probability_lower_threshold": (
                self.margin_probability_lower_threshold
            ),
            "threshold_calibration_start": self.threshold_calibration_start,
            "threshold_calibration_end": self.threshold_calibration_end,
            "threshold_calibration_days": self.threshold_calibration_days,
            "threshold_eligible_days": self.threshold_eligible_days,
        }


def build_margin_leaders(
    frame: pd.DataFrame,
    *,
    leaders_per_slot: int = DEFAULT_LEADERS_PER_SLOT,
) -> pd.DataFrame:
    leaders = build_opportunity_leaders(
        frame,
        leaders_per_slot=leaders_per_slot,
    )
    rename = {
        column: column.replace("v20_", "v21_", 1)
        for column in leaders.columns
        if column.startswith("v20_")
    }
    return leaders.rename(columns=rename)


def fit_margin_gate(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    random_seed: int,
    minimum_train_rows: int = 1_500,
    minimum_calibration_rows: int = 350,
) -> MarginGateBundle:
    prepared_train = _labeled_rows(prepare_selector_frame(train))
    prepared_calibration = _labeled_rows(
        prepare_selector_frame(calibration)
    )
    if len(prepared_train) < minimum_train_rows:
        raise ValueError(
            f"V21 has only {len(prepared_train)} train rows; "
            f"requires {minimum_train_rows}"
        )
    if len(prepared_calibration) < minimum_calibration_rows:
        raise ValueError(
            f"V21 has only {len(prepared_calibration)} calibration rows; "
            f"requires {minimum_calibration_rows}"
        )

    features = active_feature_columns(
        prepared_train,
        prepared_calibration,
        candidates=V21_FEATURES,
    )
    x_train = feature_matrix(prepared_train, features)
    x_calibration = feature_matrix(prepared_calibration, features)
    train_returns = _numeric(prepared_train, "net_return_pct")
    calibration_returns = _numeric(prepared_calibration, "net_return_pct")
    margin_train = train_returns.gt(MARGIN_TARGET_PCT).astype(int)
    margin_calibration = calibration_returns.gt(MARGIN_TARGET_PCT).astype(int)
    tail_train = train_returns.le(TAIL_LOSS_TARGET_PCT).astype(int)
    tail_calibration = calibration_returns.le(
        TAIL_LOSS_TARGET_PCT
    ).astype(int)
    for name, target in (
        ("margin", margin_train),
        ("tail", tail_train),
    ):
        if target.nunique() < 2:
            raise ValueError(f"V21 {name} target lacks both classes")

    train_weight = day_temporal_weights(prepared_train)
    calibration_weight = day_temporal_weights(prepared_calibration)
    min_leaf = max(40, min(160, len(prepared_train) // 80))

    margin_tree, margin_linear = _fit_classifier_pair(
        x_train,
        margin_train,
        train_weight,
        min_leaf=min_leaf,
        random_seed=random_seed,
    )
    margin_calibrator = ProbabilityCalibrator().fit(
        _blend_probability(margin_tree, margin_linear, x_calibration),
        margin_calibration.to_numpy(dtype=int),
        calibration_weight,
    )

    tail_tree, tail_linear = _fit_classifier_pair(
        x_train,
        tail_train,
        train_weight,
        min_leaf=min_leaf,
        random_seed=random_seed + 10_000,
    )
    tail_calibrator = ProbabilityCalibrator().fit(
        _blend_probability(tail_tree, tail_linear, x_calibration),
        tail_calibration.to_numpy(dtype=int),
        calibration_weight,
    )

    return_q20 = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.20,
        learning_rate=0.035,
        max_iter=180,
        max_leaf_nodes=11,
        min_samples_leaf=min_leaf,
        l2_regularization=15.0,
        random_state=random_seed + 20_000,
    )
    return_q20.fit(
        x_train,
        train_returns,
        sample_weight=train_weight,
    )
    return_q20_residual = (
        calibration_returns.to_numpy(dtype=float)
        - return_q20.predict(x_calibration)
    )
    return MarginGateBundle(
        margin_tree=margin_tree,
        margin_linear=margin_linear,
        tail_tree=tail_tree,
        tail_linear=tail_linear,
        return_q20=return_q20,
        margin_calibrator=margin_calibrator,
        tail_calibrator=tail_calibrator,
        return_q20_adjustment=_weighted_quantile(
            return_q20_residual,
            calibration_weight,
            0.20,
        ),
        feature_columns=features,
        train_rows=int(len(prepared_train)),
        calibration_rows=int(len(prepared_calibration)),
    )


def rolling_margin_model_segments(
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


def calibrate_margin_policy(
    scored_calibration: pd.DataFrame,
    *,
    calibration_dates: Iterable[str],
    spec: MarginPolicySpec | None = None,
) -> FrozenMarginPolicy:
    frozen_spec = spec or MarginPolicySpec()
    dates = sorted(set(map(str, calibration_dates)))
    if not dates:
        raise ValueError("V21 policy calibration has no dates")
    calibration = scored_calibration.loc[
        scored_calibration["trade_date"].astype(str).isin(dates)
    ].copy()
    eligible = _base_eligible(calibration, frozen_spec)
    daily_max = (
        eligible.groupby("trade_date", sort=False)[
            "v21_margin_probability_lower"
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
    return FrozenMarginPolicy(
        spec=frozen_spec,
        margin_probability_lower_threshold=threshold,
        threshold_calibration_start=dates[0],
        threshold_calibration_end=dates[-1],
        threshold_calibration_days=len(dates),
        threshold_eligible_days=int(len(daily_max)),
    )


def apply_margin_policy(
    scored: pd.DataFrame,
    policy: FrozenMarginPolicy,
) -> pd.DataFrame:
    eligible = _base_eligible(scored, policy.spec)
    qualified = eligible.loc[
        _numeric(eligible, "v21_margin_probability_lower").ge(
            policy.margin_probability_lower_threshold
        )
    ].copy()
    if qualified.empty:
        qualified["v21_policy_id"] = policy.policy_id
        return qualified

    qualified["_v21_slot_absolute"] = _slot_absolute(
        qualified["signal_slot"]
    )
    qualified.sort_values(
        [
            "trade_date",
            "_v21_slot_absolute",
            "v21_margin_probability_lower",
            "v21_stock_score",
            "ts_code",
        ],
        ascending=[True, True, False, False, True],
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
    ].drop(columns="_v21_slot_absolute")
    selected["v21_policy_id"] = policy.policy_id
    selected["v21_margin_probability_lower_threshold"] = (
        policy.margin_probability_lower_threshold
    )
    selected["v21_target_candidate_day_rate"] = (
        policy.spec.target_candidate_day_rate
    )
    return selected.reset_index(drop=True)


def add_economic_metrics(
    metrics: dict[str, Any],
    selected: pd.DataFrame,
) -> dict[str, Any]:
    result = dict(metrics)
    returns = _numeric(selected, "net_return_pct").dropna()
    result["margin_target_pct"] = MARGIN_TARGET_PCT
    result["margin_hit_rate"] = (
        float(returns.gt(MARGIN_TARGET_PCT).mean()) if len(returns) else 0.0
    )
    result["tail_loss_target_pct"] = TAIL_LOSS_TARGET_PCT
    result["tail_loss_rate"] = (
        float(returns.le(TAIL_LOSS_TARGET_PCT).mean()) if len(returns) else 0.0
    )
    return result


def v21_research_readiness(
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


def _base_eligible(
    scored: pd.DataFrame,
    spec: MarginPolicySpec,
) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "v21_margin_probability_lower",
        "v21_margin_model_spread",
        "v21_tail_probability_upper",
        "v21_tail_model_spread",
        "v21_return_q20_pct",
        "v21_stock_score",
        "p_severe_loss",
        "p_round_trip_fill_lower",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"V21 policy frame missing columns: {missing}")
    age = _numeric(scored, "data_age_seconds")
    fresh = age.isna() | age.le(spec.max_data_age_seconds)
    mask = (
        _numeric(scored, "v21_margin_model_spread").le(
            spec.margin_model_spread_max
        )
        & _numeric(scored, "v21_tail_model_spread").le(
            spec.tail_model_spread_max
        )
        & _numeric(scored, "v21_tail_probability_upper").le(
            spec.tail_probability_upper_max
        )
        & _numeric(scored, "v21_return_q20_pct").ge(
            spec.return_q20_min_pct
        )
        & _numeric(scored, "p_round_trip_fill_lower").ge(
            spec.round_trip_fill_min
        )
        & _numeric(scored, "p_severe_loss").le(
            spec.source_severe_loss_max
        )
        & fresh
    )
    return scored.loc[mask].copy()


def _fit_classifier_pair(
    features: pd.DataFrame,
    target: pd.Series,
    temporal_weight: np.ndarray,
    *,
    min_leaf: int,
    random_seed: int,
) -> tuple[HistGradientBoostingClassifier, Pipeline]:
    balanced_weight = _balanced_weights(target, temporal_weight)
    tree = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=180,
        max_leaf_nodes=11,
        min_samples_leaf=min_leaf,
        l2_regularization=15.0,
        random_state=random_seed,
    )
    tree.fit(features, target, sample_weight=balanced_weight)
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
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=random_seed + 1,
                ),
            ),
        ]
    )
    linear.fit(
        features,
        target,
        model__sample_weight=temporal_weight,
    )
    return tree, linear


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


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    clean = pd.DataFrame(
        {
            "value": np.asarray(values, dtype=float),
            "weight": np.asarray(weights, dtype=float),
        }
    ).dropna()
    if clean.empty:
        return 0.0
    clean.sort_values("value", kind="stable", inplace=True)
    cumulative = clean["weight"].cumsum()
    cutoff = float(clean["weight"].sum()) * float(quantile)
    index = int(cumulative.searchsorted(cutoff, side="left"))
    return float(clean.iloc[min(index, len(clean) - 1)]["value"])


def _labeled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    net = pd.to_numeric(frame.get("net_return_pct"), errors="coerce")
    return frame.loc[net.notna()].copy()


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
