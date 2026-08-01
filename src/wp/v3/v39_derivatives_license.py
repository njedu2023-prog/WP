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
from .v24_cross_section import stock_day_equalized_temporal_weights
from .v39_derivatives_daily import (
    V39_FUTURE_FEATURE_COLUMNS,
    V39_OPTION_FEATURE_COLUMNS,
)
from .v22_market_license import MARKET_AGGREGATE_FEATURES


SCHEMA_VERSION = "wp_v39_tminus1_derivatives_license_1"
MODEL_TRAIN_DAYS = 252
MODEL_CALIBRATION_DAYS = 42
MODEL_PURGE_DAYS = 2
MINIMUM_TRAIN_ROWS = 1_000
MINIMUM_CALIBRATION_ROWS = 160

FIXED_TARGET_CANDIDATE_DAY_RATE = 0.25
FIXED_MAX_CANDIDATES_PER_DAY = 3
PROBABILITY_SPREAD_MAX = 0.35
RETURN_SPREAD_MAX_PCT = 4.0
MARGIN_TARGET_PCT = 0.50
SEVERE_TARGET_PCT = -2.00

DERIVATIVE_FEATURES = (
    *V39_FUTURE_FEATURE_COLUMNS,
    *V39_OPTION_FEATURE_COLUMNS,
)
MODEL_FEATURES = (
    "v39_signal_slot_minute",
    *MARKET_AGGREGATE_FEATURES,
    *DERIVATIVE_FEATURES,
)
MODEL_FEATURE_SET = frozenset(MODEL_FEATURES)
FORBIDDEN_FEATURE_TOKENS = (
    "target",
    "label",
    "truth",
    "future_return",
    "gross_return",
    "net_return",
    "t1_",
    "exit_",
    "outcome",
)


@dataclass(frozen=True)
class DerivativesPolicySpec:
    target_candidate_day_rate: float = FIXED_TARGET_CANDIDATE_DAY_RATE
    max_candidates_per_day: int = FIXED_MAX_CANDIDATES_PER_DAY
    probability_spread_max: float = PROBABILITY_SPREAD_MAX
    return_spread_max_pct: float = RETURN_SPREAD_MAX_PCT

    @property
    def policy_id(self) -> str:
        return (
            f"v39-derivatives-rate{self.target_candidate_day_rate:.2f}-"
            f"k{self.max_candidates_per_day}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "target_candidate_day_rate": self.target_candidate_day_rate,
            "max_candidates_per_day": self.max_candidates_per_day,
            "probability_spread_max": self.probability_spread_max,
            "return_spread_max_pct": self.return_spread_max_pct,
            "first_qualifying_signal_is_immutable": True,
            "no_signal_allowed": True,
        }


@dataclass(frozen=True)
class FrozenDerivativesPolicy:
    spec: DerivativesPolicySpec
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


@dataclass
class DerivativesLicenseBundle:
    positive_tree: HistGradientBoostingClassifier
    positive_linear: Pipeline
    margin_tree: HistGradientBoostingClassifier
    margin_linear: Pipeline
    severe_tree: HistGradientBoostingClassifier
    severe_linear: Pipeline
    return_tree: HistGradientBoostingRegressor
    return_linear: Pipeline
    positive_calibrator: ProbabilityCalibrator
    margin_calibrator: ProbabilityCalibrator
    severe_calibrator: ProbabilityCalibrator
    return_calibrator: Ridge
    return_downside_residual_pct: float
    feature_columns: tuple[str, ...]
    train_rows: int
    calibration_rows: int

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        matrix = feature_matrix(result, self.feature_columns)
        raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, tree, linear in (
            ("positive", self.positive_tree, self.positive_linear),
            ("margin", self.margin_tree, self.margin_linear),
            ("severe", self.severe_tree, self.severe_linear),
        ):
            raw[name] = (
                tree.predict_proba(matrix)[:, 1],
                linear.predict_proba(matrix)[:, 1],
            )

        positive = np.clip(
            self.positive_calibrator.predict(_blend(*raw["positive"])),
            0.001,
            0.999,
        )
        margin = np.clip(
            self.margin_calibrator.predict(_blend(*raw["margin"])),
            0.001,
            0.999,
        )
        severe = np.clip(
            self.severe_calibrator.predict(_blend(*raw["severe"])),
            0.001,
            0.999,
        )
        expected_tree = self.return_tree.predict(matrix)
        expected_linear = self.return_linear.predict(matrix)
        expected = self.return_calibrator.predict(
            _blend(expected_tree, expected_linear).reshape(-1, 1)
        )

        positive_spread = np.abs(
            raw["positive"][0] - raw["positive"][1]
        )
        margin_spread = np.abs(raw["margin"][0] - raw["margin"][1])
        severe_spread = np.abs(raw["severe"][0] - raw["severe"][1])
        return_spread = np.abs(expected_tree - expected_linear)
        result["v39_p_positive"] = positive
        result["v39_p_positive_lower"] = np.clip(
            positive - 0.50 * positive_spread - 0.03,
            0.001,
            0.999,
        )
        result["v39_positive_model_spread"] = positive_spread
        result["v39_p_margin"] = margin
        result["v39_p_margin_lower"] = np.clip(
            margin - 0.50 * margin_spread - 0.03,
            0.001,
            0.999,
        )
        result["v39_margin_model_spread"] = margin_spread
        result["v39_p_severe"] = severe
        result["v39_p_severe_upper"] = np.clip(
            severe + 0.50 * severe_spread + 0.03,
            0.001,
            0.999,
        )
        result["v39_severe_model_spread"] = severe_spread
        result["v39_expected_net_return_pct"] = expected
        result["v39_return_model_spread_pct"] = return_spread
        result["v39_expected_net_return_lower_pct"] = (
            expected
            - 0.50 * return_spread
            - self.return_downside_residual_pct
        )
        result["v39_derivatives_score"] = (
            result["v39_expected_net_return_lower_pct"]
            + 1.00 * (result["v39_p_positive_lower"] - 0.50)
            + 0.60 * (result["v39_p_margin_lower"] - 0.30)
            - 1.10 * result["v39_p_severe_upper"]
        )
        return result


def join_derivative_features(
    leaders: pd.DataFrame,
    derivatives: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "trade_date",
        "source_trade_date",
        "v39_tminus1_causal",
        "v39_futures_complete",
        "v39_options_complete",
        "v39_futures_finite",
        "v39_options_finite",
        *DERIVATIVE_FEATURES,
    }
    missing = sorted(required - set(derivatives.columns))
    if missing:
        raise ValueError(f"V39 derivatives frame missing columns: {missing}")
    features = derivatives.loc[:, sorted(required)].copy()
    features["trade_date"] = features["trade_date"].astype(str)
    features["source_trade_date"] = features["source_trade_date"].astype(str)
    if features.duplicated("trade_date", keep=False).any():
        raise RuntimeError("V39 derivatives frame has duplicate trade dates")
    exact_causal = (
        features["source_trade_date"].str.fullmatch(r"\d{8}", na=False)
        & features["trade_date"].str.fullmatch(r"\d{8}", na=False)
        & features["source_trade_date"].lt(features["trade_date"])
        & _boolean(features, "v39_tminus1_causal")
    )
    if not exact_causal.all():
        raise RuntimeError("V39 derivatives frame violates T-1 causality")
    complete = (
        _boolean(features, "v39_futures_complete")
        & _boolean(features, "v39_options_complete")
        & _boolean(features, "v39_futures_finite")
        & _boolean(features, "v39_options_finite")
    )
    features["v39_derivatives_complete"] = complete
    merged = leaders.copy()
    merged["trade_date"] = merged["trade_date"].astype(str)
    merged = merged.merge(
        features,
        on="trade_date",
        how="left",
        validate="many_to_one",
    )
    merged["v39_signal_slot_minute"] = _slot_minutes(
        merged["signal_slot"]
    )
    return merged


def fit_derivatives_license(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    random_seed: int,
    minimum_train_rows: int = MINIMUM_TRAIN_ROWS,
    minimum_calibration_rows: int = MINIMUM_CALIBRATION_ROWS,
) -> DerivativesLicenseBundle:
    prepared_train = labeled_rows(train)
    prepared_calibration = labeled_rows(calibration)
    if len(prepared_train) < minimum_train_rows:
        raise ValueError(
            f"V39 has {len(prepared_train)} train rows; "
            f"requires {minimum_train_rows}"
        )
    if len(prepared_calibration) < minimum_calibration_rows:
        raise ValueError(
            f"V39 has {len(prepared_calibration)} calibration rows; "
            f"requires {minimum_calibration_rows}"
        )
    features = active_model_features(prepared_train, prepared_calibration)
    validate_feature_contract(features)
    x_train = feature_matrix(prepared_train, features)
    x_calibration = feature_matrix(prepared_calibration, features)
    train_weights = stock_day_equalized_temporal_weights(prepared_train)
    calibration_weights = stock_day_equalized_temporal_weights(
        prepared_calibration
    )
    net_train = _numeric(prepared_train, "net_return_pct")
    net_calibration = _numeric(prepared_calibration, "net_return_pct")
    targets = {
        "positive": net_train.gt(0.0).astype(int),
        "margin": net_train.gt(MARGIN_TARGET_PCT).astype(int),
        "severe": net_train.le(SEVERE_TARGET_PCT).astype(int),
    }
    calibration_targets = {
        "positive": net_calibration.gt(0.0).astype(int),
        "margin": net_calibration.gt(MARGIN_TARGET_PCT).astype(int),
        "severe": net_calibration.le(SEVERE_TARGET_PCT).astype(int),
    }
    min_leaf = max(100, min(240, len(prepared_train) // 12))
    trees: dict[str, HistGradientBoostingClassifier] = {}
    linears: dict[str, Pipeline] = {}
    calibrators: dict[str, ProbabilityCalibrator] = {}
    for offset, name in enumerate(("positive", "margin", "severe")):
        target = targets[name]
        calibration_target = calibration_targets[name]
        if target.nunique() < 2 or calibration_target.nunique() < 2:
            raise ValueError(f"V39 {name} target lacks both classes")
        tree = HistGradientBoostingClassifier(
            learning_rate=0.025,
            max_iter=180,
            max_leaf_nodes=7,
            min_samples_leaf=min_leaf,
            l2_regularization=60.0,
            random_state=random_seed + offset * 101,
        )
        tree.fit(
            x_train,
            target,
            sample_weight=_balanced_weights(target, train_weights),
        )
        linear = _linear_classifier(random_seed + offset * 101 + 1)
        linear.fit(
            x_train,
            target,
            model__sample_weight=train_weights,
        )
        calibrator = ProbabilityCalibrator().fit(
            _blend(
                tree.predict_proba(x_calibration)[:, 1],
                linear.predict_proba(x_calibration)[:, 1],
            ),
            calibration_target.to_numpy(dtype=int),
            calibration_weights,
        )
        trees[name] = tree
        linears[name] = linear
        calibrators[name] = calibrator

    return_tree = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.025,
        max_iter=180,
        max_leaf_nodes=7,
        min_samples_leaf=min_leaf,
        l2_regularization=60.0,
        random_state=random_seed + 404,
    )
    return_tree.fit(
        x_train,
        net_train.clip(-10.0, 10.0),
        sample_weight=train_weights,
    )
    return_linear = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=80.0)),
        ]
    )
    return_linear.fit(
        x_train,
        net_train.clip(-10.0, 10.0),
        model__sample_weight=train_weights,
    )
    calibration_raw = _blend(
        return_tree.predict(x_calibration),
        return_linear.predict(x_calibration),
    )
    return_calibrator = Ridge(alpha=25.0)
    return_calibrator.fit(
        calibration_raw.reshape(-1, 1),
        net_calibration.clip(-10.0, 10.0),
        sample_weight=calibration_weights,
    )
    expected = return_calibrator.predict(calibration_raw.reshape(-1, 1))
    downside = np.maximum(
        expected - net_calibration.to_numpy(dtype=float),
        0.0,
    )
    return DerivativesLicenseBundle(
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
        return_downside_residual_pct=_weighted_quantile(
            downside,
            calibration_weights,
            0.70,
        ),
        feature_columns=features,
        train_rows=int(len(prepared_train)),
        calibration_rows=int(len(prepared_calibration)),
    )


def labeled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    available = (
        _numeric(frame, "net_return_pct").notna()
        & _boolean(frame, "label_available")
        & _boolean(frame, "v39_derivatives_complete")
    )
    return frame.loc[available].copy()


def active_model_features(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
) -> tuple[str, ...]:
    combined = pd.concat(
        [
            train.reindex(columns=MODEL_FEATURES),
            calibration.reindex(columns=MODEL_FEATURES),
        ],
        ignore_index=True,
    )
    return tuple(
        column
        for column in MODEL_FEATURES
        if (
            _numeric(combined, column).notna().sum() >= 40
            and _numeric(combined, column).nunique(dropna=True) > 1
        )
    )


def feature_matrix(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    result = frame.reindex(columns=columns).copy()
    for column in columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).astype("float32")
    return result.replace([np.inf, -np.inf], np.nan)


def rolling_segments(
    prior_dates: Iterable[str],
    *,
    reserve_final_purge: bool = True,
) -> tuple[list[str], list[str]] | None:
    dates = sorted(set(map(str, prior_dates)))
    final_purge = MODEL_PURGE_DAYS if reserve_final_purge else 0
    needed = (
        MODEL_TRAIN_DAYS
        + MODEL_PURGE_DAYS
        + MODEL_CALIBRATION_DAYS
        + final_purge
    )
    if len(dates) < needed:
        return None
    selected = dates[-needed:]
    train_dates = selected[:MODEL_TRAIN_DAYS]
    calibration_start = MODEL_TRAIN_DAYS + MODEL_PURGE_DAYS
    calibration_dates = selected[
        calibration_start : calibration_start + MODEL_CALIBRATION_DAYS
    ]
    return train_dates, calibration_dates


def calibrate_policy(
    scored_calibration: pd.DataFrame,
    *,
    calibration_dates: Iterable[str],
    spec: DerivativesPolicySpec | None = None,
) -> FrozenDerivativesPolicy:
    frozen_spec = spec or DerivativesPolicySpec()
    dates = sorted(set(map(str, calibration_dates)))
    if not dates:
        raise ValueError("V39 policy calibration has no dates")
    calibration = scored_calibration.loc[
        scored_calibration["trade_date"].astype(str).isin(dates)
    ].copy()
    eligible = policy_eligible_rows(calibration, frozen_spec)
    daily_max = (
        eligible.groupby("trade_date", sort=False)[
            "v39_derivatives_score"
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
    return FrozenDerivativesPolicy(
        spec=frozen_spec,
        score_threshold=threshold,
        calibration_start=dates[0],
        calibration_end=dates[-1],
        calibration_days=len(dates),
        eligible_days=int(len(daily_max)),
    )


def apply_policy(
    scored: pd.DataFrame,
    policy: FrozenDerivativesPolicy,
) -> pd.DataFrame:
    eligible = policy_eligible_rows(scored, policy.spec)
    qualified = eligible.loc[
        _numeric(eligible, "v39_derivatives_score").ge(
            policy.score_threshold
        )
    ].copy()
    if qualified.empty:
        qualified["v39_policy_id"] = policy.policy_id
        return qualified
    qualified["_v39_slot_minute"] = _slot_minutes(
        qualified["signal_slot"]
    )
    qualified.sort_values(
        [
            "trade_date",
            "_v39_slot_minute",
            "v39_derivatives_score",
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
    ].copy()
    selected.drop(columns="_v39_slot_minute", inplace=True)
    selected["v39_policy_id"] = policy.policy_id
    selected["v39_score_threshold"] = policy.score_threshold
    selected.reset_index(drop=True, inplace=True)
    validate_selected_contract(selected, policy)
    return selected


def policy_eligible_rows(
    scored: pd.DataFrame,
    spec: DerivativesPolicySpec,
) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "v39_derivatives_complete",
        "v39_positive_model_spread",
        "v39_margin_model_spread",
        "v39_severe_model_spread",
        "v39_return_model_spread_pct",
        "v39_derivatives_score",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"V39 policy frame missing columns: {missing}")
    legal_slot = (
        scored["signal_slot"]
        .astype(str)
        .str.replace(":", "", regex=False)
        .between("1420", "1450")
    )
    eligible = (
        _boolean(scored, "v39_derivatives_complete")
        & _numeric(scored, "v39_positive_model_spread").le(
            spec.probability_spread_max
        )
        & _numeric(scored, "v39_margin_model_spread").le(
            spec.probability_spread_max
        )
        & _numeric(scored, "v39_severe_model_spread").le(
            spec.probability_spread_max
        )
        & _numeric(scored, "v39_return_model_spread_pct").le(
            spec.return_spread_max_pct
        )
        & legal_slot
    )
    return scored.loc[eligible].copy()


def validate_feature_contract(features: tuple[str, ...]) -> bool:
    invalid = sorted(set(features) - MODEL_FEATURE_SET)
    contaminated = [
        feature
        for feature in features
        if any(token in feature.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    derivatives_count = len(set(features).intersection(DERIVATIVE_FEATURES))
    aggregates_count = len(
        set(features).intersection(MARKET_AGGREGATE_FEATURES)
    )
    if (
        derivatives_count < 45
        or aggregates_count < 10
        or invalid
        or contaminated
    ):
        raise RuntimeError(
            "V39 feature contract violated: "
            f"derivatives={derivatives_count} "
            f"aggregates={aggregates_count} "
            f"invalid={invalid} contaminated={contaminated}"
        )
    return True


def validate_selected_contract(
    selected: pd.DataFrame,
    policy: FrozenDerivativesPolicy | None,
) -> None:
    if selected.empty:
        return
    if selected.duplicated(["trade_date", "ts_code"], keep=False).any():
        raise RuntimeError("V39 selected output rewrote a first signal")
    maximum = (
        policy.spec.max_candidates_per_day
        if policy is not None
        else FIXED_MAX_CANDIDATES_PER_DAY
    )
    if int(selected.groupby("trade_date").size().max()) > maximum:
        raise RuntimeError("V39 selected output exceeds the daily maximum")
    slots = (
        selected["signal_slot"]
        .astype(str)
        .str.replace(":", "", regex=False)
    )
    if not slots.between("1420", "1450").all():
        raise RuntimeError("V39 selected output contains an illegal slot")
    if not _boolean(selected, "v39_derivatives_complete").all():
        raise RuntimeError("V39 selected output has incomplete derivatives")
    if _numeric(selected, "signal_price").isna().any():
        raise RuntimeError("V39 selected output lost immutable signal prices")


def research_readiness(
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
        "minimum_nested_oos_candidates": int(metrics.get("events", 0)) >= 100,
        "minimum_nested_oos_candidate_days": (
            int(metrics.get("candidate_days", 0)) >= 70
        ),
        "practical_candidate_day_rate": (
            0.15 <= float(metrics.get("candidate_day_rate", 0.0)) <= 0.35
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
        "future_shadow_min_candidates": 45,
        "future_shadow_min_candidate_days": 30,
        "reason": (
            "historical_screen_passed_future_shadow_still_required"
            if passed
            else "historical_evidence_insufficient"
        ),
    }


def _linear_classifier(random_seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.03,
                    max_iter=1_500,
                    class_weight="balanced",
                    random_state=random_seed,
                ),
            ),
        ]
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
    array = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    mask = np.isfinite(array) & np.isfinite(weight) & (weight > 0.0)
    if not mask.any():
        return 0.0
    order = np.argsort(array[mask])
    sorted_values = array[mask][order]
    sorted_weights = weight[mask][order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = float(np.clip(quantile, 0.0, 1.0)) * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, cutoff)), len(order) - 1)
    return float(sorted_values[index])


def _blend(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return 0.70 * np.asarray(first, dtype=float) + 0.30 * np.asarray(
        second,
        dtype=float,
    )


def _slot_minutes(values: pd.Series) -> pd.Series:
    parsed = values.astype(str).str.extract(
        r"(?P<hour>\d{1,2}):?(?P<minute>\d{2})"
    )
    return (
        pd.to_numeric(parsed["hour"], errors="coerce") * 60
        + pd.to_numeric(parsed["minute"], errors="coerce")
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
    return values.astype(str).str.lower().isin({"1", "true", "yes"})
