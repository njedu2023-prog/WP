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
from .v34_intraday_path import V34_INTRADAY_PATH_FEATURE_COLUMNS


SCHEMA_VERSION = "wp_v34_full_session_path_ranker_1"
MODEL_TRAIN_DAYS = 252
MODEL_CALIBRATION_DAYS = 42
MODEL_PURGE_DAYS = 2
MINIMUM_TRAIN_ROWS = 4_000
MINIMUM_CALIBRATION_ROWS = 800

FIXED_TARGET_CANDIDATE_DAY_RATE = 0.25
FIXED_MAX_CANDIDATES_PER_DAY = 3
ROUND_TRIP_FILL_MIN = 0.95
SOURCE_SEVERE_LOSS_MAX = 0.45
MAX_DATA_AGE_SECONDS = 420.0

POSITIVE_PROBABILITY_LOWER_MIN = 0.50
MARGIN_PROBABILITY_LOWER_MIN = 0.20
SEVERE_PROBABILITY_UPPER_MAX = 0.40
EXPECTED_NET_RETURN_LOWER_MIN_PCT = -0.25
PROBABILITY_SPREAD_MAX = 0.40
RETURN_SPREAD_MAX_PCT = 5.0
MARGIN_TARGET_PCT = 0.50
SEVERE_TARGET_PCT = -2.00

DELTA_BASE_FEATURES = (
    "v34_session_return_pct",
    "v34_afternoon_return_pct",
    "v34_post_1400_return_pct",
    "v34_session_vwap_gap_pct",
    "v34_session_close_position",
    "v34_amount_acceleration_15m",
    "v34_signed_amount_imbalance",
    "v34_post_1400_signed_amount_imbalance",
    "v34_recent_vs_prior_volatility_ratio",
)
RANK_FEATURES = tuple(
    f"{column}_slot_rank" for column in V34_INTRADAY_PATH_FEATURE_COLUMNS
)
DELTA_FEATURES = tuple(f"{column}_candidate_delta" for column in DELTA_BASE_FEATURES)
CONTEXT_FEATURES = (
    "v34_signal_slot_minute",
    "v34_candidate_appearance_count",
    "v34_minutes_since_prior_candidate",
)
MODEL_FEATURES = (
    *V34_INTRADAY_PATH_FEATURE_COLUMNS,
    *RANK_FEATURES,
    *DELTA_FEATURES,
    *CONTEXT_FEATURES,
)
MODEL_FEATURE_SET = frozenset(MODEL_FEATURES)


@dataclass(frozen=True)
class PathPolicySpec:
    target_candidate_day_rate: float = FIXED_TARGET_CANDIDATE_DAY_RATE
    max_candidates_per_day: int = FIXED_MAX_CANDIDATES_PER_DAY

    @property
    def policy_id(self) -> str:
        return (
            f"v34-path-rate{self.target_candidate_day_rate:.2f}-"
            f"k{self.max_candidates_per_day}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "target_candidate_day_rate": self.target_candidate_day_rate,
            "max_candidates_per_day": self.max_candidates_per_day,
            "round_trip_fill_min": ROUND_TRIP_FILL_MIN,
            "source_severe_loss_max": SOURCE_SEVERE_LOSS_MAX,
            "positive_probability_lower_min": (
                POSITIVE_PROBABILITY_LOWER_MIN
            ),
            "margin_probability_lower_min": MARGIN_PROBABILITY_LOWER_MIN,
            "severe_probability_upper_max": SEVERE_PROBABILITY_UPPER_MAX,
            "expected_net_return_lower_min_pct": (
                EXPECTED_NET_RETURN_LOWER_MIN_PCT
            ),
            "probability_spread_max": PROBABILITY_SPREAD_MAX,
            "return_spread_max_pct": RETURN_SPREAD_MAX_PCT,
            "max_data_age_seconds": MAX_DATA_AGE_SECONDS,
            "first_qualifying_signal_is_immutable": True,
            "no_signal_allowed": True,
        }


@dataclass(frozen=True)
class FrozenPathPolicy:
    spec: PathPolicySpec
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
class PathRankBundle:
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
        matrix = path_feature_matrix(result, self.feature_columns)
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
        return_tree = self.return_tree.predict(matrix)
        return_linear = self.return_linear.predict(matrix)
        expected = self.return_calibrator.predict(
            _blend(return_tree, return_linear).reshape(-1, 1)
        )
        positive_spread = np.abs(raw["positive"][0] - raw["positive"][1])
        margin_spread = np.abs(raw["margin"][0] - raw["margin"][1])
        severe_spread = np.abs(raw["severe"][0] - raw["severe"][1])
        return_spread = np.abs(return_tree - return_linear)

        result["v34_p_positive"] = positive
        result["v34_p_positive_lower"] = np.clip(
            positive - 0.50 * positive_spread - 0.03,
            0.001,
            0.999,
        )
        result["v34_positive_model_spread"] = positive_spread
        result["v34_p_margin"] = margin
        result["v34_p_margin_lower"] = np.clip(
            margin - 0.50 * margin_spread - 0.03,
            0.001,
            0.999,
        )
        result["v34_margin_model_spread"] = margin_spread
        result["v34_p_severe_loss"] = severe
        result["v34_p_severe_loss_upper"] = np.clip(
            severe + 0.50 * severe_spread + 0.03,
            0.001,
            0.999,
        )
        result["v34_severe_model_spread"] = severe_spread
        result["v34_expected_net_return_pct"] = expected
        result["v34_expected_return_model_spread_pct"] = return_spread
        result["v34_expected_net_return_lower_pct"] = (
            expected
            - 0.50 * return_spread
            - self.return_downside_residual_pct
        )
        result["v34_path_score"] = (
            result["v34_expected_net_return_lower_pct"]
            + 1.00 * (result["v34_p_positive_lower"] - 0.50)
            + 0.60 * (result["v34_p_margin_lower"] - 0.30)
            - 1.10 * result["v34_p_severe_loss_upper"]
        )
        return result


def add_path_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {*IDENTITY_COLUMNS, *V34_INTRADAY_PATH_FEATURE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V34 context frame missing columns: {missing}")
    result = frame.copy()
    result["_v34_original_order"] = np.arange(len(result))
    result["v34_signal_slot_minute"] = _slot_minutes(result["signal_slot"])
    for column, rank_column in zip(
        V34_INTRADAY_PATH_FEATURE_COLUMNS,
        RANK_FEATURES,
        strict=True,
    ):
        values = _numeric(result, column)
        result[rank_column] = values.groupby(
            [
                result["trade_date"].astype(str),
                result["signal_slot"].astype(str),
            ],
            sort=False,
        ).rank(method="average", pct=True)
    result.sort_values(
        ["trade_date", "ts_code", "v34_signal_slot_minute"],
        kind="stable",
        inplace=True,
    )
    group = result.groupby(["trade_date", "ts_code"], sort=False)
    result["v34_candidate_appearance_count"] = group.cumcount() + 1
    result["v34_minutes_since_prior_candidate"] = (
        group["v34_signal_slot_minute"].diff().fillna(0.0)
    )
    for source, target in zip(
        DELTA_BASE_FEATURES,
        DELTA_FEATURES,
        strict=True,
    ):
        result[target] = group[source].diff().fillna(0.0)
    result.sort_values("_v34_original_order", kind="stable", inplace=True)
    result.drop(columns="_v34_original_order", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def fit_path_ranker(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    random_seed: int,
    minimum_train_rows: int = MINIMUM_TRAIN_ROWS,
    minimum_calibration_rows: int = MINIMUM_CALIBRATION_ROWS,
) -> PathRankBundle:
    prepared_train = labeled_path_rows(train)
    prepared_calibration = labeled_path_rows(calibration)
    if len(prepared_train) < minimum_train_rows:
        raise ValueError(
            f"V34 has {len(prepared_train)} train rows; "
            f"requires {minimum_train_rows}"
        )
    if len(prepared_calibration) < minimum_calibration_rows:
        raise ValueError(
            f"V34 has {len(prepared_calibration)} calibration rows; "
            f"requires {minimum_calibration_rows}"
        )
    features = active_path_features(prepared_train, prepared_calibration)
    if len(features) < 50:
        raise ValueError(f"V34 has only {len(features)} active features")
    x_train = path_feature_matrix(prepared_train, features)
    x_calibration = path_feature_matrix(prepared_calibration, features)
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
    train_weight = stock_day_equalized_temporal_weights(prepared_train)
    calibration_weight = stock_day_equalized_temporal_weights(
        prepared_calibration
    )
    min_leaf = max(80, min(220, len(prepared_train) // 35))
    trees: dict[str, HistGradientBoostingClassifier] = {}
    linears: dict[str, Pipeline] = {}
    calibrators: dict[str, ProbabilityCalibrator] = {}
    for offset, name in enumerate(("positive", "margin", "severe")):
        target = targets[name]
        if target.nunique() < 2:
            raise ValueError(f"V34 {name} target lacks both classes")
        tree = HistGradientBoostingClassifier(
            learning_rate=0.025,
            max_iter=220,
            max_leaf_nodes=9,
            min_samples_leaf=min_leaf,
            l2_regularization=40.0,
            random_state=random_seed + offset * 101,
        )
        tree.fit(
            x_train,
            target,
            sample_weight=_balanced_weights(target, train_weight),
        )
        linear = _linear_classifier(random_seed + offset * 101 + 1)
        linear.fit(
            x_train,
            target,
            model__sample_weight=train_weight,
        )
        calibrator = ProbabilityCalibrator().fit(
            _blend(
                tree.predict_proba(x_calibration)[:, 1],
                linear.predict_proba(x_calibration)[:, 1],
            ),
            calibration_targets[name].to_numpy(dtype=int),
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
    return_tree.fit(
        x_train,
        net_train.clip(-10.0, 10.0),
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
        net_train.clip(-10.0, 10.0),
        model__sample_weight=train_weight,
    )
    calibration_return_raw = _blend(
        return_tree.predict(x_calibration),
        return_linear.predict(x_calibration),
    )
    return_calibrator = Ridge(alpha=15.0)
    return_calibrator.fit(
        calibration_return_raw.reshape(-1, 1),
        net_calibration.clip(-10.0, 10.0),
        sample_weight=calibration_weight,
    )
    expected = return_calibrator.predict(
        calibration_return_raw.reshape(-1, 1)
    )
    downside = np.maximum(
        expected - net_calibration.to_numpy(dtype=float),
        0.0,
    )
    return PathRankBundle(
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
            calibration_weight,
            0.70,
        ),
        feature_columns=features,
        train_rows=int(len(prepared_train)),
        calibration_rows=int(len(prepared_calibration)),
    )


def labeled_path_rows(frame: pd.DataFrame) -> pd.DataFrame:
    net = _numeric(frame, "net_return_pct")
    target = _numeric(frame, "target_net_positive")
    available = _boolean(frame, "label_available")
    point_in_time = _boolean(frame, "v23_point_in_time_complete")
    path_complete = _boolean(frame, "v34_path_complete")
    consistent = target.eq(net.gt(0.0).astype(float))
    return frame.loc[
        available
        & net.notna()
        & target.notna()
        & consistent
        & point_in_time
        & path_complete
    ].copy()


def active_path_features(
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
            pd.to_numeric(combined[column], errors="coerce").notna().sum()
            >= 20
            and pd.to_numeric(
                combined[column],
                errors="coerce",
            ).nunique(dropna=True)
            > 1
        )
    )


def path_feature_matrix(
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


def calibrate_path_policy(
    scored_calibration: pd.DataFrame,
    *,
    calibration_dates: Iterable[str],
    spec: PathPolicySpec | None = None,
) -> FrozenPathPolicy:
    frozen_spec = spec or PathPolicySpec()
    dates = sorted(set(map(str, calibration_dates)))
    if not dates:
        raise ValueError("V34 policy calibration has no dates")
    eligible = policy_eligible_rows(scored_calibration)
    daily_max = (
        eligible.groupby("trade_date", sort=False)["v34_path_score"]
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
    return FrozenPathPolicy(
        spec=frozen_spec,
        score_threshold=threshold,
        calibration_start=dates[0],
        calibration_end=dates[-1],
        calibration_days=len(dates),
        eligible_days=int(len(daily_max)),
    )


def apply_path_policy(
    scored: pd.DataFrame,
    policy: FrozenPathPolicy,
) -> pd.DataFrame:
    eligible = policy_eligible_rows(scored)
    qualified = eligible.loc[
        _numeric(eligible, "v34_path_score").ge(policy.score_threshold)
    ].copy()
    if qualified.empty:
        qualified["v34_policy_id"] = policy.policy_id
        return qualified
    qualified["_v34_slot_minute"] = _slot_minutes(
        qualified["signal_slot"]
    )
    qualified.sort_values(
        [
            "trade_date",
            "_v34_slot_minute",
            "v34_path_score",
            "ts_code",
        ],
        ascending=[True, True, False, True],
        kind="stable",
        inplace=True,
    )
    first_signal = qualified.drop_duplicates(
        ["trade_date", "ts_code"],
        keep="first",
    )
    within_day = first_signal.groupby("trade_date", sort=False).cumcount()
    selected = first_signal.loc[
        within_day.lt(policy.spec.max_candidates_per_day)
    ].drop(columns="_v34_slot_minute")
    selected["v34_policy_id"] = policy.policy_id
    selected["v34_score_threshold"] = policy.score_threshold
    return selected.reset_index(drop=True)


def policy_eligible_rows(scored: pd.DataFrame) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "v23_point_in_time_complete",
        "v34_path_complete",
        "p_round_trip_fill_lower",
        "p_severe_loss",
        "v34_p_positive_lower",
        "v34_p_margin_lower",
        "v34_p_severe_loss_upper",
        "v34_expected_net_return_lower_pct",
        "v34_positive_model_spread",
        "v34_margin_model_spread",
        "v34_severe_model_spread",
        "v34_expected_return_model_spread_pct",
        "v34_path_score",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"V34 policy frame missing columns: {missing}")
    age = _numeric(scored, "data_age_seconds")
    fresh = age.isna() | (
        age.ge(0.0) & age.le(MAX_DATA_AGE_SECONDS)
    )
    legal_slot = (
        scored["signal_slot"]
        .astype(str)
        .str.replace(":", "", regex=False)
        .between("1420", "1450")
    )
    eligible = (
        _boolean(scored, "v23_point_in_time_complete")
        & _boolean(scored, "v34_path_complete")
        & _numeric(scored, "p_round_trip_fill_lower").ge(
            ROUND_TRIP_FILL_MIN
        )
        & _numeric(scored, "p_severe_loss").le(SOURCE_SEVERE_LOSS_MAX)
        & _numeric(scored, "v34_p_positive_lower").ge(
            POSITIVE_PROBABILITY_LOWER_MIN
        )
        & _numeric(scored, "v34_p_margin_lower").ge(
            MARGIN_PROBABILITY_LOWER_MIN
        )
        & _numeric(scored, "v34_p_severe_loss_upper").le(
            SEVERE_PROBABILITY_UPPER_MAX
        )
        & _numeric(scored, "v34_expected_net_return_lower_pct").ge(
            EXPECTED_NET_RETURN_LOWER_MIN_PCT
        )
        & _numeric(scored, "v34_positive_model_spread").le(
            PROBABILITY_SPREAD_MAX
        )
        & _numeric(scored, "v34_margin_model_spread").le(
            PROBABILITY_SPREAD_MAX
        )
        & _numeric(scored, "v34_severe_model_spread").le(
            PROBABILITY_SPREAD_MAX
        )
        & _numeric(scored, "v34_expected_return_model_spread_pct").le(
            RETURN_SPREAD_MAX_PCT
        )
        & fresh
        & legal_slot
    )
    return scored.loc[eligible].copy()


def validate_feature_contract(features: tuple[str, ...]) -> bool:
    invalid = sorted(set(features) - MODEL_FEATURE_SET)
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
    if len(features) < 50 or invalid or contaminated:
        raise RuntimeError(
            "V34 feature contract violated: "
            f"count={len(features)} invalid={invalid} "
            f"contaminated={contaminated}"
        )
    return True


def validate_selected_contract(
    selected: pd.DataFrame,
    policy: FrozenPathPolicy | None,
) -> None:
    if selected.empty:
        return
    if selected.duplicated(["trade_date", "ts_code"], keep=False).any():
        raise RuntimeError("V34 selected output rewrote a first signal")
    maximum = (
        policy.spec.max_candidates_per_day
        if policy is not None
        else FIXED_MAX_CANDIDATES_PER_DAY
    )
    if int(selected.groupby("trade_date").size().max()) > maximum:
        raise RuntimeError("V34 selected output exceeds fixed daily maximum")
    slot = (
        selected["signal_slot"]
        .astype(str)
        .str.replace(":", "", regex=False)
    )
    if not slot.between("1420", "1450").all():
        raise RuntimeError("V34 selected output contains illegal slot")
    if not _boolean(selected, "v23_point_in_time_complete").all():
        raise RuntimeError("V34 selected output contains incomplete PIT data")
    if not _boolean(selected, "v34_path_complete").all():
        raise RuntimeError("V34 selected output contains incomplete path data")


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


def _slot_minutes(values: pd.Series) -> pd.Series:
    cleaned = values.astype(str).str.replace(":", "", regex=False)
    hours = pd.to_numeric(cleaned.str[:2], errors="coerce")
    minutes = pd.to_numeric(cleaned.str[2:4], errors="coerce")
    return hours * 60.0 + minutes


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )
