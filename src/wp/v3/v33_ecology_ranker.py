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
from .v25_ranker import (
    build_pairwise_examples,
    logistic_model,
    predict_pairwise_scores,
    slot_absolute,
    stock_day_equalized_weights,
)
from .v33_limit_ecology import V33_LIMIT_ECOLOGY_FEATURE_COLUMNS


SCHEMA_VERSION = "wp_v33_ecology_ranker_1"
MODEL_TRAIN_DAYS = 252
MODEL_CALIBRATION_DAYS = 42
MODEL_PURGE_DAYS = 2
MINIMUM_TRAIN_ROWS = 4_000
MINIMUM_CALIBRATION_ROWS = 800
MINIMUM_TRAIN_PAIR_ROWS = 12_000
MINIMUM_CALIBRATION_PAIR_ROWS = 2_000

FIXED_TARGET_CANDIDATE_DAY_RATE = 0.25
FIXED_MAX_CANDIDATES_PER_DAY: int | None = None
ROUND_TRIP_FILL_MIN = 0.95
SOURCE_SEVERE_LOSS_MAX = 0.45
SOURCE_PROBABILITY_SPREAD_MAX = 0.40
SOURCE_RETURN_SPREAD_MAX_PCT = 5.0
MAX_DATA_AGE_SECONDS = 420.0

POSITIVE_PROBABILITY_MIN = 0.50
EXPECTED_NET_RETURN_MIN_PCT = 0.00
SEVERE_PROBABILITY_MAX = 0.35
PAIRWISE_SCORE_MIN = 0.50
SEVERE_TARGET_PCT = -2.00

EXPECTED_RETURN_SCORE_WEIGHT = 1.00
POSITIVE_SCORE_WEIGHT = 1.00
PAIRWISE_SCORE_WEIGHT = 0.75
SEVERE_SCORE_WEIGHT = 1.25

MODEL_FEATURES = tuple(V33_LIMIT_ECOLOGY_FEATURE_COLUMNS)
MODEL_FEATURE_SET = frozenset(MODEL_FEATURES)


@dataclass(frozen=True)
class EcologyPolicySpec:
    target_candidate_day_rate: float = FIXED_TARGET_CANDIDATE_DAY_RATE
    max_candidates_per_day: int | None = FIXED_MAX_CANDIDATES_PER_DAY

    @property
    def policy_id(self) -> str:
        candidate_limit = (
            "all"
            if self.max_candidates_per_day is None
            else f"k{self.max_candidates_per_day}"
        )
        return (
            f"v33-ecology-rate{self.target_candidate_day_rate:.2f}-"
            f"{candidate_limit}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "target_candidate_day_rate": self.target_candidate_day_rate,
            "max_candidates_per_day": self.max_candidates_per_day,
            "round_trip_fill_min": ROUND_TRIP_FILL_MIN,
            "source_severe_loss_max": SOURCE_SEVERE_LOSS_MAX,
            "source_probability_spread_max": (
                SOURCE_PROBABILITY_SPREAD_MAX
            ),
            "source_return_spread_max_pct": (
                SOURCE_RETURN_SPREAD_MAX_PCT
            ),
            "positive_probability_min": POSITIVE_PROBABILITY_MIN,
            "expected_net_return_min_pct": (
                EXPECTED_NET_RETURN_MIN_PCT
            ),
            "severe_probability_max": SEVERE_PROBABILITY_MAX,
            "pairwise_score_min": PAIRWISE_SCORE_MIN,
            "max_data_age_seconds": MAX_DATA_AGE_SECONDS,
            "current_ecology_active_required": True,
            "multiple_candidates_per_slot_allowed": True,
            "first_signal_is_immutable": True,
            "no_signal_allowed": True,
        }


@dataclass(frozen=True)
class FrozenEcologyPolicy:
    spec: EcologyPolicySpec
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
class EcologyRankBundle:
    positive_tree: HistGradientBoostingClassifier
    positive_linear: Pipeline
    severe_tree: HistGradientBoostingClassifier
    return_tree: HistGradientBoostingRegressor
    pairwise_model: Any
    positive_calibrator: ProbabilityCalibrator
    severe_calibrator: ProbabilityCalibrator
    return_calibrator: Ridge
    pairwise_calibrator: ProbabilityCalibrator
    feature_columns: tuple[str, ...]
    train_rows: int
    calibration_rows: int
    train_pair_rows: int
    calibration_pair_rows: int

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        matrix = ecology_feature_matrix(result, self.feature_columns)
        raw_positive = _blended_probability(
            self.positive_tree,
            self.positive_linear,
            matrix,
        )
        result["v33_p_positive"] = np.clip(
            self.positive_calibrator.predict(raw_positive),
            0.001,
            0.999,
        )
        raw_severe = self.severe_tree.predict_proba(matrix)[:, 1]
        result["v33_p_severe_loss"] = np.clip(
            self.severe_calibrator.predict(raw_severe),
            0.001,
            0.999,
        )
        raw_return = self.return_tree.predict(matrix)
        result["v33_expected_net_return_pct"] = (
            self.return_calibrator.predict(
                np.asarray(raw_return, dtype=float).reshape(-1, 1)
            )
        )
        result["v33_within_slot_rank_score"] = predict_pairwise_scores(
            result,
            feature_columns=self.feature_columns,
            model=self.pairwise_model,
            calibrator=self.pairwise_calibrator,
        )
        result["v33_ecology_score"] = (
            EXPECTED_RETURN_SCORE_WEIGHT
            * result["v33_expected_net_return_pct"]
            + POSITIVE_SCORE_WEIGHT * (result["v33_p_positive"] - 0.50)
            + PAIRWISE_SCORE_WEIGHT
            * (result["v33_within_slot_rank_score"] - 0.50)
            - SEVERE_SCORE_WEIGHT * result["v33_p_severe_loss"]
        )
        return result


def fit_ecology_ranker(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    random_seed: int,
    minimum_train_rows: int = MINIMUM_TRAIN_ROWS,
    minimum_calibration_rows: int = MINIMUM_CALIBRATION_ROWS,
    minimum_train_pair_rows: int = MINIMUM_TRAIN_PAIR_ROWS,
    minimum_calibration_pair_rows: int = MINIMUM_CALIBRATION_PAIR_ROWS,
) -> EcologyRankBundle:
    prepared_train = labeled_ecology_rows(train)
    prepared_calibration = labeled_ecology_rows(calibration)
    if len(prepared_train) < minimum_train_rows:
        raise ValueError(
            f"V33 has {len(prepared_train)} train rows; "
            f"requires {minimum_train_rows}"
        )
    if len(prepared_calibration) < minimum_calibration_rows:
        raise ValueError(
            f"V33 has {len(prepared_calibration)} calibration rows; "
            f"requires {minimum_calibration_rows}"
        )
    features = active_ecology_features(
        prepared_train,
        prepared_calibration,
    )
    if len(features) < 12:
        raise ValueError(f"V33 has only {len(features)} active features")
    x_train = ecology_feature_matrix(prepared_train, features)
    x_calibration = ecology_feature_matrix(
        prepared_calibration,
        features,
    )
    train_net = _numeric(prepared_train, "net_return_pct")
    calibration_net = _numeric(
        prepared_calibration,
        "net_return_pct",
    )
    train_weight = stock_day_equalized_weights(prepared_train)
    calibration_weight = stock_day_equalized_weights(
        prepared_calibration
    )

    positive_train = train_net.gt(0.0).astype(int)
    positive_calibration = calibration_net.gt(0.0).astype(int)
    _require_two_classes(positive_train, "positive")
    positive_tree = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=160,
        max_leaf_nodes=11,
        min_samples_leaf=180,
        l2_regularization=12.0,
        random_state=random_seed,
    )
    positive_tree.fit(
        x_train,
        positive_train,
        sample_weight=_balanced_weights(
            positive_train,
            train_weight,
        ),
    )
    positive_linear = _linear_probability_model()
    positive_linear.fit(
        x_train,
        positive_train,
        model__sample_weight=_balanced_weights(
            positive_train,
            train_weight,
        ),
    )
    positive_calibrator = ProbabilityCalibrator().fit(
        _blended_probability(
            positive_tree,
            positive_linear,
            x_calibration,
        ),
        positive_calibration.to_numpy(dtype=int),
        calibration_weight,
    )

    severe_train = train_net.le(SEVERE_TARGET_PCT).astype(int)
    severe_calibration = calibration_net.le(SEVERE_TARGET_PCT).astype(int)
    _require_two_classes(severe_train, "severe")
    severe_tree = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=140,
        max_leaf_nodes=9,
        min_samples_leaf=200,
        l2_regularization=15.0,
        random_state=random_seed + 1,
    )
    severe_tree.fit(
        x_train,
        severe_train,
        sample_weight=_balanced_weights(
            severe_train,
            train_weight,
        ),
    )
    severe_calibrator = ProbabilityCalibrator().fit(
        severe_tree.predict_proba(x_calibration)[:, 1],
        severe_calibration.to_numpy(dtype=int),
        calibration_weight,
    )

    return_tree = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.035,
        max_iter=180,
        max_leaf_nodes=11,
        min_samples_leaf=180,
        l2_regularization=15.0,
        random_state=random_seed + 2,
    )
    return_tree.fit(
        x_train,
        train_net.clip(-10.0, 15.0),
        sample_weight=train_weight,
    )
    return_calibrator = Ridge(alpha=25.0)
    return_calibrator.fit(
        return_tree.predict(x_calibration).reshape(-1, 1),
        calibration_net.clip(-10.0, 15.0),
        sample_weight=calibration_weight,
    )

    train_pairs, train_pair_target, train_pair_weight = (
        build_pairwise_examples(prepared_train, features)
    )
    calibration_pairs, calibration_pair_target, calibration_pair_weight = (
        build_pairwise_examples(prepared_calibration, features)
    )
    if len(train_pairs) < minimum_train_pair_rows:
        raise ValueError(
            f"V33 has {len(train_pairs)} train pair rows; "
            f"requires {minimum_train_pair_rows}"
        )
    if len(calibration_pairs) < minimum_calibration_pair_rows:
        raise ValueError(
            f"V33 has {len(calibration_pairs)} calibration pair rows; "
            f"requires {minimum_calibration_pair_rows}"
        )
    pairwise_model = logistic_model(random_seed + 3)
    pairwise_model.fit(
        train_pairs,
        train_pair_target,
        model__sample_weight=train_pair_weight,
    )
    pairwise_calibrator = ProbabilityCalibrator().fit(
        pairwise_model.predict_proba(calibration_pairs)[:, 1],
        calibration_pair_target,
        calibration_pair_weight,
    )
    return EcologyRankBundle(
        positive_tree=positive_tree,
        positive_linear=positive_linear,
        severe_tree=severe_tree,
        return_tree=return_tree,
        pairwise_model=pairwise_model,
        positive_calibrator=positive_calibrator,
        severe_calibrator=severe_calibrator,
        return_calibrator=return_calibrator,
        pairwise_calibrator=pairwise_calibrator,
        feature_columns=features,
        train_rows=int(len(prepared_train)),
        calibration_rows=int(len(prepared_calibration)),
        train_pair_rows=int(len(train_pairs)),
        calibration_pair_rows=int(len(calibration_pairs)),
    )


def labeled_ecology_rows(frame: pd.DataFrame) -> pd.DataFrame:
    net = _numeric(frame, "net_return_pct")
    target = _numeric(frame, "target_net_positive")
    available = _boolean(frame, "label_available")
    point_in_time = _boolean(frame, "v23_point_in_time_complete")
    ecology_complete = _boolean(frame, "v33_ecology_features_complete")
    consistent = target.eq(net.gt(0.0).astype(float))
    return frame.loc[
        available
        & net.notna()
        & target.notna()
        & consistent
        & point_in_time
        & ecology_complete
    ].copy()


def active_ecology_features(
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
    active = []
    for column in MODEL_FEATURES:
        values = pd.to_numeric(combined[column], errors="coerce")
        if (
            int(values.notna().sum()) >= 20
            and int(values.nunique(dropna=True)) > 1
        ):
            active.append(column)
    return tuple(active)


def ecology_feature_matrix(
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


def calibrate_ecology_policy(
    scored_calibration: pd.DataFrame,
    *,
    calibration_dates: Iterable[str],
    spec: EcologyPolicySpec | None = None,
) -> FrozenEcologyPolicy:
    frozen_spec = spec or EcologyPolicySpec()
    dates = sorted(set(map(str, calibration_dates)))
    if not dates:
        raise ValueError("V33 policy calibration has no dates")
    eligible = policy_eligible_rows(scored_calibration)
    daily_max = (
        eligible.groupby("trade_date", sort=False)["v33_ecology_score"]
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
    return FrozenEcologyPolicy(
        spec=frozen_spec,
        score_threshold=threshold,
        calibration_start=dates[0],
        calibration_end=dates[-1],
        calibration_days=len(dates),
        eligible_days=int(len(daily_max)),
    )


def apply_ecology_policy(
    scored: pd.DataFrame,
    policy: FrozenEcologyPolicy,
) -> pd.DataFrame:
    eligible = policy_eligible_rows(scored)
    qualified = eligible.loc[
        _numeric(eligible, "v33_ecology_score").ge(
            policy.score_threshold
        )
    ].copy()
    if qualified.empty:
        qualified["v33_policy_id"] = policy.policy_id
        return qualified
    qualified["_slot_absolute"] = slot_absolute(
        qualified["signal_slot"]
    )
    qualified.sort_values(
        [
            "trade_date",
            "_slot_absolute",
            "v33_ecology_score",
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
    selected = first_signal.drop(columns="_slot_absolute")
    if policy.spec.max_candidates_per_day is not None:
        within_day = selected.groupby(
            "trade_date",
            sort=False,
        ).cumcount()
        selected = selected.loc[
            within_day.lt(policy.spec.max_candidates_per_day)
        ]
    selected["v33_policy_id"] = policy.policy_id
    selected["v33_score_threshold"] = policy.score_threshold
    return selected.reset_index(drop=True)


def policy_eligible_rows(scored: pd.DataFrame) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "v23_point_in_time_complete",
        "v33_ecology_features_complete",
        "v33_ecology_active_before_signal",
        "p_round_trip_fill_lower",
        "p_severe_loss",
        "v23_positive_model_spread",
        "v23_margin_model_spread",
        "v23_severe_model_spread",
        "v23_expected_return_model_spread_pct",
        "data_age_seconds",
        "v33_p_positive",
        "v33_expected_net_return_pct",
        "v33_p_severe_loss",
        "v33_within_slot_rank_score",
        "v33_ecology_score",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"V33 policy frame missing columns: {missing}")
    age = _numeric(scored, "data_age_seconds")
    fresh = age.notna() & age.ge(0.0) & age.le(MAX_DATA_AGE_SECONDS)
    legal_slot = (
        scored["signal_slot"]
        .astype(str)
        .str.replace(":", "", regex=False)
        .between("1420", "1450")
    )
    eligible = (
        _boolean(scored, "v23_point_in_time_complete")
        & _boolean(scored, "v33_ecology_features_complete")
        & _numeric(
            scored,
            "v33_ecology_active_before_signal",
        ).gt(0)
        & _numeric(scored, "p_round_trip_fill_lower").ge(
            ROUND_TRIP_FILL_MIN
        )
        & _numeric(scored, "p_severe_loss").le(
            SOURCE_SEVERE_LOSS_MAX
        )
        & _numeric(scored, "v23_positive_model_spread").le(
            SOURCE_PROBABILITY_SPREAD_MAX
        )
        & _numeric(scored, "v23_margin_model_spread").le(
            SOURCE_PROBABILITY_SPREAD_MAX
        )
        & _numeric(scored, "v23_severe_model_spread").le(
            SOURCE_PROBABILITY_SPREAD_MAX
        )
        & _numeric(
            scored,
            "v23_expected_return_model_spread_pct",
        ).le(SOURCE_RETURN_SPREAD_MAX_PCT)
        & _numeric(scored, "v33_p_positive").ge(
            POSITIVE_PROBABILITY_MIN
        )
        & _numeric(scored, "v33_expected_net_return_pct").ge(
            EXPECTED_NET_RETURN_MIN_PCT
        )
        & _numeric(scored, "v33_p_severe_loss").le(
            SEVERE_PROBABILITY_MAX
        )
        & _numeric(scored, "v33_within_slot_rank_score").ge(
            PAIRWISE_SCORE_MIN
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
                "l2_code",
                "l3_code",
            )
        )
    ]
    if len(features) < 12 or invalid or contaminated:
        raise RuntimeError(
            "V33 feature contract violated: "
            f"count={len(features)} invalid={invalid} "
            f"contaminated={contaminated}"
        )
    return True


def validate_selected_contract(
    selected: pd.DataFrame,
    policy: FrozenEcologyPolicy | None,
) -> None:
    if selected.empty:
        return
    if selected.duplicated(["trade_date", "ts_code"], keep=False).any():
        raise RuntimeError("V33 selected output rewrote a first signal")
    maximum = (
        policy.spec.max_candidates_per_day
        if policy is not None
        else FIXED_MAX_CANDIDATES_PER_DAY
    )
    if (
        maximum is not None
        and int(selected.groupby("trade_date").size().max()) > maximum
    ):
        raise RuntimeError("V33 selected output exceeds daily maximum")
    slot = (
        selected["signal_slot"]
        .astype(str)
        .str.replace(":", "", regex=False)
    )
    if not slot.between("1420", "1450").all():
        raise RuntimeError("V33 selected output contains illegal slot")
    if not _boolean(selected, "v23_point_in_time_complete").all():
        raise RuntimeError("V33 selected output contains incomplete PIT data")
    if not _boolean(selected, "v33_ecology_features_complete").all():
        raise RuntimeError(
            "V33 selected output contains incomplete ecology data"
        )
    if not _numeric(
        selected,
        "v33_ecology_active_before_signal",
    ).gt(0).all():
        raise RuntimeError(
            "V33 selected output contains no active current ecology"
        )


def _linear_probability_model() -> Pipeline:
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
                    C=0.10,
                    max_iter=2_000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _blended_probability(
    tree: HistGradientBoostingClassifier,
    linear: Pipeline,
    matrix: pd.DataFrame,
) -> np.ndarray:
    return (
        0.50 * tree.predict_proba(matrix)[:, 1]
        + 0.50 * linear.predict_proba(matrix)[:, 1]
    )


def _require_two_classes(target: pd.Series, name: str) -> None:
    if target.nunique() < 2:
        raise ValueError(f"V33 {name} target lacks both classes")


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
