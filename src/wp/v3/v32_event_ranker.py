from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .meta_alpha import IDENTITY_COLUMNS, ProbabilityCalibrator
from .v25_ranker import (
    build_pairwise_examples,
    logistic_model,
    predict_pairwise_scores,
    slot_absolute,
    stock_day_equalized_weights,
)
from .v32_event_features import (
    ADMITTED_SOURCES,
    V32_EVENT_FEATURE_COLUMNS,
)


SCHEMA_VERSION = "wp_v32_public_event_ranker_1"
MODEL_TRAIN_DAYS = 252
MODEL_CALIBRATION_DAYS = 42
MODEL_PURGE_DAYS = 2
MINIMUM_TRAIN_ROWS = 4_000
MINIMUM_CALIBRATION_ROWS = 800
MINIMUM_TRAIN_PAIR_ROWS = 12_000
MINIMUM_CALIBRATION_PAIR_ROWS = 2_000

FIXED_TARGET_CANDIDATE_DAY_RATE = 0.25
FIXED_MAX_CANDIDATES_PER_DAY = 3
ROUND_TRIP_FILL_MIN = 0.95
SOURCE_SEVERE_LOSS_MAX = 0.45
MAX_DATA_AGE_SECONDS = 420.0

POSITIVE_PROBABILITY_MIN = 0.50
MARGIN_PROBABILITY_MIN = 0.35
SEVERE_PROBABILITY_MAX = 0.30
PAIRWISE_SCORE_MIN = 0.50
POSITIVE_SCORE_WEIGHT = 1.00
MARGIN_SCORE_WEIGHT = 0.75
PAIRWISE_SCORE_WEIGHT = 1.00
SEVERE_SCORE_WEIGHT = 1.00
MARGIN_TARGET_PCT = 0.50
SEVERE_TARGET_PCT = -2.00

MODEL_FEATURES = tuple(V32_EVENT_FEATURE_COLUMNS)
MODEL_FEATURE_SET = frozenset(MODEL_FEATURES)


@dataclass(frozen=True)
class EventPolicySpec:
    target_candidate_day_rate: float = FIXED_TARGET_CANDIDATE_DAY_RATE
    max_candidates_per_day: int = FIXED_MAX_CANDIDATES_PER_DAY

    @property
    def policy_id(self) -> str:
        return (
            f"v32-event-rate{self.target_candidate_day_rate:.2f}-"
            f"k{self.max_candidates_per_day}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "target_candidate_day_rate": self.target_candidate_day_rate,
            "max_candidates_per_day": self.max_candidates_per_day,
            "round_trip_fill_min": ROUND_TRIP_FILL_MIN,
            "source_severe_loss_max": SOURCE_SEVERE_LOSS_MAX,
            "positive_probability_min": POSITIVE_PROBABILITY_MIN,
            "margin_probability_min": MARGIN_PROBABILITY_MIN,
            "severe_probability_max": SEVERE_PROBABILITY_MAX,
            "pairwise_score_min": PAIRWISE_SCORE_MIN,
            "max_data_age_seconds": MAX_DATA_AGE_SECONDS,
            "event_active_required": True,
            "one_candidate_per_slot_before_daily_cap": True,
            "first_signal_is_immutable": True,
            "no_signal_allowed": True,
        }


@dataclass(frozen=True)
class FrozenEventPolicy:
    spec: EventPolicySpec
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
class EventRankBundle:
    pairwise_model: Any
    positive_model: Any
    margin_model: Any
    severe_model: Any
    pairwise_calibrator: ProbabilityCalibrator
    positive_calibrator: ProbabilityCalibrator
    margin_calibrator: ProbabilityCalibrator
    severe_calibrator: ProbabilityCalibrator
    feature_columns: tuple[str, ...]
    train_rows: int
    calibration_rows: int
    train_pair_rows: int
    calibration_pair_rows: int

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        matrix = event_feature_matrix(result, self.feature_columns)
        result["v32_p_positive"] = calibrated_probability(
            self.positive_model,
            self.positive_calibrator,
            matrix,
        )
        result["v32_p_margin"] = calibrated_probability(
            self.margin_model,
            self.margin_calibrator,
            matrix,
        )
        result["v32_p_severe_loss"] = calibrated_probability(
            self.severe_model,
            self.severe_calibrator,
            matrix,
        )
        result["v32_within_slot_rank_score"] = predict_pairwise_scores(
            result,
            feature_columns=self.feature_columns,
            model=self.pairwise_model,
            calibrator=self.pairwise_calibrator,
        )
        result["v32_event_score"] = (
            POSITIVE_SCORE_WEIGHT * (result["v32_p_positive"] - 0.50)
            + MARGIN_SCORE_WEIGHT * (result["v32_p_margin"] - 0.35)
            + PAIRWISE_SCORE_WEIGHT
            * (result["v32_within_slot_rank_score"] - 0.50)
            - SEVERE_SCORE_WEIGHT * result["v32_p_severe_loss"]
        )
        return result


def fit_event_ranker(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    random_seed: int,
    minimum_train_rows: int = MINIMUM_TRAIN_ROWS,
    minimum_calibration_rows: int = MINIMUM_CALIBRATION_ROWS,
    minimum_train_pair_rows: int = MINIMUM_TRAIN_PAIR_ROWS,
    minimum_calibration_pair_rows: int = MINIMUM_CALIBRATION_PAIR_ROWS,
) -> EventRankBundle:
    prepared_train = labeled_event_rows(train)
    prepared_calibration = labeled_event_rows(calibration)
    if len(prepared_train) < minimum_train_rows:
        raise ValueError(
            f"V32 has {len(prepared_train)} train rows; "
            f"requires {minimum_train_rows}"
        )
    if len(prepared_calibration) < minimum_calibration_rows:
        raise ValueError(
            f"V32 has {len(prepared_calibration)} calibration rows; "
            f"requires {minimum_calibration_rows}"
        )
    features = active_event_features(
        prepared_train,
        prepared_calibration,
    )
    if len(features) < 12:
        raise ValueError(f"V32 has only {len(features)} active features")
    train_matrix = event_feature_matrix(prepared_train, features)
    calibration_matrix = event_feature_matrix(
        prepared_calibration,
        features,
    )
    train_net = _numeric(prepared_train, "net_return_pct")
    calibration_net = _numeric(prepared_calibration, "net_return_pct")
    train_weight = stock_day_equalized_weights(prepared_train)
    calibration_weight = stock_day_equalized_weights(
        prepared_calibration
    )
    targets = {
        "positive": train_net.gt(0.0).astype(int),
        "margin": train_net.gt(MARGIN_TARGET_PCT).astype(int),
        "severe": train_net.le(SEVERE_TARGET_PCT).astype(int),
    }
    calibration_targets = {
        "positive": calibration_net.gt(0.0).astype(int),
        "margin": calibration_net.gt(MARGIN_TARGET_PCT).astype(int),
        "severe": calibration_net.le(SEVERE_TARGET_PCT).astype(int),
    }
    models: dict[str, Any] = {}
    calibrators: dict[str, ProbabilityCalibrator] = {}
    for offset, name in enumerate(("positive", "margin", "severe"), start=1):
        target = targets[name]
        if target.nunique() < 2:
            raise ValueError(f"V32 {name} target lacks both classes")
        model = logistic_model(random_seed + offset)
        model.fit(
            train_matrix,
            target,
            model__sample_weight=_balanced_weights(
                target,
                train_weight,
            ),
        )
        calibrator = ProbabilityCalibrator().fit(
            model.predict_proba(calibration_matrix)[:, 1],
            calibration_targets[name].to_numpy(dtype=int),
            calibration_weight,
        )
        models[name] = model
        calibrators[name] = calibrator

    train_pairs, train_pair_target, train_pair_weight = (
        build_pairwise_examples(prepared_train, features)
    )
    calibration_pairs, calibration_pair_target, calibration_pair_weight = (
        build_pairwise_examples(prepared_calibration, features)
    )
    if len(train_pairs) < minimum_train_pair_rows:
        raise ValueError(
            f"V32 has {len(train_pairs)} train pair rows; "
            f"requires {minimum_train_pair_rows}"
        )
    if len(calibration_pairs) < minimum_calibration_pair_rows:
        raise ValueError(
            f"V32 has {len(calibration_pairs)} calibration pair rows; "
            f"requires {minimum_calibration_pair_rows}"
        )
    pairwise_model = logistic_model(random_seed + 4)
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
    return EventRankBundle(
        pairwise_model=pairwise_model,
        positive_model=models["positive"],
        margin_model=models["margin"],
        severe_model=models["severe"],
        pairwise_calibrator=pairwise_calibrator,
        positive_calibrator=calibrators["positive"],
        margin_calibrator=calibrators["margin"],
        severe_calibrator=calibrators["severe"],
        feature_columns=features,
        train_rows=int(len(prepared_train)),
        calibration_rows=int(len(prepared_calibration)),
        train_pair_rows=int(len(train_pairs)),
        calibration_pair_rows=int(len(calibration_pairs)),
    )


def labeled_event_rows(frame: pd.DataFrame) -> pd.DataFrame:
    net = _numeric(frame, "net_return_pct")
    target = _numeric(frame, "target_net_positive")
    available = _boolean(frame, "label_available")
    point_in_time = _boolean(frame, "v23_point_in_time_complete")
    event_complete = _boolean(frame, "v32_event_features_complete")
    consistent = target.eq(net.gt(0.0).astype(float))
    return frame.loc[
        available
        & net.notna()
        & target.notna()
        & consistent
        & point_in_time
        & event_complete
    ].copy()


def active_event_features(
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


def event_feature_matrix(
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


def calibrate_event_policy(
    scored_calibration: pd.DataFrame,
    *,
    calibration_dates: Iterable[str],
    spec: EventPolicySpec | None = None,
) -> FrozenEventPolicy:
    frozen_spec = spec or EventPolicySpec()
    dates = sorted(set(map(str, calibration_dates)))
    if not dates:
        raise ValueError("V32 policy calibration has no dates")
    eligible = event_slot_leaders(
        policy_eligible_rows(scored_calibration)
    )
    daily_max = (
        eligible.groupby("trade_date", sort=False)["v32_event_score"]
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
    return FrozenEventPolicy(
        spec=frozen_spec,
        score_threshold=threshold,
        calibration_start=dates[0],
        calibration_end=dates[-1],
        calibration_days=len(dates),
        eligible_days=int(len(daily_max)),
    )


def apply_event_policy(
    scored: pd.DataFrame,
    policy: FrozenEventPolicy,
) -> pd.DataFrame:
    eligible = event_slot_leaders(policy_eligible_rows(scored))
    qualified = eligible.loc[
        _numeric(eligible, "v32_event_score").ge(policy.score_threshold)
    ].copy()
    if qualified.empty:
        qualified["v32_policy_id"] = policy.policy_id
        return qualified
    qualified["_slot_absolute"] = slot_absolute(
        qualified["signal_slot"]
    )
    qualified.sort_values(
        [
            "trade_date",
            "_slot_absolute",
            "v32_event_score",
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
    within_day = first_signal.groupby(
        "trade_date",
        sort=False,
    ).cumcount()
    selected = first_signal.loc[
        within_day.lt(policy.spec.max_candidates_per_day)
    ].drop(columns="_slot_absolute")
    selected["v32_policy_id"] = policy.policy_id
    selected["v32_score_threshold"] = policy.score_threshold
    return selected.reset_index(drop=True)


def policy_eligible_rows(scored: pd.DataFrame) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "v23_point_in_time_complete",
        "v32_event_features_complete",
        "v32_event_any",
        "p_round_trip_fill_lower",
        "p_severe_loss",
        "v32_p_positive",
        "v32_p_margin",
        "v32_p_severe_loss",
        "v32_within_slot_rank_score",
        "v32_event_score",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"V32 policy frame missing columns: {missing}")
    age = _numeric(scored, "data_age_seconds")
    fresh = age.isna() | age.le(MAX_DATA_AGE_SECONDS)
    legal_slot = (
        scored["signal_slot"]
        .astype(str)
        .str.replace(":", "", regex=False)
        .between("1420", "1450")
    )
    eligible = (
        _boolean(scored, "v23_point_in_time_complete")
        & _boolean(scored, "v32_event_features_complete")
        & _boolean(scored, "v32_event_any")
        & _numeric(scored, "p_round_trip_fill_lower").ge(
            ROUND_TRIP_FILL_MIN
        )
        & _numeric(scored, "p_severe_loss").le(SOURCE_SEVERE_LOSS_MAX)
        & _numeric(scored, "v32_p_positive").ge(
            POSITIVE_PROBABILITY_MIN
        )
        & _numeric(scored, "v32_p_margin").ge(
            MARGIN_PROBABILITY_MIN
        )
        & _numeric(scored, "v32_p_severe_loss").le(
            SEVERE_PROBABILITY_MAX
        )
        & _numeric(scored, "v32_within_slot_rank_score").ge(
            PAIRWISE_SCORE_MIN
        )
        & fresh
        & legal_slot
    )
    return scored.loc[eligible].copy()


def event_slot_leaders(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ordered = frame.sort_values(
        [
            "trade_date",
            "signal_slot",
            "v32_event_score",
            "ts_code",
        ],
        ascending=[True, True, False, True],
        kind="stable",
    )
    return ordered.drop_duplicates(
        ["trade_date", "signal_slot"],
        keep="first",
    )


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
    if len(features) < 12 or invalid or contaminated:
        raise RuntimeError(
            "V32 feature contract violated: "
            f"count={len(features)} invalid={invalid} "
            f"contaminated={contaminated}"
        )
    return True


def validate_selected_contract(
    selected: pd.DataFrame,
    policy: FrozenEventPolicy | None,
) -> None:
    if selected.empty:
        return
    if selected.duplicated(["trade_date", "ts_code"], keep=False).any():
        raise RuntimeError("V32 selected output rewrote a first signal")
    maximum = (
        policy.spec.max_candidates_per_day
        if policy is not None
        else FIXED_MAX_CANDIDATES_PER_DAY
    )
    if int(selected.groupby("trade_date").size().max()) > maximum:
        raise RuntimeError("V32 selected output exceeds daily maximum")
    slot = (
        selected["signal_slot"]
        .astype(str)
        .str.replace(":", "", regex=False)
    )
    if not slot.between("1420", "1450").all():
        raise RuntimeError("V32 selected output contains illegal slot")
    if not _boolean(selected, "v23_point_in_time_complete").all():
        raise RuntimeError("V32 selected output contains incomplete PIT data")
    if not _boolean(selected, "v32_event_features_complete").all():
        raise RuntimeError(
            "V32 selected output contains incomplete event data"
        )
    if not _boolean(selected, "v32_event_any").all():
        raise RuntimeError("V32 selected output contains no-event candidate")


def event_any(frame: pd.DataFrame) -> pd.Series:
    columns = [
        f"v32_{source}_active_5d" for source in ADMITTED_SOURCES
    ]
    return pd.concat(
        [_numeric(frame, column).gt(0) for column in columns],
        axis=1,
    ).any(axis=1)


def calibrated_probability(
    model: Any,
    calibrator: ProbabilityCalibrator,
    matrix: pd.DataFrame,
) -> np.ndarray:
    return np.clip(
        calibrator.predict(model.predict_proba(matrix)[:, 1]),
        0.001,
        0.999,
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
