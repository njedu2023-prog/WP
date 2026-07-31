from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .meta_alpha import IDENTITY_COLUMNS, ProbabilityCalibrator
from .v25_positioning import V25_FEATURE_COLUMNS


SCHEMA_VERSION = "wp_v25_positioning_ranker_1"
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
V25_POSITIVE_PROBABILITY_MIN = 0.50
V25_SEVERE_PROBABILITY_MAX = 0.25
V25_PAIRWISE_SCORE_MIN = 0.50
MAX_DATA_AGE_SECONDS = 420.0

MODEL_FEATURES = tuple(V25_FEATURE_COLUMNS)
MODEL_FEATURE_SET = frozenset(MODEL_FEATURES)


@dataclass(frozen=True)
class PositioningPolicySpec:
    target_candidate_day_rate: float = FIXED_TARGET_CANDIDATE_DAY_RATE
    max_candidates_per_day: int = FIXED_MAX_CANDIDATES_PER_DAY

    @property
    def policy_id(self) -> str:
        return (
            f"v25-positioning-rate{self.target_candidate_day_rate:.2f}-"
            f"k{self.max_candidates_per_day}-top5"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "target_candidate_day_rate": self.target_candidate_day_rate,
            "max_candidates_per_day": self.max_candidates_per_day,
            "round_trip_fill_min": ROUND_TRIP_FILL_MIN,
            "source_severe_loss_max": SOURCE_SEVERE_LOSS_MAX,
            "v25_positive_probability_min": (
                V25_POSITIVE_PROBABILITY_MIN
            ),
            "v25_severe_probability_max": V25_SEVERE_PROBABILITY_MAX,
            "v25_pairwise_score_min": V25_PAIRWISE_SCORE_MIN,
            "max_data_age_seconds": MAX_DATA_AGE_SECONDS,
            "one_candidate_per_slot_before_daily_cap": True,
            "first_signal_is_immutable": True,
        }


@dataclass(frozen=True)
class FrozenPositioningPolicy:
    spec: PositioningPolicySpec
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
class PositioningRankBundle:
    pairwise_model: Pipeline
    positive_model: Pipeline
    severe_model: Pipeline
    pairwise_calibrator: ProbabilityCalibrator
    positive_calibrator: ProbabilityCalibrator
    severe_calibrator: ProbabilityCalibrator
    feature_columns: tuple[str, ...]
    train_rows: int
    calibration_rows: int
    train_pair_rows: int
    calibration_pair_rows: int

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        matrix = positioning_matrix(result, self.feature_columns)
        result["v25_p_positive"] = np.clip(
            self.positive_calibrator.predict(
                self.positive_model.predict_proba(matrix)[:, 1]
            ),
            0.001,
            0.999,
        )
        result["v25_p_severe_loss"] = np.clip(
            self.severe_calibrator.predict(
                self.severe_model.predict_proba(matrix)[:, 1]
            ),
            0.001,
            0.999,
        )
        result["v25_within_slot_rank_score"] = (
            predict_pairwise_scores(
                result,
                feature_columns=self.feature_columns,
                model=self.pairwise_model,
                calibrator=self.pairwise_calibrator,
            )
        )
        result["v25_positioning_score"] = (
            1.25 * (result["v25_p_positive"] - 0.50)
            + 1.00 * (result["v25_within_slot_rank_score"] - 0.50)
            - 1.00 * result["v25_p_severe_loss"]
        )
        return result


def fit_positioning_ranker(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    random_seed: int,
    minimum_train_rows: int = MINIMUM_TRAIN_ROWS,
    minimum_calibration_rows: int = MINIMUM_CALIBRATION_ROWS,
    minimum_train_pair_rows: int = MINIMUM_TRAIN_PAIR_ROWS,
    minimum_calibration_pair_rows: int = MINIMUM_CALIBRATION_PAIR_ROWS,
) -> PositioningRankBundle:
    prepared_train = labeled_positioning_rows(train)
    prepared_calibration = labeled_positioning_rows(calibration)
    if len(prepared_train) < minimum_train_rows:
        raise ValueError(
            f"V25 has {len(prepared_train)} train rows; "
            f"requires {minimum_train_rows}"
        )
    if len(prepared_calibration) < minimum_calibration_rows:
        raise ValueError(
            f"V25 has {len(prepared_calibration)} calibration rows; "
            f"requires {minimum_calibration_rows}"
        )
    features = active_positioning_features(
        prepared_train,
        prepared_calibration,
    )
    if len(features) < 12:
        raise ValueError(f"V25 has only {len(features)} active features")

    train_matrix = positioning_matrix(prepared_train, features)
    calibration_matrix = positioning_matrix(prepared_calibration, features)
    train_net = _numeric(prepared_train, "net_return_pct")
    calibration_net = _numeric(prepared_calibration, "net_return_pct")
    absolute_weight = stock_day_equalized_weights(prepared_train)
    calibration_weight = stock_day_equalized_weights(
        prepared_calibration
    )

    positive_target = train_net.gt(0.0).astype(int)
    severe_target = train_net.le(-2.0).astype(int)
    calibration_positive = calibration_net.gt(0.0).astype(int)
    calibration_severe = calibration_net.le(-2.0).astype(int)
    for name, target in (
        ("positive", positive_target),
        ("severe", severe_target),
    ):
        if target.nunique() < 2:
            raise ValueError(f"V25 {name} target lacks both classes")

    positive_model = logistic_model(random_seed + 1)
    positive_model.fit(
        train_matrix,
        positive_target,
        model__sample_weight=_balanced_weights(
            positive_target,
            absolute_weight,
        ),
    )
    severe_model = logistic_model(random_seed + 2)
    severe_model.fit(
        train_matrix,
        severe_target,
        model__sample_weight=_balanced_weights(
            severe_target,
            absolute_weight,
        ),
    )
    positive_calibrator = ProbabilityCalibrator().fit(
        positive_model.predict_proba(calibration_matrix)[:, 1],
        calibration_positive.to_numpy(dtype=int),
        calibration_weight,
    )
    severe_calibrator = ProbabilityCalibrator().fit(
        severe_model.predict_proba(calibration_matrix)[:, 1],
        calibration_severe.to_numpy(dtype=int),
        calibration_weight,
    )

    train_pairs, train_pair_target, train_pair_weight = (
        build_pairwise_examples(prepared_train, features)
    )
    calibration_pairs, calibration_pair_target, calibration_pair_weight = (
        build_pairwise_examples(prepared_calibration, features)
    )
    if len(train_pairs) < minimum_train_pair_rows:
        raise ValueError(
            f"V25 has {len(train_pairs)} train pair rows; "
            f"requires {minimum_train_pair_rows}"
        )
    if len(calibration_pairs) < minimum_calibration_pair_rows:
        raise ValueError(
            f"V25 has {len(calibration_pairs)} calibration pair rows; "
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
    return PositioningRankBundle(
        pairwise_model=pairwise_model,
        positive_model=positive_model,
        severe_model=severe_model,
        pairwise_calibrator=pairwise_calibrator,
        positive_calibrator=positive_calibrator,
        severe_calibrator=severe_calibrator,
        feature_columns=features,
        train_rows=int(len(prepared_train)),
        calibration_rows=int(len(prepared_calibration)),
        train_pair_rows=int(len(train_pairs)),
        calibration_pair_rows=int(len(calibration_pairs)),
    )


def labeled_positioning_rows(frame: pd.DataFrame) -> pd.DataFrame:
    net = _numeric(frame, "net_return_pct")
    target = _numeric(frame, "target_net_positive")
    label_available = _boolean(frame, "label_available")
    point_in_time = _boolean(frame, "v23_point_in_time_complete")
    positioning = _boolean(frame, "v25_positioning_core_complete")
    consistent = target.eq(net.gt(0.0).astype(float))
    return frame.loc[
        label_available
        & net.notna()
        & target.notna()
        & consistent
        & point_in_time
        & positioning
    ].copy()


def active_positioning_features(
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
        if int(values.notna().sum()) >= 20 and int(values.nunique()) > 1:
            active.append(column)
    return tuple(active)


def positioning_matrix(
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


def build_pairwise_examples(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    matrix = positioning_matrix(frame, feature_columns)
    returns = _numeric(frame, "net_return_pct")
    rows: list[np.ndarray] = []
    targets: list[int] = []
    weights: list[float] = []
    grouped = frame.groupby(
        ["trade_date", "signal_slot"],
        sort=False,
    ).indices
    for indices in grouped.values():
        positions = np.asarray(indices, dtype=int)
        valid = positions[returns.iloc[positions].notna().to_numpy()]
        pair_count = len(valid) * (len(valid) - 1) // 2
        if pair_count == 0:
            continue
        group_weight = 1.0 / (2.0 * pair_count)
        for left_offset in range(len(valid) - 1):
            left = int(valid[left_offset])
            for right_offset in range(left_offset + 1, len(valid)):
                right = int(valid[right_offset])
                left_return = float(returns.iloc[left])
                right_return = float(returns.iloc[right])
                if np.isclose(left_return, right_return):
                    continue
                difference = (
                    matrix.iloc[left].to_numpy(dtype=float)
                    - matrix.iloc[right].to_numpy(dtype=float)
                )
                label = int(left_return > right_return)
                rows.extend((difference, -difference))
                targets.extend((label, 1 - label))
                weights.extend((group_weight, group_weight))
    if not rows:
        return (
            pd.DataFrame(columns=feature_columns, dtype=float),
            np.asarray([], dtype=int),
            np.asarray([], dtype=float),
        )
    pair_matrix = pd.DataFrame(rows, columns=feature_columns)
    pair_weights = np.asarray(weights, dtype=float)
    pair_weights /= max(float(pair_weights.mean()), np.finfo(float).eps)
    return pair_matrix, np.asarray(targets, dtype=int), pair_weights


def predict_pairwise_scores(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    model: Pipeline,
    calibrator: ProbabilityCalibrator,
) -> pd.Series:
    matrix = positioning_matrix(frame, feature_columns)
    scores = pd.Series(0.5, index=frame.index, dtype=float)
    grouped = frame.groupby(
        ["trade_date", "signal_slot"],
        sort=False,
    ).indices
    for indices in grouped.values():
        positions = np.asarray(indices, dtype=int)
        if len(positions) < 2:
            continue
        totals = np.zeros(len(positions), dtype=float)
        counts = np.zeros(len(positions), dtype=float)
        for left_offset in range(len(positions) - 1):
            left = int(positions[left_offset])
            for right_offset in range(left_offset + 1, len(positions)):
                right = int(positions[right_offset])
                difference = (
                    matrix.iloc[left].to_numpy(dtype=float)
                    - matrix.iloc[right].to_numpy(dtype=float)
                )
                pair = pd.DataFrame(
                    [difference, -difference],
                    columns=feature_columns,
                )
                raw = model.predict_proba(pair)[:, 1]
                calibrated = calibrator.predict(raw)
                left_probability = float(
                    0.5 * (calibrated[0] + 1.0 - calibrated[1])
                )
                totals[left_offset] += left_probability
                totals[right_offset] += 1.0 - left_probability
                counts[left_offset] += 1.0
                counts[right_offset] += 1.0
        values = np.divide(
            totals,
            counts,
            out=np.full(len(positions), 0.5, dtype=float),
            where=counts > 0.0,
        )
        scores.iloc[positions] = values
    return scores


def calibrate_positioning_policy(
    scored_calibration: pd.DataFrame,
    *,
    calibration_dates: Iterable[str],
    spec: PositioningPolicySpec | None = None,
) -> FrozenPositioningPolicy:
    frozen_spec = spec or PositioningPolicySpec()
    dates = sorted(set(map(str, calibration_dates)))
    if not dates:
        raise ValueError("V25 policy calibration has no dates")
    eligible = slot_leaders(policy_eligible_rows(scored_calibration))
    daily_max = (
        eligible.groupby("trade_date", sort=False)["v25_positioning_score"]
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
    return FrozenPositioningPolicy(
        spec=frozen_spec,
        score_threshold=threshold,
        calibration_start=dates[0],
        calibration_end=dates[-1],
        calibration_days=len(dates),
        eligible_days=int(len(daily_max)),
    )


def apply_positioning_policy(
    scored: pd.DataFrame,
    policy: FrozenPositioningPolicy,
) -> pd.DataFrame:
    eligible = slot_leaders(policy_eligible_rows(scored))
    qualified = eligible.loc[
        _numeric(eligible, "v25_positioning_score").ge(
            policy.score_threshold
        )
    ].copy()
    if qualified.empty:
        qualified["v25_policy_id"] = policy.policy_id
        return qualified
    qualified["_slot_absolute"] = slot_absolute(
        qualified["signal_slot"]
    )
    qualified.sort_values(
        [
            "trade_date",
            "_slot_absolute",
            "v25_positioning_score",
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
    within_day = first_signal.groupby(
        "trade_date",
        sort=False,
    ).cumcount()
    selected = first_signal.loc[
        within_day.lt(policy.spec.max_candidates_per_day)
    ].drop(columns="_slot_absolute")
    selected["v25_policy_id"] = policy.policy_id
    selected["v25_score_threshold"] = policy.score_threshold
    return selected.reset_index(drop=True)


def policy_eligible_rows(scored: pd.DataFrame) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "v23_point_in_time_complete",
        "v25_positioning_core_complete",
        "p_round_trip_fill_lower",
        "p_severe_loss",
        "v25_p_positive",
        "v25_p_severe_loss",
        "v25_within_slot_rank_score",
        "v25_positioning_score",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"V25 policy frame missing columns: {missing}")
    age = _numeric(scored, "data_age_seconds")
    fresh = age.isna() | age.le(MAX_DATA_AGE_SECONDS)
    slot = scored["signal_slot"].astype(str).str.replace(
        ":",
        "",
        regex=False,
    )
    eligible = (
        _boolean(scored, "v23_point_in_time_complete")
        & _boolean(scored, "v25_positioning_core_complete")
        & _numeric(scored, "p_round_trip_fill_lower").ge(
            ROUND_TRIP_FILL_MIN
        )
        & _numeric(scored, "p_severe_loss").le(SOURCE_SEVERE_LOSS_MAX)
        & _numeric(scored, "v25_p_positive").ge(
            V25_POSITIVE_PROBABILITY_MIN
        )
        & _numeric(scored, "v25_p_severe_loss").le(
            V25_SEVERE_PROBABILITY_MAX
        )
        & _numeric(scored, "v25_within_slot_rank_score").ge(
            V25_PAIRWISE_SCORE_MIN
        )
        & fresh
        & slot.between("1420", "1450")
    )
    return scored.loc[eligible].copy()


def slot_leaders(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ordered = frame.sort_values(
        [
            "trade_date",
            "signal_slot",
            "v25_positioning_score",
            "ts_code",
        ],
        ascending=[True, True, False, True],
        kind="stable",
    )
    return ordered.drop_duplicates(
        ["trade_date", "signal_slot"],
        keep="first",
    )


def within_slot_rank_diagnostics(
    scored: pd.DataFrame,
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    grouped = scored.groupby(
        ["trade_date", "signal_slot"],
        sort=False,
    )
    for (date, slot), group in grouped:
        valid = group.loc[
            _numeric(group, "v25_within_slot_rank_score").notna()
            & _numeric(group, "net_return_pct").notna()
        ].copy()
        if len(valid) < 2:
            continue
        score = _numeric(valid, "v25_within_slot_rank_score")
        net = _numeric(valid, "net_return_pct")
        correlation = spearmanr(score, net).statistic
        ordered = valid.assign(_score=score).sort_values(
            ["_score", "ts_code"],
            ascending=[False, True],
            kind="stable",
        )
        rows.append(
            {
                "trade_date": str(date),
                "signal_slot": str(slot),
                "ic": (
                    float(correlation)
                    if np.isfinite(correlation)
                    else 0.0
                ),
                "spread_pct": float(
                    _numeric(ordered, "net_return_pct").iloc[0]
                    - _numeric(ordered, "net_return_pct").iloc[-1]
                ),
            }
        )
    details = pd.DataFrame(rows)
    if details.empty:
        return {
            "groups": 0,
            "days": 0,
            "mean_within_slot_ic": float("nan"),
            "mean_top_minus_bottom_return_pct": float("nan"),
            "clustered_spread_lower_pct": float("nan"),
            "clustered_spread_upper_pct": float("nan"),
            "yearly": [],
        }
    daily = details.groupby("trade_date", sort=True).agg(
        ic=("ic", "mean"),
        spread_pct=("spread_pct", "mean"),
    )
    rng = np.random.default_rng(seed)
    values = daily["spread_pct"].to_numpy(dtype=float)
    choices = rng.integers(
        0,
        len(values),
        size=(max(bootstrap_samples, 1), len(values)),
    )
    means = values[choices].mean(axis=1)
    yearly = []
    years = details["trade_date"].str[:4]
    for year in sorted(years.unique()):
        subset = details.loc[years.eq(year)]
        yearly.append(
            {
                "year": year,
                "groups": int(len(subset)),
                "mean_within_slot_ic": float(subset["ic"].mean()),
                "mean_top_minus_bottom_return_pct": float(
                    subset["spread_pct"].mean()
                ),
            }
        )
    return {
        "groups": int(len(details)),
        "days": int(len(daily)),
        "mean_within_slot_ic": float(details["ic"].mean()),
        "mean_top_minus_bottom_return_pct": float(
            details["spread_pct"].mean()
        ),
        "clustered_spread_lower_pct": float(np.quantile(means, 0.025)),
        "clustered_spread_upper_pct": float(np.quantile(means, 0.975)),
        "yearly": yearly,
    }


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
                "exit_",
            )
        )
    ]
    if len(features) < 12 or invalid or contaminated:
        raise RuntimeError(
            "V25 feature contract violated: "
            f"count={len(features)} invalid={invalid} "
            f"contaminated={contaminated}"
        )
    return True


def validate_selected_contract(
    selected: pd.DataFrame,
    policy: FrozenPositioningPolicy | None,
) -> None:
    if selected.empty:
        return
    if selected.duplicated(["trade_date", "ts_code"], keep=False).any():
        raise RuntimeError("V25 selected output rewrote a first signal")
    maximum = (
        policy.spec.max_candidates_per_day
        if policy is not None
        else FIXED_MAX_CANDIDATES_PER_DAY
    )
    if int(selected.groupby("trade_date").size().max()) > maximum:
        raise RuntimeError("V25 selected output exceeds fixed daily maximum")
    slot = selected["signal_slot"].astype(str).str.replace(
        ":",
        "",
        regex=False,
    )
    if not slot.between("1420", "1450").all():
        raise RuntimeError("V25 selected output contains an illegal slot")
    if not _boolean(selected, "v25_positioning_core_complete").all():
        raise RuntimeError("V25 selected output lacks core positioning data")


def stock_day_equalized_weights(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    counts = (
        frame.groupby(["trade_date", "ts_code"], sort=False)["ts_code"]
        .transform("size")
        .to_numpy(dtype=float)
    )
    weights = 1.0 / np.maximum(counts, 1.0)
    return weights / max(float(weights.mean()), np.finfo(float).eps)


def logistic_model(random_seed: int) -> Pipeline:
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
                    class_weight=None,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def slot_absolute(slot: pd.Series) -> pd.Series:
    parsed = slot.astype(str).str.extract(
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2})"
    )
    return (
        pd.to_numeric(parsed["hour"], errors="coerce") * 60
        + pd.to_numeric(parsed["minute"], errors="coerce")
    )


def _balanced_weights(
    target: pd.Series,
    base_weight: np.ndarray,
) -> np.ndarray:
    labels = target.to_numpy(dtype=int)
    counts = np.bincount(labels, minlength=2).astype(float)
    class_weight = np.divide(
        len(labels),
        2.0 * counts,
        out=np.ones(2, dtype=float),
        where=counts > 0.0,
    )
    return np.asarray(base_weight, dtype=float) * class_weight[labels]


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)
