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

from .meta_alpha import ProbabilityCalibrator


SCHEMA_VERSION = "wp_v35_full_session_regime_license_1"
MODEL_TRAIN_DAYS = 252
MODEL_CALIBRATION_DAYS = 42
MODEL_PURGE_DAYS = 2
MINIMUM_TRAIN_ROWS = 1_000
MINIMUM_CALIBRATION_ROWS = 200

FIXED_TARGET_CANDIDATE_DAY_RATE = 0.20
FIXED_BASKET_SIZE = 3
FIXED_MAX_CANDIDATES_PER_DAY = 3
MINIMUM_BASKET_MEMBERS = 2

ROUND_TRIP_FILL_MIN = 0.95
SOURCE_SEVERE_LOSS_MAX = 0.45
SOURCE_MODEL_SPREAD_MAX = 0.30
MAX_DATA_AGE_SECONDS = 420.0
LICENSE_PROBABILITY_SPREAD_MAX = 0.40
LICENSE_RETURN_SPREAD_MAX_PCT = 5.0

MARGIN_TARGET_PCT = 0.50
SEVERE_TARGET_PCT = -2.00

PATH_MEDIAN_BASE_FEATURES = (
    "v34_session_return_pct",
    "v34_opening_30m_return_pct",
    "v34_morning_return_pct",
    "v34_afternoon_return_pct",
    "v34_lunch_gap_pct",
    "v34_post_1400_return_pct",
    "v34_session_realized_volatility_pct",
    "v34_session_downside_volatility_pct",
    "v34_directional_efficiency",
    "v34_max_drawdown_pct",
    "v34_rebound_from_low_pct",
    "v34_reversal_from_high_pct",
    "v34_session_vwap_gap_pct",
    "v34_above_vwap_share",
    "v34_session_close_position",
    "v34_amount_acceleration_15m",
    "v34_signed_amount_imbalance",
    "v34_afternoon_signed_amount_imbalance",
    "v34_post_1400_signed_amount_imbalance",
    "v34_flow_regime_agreement",
    "v34_price_amount_correlation",
    "v34_amihud_proxy",
    "v34_recent_vs_prior_volatility_ratio",
)
PATH_IQR_BASE_FEATURES = (
    "v34_session_return_pct",
    "v34_afternoon_return_pct",
    "v34_post_1400_return_pct",
    "v34_session_vwap_gap_pct",
    "v34_amount_acceleration_15m",
    "v34_signed_amount_imbalance",
    "v34_post_1400_signed_amount_imbalance",
    "v34_recent_vs_prior_volatility_ratio",
)
PATH_BREADTH_SPECS = (
    ("v35_breadth_session_positive", "v34_session_return_pct", 0.0),
    ("v35_breadth_afternoon_positive", "v34_afternoon_return_pct", 0.0),
    ("v35_breadth_post_1400_positive", "v34_post_1400_return_pct", 0.0),
    ("v35_breadth_above_vwap", "v34_above_vwap_share", 0.50),
    ("v35_breadth_positive_vwap_gap", "v34_session_vwap_gap_pct", 0.0),
    (
        "v35_breadth_positive_amount_imbalance",
        "v34_signed_amount_imbalance",
        0.0,
    ),
    (
        "v35_breadth_flow_regime_agreement",
        "v34_flow_regime_agreement",
        0.0,
    ),
)

PATH_MEDIAN_FEATURES = tuple(
    f"v35_median_{column.removeprefix('v34_')}"
    for column in PATH_MEDIAN_BASE_FEATURES
)
PATH_IQR_FEATURES = tuple(
    f"v35_iqr_{column.removeprefix('v34_')}"
    for column in PATH_IQR_BASE_FEATURES
)
V35_REGIME_FEATURES = (
    "v35_slot_minute",
    "v35_basket_member_count",
    *PATH_MEDIAN_FEATURES,
    *PATH_IQR_FEATURES,
    *(name for name, _, _ in PATH_BREADTH_SPECS),
)
V35_REGIME_FEATURE_SET = frozenset(V35_REGIME_FEATURES)

FORBIDDEN_FEATURE_TOKENS = (
    "target",
    "label",
    "truth",
    "future",
    "gross_return",
    "net_return",
    "t1_",
    "exit_",
)


@dataclass(frozen=True)
class RegimePolicySpec:
    target_candidate_day_rate: float = FIXED_TARGET_CANDIDATE_DAY_RATE
    basket_size: int = FIXED_BASKET_SIZE
    max_candidates_per_day: int = FIXED_MAX_CANDIDATES_PER_DAY
    probability_spread_max: float = LICENSE_PROBABILITY_SPREAD_MAX
    return_spread_max_pct: float = LICENSE_RETURN_SPREAD_MAX_PCT

    @property
    def policy_id(self) -> str:
        return (
            f"v35-regime-rate{self.target_candidate_day_rate:.2f}-"
            f"basket{self.basket_size}-k{self.max_candidates_per_day}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "target_candidate_day_rate": self.target_candidate_day_rate,
            "basket_size": self.basket_size,
            "max_candidates_per_day": self.max_candidates_per_day,
            "minimum_basket_members": MINIMUM_BASKET_MEMBERS,
            "round_trip_fill_min": ROUND_TRIP_FILL_MIN,
            "source_severe_loss_max": SOURCE_SEVERE_LOSS_MAX,
            "source_model_spread_max": SOURCE_MODEL_SPREAD_MAX,
            "max_data_age_seconds": MAX_DATA_AGE_SECONDS,
            "probability_spread_max": self.probability_spread_max,
            "return_spread_max_pct": self.return_spread_max_pct,
            "first_qualifying_slot_is_immutable": True,
            "first_signal_price_is_immutable": True,
            "no_signal_allowed": True,
        }


@dataclass(frozen=True)
class FrozenRegimePolicy:
    spec: RegimePolicySpec
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
class RegimeLicenseBundle:
    good_tree: HistGradientBoostingClassifier
    good_linear: Pipeline
    margin_tree: HistGradientBoostingClassifier
    margin_linear: Pipeline
    severe_tree: HistGradientBoostingClassifier
    severe_linear: Pipeline
    return_tree: HistGradientBoostingRegressor
    return_linear: Pipeline
    good_calibrator: ProbabilityCalibrator
    margin_calibrator: ProbabilityCalibrator
    severe_calibrator: ProbabilityCalibrator
    return_calibrator: Ridge
    return_downside_residual_pct: float
    feature_columns: tuple[str, ...]
    train_rows: int
    calibration_rows: int

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        matrix = regime_feature_matrix(result, self.feature_columns)
        raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, tree, linear in (
            ("good", self.good_tree, self.good_linear),
            ("margin", self.margin_tree, self.margin_linear),
            ("severe", self.severe_tree, self.severe_linear),
        ):
            raw[name] = (
                tree.predict_proba(matrix)[:, 1],
                linear.predict_proba(matrix)[:, 1],
            )

        good = np.clip(
            self.good_calibrator.predict(_blend(*raw["good"])),
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

        good_spread = np.abs(raw["good"][0] - raw["good"][1])
        margin_spread = np.abs(raw["margin"][0] - raw["margin"][1])
        severe_spread = np.abs(raw["severe"][0] - raw["severe"][1])
        return_spread = np.abs(expected_tree - expected_linear)
        result["v35_p_good"] = good
        result["v35_p_good_lower"] = np.clip(
            good - 0.50 * good_spread - 0.03,
            0.001,
            0.999,
        )
        result["v35_good_model_spread"] = good_spread
        result["v35_p_margin"] = margin
        result["v35_p_margin_lower"] = np.clip(
            margin - 0.50 * margin_spread - 0.03,
            0.001,
            0.999,
        )
        result["v35_margin_model_spread"] = margin_spread
        result["v35_p_severe"] = severe
        result["v35_p_severe_upper"] = np.clip(
            severe + 0.50 * severe_spread + 0.03,
            0.001,
            0.999,
        )
        result["v35_severe_model_spread"] = severe_spread
        result["v35_expected_basket_mean_pct"] = expected
        result["v35_return_model_spread_pct"] = return_spread
        result["v35_expected_basket_mean_lower_pct"] = (
            expected
            - 0.50 * return_spread
            - self.return_downside_residual_pct
        )
        result["v35_regime_score"] = (
            result["v35_expected_basket_mean_lower_pct"]
            + 1.00 * (result["v35_p_good_lower"] - 0.50)
            + 0.60 * (result["v35_p_margin_lower"] - 0.30)
            - 1.10 * result["v35_p_severe_upper"]
        )
        return result


def build_regime_slot_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {
        "trade_date",
        "signal_slot",
        "ts_code",
        "fold",
        "signal_price",
        "v20_stock_rank_in_slot",
        "selection_score",
        "p_net_positive_lower",
        "execution_eligible",
        "p_round_trip_fill_lower",
        "p_severe_loss",
        "probability_model_spread",
        "expected_return_model_spread",
        "v23_point_in_time_complete",
        "v34_path_complete",
        *PATH_MEDIAN_BASE_FEATURES,
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"V35 candidate frame missing columns: {missing}")
    if candidates.empty:
        return pd.DataFrame(columns=("trade_date", "signal_slot", "fold"))

    eligible = eligible_source_candidates(candidates)
    eligible.sort_values(
        [
            "trade_date",
            "signal_slot",
            "v20_stock_rank_in_slot",
            "selection_score",
            "p_net_positive_lower",
            "ts_code",
        ],
        ascending=[True, True, True, False, False, True],
        kind="stable",
        inplace=True,
    )
    eligible["_v35_basket_rank"] = (
        eligible.groupby(["trade_date", "signal_slot"], sort=False).cumcount()
        + 1
    )
    basket = eligible.loc[
        eligible["_v35_basket_rank"].le(FIXED_BASKET_SIZE)
    ].copy()

    rows: list[dict[str, Any]] = []
    for (trade_date, signal_slot), group in basket.groupby(
        ["trade_date", "signal_slot"],
        sort=True,
    ):
        if len(group) < MINIMUM_BASKET_MEMBERS:
            continue
        folds = pd.to_numeric(group["fold"], errors="coerce").dropna().unique()
        if len(folds) != 1:
            raise RuntimeError("V35 basket contains inconsistent source folds")
        row: dict[str, Any] = {
            "trade_date": str(trade_date),
            "signal_slot": str(signal_slot),
            "fold": int(folds[0]),
            "v35_slot_minute": float(_slot_minute(str(signal_slot))),
            "v35_basket_member_count": int(len(group)),
            "v35_basket_member_codes": "|".join(
                group["ts_code"].astype(str).tolist()
            ),
        }
        for source, target in zip(
            PATH_MEDIAN_BASE_FEATURES,
            PATH_MEDIAN_FEATURES,
            strict=True,
        ):
            row[target] = float(_numeric(group, source).median())
        for source, target in zip(
            PATH_IQR_BASE_FEATURES,
            PATH_IQR_FEATURES,
            strict=True,
        ):
            values = _numeric(group, source)
            row[target] = float(values.quantile(0.75) - values.quantile(0.25))
        for target, source, threshold in PATH_BREADTH_SPECS:
            row[target] = float(_numeric(group, source).gt(threshold).mean())

        net = _numeric(group, "net_return_pct")
        label_available = _boolean(group, "label_available")
        if net.notna().all() and label_available.all():
            mean_net = float(net.mean())
            positive_share = float(net.gt(0.0).mean())
            row.update(
                {
                    "v35_basket_mean_net_return_pct": mean_net,
                    "v35_basket_min_net_return_pct": float(net.min()),
                    "v35_basket_positive_share": positive_share,
                    "v35_target_good": float(
                        mean_net > 0.0 and positive_share > 0.50
                    ),
                    "v35_target_margin": float(
                        mean_net > MARGIN_TARGET_PCT
                        and positive_share > 0.50
                    ),
                    "v35_target_severe": float(
                        bool(net.le(SEVERE_TARGET_PCT).any())
                    ),
                }
            )
        else:
            row.update(
                {
                    "v35_basket_mean_net_return_pct": np.nan,
                    "v35_basket_min_net_return_pct": np.nan,
                    "v35_basket_positive_share": np.nan,
                    "v35_target_good": np.nan,
                    "v35_target_margin": np.nan,
                    "v35_target_severe": np.nan,
                }
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result.sort_values(["fold", "trade_date", "signal_slot"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    if result.duplicated(["trade_date", "signal_slot"], keep=False).any():
        raise RuntimeError("V35 slot frame contains duplicate identities")
    numeric = result.reindex(columns=V35_REGIME_FEATURES).apply(
        pd.to_numeric,
        errors="coerce",
    )
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise RuntimeError("V35 slot features contain infinite values")
    return result


def eligible_source_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "trade_date",
        "signal_slot",
        "ts_code",
        "v20_stock_rank_in_slot",
        "execution_eligible",
        "p_round_trip_fill_lower",
        "p_severe_loss",
        "probability_model_spread",
        "expected_return_model_spread",
        "v23_point_in_time_complete",
        "v34_path_complete",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V35 source eligibility missing columns: {missing}")
    age = _numeric(frame, "data_age_seconds")
    fresh = age.isna() | (
        age.ge(0.0) & age.le(MAX_DATA_AGE_SECONDS)
    )
    legal_slot = (
        frame["signal_slot"]
        .astype(str)
        .str.replace(":", "", regex=False)
        .between("1420", "1450")
    )
    eligible = (
        _boolean(frame, "execution_eligible")
        & _boolean(frame, "v23_point_in_time_complete")
        & _boolean(frame, "v34_path_complete")
        & _numeric(frame, "p_round_trip_fill_lower").ge(ROUND_TRIP_FILL_MIN)
        & _numeric(frame, "p_severe_loss").le(SOURCE_SEVERE_LOSS_MAX)
        & _numeric(frame, "probability_model_spread").le(
            SOURCE_MODEL_SPREAD_MAX
        )
        & _numeric(frame, "expected_return_model_spread").le(
            SOURCE_MODEL_SPREAD_MAX
        )
        & fresh
        & legal_slot
    )
    return frame.loc[eligible].copy()


def fit_regime_license(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    random_seed: int,
    minimum_train_rows: int = MINIMUM_TRAIN_ROWS,
    minimum_calibration_rows: int = MINIMUM_CALIBRATION_ROWS,
) -> RegimeLicenseBundle:
    prepared_train = labeled_regime_slots(train)
    prepared_calibration = labeled_regime_slots(calibration)
    if len(prepared_train) < minimum_train_rows:
        raise ValueError(
            f"V35 has {len(prepared_train)} train slots; "
            f"requires {minimum_train_rows}"
        )
    if len(prepared_calibration) < minimum_calibration_rows:
        raise ValueError(
            f"V35 has {len(prepared_calibration)} calibration slots; "
            f"requires {minimum_calibration_rows}"
        )
    features = active_regime_features(prepared_train, prepared_calibration)
    if len(features) < 25:
        raise ValueError(f"V35 has only {len(features)} active features")
    x_train = regime_feature_matrix(prepared_train, features)
    x_calibration = regime_feature_matrix(prepared_calibration, features)
    weights = day_equalized_temporal_weights(prepared_train)
    calibration_weights = day_equalized_temporal_weights(
        prepared_calibration
    )

    targets = {
        name: _numeric(prepared_train, f"v35_target_{name}").astype(int)
        for name in ("good", "margin", "severe")
    }
    calibration_targets = {
        name: _numeric(
            prepared_calibration,
            f"v35_target_{name}",
        ).astype(int)
        for name in ("good", "margin", "severe")
    }
    min_leaf = max(35, min(100, len(prepared_train) // 30))
    trees: dict[str, HistGradientBoostingClassifier] = {}
    linears: dict[str, Pipeline] = {}
    calibrators: dict[str, ProbabilityCalibrator] = {}
    for offset, name in enumerate(("good", "margin", "severe")):
        target = targets[name]
        calibration_target = calibration_targets[name]
        if target.nunique() < 2 or calibration_target.nunique() < 2:
            raise ValueError(f"V35 {name} target lacks both classes")
        tree = HistGradientBoostingClassifier(
            learning_rate=0.025,
            max_iter=180,
            max_leaf_nodes=7,
            min_samples_leaf=min_leaf,
            l2_regularization=50.0,
            random_state=random_seed + offset * 101,
        )
        tree.fit(
            x_train,
            target,
            sample_weight=_balanced_weights(target, weights),
        )
        linear = _linear_classifier(random_seed + offset * 101 + 1)
        linear.fit(
            x_train,
            target,
            model__sample_weight=weights,
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

    net_train = _numeric(
        prepared_train,
        "v35_basket_mean_net_return_pct",
    ).clip(-10.0, 10.0)
    net_calibration = _numeric(
        prepared_calibration,
        "v35_basket_mean_net_return_pct",
    ).clip(-10.0, 10.0)
    return_tree = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.025,
        max_iter=180,
        max_leaf_nodes=7,
        min_samples_leaf=min_leaf,
        l2_regularization=50.0,
        random_state=random_seed + 404,
    )
    return_tree.fit(x_train, net_train, sample_weight=weights)
    return_linear = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=60.0)),
        ]
    )
    return_linear.fit(
        x_train,
        net_train,
        model__sample_weight=weights,
    )
    calibration_raw = _blend(
        return_tree.predict(x_calibration),
        return_linear.predict(x_calibration),
    )
    return_calibrator = Ridge(alpha=20.0)
    return_calibrator.fit(
        calibration_raw.reshape(-1, 1),
        net_calibration,
        sample_weight=calibration_weights,
    )
    expected = return_calibrator.predict(calibration_raw.reshape(-1, 1))
    downside = np.maximum(
        expected - net_calibration.to_numpy(dtype=float),
        0.0,
    )
    return RegimeLicenseBundle(
        good_tree=trees["good"],
        good_linear=linears["good"],
        margin_tree=trees["margin"],
        margin_linear=linears["margin"],
        severe_tree=trees["severe"],
        severe_linear=linears["severe"],
        return_tree=return_tree,
        return_linear=return_linear,
        good_calibrator=calibrators["good"],
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


def labeled_regime_slots(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "v35_basket_mean_net_return_pct",
        "v35_target_good",
        "v35_target_margin",
        "v35_target_severe",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V35 labeled slot frame missing columns: {missing}")
    available = pd.Series(True, index=frame.index)
    for column in required:
        available &= _numeric(frame, column).notna()
    return frame.loc[available].copy()


def active_regime_features(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
) -> tuple[str, ...]:
    combined = pd.concat(
        [
            train.reindex(columns=V35_REGIME_FEATURES),
            calibration.reindex(columns=V35_REGIME_FEATURES),
        ],
        ignore_index=True,
    )
    return tuple(
        column
        for column in V35_REGIME_FEATURES
        if (
            _numeric(combined, column).notna().sum() >= 20
            and _numeric(combined, column).nunique(dropna=True) > 1
        )
    )


def regime_feature_matrix(
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


def rolling_regime_segments(
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


def calibrate_regime_policy(
    scored_calibration: pd.DataFrame,
    *,
    calibration_dates: Iterable[str],
    spec: RegimePolicySpec | None = None,
) -> FrozenRegimePolicy:
    frozen_spec = spec or RegimePolicySpec()
    dates = sorted(set(map(str, calibration_dates)))
    if not dates:
        raise ValueError("V35 policy calibration has no dates")
    calibration = scored_calibration.loc[
        scored_calibration["trade_date"].astype(str).isin(dates)
    ].copy()
    eligible = slot_policy_eligible(calibration, frozen_spec)
    daily_max = (
        eligible.groupby("trade_date", sort=False)["v35_regime_score"]
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
    return FrozenRegimePolicy(
        spec=frozen_spec,
        score_threshold=threshold,
        calibration_start=dates[0],
        calibration_end=dates[-1],
        calibration_days=len(dates),
        eligible_days=int(len(daily_max)),
    )


def apply_regime_policy_to_slots(
    scored_slots: pd.DataFrame,
    policy: FrozenRegimePolicy,
) -> pd.DataFrame:
    eligible = slot_policy_eligible(scored_slots, policy.spec)
    qualified = eligible.loc[
        _numeric(eligible, "v35_regime_score").ge(policy.score_threshold)
    ].copy()
    if qualified.empty:
        qualified["v35_policy_id"] = policy.policy_id
        return qualified
    qualified["_v35_slot_minute"] = _slot_minutes(
        qualified["signal_slot"]
    )
    qualified.sort_values(
        ["trade_date", "_v35_slot_minute", "v35_regime_score"],
        ascending=[True, True, False],
        kind="stable",
        inplace=True,
    )
    licensed = qualified.drop_duplicates("trade_date", keep="first").copy()
    licensed.drop(columns="_v35_slot_minute", inplace=True)
    licensed["v35_policy_id"] = policy.policy_id
    licensed["v35_score_threshold"] = policy.score_threshold
    return licensed.reset_index(drop=True)


def select_regime_candidates(
    candidates: pd.DataFrame,
    licensed_slots: pd.DataFrame,
    policy: FrozenRegimePolicy,
) -> pd.DataFrame:
    if licensed_slots.empty:
        result = candidates.head(0).copy()
        result["v35_policy_id"] = policy.policy_id
        return result
    slot_columns = [
        "trade_date",
        "signal_slot",
        "v35_regime_score",
        "v35_score_threshold",
        "v35_p_good_lower",
        "v35_p_margin_lower",
        "v35_p_severe_upper",
        "v35_expected_basket_mean_lower_pct",
    ]
    missing = sorted(set(slot_columns) - set(licensed_slots.columns))
    if missing:
        raise ValueError(f"V35 licensed slots missing columns: {missing}")
    eligible = eligible_source_candidates(candidates)
    eligible = eligible.merge(
        licensed_slots.loc[:, slot_columns],
        on=["trade_date", "signal_slot"],
        how="inner",
        validate="many_to_one",
    )
    eligible.sort_values(
        [
            "trade_date",
            "v20_stock_rank_in_slot",
            "selection_score",
            "p_net_positive_lower",
            "ts_code",
        ],
        ascending=[True, True, False, False, True],
        kind="stable",
        inplace=True,
    )
    eligible["_v35_candidate_rank"] = (
        eligible.groupby("trade_date", sort=False).cumcount() + 1
    )
    selected = eligible.loc[
        eligible["_v35_candidate_rank"].le(policy.spec.basket_size)
        & eligible["_v35_candidate_rank"].le(
            policy.spec.max_candidates_per_day
        )
    ].copy()
    selected["v35_policy_id"] = policy.policy_id
    selected["v35_source_rank"] = selected["_v35_candidate_rank"]
    selected.drop(columns="_v35_candidate_rank", inplace=True)
    selected.reset_index(drop=True, inplace=True)
    validate_selected_contract(selected, policy)
    return selected


def slot_policy_eligible(
    scored: pd.DataFrame,
    spec: RegimePolicySpec,
) -> pd.DataFrame:
    required = {
        "trade_date",
        "signal_slot",
        "v35_basket_member_count",
        "v35_good_model_spread",
        "v35_margin_model_spread",
        "v35_severe_model_spread",
        "v35_return_model_spread_pct",
        "v35_regime_score",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"V35 policy frame missing columns: {missing}")
    legal_slot = (
        scored["signal_slot"]
        .astype(str)
        .str.replace(":", "", regex=False)
        .between("1420", "1450")
    )
    eligible = (
        _numeric(scored, "v35_basket_member_count").ge(
            MINIMUM_BASKET_MEMBERS
        )
        & _numeric(scored, "v35_good_model_spread").le(
            spec.probability_spread_max
        )
        & _numeric(scored, "v35_margin_model_spread").le(
            spec.probability_spread_max
        )
        & _numeric(scored, "v35_severe_model_spread").le(
            spec.probability_spread_max
        )
        & _numeric(scored, "v35_return_model_spread_pct").le(
            spec.return_spread_max_pct
        )
        & legal_slot
    )
    return scored.loc[eligible].copy()


def validate_feature_contract(features: tuple[str, ...]) -> bool:
    invalid = sorted(set(features) - V35_REGIME_FEATURE_SET)
    contaminated = [
        feature
        for feature in features
        if any(token in feature.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if len(features) < 25 or invalid or contaminated:
        raise RuntimeError(
            "V35 feature contract violated: "
            f"count={len(features)} invalid={invalid} "
            f"contaminated={contaminated}"
        )
    return True


def validate_selected_contract(
    selected: pd.DataFrame,
    policy: FrozenRegimePolicy | None,
) -> None:
    if selected.empty:
        return
    if selected.duplicated(["trade_date", "ts_code"], keep=False).any():
        raise RuntimeError("V35 selected output rewrote a first signal")
    maximum = (
        policy.spec.max_candidates_per_day
        if policy is not None
        else FIXED_MAX_CANDIDATES_PER_DAY
    )
    if int(selected.groupby("trade_date").size().max()) > maximum:
        raise RuntimeError("V35 selected output exceeds the daily maximum")
    slots_per_day = selected.groupby("trade_date")["signal_slot"].nunique()
    if not slots_per_day.eq(1).all():
        raise RuntimeError("V35 selected output spans multiple daily slots")
    slot = (
        selected["signal_slot"]
        .astype(str)
        .str.replace(":", "", regex=False)
    )
    if not slot.between("1420", "1450").all():
        raise RuntimeError("V35 selected output contains an illegal slot")
    if not _boolean(selected, "v23_point_in_time_complete").all():
        raise RuntimeError("V35 selected output contains incomplete PIT data")
    if not _boolean(selected, "v34_path_complete").all():
        raise RuntimeError("V35 selected output contains incomplete path data")
    if _numeric(selected, "signal_price").isna().any():
        raise RuntimeError("V35 selected output lost an immutable signal price")


def day_equalized_temporal_weights(frame: pd.DataFrame) -> np.ndarray:
    dates = frame["trade_date"].astype(str)
    unique_dates = sorted(dates.unique())
    recency = {
        date: 0.50 + 0.50 * ((index + 1) / max(len(unique_dates), 1))
        for index, date in enumerate(unique_dates)
    }
    counts = dates.value_counts()
    weights = np.asarray(
        [recency[date] / float(counts[date]) for date in dates],
        dtype=float,
    )
    mean = float(weights.mean()) if len(weights) else 1.0
    return weights / max(mean, 1e-12)


def v35_research_readiness(
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
    minimum_year_events = min(
        (int(row.get("events", 0)) for row in active_years),
        default=0,
    )
    worst_year = min(
        (
            float(row.get("mean_net_return_pct") or -999.0)
            for row in active_years
        ),
        default=-999.0,
    )
    gates = {
        "minimum_nested_oos_candidates": int(metrics.get("events", 0)) >= 180,
        "minimum_nested_oos_candidate_days": (
            int(metrics.get("candidate_days", 0)) >= 100
        ),
        "practical_candidate_day_rate": (
            0.12 <= float(metrics.get("candidate_day_rate", 0.0)) <= 0.28
        ),
        "minimum_win_rate": float(metrics.get("win_rate", 0.0)) >= 0.55,
        "minimum_wilson_lower": (
            float(metrics.get("win_rate_wilson_lower", 0.0)) >= 0.50
        ),
        "minimum_clustered_win_rate_lower": (
            float(metrics.get("clustered_win_rate_lower", 0.0)) >= 0.48
        ),
        "minimum_margin_hit_rate": (
            float(metrics.get("margin_hit_rate", 0.0)) >= 0.35
        ),
        "maximum_tail_loss_rate": (
            float(metrics.get("tail_loss_rate", 1.0)) <= 0.20
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
        "minimum_25_candidates_each_active_year": minimum_year_events >= 25,
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
        "future_shadow_min_candidate_days": 25,
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
    threshold = float(quantile) * float(cumulative[-1])
    index = min(int(np.searchsorted(cumulative, threshold)), len(order) - 1)
    return float(sorted_values[index])


def _blend(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (
        0.70 * np.asarray(first, dtype=float)
        + 0.30 * np.asarray(second, dtype=float)
    )


def _slot_minute(value: str) -> int:
    cleaned = str(value).replace(":", "")
    return int(cleaned[:2]) * 60 + int(cleaned[2:4])


def _slot_minutes(values: pd.Series) -> pd.Series:
    return values.astype(str).map(_slot_minute).astype(float)


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
