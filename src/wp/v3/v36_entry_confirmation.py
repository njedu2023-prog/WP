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
from .v34_intraday_path import normalize_historical_minutes


SCHEMA_VERSION = "wp_v36_post_alert_entry_confirmation_1"
BASE_ALERT_SLOTS = (
    "14:20",
    "14:25",
    "14:30",
    "14:35",
    "14:40",
    "14:45",
)
CONFIRMATION_DELAY_MINUTES = 4
ENTRY_DELAY_MINUTES = 5
EXPECTED_CONFIRMATION_ROWS = 4
MAX_ENTRY_PRICE_ERROR_BPS = 10.0
MINIMUM_DATASET_COVERAGE = 0.98

MODEL_TRAIN_DAYS = 252
MODEL_CALIBRATION_DAYS = 42
MODEL_PURGE_DAYS = 2
MINIMUM_TRAIN_ROWS = 4_000
MINIMUM_CALIBRATION_ROWS = 700

FIXED_TARGET_CANDIDATE_DAY_RATE = 0.30
FIXED_MAX_CANDIDATES_PER_DAY = 3
ROUND_TRIP_FILL_MIN = 0.95
SOURCE_SEVERE_LOSS_MAX = 0.45
MAX_DATA_AGE_SECONDS = 420.0

POSITIVE_PROBABILITY_LOWER_MIN = 0.50
MARGIN_PROBABILITY_LOWER_MIN = 0.20
SEVERE_PROBABILITY_UPPER_MAX = 0.40
EXPECTED_NET_RETURN_LOWER_MIN_PCT = -0.25
PROBABILITY_SPREAD_MAX = 0.35
RETURN_SPREAD_MAX_PCT = 3.0
MARGIN_TARGET_PCT = 0.50
SEVERE_TARGET_PCT = -2.00

SOURCE_PRIOR_FEATURES = (
    "v36_base_slot_minute",
    "ret_from_prev_close_pct",
    "p_net_positive_lower",
    "p_conditional_net_positive",
    "p_severe_loss",
    "p_round_trip_fill_lower",
    "probability_model_spread",
    "expected_return_model_spread",
    "expected_utility_lower_pct",
    "selection_score",
)

POST_ALERT_FEATURES = (
    "v36_confirmation_return_pct",
    "v36_confirmation_range_pct",
    "v36_confirmation_max_extension_pct",
    "v36_confirmation_max_drawdown_pct",
    "v36_confirmation_rebound_from_low_pct",
    "v36_confirmation_reversal_from_high_pct",
    "v36_confirmation_close_position",
    "v36_confirmation_vwap_gap_pct",
    "v36_confirmation_up_minute_share",
    "v36_confirmation_directional_efficiency",
    "v36_confirmation_signed_amount_imbalance",
    "v36_confirmation_amount_ratio_prior20",
    "v36_confirmation_amount_acceleration",
    "v36_confirmation_last2_return_pct",
    "v36_confirmation_zero_amount_share",
)

QUALITY_COLUMNS = (
    "v36_confirmation_observed_rows",
    "v36_confirmation_expected_rows",
    "v36_confirmation_time",
    "v36_entry_benchmark_time",
    "v36_confirmation_latest_time",
    "v36_public_signal_price",
    "v36_entry_audit_close",
    "v36_entry_price_error_bps",
    "v36_causal_ok",
    "v36_entry_price_parity_ok",
    "v36_path_complete",
)

MODEL_FEATURES = (*SOURCE_PRIOR_FEATURES, *POST_ALERT_FEATURES)
MODEL_FEATURE_SET = frozenset(MODEL_FEATURES)
FORBIDDEN_FEATURE_TOKENS = (
    "target",
    "label",
    "truth",
    "future",
    "gross_return",
    "net_return",
    "t1_",
    "entry_price",
    "entry_audit",
    "exit_",
)


@dataclass(frozen=True)
class EntryConfirmationPolicySpec:
    target_candidate_day_rate: float = FIXED_TARGET_CANDIDATE_DAY_RATE
    max_candidates_per_day: int = FIXED_MAX_CANDIDATES_PER_DAY

    @property
    def policy_id(self) -> str:
        return (
            f"v36-confirm-rate{self.target_candidate_day_rate:.2f}-"
            f"k{self.max_candidates_per_day}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "target_candidate_day_rate": self.target_candidate_day_rate,
            "max_candidates_per_day": self.max_candidates_per_day,
            "confirmation_delay_minutes": CONFIRMATION_DELAY_MINUTES,
            "entry_delay_minutes": ENTRY_DELAY_MINUTES,
            "round_trip_fill_min": ROUND_TRIP_FILL_MIN,
            "source_severe_loss_max": SOURCE_SEVERE_LOSS_MAX,
            "positive_probability_lower_min": (
                POSITIVE_PROBABILITY_LOWER_MIN
            ),
            "margin_probability_lower_min": MARGIN_PROBABILITY_LOWER_MIN,
            "severe_probability_upper_max": (
                SEVERE_PROBABILITY_UPPER_MAX
            ),
            "expected_net_return_lower_min_pct": (
                EXPECTED_NET_RETURN_LOWER_MIN_PCT
            ),
            "probability_spread_max": PROBABILITY_SPREAD_MAX,
            "return_spread_max_pct": RETURN_SPREAD_MAX_PCT,
            "max_data_age_seconds": MAX_DATA_AGE_SECONDS,
            "first_passing_confirmation_is_immutable": True,
            "no_signal_allowed": True,
        }


@dataclass(frozen=True)
class FrozenEntryConfirmationPolicy:
    spec: EntryConfirmationPolicySpec
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
class EntryConfirmationBundle:
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
        matrix = confirmation_feature_matrix(result, self.feature_columns)
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

        result["v36_p_positive"] = positive
        result["v36_p_positive_lower"] = np.clip(
            positive - 0.50 * positive_spread - 0.03,
            0.001,
            0.999,
        )
        result["v36_positive_model_spread"] = positive_spread
        result["v36_p_margin"] = margin
        result["v36_p_margin_lower"] = np.clip(
            margin - 0.50 * margin_spread - 0.03,
            0.001,
            0.999,
        )
        result["v36_margin_model_spread"] = margin_spread
        result["v36_p_severe_loss"] = severe
        result["v36_p_severe_loss_upper"] = np.clip(
            severe + 0.50 * severe_spread + 0.03,
            0.001,
            0.999,
        )
        result["v36_severe_model_spread"] = severe_spread
        result["v36_expected_net_return_pct"] = expected
        result["v36_expected_return_model_spread_pct"] = return_spread
        result["v36_expected_net_return_lower_pct"] = (
            expected
            - 0.50 * return_spread
            - self.return_downside_residual_pct
        )
        result["v36_confirmation_score"] = (
            result["v36_expected_net_return_lower_pct"]
            + 1.00 * (result["v36_p_positive_lower"] - 0.50)
            + 0.60 * (result["v36_p_margin_lower"] - 0.30)
            - 1.10 * result["v36_p_severe_loss_upper"]
        )
        return result


def confirmation_timing(signal_slot: str) -> tuple[str, str]:
    if signal_slot not in BASE_ALERT_SLOTS:
        raise ValueError(f"unsupported V36 base alert slot: {signal_slot}")
    base = pd.Timestamp(f"2000-01-01 {signal_slot}:00")
    confirmation = base + pd.Timedelta(f"{CONFIRMATION_DELAY_MINUTES}min")
    entry = base + pd.Timedelta(f"{ENTRY_DELAY_MINUTES}min")
    return confirmation.strftime("%H:%M"), entry.strftime("%H:%M")


def build_post_alert_confirmation_features(
    candidates: pd.DataFrame,
    minutes: pd.DataFrame,
    *,
    entry_slippage_bps: float,
) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "fold",
        "signal_price",
        "entry_price",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"V36 candidates missing columns: {missing}")
    source = candidates.loc[
        candidates["signal_slot"].astype(str).isin(BASE_ALERT_SLOTS),
        [*IDENTITY_COLUMNS, "fold", "signal_price", "entry_price"],
    ].copy()
    for column in IDENTITY_COLUMNS:
        source[column] = source[column].astype(str)
    if source.duplicated(list(IDENTITY_COLUMNS), keep=False).any():
        raise ValueError("V36 candidate identities are duplicated")

    normalized = normalize_historical_minutes(minutes)
    grouped = {
        (str(code), str(date)): group.reset_index(drop=True)
        for (code, date), group in normalized.groupby(
            ["ts_code", "trade_date"],
            sort=False,
        )
    }
    empty = normalized.head(0)
    rows: list[dict[str, Any]] = []
    slippage = float(entry_slippage_bps) / 10_000.0

    for record in source.to_dict(orient="records"):
        trade_date = str(record["trade_date"])
        signal_slot = str(record["signal_slot"])
        signal_time = _timestamp(trade_date, signal_slot)
        confirmation_time = signal_time + pd.Timedelta(
            f"{CONFIRMATION_DELAY_MINUTES}min"
        )
        entry_time = signal_time + pd.Timedelta(f"{ENTRY_DELAY_MINUTES}min")
        stock_day = grouped.get((str(record["ts_code"]), trade_date), empty)
        prior = stock_day.loc[
            stock_day["trade_time"].le(signal_time)
        ].tail(20)
        confirmation = stock_day.loc[
            stock_day["trade_time"].gt(signal_time)
            & stock_day["trade_time"].le(confirmation_time)
        ].copy()
        confirmation.sort_values("trade_time", kind="stable", inplace=True)
        entry_bar = stock_day.loc[
            stock_day["trade_time"].eq(entry_time)
        ].tail(1)
        latest = (
            confirmation["trade_time"].max()
            if not confirmation.empty
            else pd.NaT
        )
        observed = int(len(confirmation))
        causal = bool(
            pd.isna(latest)
            or (
                latest <= confirmation_time
                and confirmation["trade_time"].le(confirmation_time).all()
                and confirmation["trade_time"].gt(signal_time).all()
            )
        )
        source_entry = _finite_float(record.get("entry_price"))
        entry_close = (
            _finite_float(entry_bar["close"].iloc[-1])
            if not entry_bar.empty
            else np.nan
        )
        audited_entry = (
            entry_close * (1.0 + slippage)
            if np.isfinite(entry_close)
            else np.nan
        )
        entry_error = _ratio_distance_bps(audited_entry, source_entry)
        entry_parity = bool(
            np.isfinite(entry_error)
            and entry_error <= MAX_ENTRY_PRICE_ERROR_BPS
        )
        values = _confirmation_feature_values(
            confirmation,
            prior,
            signal_price=_finite_float(record.get("signal_price")),
        )
        finite = bool(
            all(np.isfinite(values[column]) for column in POST_ALERT_FEATURES)
        )
        complete = bool(
            observed == EXPECTED_CONFIRMATION_ROWS
            and causal
            and not pd.isna(latest)
            and latest == confirmation_time
            and len(entry_bar) == 1
            and entry_parity
            and finite
        )
        public_signal_price = (
            _finite_float(confirmation["close"].iloc[-1])
            if observed
            else np.nan
        )
        rows.append(
            {
                **record,
                "v36_base_slot_minute": float(_slot_minute(signal_slot)),
                "v36_confirmation_observed_rows": observed,
                "v36_confirmation_expected_rows": (
                    EXPECTED_CONFIRMATION_ROWS
                ),
                "v36_confirmation_time": confirmation_time.strftime("%H:%M"),
                "v36_entry_benchmark_time": entry_time.strftime("%H:%M"),
                "v36_confirmation_latest_time": (
                    latest.isoformat() if not pd.isna(latest) else None
                ),
                "v36_public_signal_price": public_signal_price,
                "v36_entry_audit_close": (
                    float(entry_close) if np.isfinite(entry_close) else np.nan
                ),
                "v36_entry_price_error_bps": (
                    float(entry_error)
                    if np.isfinite(entry_error)
                    else np.nan
                ),
                "v36_causal_ok": causal,
                "v36_entry_price_parity_ok": entry_parity,
                "v36_path_complete": complete,
                **values,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result.sort_values(
        ["fold", *IDENTITY_COLUMNS],
        kind="stable",
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)
    if not result["v36_causal_ok"].all():
        raise RuntimeError("V36 confirmation features crossed their cutoff")
    return result


def audit_confirmation_feature_coverage(
    features: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    identity = list(IDENTITY_COLUMNS)
    expected = candidates.loc[
        candidates["signal_slot"].astype(str).isin(BASE_ALERT_SLOTS),
        identity,
    ].copy()
    for column in identity:
        expected[column] = expected[column].astype(str)
    actual = features.reindex(columns=identity).copy()
    for column in identity:
        actual[column] = actual[column].astype(str)
    expected.sort_values(identity, inplace=True)
    actual.sort_values(identity, inplace=True)
    expected.reset_index(drop=True, inplace=True)
    actual.reset_index(drop=True, inplace=True)
    identity_exact = bool(
        len(expected) == len(actual)
        and expected.equals(actual)
        and not features.duplicated(identity, keep=False).any()
    )
    complete = _boolean(features, "v36_path_complete")
    parity = _boolean(features, "v36_entry_price_parity_ok")
    causal = _boolean(features, "v36_causal_ok")
    numeric = features.reindex(columns=POST_ALERT_FEATURES).apply(
        pd.to_numeric,
        errors="coerce",
    )
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    complete_rate = float(complete.mean()) if len(features) else 0.0
    parity_rate = float(parity.mean()) if len(features) else 0.0
    finite_rate = float(finite.mean()) if len(features) else 0.0
    passed = bool(
        identity_exact
        and complete_rate >= MINIMUM_DATASET_COVERAGE
        and parity_rate >= MINIMUM_DATASET_COVERAGE
        and finite_rate >= MINIMUM_DATASET_COVERAGE
        and causal.all()
    )
    return {
        "expected_rows": int(len(expected)),
        "feature_rows": int(len(features)),
        "identity_exact": identity_exact,
        "duplicate_identities": int(features.duplicated(identity).sum()),
        "complete_row_coverage": complete_rate,
        "entry_price_parity_rate": parity_rate,
        "finite_feature_row_rate": finite_rate,
        "causal_timestamps": bool(causal.all()),
        "coverage_passed": passed,
    }


def join_confirmation_features(
    candidates: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    identity = list(IDENTITY_COLUMNS)
    source = candidates.loc[
        candidates["signal_slot"].astype(str).isin(BASE_ALERT_SLOTS)
    ].copy()
    for column in identity:
        source[column] = source[column].astype(str)
    feature = features.copy()
    for column in identity:
        feature[column] = feature[column].astype(str)
    feature.rename(
        columns={
            "fold": "_v36_feature_fold",
            "signal_price": "_v36_feature_signal_price",
            "entry_price": "_v36_feature_entry_price",
        },
        inplace=True,
    )
    joined = source.merge(
        feature,
        on=identity,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise RuntimeError("V36 confirmation join missed source identities")
    if not _numeric(joined, "fold").equals(
        _numeric(joined, "_v36_feature_fold")
    ):
        raise RuntimeError("V36 confirmation folds differ from source folds")
    for source_column, feature_column, name in (
        ("signal_price", "_v36_feature_signal_price", "signal"),
        ("entry_price", "_v36_feature_entry_price", "entry"),
    ):
        if not np.allclose(
            _numeric(joined, source_column).to_numpy(dtype=float),
            _numeric(joined, feature_column).to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise RuntimeError(f"V36 {name} prices differ from source truth")
    return joined.drop(
        columns=[
            "_v36_feature_fold",
            "_v36_feature_signal_price",
            "_v36_feature_entry_price",
            "_merge",
        ]
    )


def fit_entry_confirmation_gate(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    random_seed: int,
    minimum_train_rows: int = MINIMUM_TRAIN_ROWS,
    minimum_calibration_rows: int = MINIMUM_CALIBRATION_ROWS,
) -> EntryConfirmationBundle:
    prepared_train = labeled_confirmation_rows(train)
    prepared_calibration = labeled_confirmation_rows(calibration)
    if len(prepared_train) < minimum_train_rows:
        raise ValueError(
            f"V36 has {len(prepared_train)} train rows; "
            f"requires {minimum_train_rows}"
        )
    if len(prepared_calibration) < minimum_calibration_rows:
        raise ValueError(
            f"V36 has {len(prepared_calibration)} calibration rows; "
            f"requires {minimum_calibration_rows}"
        )
    features = active_confirmation_features(
        prepared_train,
        prepared_calibration,
    )
    if len(features) < 15:
        raise ValueError(f"V36 has only {len(features)} active features")
    x_train = confirmation_feature_matrix(prepared_train, features)
    x_calibration = confirmation_feature_matrix(
        prepared_calibration,
        features,
    )
    net_train = _numeric(prepared_train, "net_return_pct")
    net_calibration = _numeric(prepared_calibration, "net_return_pct")
    targets_train = {
        "positive": net_train.gt(0.0).astype(int),
        "margin": net_train.gt(MARGIN_TARGET_PCT).astype(int),
        "severe": net_train.le(SEVERE_TARGET_PCT).astype(int),
    }
    targets_calibration = {
        "positive": net_calibration.gt(0.0).astype(int),
        "margin": net_calibration.gt(MARGIN_TARGET_PCT).astype(int),
        "severe": net_calibration.le(SEVERE_TARGET_PCT).astype(int),
    }
    for name in targets_train:
        if (
            targets_train[name].nunique() < 2
            or targets_calibration[name].nunique() < 2
        ):
            raise ValueError(f"V36 {name} target lacks both classes")

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
            max_iter=180,
            max_leaf_nodes=7,
            min_samples_leaf=min_leaf,
            l2_regularization=50.0,
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
        max_iter=180,
        max_leaf_nodes=7,
        min_samples_leaf=min_leaf,
        l2_regularization=50.0,
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
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=50.0)),
        ]
    )
    return_linear.fit(
        x_train,
        clipped_train,
        model__sample_weight=train_weight,
    )
    raw_return = _blend(
        return_tree.predict(x_calibration),
        return_linear.predict(x_calibration),
    )
    return_calibrator = Ridge(alpha=20.0)
    return_calibrator.fit(
        raw_return.reshape(-1, 1),
        net_calibration.clip(-10.0, 10.0),
        sample_weight=calibration_weight,
    )
    expected = return_calibrator.predict(raw_return.reshape(-1, 1))
    downside = np.maximum(
        expected - net_calibration.to_numpy(dtype=float),
        0.0,
    )
    return EntryConfirmationBundle(
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


def labeled_confirmation_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "v36_path_complete",
        "label_available",
        "net_return_pct",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V36 labeled frame missing columns: {missing}")
    available = (
        _boolean(frame, "v36_path_complete")
        & _boolean(frame, "label_available")
        & _numeric(frame, "net_return_pct").notna()
    )
    return frame.loc[available].copy()


def active_confirmation_features(
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
            _numeric(combined, column).notna().sum() >= 100
            and _numeric(combined, column).nunique(dropna=True) > 1
        )
    )


def confirmation_feature_matrix(
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


def rolling_confirmation_segments(
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
    train = selected[:MODEL_TRAIN_DAYS]
    calibration_start = MODEL_TRAIN_DAYS + MODEL_PURGE_DAYS
    calibration = selected[
        calibration_start : calibration_start + MODEL_CALIBRATION_DAYS
    ]
    return train, calibration


def calibrate_entry_confirmation_policy(
    scored_calibration: pd.DataFrame,
    *,
    calibration_dates: Iterable[str],
    spec: EntryConfirmationPolicySpec | None = None,
) -> FrozenEntryConfirmationPolicy:
    frozen_spec = spec or EntryConfirmationPolicySpec()
    dates = sorted(set(map(str, calibration_dates)))
    if not dates:
        raise ValueError("V36 policy calibration has no dates")
    calibration = scored_calibration.loc[
        scored_calibration["trade_date"].astype(str).isin(dates)
    ].copy()
    eligible = confirmation_policy_eligible(calibration)
    daily_max = (
        eligible.groupby("trade_date", sort=False)[
            "v36_confirmation_score"
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
    return FrozenEntryConfirmationPolicy(
        spec=frozen_spec,
        score_threshold=threshold,
        calibration_start=dates[0],
        calibration_end=dates[-1],
        calibration_days=len(dates),
        eligible_days=int(len(daily_max)),
    )


def apply_entry_confirmation_policy(
    scored: pd.DataFrame,
    policy: FrozenEntryConfirmationPolicy,
) -> pd.DataFrame:
    eligible = confirmation_policy_eligible(scored)
    qualified = eligible.loc[
        _numeric(eligible, "v36_confirmation_score").ge(
            policy.score_threshold
        )
    ].copy()
    if qualified.empty:
        qualified["v36_policy_id"] = policy.policy_id
        return qualified
    qualified["_v36_confirmation_minute"] = qualified[
        "v36_confirmation_time"
    ].map(_slot_minute)
    qualified.sort_values(
        [
            "trade_date",
            "_v36_confirmation_minute",
            "v36_confirmation_score",
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
    selected.drop(columns="_v36_confirmation_minute", inplace=True)
    selected["v36_policy_id"] = policy.policy_id
    selected["v36_score_threshold"] = policy.score_threshold
    return selected.reset_index(drop=True)


def confirmation_policy_eligible(scored: pd.DataFrame) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "v36_path_complete",
        "v36_confirmation_time",
        "v36_entry_benchmark_time",
        "v36_confirmation_score",
        "v36_p_positive_lower",
        "v36_p_margin_lower",
        "v36_p_severe_loss_upper",
        "v36_expected_net_return_lower_pct",
        "v36_positive_model_spread",
        "v36_margin_model_spread",
        "v36_severe_model_spread",
        "v36_expected_return_model_spread_pct",
        "p_round_trip_fill_lower",
        "p_severe_loss",
        "execution_eligible",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"V36 policy frame missing columns: {missing}")
    age = _numeric(scored, "data_age_seconds")
    fresh = age.isna() | (
        age.ge(0.0) & age.le(MAX_DATA_AGE_SECONDS)
    )
    legal_base = scored["signal_slot"].astype(str).isin(BASE_ALERT_SLOTS)
    legal_confirmation = (
        scored["v36_confirmation_time"].astype(str).between("14:24", "14:49")
    )
    legal_entry = (
        scored["v36_entry_benchmark_time"].astype(str).between(
            "14:25",
            "14:50",
        )
    )
    eligible = (
        _boolean(scored, "v36_path_complete")
        & _boolean(scored, "execution_eligible")
        & _numeric(scored, "p_round_trip_fill_lower").ge(
            ROUND_TRIP_FILL_MIN
        )
        & _numeric(scored, "p_severe_loss").le(SOURCE_SEVERE_LOSS_MAX)
        & _numeric(scored, "v36_p_positive_lower").ge(
            POSITIVE_PROBABILITY_LOWER_MIN
        )
        & _numeric(scored, "v36_p_margin_lower").ge(
            MARGIN_PROBABILITY_LOWER_MIN
        )
        & _numeric(scored, "v36_p_severe_loss_upper").le(
            SEVERE_PROBABILITY_UPPER_MAX
        )
        & _numeric(scored, "v36_expected_net_return_lower_pct").ge(
            EXPECTED_NET_RETURN_LOWER_MIN_PCT
        )
        & _numeric(scored, "v36_positive_model_spread").le(
            PROBABILITY_SPREAD_MAX
        )
        & _numeric(scored, "v36_margin_model_spread").le(
            PROBABILITY_SPREAD_MAX
        )
        & _numeric(scored, "v36_severe_model_spread").le(
            PROBABILITY_SPREAD_MAX
        )
        & _numeric(scored, "v36_expected_return_model_spread_pct").le(
            RETURN_SPREAD_MAX_PCT
        )
        & fresh
        & legal_base
        & legal_confirmation
        & legal_entry
    )
    return scored.loc[eligible].copy()


def validate_feature_contract(features: tuple[str, ...]) -> bool:
    invalid = sorted(set(features) - MODEL_FEATURE_SET)
    contaminated = [
        feature
        for feature in features
        if any(token in feature.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if len(features) < 15 or invalid or contaminated:
        raise RuntimeError(
            "V36 feature contract violated: "
            f"count={len(features)} invalid={invalid} "
            f"contaminated={contaminated}"
        )
    return True


def validate_selected_contract(
    selected: pd.DataFrame,
    policy: FrozenEntryConfirmationPolicy | None,
) -> None:
    if selected.empty:
        return
    if selected.duplicated(["trade_date", "ts_code"], keep=False).any():
        raise RuntimeError("V36 selected output rewrote a first confirmation")
    maximum = (
        policy.spec.max_candidates_per_day
        if policy is not None
        else FIXED_MAX_CANDIDATES_PER_DAY
    )
    if int(selected.groupby("trade_date").size().max()) > maximum:
        raise RuntimeError("V36 selected output exceeds fixed daily maximum")
    if not selected["signal_slot"].astype(str).isin(BASE_ALERT_SLOTS).all():
        raise RuntimeError("V36 selected output has an illegal base alert")
    if not selected["v36_confirmation_time"].astype(str).between(
        "14:24",
        "14:49",
    ).all():
        raise RuntimeError("V36 selected output has an illegal public signal")
    if not selected["v36_entry_benchmark_time"].astype(str).between(
        "14:25",
        "14:50",
    ).all():
        raise RuntimeError("V36 selected output has an illegal entry time")
    if not _boolean(selected, "v36_path_complete").all():
        raise RuntimeError("V36 selected output contains incomplete paths")


def v36_research_readiness(
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
        "minimum_nested_oos_candidates": int(metrics.get("events", 0)) >= 150,
        "minimum_nested_oos_candidate_days": (
            int(metrics.get("candidate_days", 0)) >= 100
        ),
        "practical_candidate_day_rate": (
            0.15 <= float(metrics.get("candidate_day_rate", 0.0)) <= 0.40
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
        "minimum_twenty_five_candidates_each_active_year": (
            minimum_year_events >= 25
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
        "future_shadow_min_candidates": 75,
        "future_shadow_min_candidate_days": 50,
        "reason": (
            "historical_screen_passed_future_shadow_still_required"
            if passed
            else "historical_evidence_insufficient"
        ),
    }


def _confirmation_feature_values(
    confirmation: pd.DataFrame,
    prior: pd.DataFrame,
    *,
    signal_price: float,
) -> dict[str, float]:
    empty = {column: np.nan for column in POST_ALERT_FEATURES}
    if confirmation.empty or not np.isfinite(signal_price) or signal_price <= 0:
        return empty
    close = _numeric(confirmation, "close")
    high = _numeric(confirmation, "high")
    low = _numeric(confirmation, "low")
    amount = _numeric(confirmation, "amount").clip(lower=0.0)
    sequence = pd.Series(
        [signal_price, *close.tolist()],
        dtype=float,
    )
    changes = sequence.pct_change().iloc[1:]
    total_amount = float(amount.sum())
    interval_high = float(high.max())
    interval_low = float(low.min())
    final_close = float(close.iloc[-1])
    price_range = interval_high - interval_low
    weighted_price = (
        float((close * amount).sum() / total_amount)
        if total_amount > 0
        else float(close.mean())
    )
    path_denominator = float(sequence.diff().abs().sum())
    prior_amount = _numeric(prior, "amount").clip(lower=0.0)
    prior_mean = float(prior_amount.mean()) if len(prior_amount) else np.nan
    first_half = float(amount.iloc[:2].mean()) if len(amount) >= 2 else np.nan
    second_half = float(amount.iloc[-2:].mean()) if len(amount) >= 2 else np.nan
    signed_amount = np.sign(changes.to_numpy(dtype=float)) * amount.to_numpy(
        dtype=float
    )
    values = {
        "v36_confirmation_return_pct": _return_pct(
            final_close,
            signal_price,
        ),
        "v36_confirmation_range_pct": _return_pct(
            interval_high,
            interval_low,
        ),
        "v36_confirmation_max_extension_pct": _return_pct(
            interval_high,
            signal_price,
        ),
        "v36_confirmation_max_drawdown_pct": _return_pct(
            interval_low,
            signal_price,
        ),
        "v36_confirmation_rebound_from_low_pct": _return_pct(
            final_close,
            interval_low,
        ),
        "v36_confirmation_reversal_from_high_pct": _return_pct(
            final_close,
            interval_high,
        ),
        "v36_confirmation_close_position": (
            float((final_close - interval_low) / price_range)
            if price_range > 0
            else 0.5
        ),
        "v36_confirmation_vwap_gap_pct": _return_pct(
            final_close,
            weighted_price,
        ),
        "v36_confirmation_up_minute_share": float(changes.gt(0.0).mean()),
        "v36_confirmation_directional_efficiency": (
            float((final_close - signal_price) / path_denominator)
            if path_denominator > 0
            else 0.0
        ),
        "v36_confirmation_signed_amount_imbalance": (
            float(signed_amount.sum() / total_amount)
            if total_amount > 0
            else np.nan
        ),
        "v36_confirmation_amount_ratio_prior20": (
            float(amount.mean() / prior_mean)
            if np.isfinite(prior_mean) and prior_mean > 0
            else np.nan
        ),
        "v36_confirmation_amount_acceleration": (
            float(second_half / first_half)
            if np.isfinite(first_half) and first_half > 0
            else np.nan
        ),
        "v36_confirmation_last2_return_pct": (
            _return_pct(final_close, float(sequence.iloc[-3]))
            if len(sequence) >= 3
            else np.nan
        ),
        "v36_confirmation_zero_amount_share": float(amount.le(0.0).mean()),
    }
    return {**empty, **values}


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
    order = np.argsort(values)
    ordered_values = np.asarray(values, dtype=float)[order]
    ordered_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(ordered_weights)
    if not len(cumulative) or cumulative[-1] <= 0:
        return 0.0
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1]))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _blend(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 0.50 * np.asarray(left, dtype=float) + 0.50 * np.asarray(
        right,
        dtype=float,
    )


def _timestamp(trade_date: str, slot: str) -> pd.Timestamp:
    return pd.Timestamp(
        f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} {slot}:00"
    )


def _slot_minute(slot: str) -> int:
    hours, minutes = map(int, str(slot).split(":"))
    return hours * 60 + minutes


def _return_pct(numerator: float, denominator: float) -> float:
    if (
        not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or denominator <= 0
    ):
        return np.nan
    return float((numerator / denominator - 1.0) * 100.0)


def _ratio_distance_bps(left: float, right: float) -> float:
    if (
        not np.isfinite(left)
        or not np.isfinite(right)
        or left <= 0
        or right <= 0
    ):
        return np.nan
    return float(abs(left / right - 1.0) * 10_000.0)


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


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
