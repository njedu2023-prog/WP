from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .io import file_sha256
from .meta_alpha import IDENTITY_COLUMNS, ProbabilityCalibrator
from .sharding import (
    SHARD_MANIFEST_NAME,
    SHARD_PREDICTIONS_NAME,
    SHARD_SCHEMA_VERSION,
)
from .features import slot_to_minute
from .v17_selector import active_feature_columns, day_temporal_weights
from .v23_data import (
    OPTIONAL_SOURCE_SELECTION_COLUMNS,
    REQUIRED_SOURCE_SELECTION_COLUMNS,
    SOURCE_SELECTION_COLUMNS,
    V23_FEATURE_COLUMNS,
)


SCHEMA_VERSION = "wp_v23_microstructure_gate_1"
MODEL_TRAIN_DAYS = 252
MODEL_CALIBRATION_DAYS = 42
MODEL_PURGE_DAYS = 2

FIXED_TARGET_CANDIDATE_DAY_RATE = 0.20
FIXED_MAX_CANDIDATES_PER_DAY = 3
POSITIVE_SPREAD_MAX = 0.25
MARGIN_SPREAD_MAX = 0.25
SEVERE_SPREAD_MAX = 0.25
RETURN_SPREAD_MAX_PCT = 3.0
ROUND_TRIP_FILL_MIN = 0.95
SOURCE_SEVERE_LOSS_MAX = 0.40

SOURCE_PRIOR_FEATURES = (
    "slot_minute",
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
MODEL_FEATURES = (*V23_FEATURE_COLUMNS, *SOURCE_PRIOR_FEATURES)
RESEARCH_SOURCE_COLUMNS = tuple(
    dict.fromkeys(
        (
            *SOURCE_SELECTION_COLUMNS,
            "target_trade_date",
            "name",
            "board",
            "entry_price",
            "t1_close",
            "gross_return_pct",
            "net_return_pct",
            "entry_fillable",
            "exit_fillable",
            "execution_success",
            "label_available",
            "target_net_positive",
            "test_start",
            "test_end",
        )
    )
)
REQUIRED_RESEARCH_SOURCE_COLUMNS = (
    *REQUIRED_SOURCE_SELECTION_COLUMNS,
    "net_return_pct",
    "label_available",
    "target_net_positive",
    "test_start",
    "test_end",
)


@dataclass
class MicrostructureGateBundle:
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
        scored = frame.copy()
        features = feature_matrix(scored, self.feature_columns)
        positive_tree = self.positive_tree.predict_proba(features)[:, 1]
        positive_linear = self.positive_linear.predict_proba(features)[:, 1]
        margin_tree = self.margin_tree.predict_proba(features)[:, 1]
        margin_linear = self.margin_linear.predict_proba(features)[:, 1]
        severe_tree = self.severe_tree.predict_proba(features)[:, 1]
        severe_linear = self.severe_linear.predict_proba(features)[:, 1]
        return_tree = self.return_tree.predict(features)
        return_linear = self.return_linear.predict(features)

        positive_raw = _blend(positive_tree, positive_linear)
        margin_raw = _blend(margin_tree, margin_linear)
        severe_raw = _blend(severe_tree, severe_linear)
        return_raw = _blend(return_tree, return_linear)
        positive = np.clip(
            self.positive_calibrator.predict(positive_raw),
            0.001,
            0.999,
        )
        margin = np.clip(
            self.margin_calibrator.predict(margin_raw),
            0.001,
            0.999,
        )
        severe = np.clip(
            self.severe_calibrator.predict(severe_raw),
            0.001,
            0.999,
        )
        expected = self.return_calibrator.predict(
            np.asarray(return_raw, dtype=float).reshape(-1, 1)
        )
        positive_spread = np.abs(positive_tree - positive_linear)
        margin_spread = np.abs(margin_tree - margin_linear)
        severe_spread = np.abs(severe_tree - severe_linear)
        return_spread = np.abs(return_tree - return_linear)

        scored["v23_p_positive"] = positive
        scored["v23_p_positive_lower"] = np.clip(
            positive - 0.50 * positive_spread - 0.03,
            0.001,
            0.999,
        )
        scored["v23_positive_model_spread"] = positive_spread
        scored["v23_p_margin"] = margin
        scored["v23_p_margin_lower"] = np.clip(
            margin - 0.50 * margin_spread - 0.03,
            0.001,
            0.999,
        )
        scored["v23_margin_model_spread"] = margin_spread
        scored["v23_p_severe_loss"] = severe
        scored["v23_p_severe_loss_upper"] = np.clip(
            severe + 0.50 * severe_spread + 0.03,
            0.001,
            0.999,
        )
        scored["v23_severe_model_spread"] = severe_spread
        scored["v23_expected_net_return_pct"] = expected
        scored["v23_expected_return_model_spread_pct"] = return_spread
        scored["v23_expected_net_return_lower_pct"] = (
            expected
            - 0.50 * return_spread
            - self.return_downside_residual_pct
        )
        scored["v23_economic_score"] = (
            scored["v23_expected_net_return_lower_pct"]
            + 0.80 * (scored["v23_p_positive_lower"] - 0.50)
            + 0.60 * (scored["v23_p_margin_lower"] - 0.30)
            - 1.00 * scored["v23_p_severe_loss_upper"]
        )
        return scored


@dataclass(frozen=True)
class MicrostructurePolicySpec:
    target_candidate_day_rate: float = FIXED_TARGET_CANDIDATE_DAY_RATE
    max_candidates_per_day: int = FIXED_MAX_CANDIDATES_PER_DAY
    positive_probability_lower_min: float = 0.50
    margin_probability_lower_min: float = 0.25
    severe_probability_upper_max: float = 0.35
    expected_net_return_lower_min_pct: float = -0.10

    @property
    def policy_id(self) -> str:
        return (
            f"v23-rate{self.target_candidate_day_rate:.2f}-"
            f"p{self.positive_probability_lower_min:.2f}-"
            f"m{self.margin_probability_lower_min:.2f}-"
            f"s{self.severe_probability_upper_max:.2f}-"
            f"e{self.expected_net_return_lower_min_pct:.2f}-"
            f"k{self.max_candidates_per_day}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "target_candidate_day_rate": self.target_candidate_day_rate,
            "max_candidates_per_day": self.max_candidates_per_day,
            "positive_probability_lower_min": (
                self.positive_probability_lower_min
            ),
            "margin_probability_lower_min": (
                self.margin_probability_lower_min
            ),
            "severe_probability_upper_max": (
                self.severe_probability_upper_max
            ),
            "expected_net_return_lower_min_pct": (
                self.expected_net_return_lower_min_pct
            ),
            "positive_spread_max": POSITIVE_SPREAD_MAX,
            "margin_spread_max": MARGIN_SPREAD_MAX,
            "severe_spread_max": SEVERE_SPREAD_MAX,
            "return_spread_max_pct": RETURN_SPREAD_MAX_PCT,
            "round_trip_fill_min": ROUND_TRIP_FILL_MIN,
            "source_severe_loss_max": SOURCE_SEVERE_LOSS_MAX,
        }


@dataclass(frozen=True)
class FrozenMicrostructurePolicy:
    spec: MicrostructurePolicySpec
    economic_score_threshold: float
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
            "economic_score_threshold": self.economic_score_threshold,
            "calibration_start": self.calibration_start,
            "calibration_end": self.calibration_end,
            "calibration_days": self.calibration_days,
            "eligible_days": self.eligible_days,
        }


def fit_microstructure_gate(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    random_seed: int,
    minimum_train_rows: int = 1_200,
    minimum_calibration_rows: int = 200,
) -> MicrostructureGateBundle:
    prepared_train = labeled_complete_rows(train)
    prepared_calibration = labeled_complete_rows(calibration)
    if len(prepared_train) < minimum_train_rows:
        raise ValueError(
            f"V23 has {len(prepared_train)} train rows; "
            f"requires {minimum_train_rows}"
        )
    if len(prepared_calibration) < minimum_calibration_rows:
        raise ValueError(
            f"V23 has {len(prepared_calibration)} calibration rows; "
            f"requires {minimum_calibration_rows}"
        )
    features = active_feature_columns(
        prepared_train,
        prepared_calibration,
        candidates=MODEL_FEATURES,
    )
    if len(features) < 20:
        raise ValueError(f"V23 has only {len(features)} active features")
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
            raise ValueError(f"V23 {name} target lacks both classes")

    temporal_weight = day_temporal_weights(prepared_train)
    min_leaf = max(40, min(100, len(prepared_train) // 25))
    trees: dict[str, HistGradientBoostingClassifier] = {}
    linears: dict[str, Pipeline] = {}
    calibrators: dict[str, ProbabilityCalibrator] = {}
    for index, name in enumerate(("positive", "margin", "severe")):
        target = targets_train[name]
        tree = HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=180,
            max_leaf_nodes=7,
            min_samples_leaf=min_leaf,
            l2_regularization=30.0,
            random_state=random_seed + index * 101,
        )
        tree.fit(
            x_train,
            target,
            sample_weight=_balanced_weights(target, temporal_weight),
        )
        linear = _linear_classifier(random_seed + index * 101 + 1)
        linear.fit(
            x_train,
            target,
            model__sample_weight=temporal_weight,
        )
        raw = _blend(
            tree.predict_proba(x_calibration)[:, 1],
            linear.predict_proba(x_calibration)[:, 1],
        )
        calibrator = ProbabilityCalibrator().fit(
            raw,
            targets_calibration[name].to_numpy(dtype=int),
            day_temporal_weights(prepared_calibration),
        )
        trees[name] = tree
        linears[name] = linear
        calibrators[name] = calibrator

    return_tree = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.03,
        max_iter=180,
        max_leaf_nodes=7,
        min_samples_leaf=min_leaf,
        l2_regularization=30.0,
        random_state=random_seed + 404,
    )
    clipped_train = net_train.clip(-10.0, 10.0)
    return_tree.fit(
        x_train,
        clipped_train,
        sample_weight=temporal_weight,
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
            ("model", Ridge(alpha=30.0)),
        ]
    )
    return_linear.fit(
        x_train,
        clipped_train,
        model__sample_weight=temporal_weight,
    )
    calibration_return_raw = _blend(
        return_tree.predict(x_calibration),
        return_linear.predict(x_calibration),
    )
    return_calibrator = Ridge(alpha=10.0)
    return_calibrator.fit(
        np.asarray(calibration_return_raw).reshape(-1, 1),
        net_calibration.clip(-10.0, 10.0),
        sample_weight=day_temporal_weights(prepared_calibration),
    )
    calibration_expected = return_calibrator.predict(
        np.asarray(calibration_return_raw).reshape(-1, 1)
    )
    downside_residual = float(
        np.quantile(
            np.maximum(
                calibration_expected
                - net_calibration.to_numpy(dtype=float),
                0.0,
            ),
            0.70,
        )
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
        return_downside_residual_pct=downside_residual,
        feature_columns=features,
        train_rows=int(len(prepared_train)),
        calibration_rows=int(len(prepared_calibration)),
    )


def rolling_microstructure_segments(
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


def calibrate_microstructure_policy(
    scored_calibration: pd.DataFrame,
    *,
    calibration_dates: Iterable[str],
    spec: MicrostructurePolicySpec | None = None,
) -> FrozenMicrostructurePolicy:
    frozen_spec = spec or MicrostructurePolicySpec()
    dates = sorted(set(map(str, calibration_dates)))
    if not dates:
        raise ValueError("V23 policy calibration has no dates")
    calibration = scored_calibration.loc[
        scored_calibration["trade_date"].astype(str).isin(dates)
    ].copy()
    eligible = policy_eligible_rows(calibration, frozen_spec)
    daily_max = (
        eligible.groupby("trade_date", sort=False)["v23_economic_score"]
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
    return FrozenMicrostructurePolicy(
        spec=frozen_spec,
        economic_score_threshold=threshold,
        calibration_start=dates[0],
        calibration_end=dates[-1],
        calibration_days=len(dates),
        eligible_days=int(len(daily_max)),
    )


def apply_microstructure_policy(
    scored: pd.DataFrame,
    policy: FrozenMicrostructurePolicy,
) -> pd.DataFrame:
    eligible = policy_eligible_rows(scored, policy.spec)
    qualified = eligible.loc[
        _numeric(eligible, "v23_economic_score").ge(
            policy.economic_score_threshold
        )
    ].copy()
    if qualified.empty:
        qualified["v23_policy_id"] = policy.policy_id
        return qualified
    qualified["_slot_absolute"] = _slot_absolute(qualified["signal_slot"])
    qualified.sort_values(
        [
            "trade_date",
            "_slot_absolute",
            "v23_economic_score",
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
    selected["v23_policy_id"] = policy.policy_id
    selected["v23_economic_score_threshold"] = (
        policy.economic_score_threshold
    )
    return selected.reset_index(drop=True)


def policy_eligible_rows(
    scored: pd.DataFrame,
    spec: MicrostructurePolicySpec,
) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "v23_point_in_time_complete",
        "v23_p_positive_lower",
        "v23_positive_model_spread",
        "v23_p_margin_lower",
        "v23_margin_model_spread",
        "v23_p_severe_loss_upper",
        "v23_severe_model_spread",
        "v23_expected_net_return_lower_pct",
        "v23_expected_return_model_spread_pct",
        "v23_economic_score",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"V23 policy frame missing columns: {missing}")
    complete = _boolean(scored, "v23_point_in_time_complete")
    eligible = (
        complete
        & _numeric(scored, "p_round_trip_fill_lower").ge(
            ROUND_TRIP_FILL_MIN
        )
        & _numeric(scored, "p_severe_loss").le(SOURCE_SEVERE_LOSS_MAX)
        & _numeric(scored, "v23_positive_model_spread").le(
            POSITIVE_SPREAD_MAX
        )
        & _numeric(scored, "v23_margin_model_spread").le(MARGIN_SPREAD_MAX)
        & _numeric(scored, "v23_severe_model_spread").le(SEVERE_SPREAD_MAX)
        & _numeric(scored, "v23_expected_return_model_spread_pct").le(
            RETURN_SPREAD_MAX_PCT
        )
        & _numeric(scored, "v23_p_positive_lower").ge(
            spec.positive_probability_lower_min
        )
        & _numeric(scored, "v23_p_margin_lower").ge(
            spec.margin_probability_lower_min
        )
        & _numeric(scored, "v23_p_severe_loss_upper").le(
            spec.severe_probability_upper_max
        )
        & _numeric(scored, "v23_expected_net_return_lower_pct").ge(
            spec.expected_net_return_lower_min_pct
        )
    )
    return scored.loc[eligible].copy()


def v23_research_readiness(
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
            0.12 <= float(metrics.get("candidate_day_rate", 0.0)) <= 0.30
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
        "minimum_fifteen_candidates_each_active_year": (
            minimum_year_events >= 15
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


def load_v23_research_source(
    shard_dir: str | Path,
    *,
    evaluation_end: str,
    features: pd.DataFrame,
    data_manifest: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(shard_dir)
    source_contract = data_manifest.get("source") or {}
    if (
        source_contract.get("profit_outcomes_read") is not False
        or not source_contract.get("source_integrity")
    ):
        raise RuntimeError("V23 data source was not outcome blind")
    shard_contracts = source_contract.get("shards") or []
    if not shard_contracts:
        raise RuntimeError("V23 data source has no immutable shard contracts")

    frames: list[pd.DataFrame] = []
    folds: set[int] = set()
    dataset_digest = str(
        source_contract.get("dataset_manifest_sha256") or ""
    )
    if not dataset_digest:
        raise RuntimeError("V23 data source has no dataset digest")
    for shard_contract in shard_contracts:
        manifest_relative = str(shard_contract.get("manifest") or "")
        manifest_path = root / manifest_relative
        if (
            not manifest_relative
            or manifest_path.name != SHARD_MANIFEST_NAME
            or not manifest_path.exists()
        ):
            raise RuntimeError(
                f"V23 immutable source manifest missing: {manifest_relative}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SHARD_SCHEMA_VERSION:
            raise RuntimeError(
                f"V23 immutable source schema mismatch: {manifest_path}"
            )
        if (
            str(manifest.get("dataset_manifest_sha256") or "")
            != dataset_digest
        ):
            raise RuntimeError(
                f"V23 dataset digest mismatch: {manifest_path}"
            )
        prediction_path = manifest_path.with_name(SHARD_PREDICTIONS_NAME)
        expected_sha = str(shard_contract.get("prediction_sha256") or "")
        actual_sha = file_sha256(prediction_path)
        if (
            not expected_sha
            or actual_sha != expected_sha
            or actual_sha != str(manifest.get("prediction_sha256") or "")
        ):
            raise RuntimeError(
                f"V23 immutable source digest mismatch: {prediction_path}"
            )
        available = set(pq.read_schema(prediction_path).names)
        missing = sorted(
            set(REQUIRED_RESEARCH_SOURCE_COLUMNS) - available
        )
        if missing:
            raise RuntimeError(
                f"V23 research source missing {missing}: {prediction_path}"
            )
        projected = [
            column
            for column in RESEARCH_SOURCE_COLUMNS
            if column in available
        ]
        frame = pq.read_table(
            prediction_path,
            columns=projected,
        ).to_pandas()
        if len(frame) != int(manifest.get("prediction_rows", -1)):
            raise RuntimeError(
                f"V23 research source row count mismatch: {prediction_path}"
            )
        for column in OPTIONAL_SOURCE_SELECTION_COLUMNS:
            if column not in frame:
                frame[column] = float("nan")
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame = frame.loc[
            frame["trade_date"].le(str(evaluation_end))
        ].copy()
        folds.update(
            int(value)
            for value in pd.to_numeric(frame["fold"], errors="coerce")
            .dropna()
            .astype(int)
        )
        frames.append(frame)

    source_rows = pd.concat(frames, ignore_index=True)
    identities = list(IDENTITY_COLUMNS)
    if source_rows.duplicated(identities, keep=False).any():
        raise RuntimeError("V23 research source has duplicate identities")
    requested = features.loc[:, [*identities, "fold"]].copy()
    requested.rename(columns={"fold": "_v23_data_fold"}, inplace=True)
    source_rows.rename(columns={"fold": "_v23_source_fold"}, inplace=True)
    leaders = requested.merge(
        source_rows,
        on=identities,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not leaders["_merge"].eq("both").all():
        raise RuntimeError(
            "V23 outcome-blind leader identities are missing from V9 source"
        )
    data_fold = pd.to_numeric(
        leaders["_v23_data_fold"],
        errors="coerce",
    )
    source_fold = pd.to_numeric(
        leaders["_v23_source_fold"],
        errors="coerce",
    )
    if not data_fold.equals(source_fold):
        raise RuntimeError("V23 outcome-blind leader folds do not match")
    leaders["fold"] = data_fold.astype(int)
    leaders.drop(
        columns=["_v23_data_fold", "_v23_source_fold", "_merge"],
        inplace=True,
    )
    net = _numeric(leaders, "net_return_pct")
    target = _numeric(leaders, "target_net_positive")
    label_available = _boolean(leaders, "label_available")
    has_outcome = net.notna() & target.notna()
    inconsistent_availability = label_available.ne(has_outcome)
    inconsistent_target = (
        has_outcome
        & ~target.eq(net.gt(0.0).astype(float))
    )
    if inconsistent_availability.any() or inconsistent_target.any():
        raise RuntimeError(
            "V23 research source outcome availability contract is inconsistent"
        )

    contract_folds = {
        int(value) for value in source_contract.get("folds", [])
    }
    if folds != contract_folds:
        raise RuntimeError("V23 source folds do not match the data build")
    return leaders, {
        **source_contract,
        "schema_version": "wp_v23_research_source_1",
        "profit_outcomes_read": True,
        "outcome_blind_identity_selection": True,
        "research_source_rows": int(len(source_rows)),
        "leader_rows": int(len(leaders)),
        "label_available_rows": int(
            pd.to_numeric(
                leaders["net_return_pct"],
                errors="coerce",
            )
            .notna()
            .sum()
        ),
        "source_integrity": True,
    }


def load_full_trade_calendar(data_manifest: dict[str, Any]) -> list[str]:
    calendar = data_manifest.get("trade_calendar") or {}
    dates = [str(value) for value in calendar.get("open_dates", [])]
    if (
        not dates
        or dates != sorted(set(dates))
        or int(calendar.get("open_date_count", -1)) != len(dates)
    ):
        raise RuntimeError("V23 data manifest trade calendar is invalid")
    return dates


def load_evaluation_calendar(
    data_manifest: dict[str, Any],
    *,
    start_date: str,
    end_date: str,
) -> list[str]:
    return [
        date
        for date in load_full_trade_calendar(data_manifest)
        if str(start_date) <= date <= str(end_date)
    ]


def fold_test_window(frame: pd.DataFrame) -> tuple[str, str]:
    starts = sorted(frame["test_start"].dropna().astype(str).unique())
    ends = sorted(frame["test_end"].dropna().astype(str).unique())
    if len(starts) != 1 or len(ends) != 1 or starts[0] > ends[0]:
        raise RuntimeError(
            f"V23 fold test window is invalid: starts={starts} ends={ends}"
        )
    return starts[0], ends[0]


def labeled_complete_rows(frame: pd.DataFrame) -> pd.DataFrame:
    net = _numeric(frame, "net_return_pct")
    target = _numeric(frame, "target_net_positive")
    label_available = _boolean(frame, "label_available")
    complete = _boolean(frame, "v23_point_in_time_complete")
    consistent = target.eq(net.gt(0.0).astype(float))
    return frame.loc[
        label_available
        & net.notna()
        & target.notna()
        & consistent
        & complete
    ].copy()


def selected_outcome_audit(
    selected: pd.DataFrame,
    *,
    total_days: int,
) -> dict[str, Any]:
    if selected.empty:
        return {
            "selected_rows": 0,
            "selected_days": 0,
            "selected_day_rate": 0.0,
            "verified_outcome_rows": 0,
            "missing_outcome_rows": 0,
            "inconsistent_outcome_rows": 0,
            "all_selected_outcomes_verified": True,
        }
    net = _numeric(selected, "net_return_pct")
    target = _numeric(selected, "target_net_positive")
    label_available = _boolean(selected, "label_available")
    has_outcome = label_available & net.notna() & target.notna()
    consistent = target.eq(net.gt(0.0).astype(float))
    inconsistent = has_outcome & ~consistent
    selected_days = int(selected["trade_date"].astype(str).nunique())
    return {
        "selected_rows": int(len(selected)),
        "selected_days": selected_days,
        "selected_day_rate": selected_days / max(int(total_days), 1),
        "verified_outcome_rows": int((has_outcome & consistent).sum()),
        "missing_outcome_rows": int((~has_outcome).sum()),
        "inconsistent_outcome_rows": int(inconsistent.sum()),
        "all_selected_outcomes_verified": bool(
            has_outcome.all() and consistent.loc[has_outcome].all()
        ),
    }


def feature_matrix(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    values = frame.reindex(columns=columns).copy()
    if "slot_minute" in columns and "signal_slot" in frame:
        values["slot_minute"] = slot_to_minute(frame["signal_slot"])
    for column in columns:
        values[column] = pd.to_numeric(values[column], errors="coerce").astype(
            "float32"
        )
    return values.replace([np.inf, -np.inf], np.nan)


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
                    C=0.05,
                    max_iter=1_000,
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


def _blend(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 0.70 * np.asarray(left, dtype=float) + 0.30 * np.asarray(
        right,
        dtype=float,
    )


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


def _boolean(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )
