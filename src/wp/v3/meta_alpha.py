from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import V3Config
from .statistics import wilson_interval


IDENTITY_COLUMNS = ("trade_date", "signal_slot", "ts_code")
PRUNE_SCORE_COLUMNS = (
    "p_net_positive",
    "expected_utility_pct",
    "selection_score",
    "p_conditional_net_positive",
)
BASE_NUMERIC_COLUMNS = (
    "ret_from_prev_close_pct",
    "data_age_seconds",
    "p_entry_fill",
    "p_exit_fill_given_entry",
    "p_round_trip_fill_lower",
    "p_net_positive",
    "p_net_positive_lower",
    "p_conditional_net_positive",
    "p_severe_loss",
    "selection_score",
    "selection_rank_pct",
    "expected_utility_pct",
    "expected_utility_lower_pct",
    "downside_q10_pct",
    "probability_model_spread",
    "fill_probability_model_spread",
    "selection_rank_spread",
    "expected_return_model_spread",
)
META_FEATURE_COLUMNS = (
    "slot_minute",
    "ret_from_prev_close_pct",
    "ret_abs_pct",
    "p_entry_fill",
    "p_exit_fill_given_entry",
    "p_round_trip_fill_lower",
    "p_net_positive",
    "p_net_positive_lower",
    "p_conditional_net_positive",
    "p_severe_loss",
    "selection_score",
    "selection_rank_pct",
    "expected_utility_pct",
    "expected_utility_lower_pct",
    "downside_q10_pct",
    "probability_model_spread",
    "fill_probability_model_spread",
    "selection_rank_spread",
    "expected_return_model_spread",
    "data_age_seconds",
    "probability_uncertainty",
    "utility_uncertainty_pct",
    "context_universe_size",
    "context_return_mean_pct",
    "context_return_median_pct",
    "context_return_dispersion_pct",
    "context_breadth_positive",
    "context_breadth_above_2pct",
    "context_breadth_above_5pct",
    "context_breadth_above_7pct",
    "context_probability_mean",
    "context_probability_dispersion",
    "context_utility_mean_pct",
    "context_return_change_from_1420_pct",
    "return_context_relative_pct",
    "return_context_zscore",
    "return_rank_pct",
    "probability_rank_pct",
    "utility_rank_pct",
    "selection_rank_recomputed_pct",
    "severe_quality_rank_pct",
)


@dataclass(frozen=True)
class MetaPolicy:
    probability_min: float
    expected_return_min_pct: float
    severe_loss_max: float
    round_trip_fill_min: float
    meta_rank_min: float
    max_candidates_per_day: int
    slot_group: str

    @property
    def policy_id(self) -> str:
        return (
            f"p{self.probability_min:.2f}-"
            f"e{self.expected_return_min_pct:.2f}-"
            f"s{self.severe_loss_max:.2f}-"
            f"f{self.round_trip_fill_min:.2f}-"
            f"r{self.meta_rank_min:.2f}-"
            f"k{self.max_candidates_per_day}-"
            f"{self.slot_group}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "probability_min": self.probability_min,
            "expected_return_min_pct": self.expected_return_min_pct,
            "severe_loss_max": self.severe_loss_max,
            "round_trip_fill_min": self.round_trip_fill_min,
            "meta_rank_min": self.meta_rank_min,
            "max_candidates_per_day": self.max_candidates_per_day,
            "slot_group": self.slot_group,
        }


@dataclass
class ProbabilityCalibrator:
    model: IsotonicRegression | None = None
    platt_model: LogisticRegression | None = None
    constant: float | None = None
    method: str = "isotonic"
    one_sided_margin: float = 0.0

    def fit(
        self,
        raw_probability: np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray,
        *,
        dates: np.ndarray | None = None,
        margin_seed: int = 20_260_804,
    ) -> "ProbabilityCalibrator":
        values = np.clip(
            np.asarray(raw_probability, dtype=float),
            0.001,
            0.999,
        )
        labels = np.asarray(target, dtype=int)
        if len(labels) == 0:
            raise ValueError("calibration labels are empty")
        if len(np.unique(labels)) < 2:
            self.constant = float(np.average(labels, weights=sample_weight))
            self.model = None
            self.platt_model = None
            self.one_sided_margin = _clustered_overconfidence_margin(
                np.full(len(labels), self.constant, dtype=float),
                labels,
                dates=dates,
                sample_weight=sample_weight,
                seed=margin_seed,
            )
            return self
        if self.method == "platt":
            self.platt_model = LogisticRegression(
                C=1.0,
                max_iter=2_000,
                solver="lbfgs",
            )
            self.platt_model.fit(
                _probability_logit(values),
                labels,
                sample_weight=sample_weight,
            )
            self.model = None
        elif self.method == "isotonic":
            self.model = IsotonicRegression(
                y_min=0.001,
                y_max=0.999,
                out_of_bounds="clip",
            )
            self.model.fit(values, labels, sample_weight=sample_weight)
            self.platt_model = None
        else:
            raise ValueError(f"unknown probability calibration method: {self.method}")
        self.constant = None
        self.one_sided_margin = _clustered_overconfidence_margin(
            self.predict(values),
            labels,
            dates=dates,
            sample_weight=sample_weight,
            seed=margin_seed,
        )
        return self

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        values = np.clip(
            np.asarray(raw_probability, dtype=float),
            0.001,
            0.999,
        )
        if self.constant is not None:
            return np.full(
                len(values),
                np.clip(self.constant, 0.001, 0.999),
                dtype=float,
            )
        if self.method == "platt" and self.platt_model is not None:
            return np.clip(
                np.asarray(
                    self.platt_model.predict_proba(
                        _probability_logit(values)
                    )[:, 1],
                    dtype=float,
                ),
                0.001,
                0.999,
            )
        if self.model is None:
            constant = 0.5 if self.constant is None else self.constant
            return np.full(
                len(values),
                np.clip(constant, 0.001, 0.999),
                dtype=float,
            )
        return np.clip(
            np.asarray(self.model.predict(values), dtype=float),
            0.001,
            0.999,
        )

    def predict_lower(
        self,
        raw_probability: np.ndarray,
        *,
        member_probabilities: np.ndarray | None = None,
    ) -> np.ndarray:
        point = self.predict(raw_probability)
        conservative = point.copy()
        if member_probabilities is not None:
            members = np.asarray(member_probabilities, dtype=float)
            if members.ndim != 2 or members.shape[0] != len(point):
                raise ValueError("member probabilities must be rows by members")
            calibrated_members = np.column_stack(
                [self.predict(members[:, index]) for index in range(members.shape[1])]
            )
            conservative = np.minimum(
                conservative,
                calibrated_members.min(axis=1),
            )
        return np.clip(
            conservative - max(float(self.one_sided_margin), 0.0),
            0.001,
            0.999,
        )


@dataclass
class MetaAlphaBundle:
    probability_tree: HistGradientBoostingClassifier
    probability_linear: Pipeline
    severe_tree: HistGradientBoostingClassifier
    return_tree: HistGradientBoostingRegressor
    probability_calibrator: ProbabilityCalibrator
    severe_calibrator: ProbabilityCalibrator
    return_calibrator: Ridge
    feature_columns: tuple[str, ...]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        scored = frame.copy()
        features = meta_feature_matrix(scored, self.feature_columns)
        tree_probability, linear_probability = _probability_components(
            self.probability_tree,
            self.probability_linear,
            features,
        )
        raw_probability = _blend_probability_components(
            tree_probability,
            linear_probability,
        )
        raw_severe = self.severe_tree.predict_proba(features)[:, 1]
        raw_return = self.return_tree.predict(features)
        scored["meta_p_positive_raw"] = raw_probability
        scored["meta_p_positive_tree_raw"] = tree_probability
        scored["meta_p_positive_linear_raw"] = linear_probability
        scored["meta_p_positive"] = np.clip(
            self.probability_calibrator.predict(raw_probability),
            0.001,
            0.999,
        )
        scored["meta_p_positive_lower"] = (
            self.probability_calibrator.predict_lower(
                raw_probability,
                member_probabilities=np.column_stack(
                    [tree_probability, linear_probability]
                ),
            )
        )
        scored["meta_probability_calibration_margin"] = float(
            self.probability_calibrator.one_sided_margin
        )
        scored["meta_p_severe_loss"] = np.clip(
            self.severe_calibrator.predict(raw_severe),
            0.001,
            0.999,
        )
        scored["meta_expected_net_return_pct"] = self.return_calibrator.predict(
            np.asarray(raw_return, dtype=float).reshape(-1, 1)
        )
        scored["meta_score"] = (
            scored["meta_expected_net_return_pct"]
            + 1.25 * (scored["meta_p_positive"] - 0.50)
            - 1.50 * scored["meta_p_severe_loss"]
            - 0.50
            * (
                1.0
                - _numeric(scored, "p_round_trip_fill_lower").clip(0.0, 1.0)
            )
        )
        scored["meta_rank_pct"] = scored.groupby(
            ["trade_date", "signal_slot"],
            sort=False,
        )["meta_score"].rank(method="average", pct=True)
        return scored


@dataclass(frozen=True)
class MetaPolicySelection:
    policy: MetaPolicy | None
    design: dict[str, Any]
    confirmation: dict[str, Any]
    tested: int
    design_passed: int
    confirmation_passed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.as_dict() if self.policy else None,
            "design": self.design,
            "confirmation": self.confirmation,
            "search": {
                "tested": self.tested,
                "design_passed": self.design_passed,
                "confirmation_passed": self.confirmation_passed,
            },
        }


def attach_meta_context(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(IDENTITY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"missing identity columns: {missing}")
    result = frame.copy()
    result["trade_date"] = result["trade_date"].astype(str)
    result["signal_slot"] = result["signal_slot"].astype(str)
    result["ts_code"] = result["ts_code"].astype(str)
    for column in BASE_NUMERIC_COLUMNS:
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")

    parsed = result["signal_slot"].str.extract(
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2})"
    )
    result["slot_minute"] = (
        pd.to_numeric(parsed["hour"], errors="coerce") * 60
        + pd.to_numeric(parsed["minute"], errors="coerce")
        - (14 * 60 + 20)
    )
    result["ret_abs_pct"] = result["ret_from_prev_close_pct"].abs()
    result["probability_uncertainty"] = (
        result["p_net_positive"] - result["p_net_positive_lower"]
    )
    result["utility_uncertainty_pct"] = (
        result["expected_utility_pct"]
        - result["expected_utility_lower_pct"]
    )

    keys = ["trade_date", "signal_slot"]
    grouped = result.groupby(keys, sort=False)
    returns = result["ret_from_prev_close_pct"]
    result["context_universe_size"] = grouped["ts_code"].transform("size")
    result["context_return_mean_pct"] = grouped[
        "ret_from_prev_close_pct"
    ].transform("mean")
    result["context_return_median_pct"] = grouped[
        "ret_from_prev_close_pct"
    ].transform("median")
    result["context_return_dispersion_pct"] = grouped[
        "ret_from_prev_close_pct"
    ].transform("std")
    for threshold, suffix in (
        (0.0, "positive"),
        (2.0, "above_2pct"),
        (5.0, "above_5pct"),
        (7.0, "above_7pct"),
    ):
        indicator = returns.gt(threshold).astype(float)
        result[f"context_breadth_{suffix}"] = indicator.groupby(
            [result["trade_date"], result["signal_slot"]],
            sort=False,
        ).transform("mean")
    result["context_probability_mean"] = grouped["p_net_positive"].transform(
        "mean"
    )
    result["context_probability_dispersion"] = grouped[
        "p_net_positive"
    ].transform("std")
    result["context_utility_mean_pct"] = grouped[
        "expected_utility_pct"
    ].transform("mean")

    first_market = (
        result.loc[
            result["signal_slot"].eq("14:20"),
            ["trade_date", "context_return_mean_pct"],
        ]
        .drop_duplicates("trade_date")
        .set_index("trade_date")["context_return_mean_pct"]
    )
    result["context_return_change_from_1420_pct"] = (
        result["context_return_mean_pct"]
        - result["trade_date"].map(first_market)
    )
    result["return_context_relative_pct"] = (
        result["ret_from_prev_close_pct"]
        - result["context_return_mean_pct"]
    )
    dispersion = result["context_return_dispersion_pct"].replace(0.0, np.nan)
    result["return_context_zscore"] = (
        result["return_context_relative_pct"] / dispersion
    ).clip(-10.0, 10.0)

    rank_contracts = {
        "return_rank_pct": ("ret_from_prev_close_pct", True),
        "probability_rank_pct": ("p_net_positive", True),
        "utility_rank_pct": ("expected_utility_pct", True),
        "selection_rank_recomputed_pct": ("selection_score", True),
        "severe_quality_rank_pct": ("p_severe_loss", False),
    }
    for output, (source, ascending) in rank_contracts.items():
        ranked = result[source] if ascending else -result[source]
        result[output] = ranked.groupby(
            [result["trade_date"], result["signal_slot"]],
            sort=False,
        ).rank(method="average", pct=True)
    return result.replace([np.inf, -np.inf], np.nan)


def prune_candidate_universe(
    frame: pd.DataFrame,
    *,
    top_per_score: int = 12,
    require_label: bool = True,
) -> pd.DataFrame:
    if top_per_score < 1:
        raise ValueError("top_per_score must be positive")
    contextual = attach_meta_context(frame)
    execution = _boolean(
        contextual.get(
            "execution_eligible",
            pd.Series(False, index=contextual.index),
        )
    )
    labelled = (
        _boolean(
            contextual.get(
                "label_available",
                pd.Series(True, index=contextual.index),
            )
        )
        if require_label
        else pd.Series(True, index=contextual.index, dtype=bool)
    )
    eligible = contextual.loc[execution & labelled].copy()
    selected_indices: set[int] = set()
    keys = ["trade_date", "signal_slot"]
    for score in PRUNE_SCORE_COLUMNS:
        if score not in eligible:
            continue
        ranked = eligible.dropna(subset=[score]).sort_values(
            [*keys, score, "ts_code"],
            ascending=[True, True, False, True],
            kind="stable",
        )
        selected_indices.update(
            ranked.groupby(keys, sort=False).head(top_per_score).index.tolist()
        )
    if "p_severe_loss" in eligible:
        safest = eligible.dropna(subset=["p_severe_loss"]).sort_values(
            [*keys, "p_severe_loss", "ts_code"],
            ascending=[True, True, True, True],
            kind="stable",
        )
        selected_indices.update(
            safest.groupby(keys, sort=False).head(top_per_score).index.tolist()
        )
    if not selected_indices:
        return eligible.head(0).reset_index(drop=True)
    return (
        eligible.loc[sorted(selected_indices)]
        .sort_values([*IDENTITY_COLUMNS], kind="stable")
        .reset_index(drop=True)
    )


def fit_meta_alpha(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    random_seed: int,
) -> MetaAlphaBundle:
    if len(train) < 1_000 or len(calibration) < 200:
        raise ValueError("insufficient rows for meta-alpha fit")
    feature_columns = _active_feature_columns(train)
    x_train = meta_feature_matrix(train, feature_columns)
    x_calibration = meta_feature_matrix(calibration, feature_columns)
    y_train = _numeric(train, "target_net_positive").fillna(0).astype(int)
    y_calibration = (
        _numeric(calibration, "target_net_positive").fillna(0).astype(int)
    )
    train_weight = _day_equal_weights(train)
    calibration_weight = _day_equal_weights(calibration)

    tree = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=140,
        max_leaf_nodes=15,
        min_samples_leaf=120,
        l2_regularization=8.0,
        random_state=random_seed,
    )
    tree.fit(x_train, y_train, sample_weight=train_weight)
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
                    C=0.15,
                    max_iter=2_000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    linear.fit(x_train, y_train, model__sample_weight=train_weight)
    raw_calibration = _blended_probability(tree, linear, x_calibration)
    probability_calibrator = ProbabilityCalibrator(method="platt").fit(
        raw_calibration,
        y_calibration.to_numpy(),
        calibration_weight,
        dates=calibration["trade_date"].astype(str).to_numpy(),
        margin_seed=random_seed + 10,
    )

    severe_train = _severe_target(train)
    severe_calibration = _severe_target(calibration)
    severe_tree = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=11,
        min_samples_leaf=150,
        l2_regularization=10.0,
        random_state=random_seed + 1,
    )
    severe_tree.fit(
        x_train,
        severe_train,
        sample_weight=train_weight,
    )
    raw_severe_calibration = severe_tree.predict_proba(x_calibration)[:, 1]
    severe_calibrator = ProbabilityCalibrator().fit(
        raw_severe_calibration,
        severe_calibration,
        calibration_weight,
    )

    return_train = _numeric(train, "net_return_pct").clip(-10.0, 15.0)
    return_calibration = _numeric(
        calibration,
        "net_return_pct",
    ).clip(-10.0, 15.0)
    return_tree = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.04,
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=120,
        l2_regularization=10.0,
        random_state=random_seed + 2,
    )
    return_tree.fit(
        x_train,
        return_train,
        sample_weight=train_weight,
    )
    raw_return_calibration = return_tree.predict(x_calibration)
    return_calibrator = Ridge(alpha=25.0)
    return_calibrator.fit(
        raw_return_calibration.reshape(-1, 1),
        return_calibration,
        sample_weight=calibration_weight,
    )
    return MetaAlphaBundle(
        probability_tree=tree,
        probability_linear=linear,
        severe_tree=severe_tree,
        return_tree=return_tree,
        probability_calibrator=probability_calibrator,
        severe_calibrator=severe_calibrator,
        return_calibrator=return_calibrator,
        feature_columns=feature_columns,
    )


def meta_policy_grid() -> tuple[MetaPolicy, ...]:
    policies = []
    for values in product(
        (0.46, 0.50, 0.54),
        (0.00, 0.10, 0.20),
        (0.25, 0.35),
        (0.90, 0.95),
        (0.95, 0.98),
        (1, 2, 3),
        ("all", "early", "late"),
    ):
        policies.append(MetaPolicy(*values))
    return tuple(policies)


def apply_meta_policy(
    frame: pd.DataFrame,
    policy: MetaPolicy,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = (
        _numeric(frame, "meta_p_positive").ge(policy.probability_min)
        & _numeric(frame, "meta_expected_net_return_pct").ge(
            policy.expected_return_min_pct
        )
        & _numeric(frame, "meta_p_severe_loss").le(
            policy.severe_loss_max
        )
        & _numeric(frame, "p_round_trip_fill_lower").ge(
            policy.round_trip_fill_min
        )
        & _numeric(frame, "meta_rank_pct").ge(policy.meta_rank_min)
    )
    minute = _slot_minute(frame["signal_slot"])
    if policy.slot_group == "early":
        mask &= minute.le(15)
    elif policy.slot_group == "late":
        mask &= minute.ge(20)
    elif policy.slot_group != "all":
        raise ValueError(f"unknown slot group: {policy.slot_group}")

    qualified = frame.loc[mask].copy()
    if qualified.empty:
        return qualified
    qualified["_slot_minute"] = _slot_minute(qualified["signal_slot"])
    qualified.sort_values(
        ["trade_date", "_slot_minute", "meta_score", "ts_code"],
        ascending=[True, True, False, True],
        kind="stable",
        inplace=True,
    )
    selected: list[int] = []
    for _, day in qualified.groupby("trade_date", sort=False):
        seen: set[str] = set()
        count = 0
        for index, row in day.iterrows():
            code = str(row["ts_code"])
            if code in seen:
                continue
            selected.append(index)
            seen.add(code)
            count += 1
            if count >= policy.max_candidates_per_day:
                break
    return (
        qualified.loc[selected]
        .drop(columns="_slot_minute")
        .reset_index(drop=True)
    )


def select_meta_policy(
    design: pd.DataFrame,
    confirmation: pd.DataFrame,
    config: V3Config,
) -> MetaPolicySelection:
    design_passed: list[tuple[MetaPolicy, dict[str, Any]]] = []
    policies = meta_policy_grid()
    for policy in policies:
        metrics = fast_policy_metrics(
            apply_meta_policy(design, policy),
            config,
        )
        if _passes_design(metrics):
            design_passed.append((policy, metrics))

    confirmed: list[
        tuple[MetaPolicy, dict[str, Any], dict[str, Any]]
    ] = []
    for policy, design_metrics in design_passed:
        confirmation_metrics = fast_policy_metrics(
            apply_meta_policy(confirmation, policy),
            config,
        )
        if _passes_confirmation(confirmation_metrics):
            confirmed.append(
                (policy, design_metrics, confirmation_metrics)
            )
    if not confirmed:
        return MetaPolicySelection(
            policy=None,
            design={},
            confirmation={},
            tested=len(policies),
            design_passed=len(design_passed),
            confirmation_passed=0,
        )
    confirmed.sort(
        key=lambda item: (
            min(
                item[1]["mean_net_return_pct"],
                item[2]["mean_net_return_pct"],
            ),
            min(item[1]["profit_factor"], item[2]["profit_factor"]),
            min(item[1]["win_rate"], item[2]["win_rate"]),
            item[2]["events"],
            item[0].policy_id,
        ),
        reverse=True,
    )
    policy, design_metrics, confirmation_metrics = confirmed[0]
    return MetaPolicySelection(
        policy=policy,
        design=design_metrics,
        confirmation=confirmation_metrics,
        tested=len(policies),
        design_passed=len(design_passed),
        confirmation_passed=len(confirmed),
    )


def fast_policy_metrics(
    frame: pd.DataFrame,
    config: V3Config,
) -> dict[str, Any]:
    clean = frame.copy()
    returns = _numeric(clean, "net_return_pct").dropna()
    clean = clean.loc[returns.index]
    total = int(len(clean))
    wins = int(returns.gt(0).sum())
    lower, upper = wilson_interval(wins, total)
    profits = float(returns.loc[returns > 0].sum())
    losses = float(-returns.loc[returns < 0].sum())
    entry = _boolean(
        clean.get("entry_fillable", pd.Series(False, index=clean.index))
    )
    exit_fill = _boolean(
        clean.get("exit_fillable", pd.Series(False, index=clean.index))
    )
    entered = int(entry.sum())
    stress_extra_pct = (
        50.0 - config.execution.baseline_all_in_cost_bps
    ) / 100.0
    stressed = returns - stress_extra_pct * entry.astype(float)
    return {
        "events": total,
        "trade_days": int(
            clean["trade_date"].astype(str).nunique()
        )
        if total
        else 0,
        "wins": wins,
        "win_rate": wins / total if total else 0.0,
        "win_rate_wilson_lower": lower,
        "win_rate_wilson_upper": upper,
        "mean_net_return_pct": float(returns.mean()) if total else None,
        "median_net_return_pct": float(returns.median()) if total else None,
        "profit_factor": (
            profits / losses
            if losses > 0
            else (float("inf") if profits > 0 else 0.0)
        ),
        "entry_fill_rate": entered / total if total else 0.0,
        "exit_fill_rate_given_entry": (
            int((entry & exit_fill).sum()) / entered if entered else 0.0
        ),
        "stress_50bps_mean_net_return_pct": (
            float(stressed.mean()) if total else None
        ),
    }


def meta_feature_matrix(
    frame: pd.DataFrame,
    columns: tuple[str, ...] = META_FEATURE_COLUMNS,
) -> pd.DataFrame:
    values = frame.reindex(columns=columns).copy()
    for column in columns:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan).astype("float32")


def _active_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    values = meta_feature_matrix(frame)
    active = tuple(
        column
        for column in META_FEATURE_COLUMNS
        if values[column].dropna().nunique() >= 2
    )
    if not active:
        raise ValueError("meta-alpha training window has no varying features")
    return active


def _passes_design(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["events"] >= 40
        and metrics["trade_days"] >= 20
        and metrics["win_rate"] >= 0.50
        and (metrics["mean_net_return_pct"] or -999.0) >= 0.10
        and metrics["profit_factor"] >= 1.10
        and metrics["entry_fill_rate"] >= 0.90
        and metrics["exit_fill_rate_given_entry"] >= 0.95
        and (
            metrics["stress_50bps_mean_net_return_pct"]
            or -999.0
        )
        >= 0.0
    )


def _passes_confirmation(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["events"] >= 20
        and metrics["trade_days"] >= 10
        and metrics["win_rate"] >= 0.48
        and (metrics["mean_net_return_pct"] or -999.0) > 0.0
        and metrics["profit_factor"] > 1.0
        and metrics["entry_fill_rate"] >= 0.88
        and metrics["exit_fill_rate_given_entry"] >= 0.95
        and (
            metrics["stress_50bps_mean_net_return_pct"]
            or -999.0
        )
        >= 0.0
    )


def _blended_probability(
    tree: HistGradientBoostingClassifier,
    linear: Pipeline,
    features: pd.DataFrame,
) -> np.ndarray:
    tree_probability, linear_probability = _probability_components(
        tree,
        linear,
        features,
    )
    return _blend_probability_components(tree_probability, linear_probability)


def _probability_components(
    tree: HistGradientBoostingClassifier,
    linear: Pipeline,
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(tree.predict_proba(features)[:, 1], dtype=float),
        np.asarray(linear.predict_proba(features)[:, 1], dtype=float),
    )


def _blend_probability_components(
    tree_probability: np.ndarray,
    linear_probability: np.ndarray,
) -> np.ndarray:
    return np.clip(
        0.65 * tree_probability + 0.35 * linear_probability,
        0.001,
        0.999,
    )


def _probability_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 0.001, 0.999)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def _clustered_overconfidence_margin(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    dates: np.ndarray | None,
    sample_weight: np.ndarray,
    seed: int,
    samples: int = 2_000,
) -> float:
    values = np.asarray(probability, dtype=float)
    labels = np.asarray(target, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    date_values = (
        np.asarray(dates).astype(str)
        if dates is not None
        else np.arange(len(values)).astype(str)
    )
    calibration = pd.DataFrame(
        {
            "trade_date": date_values,
            "weighted_residual": weights * (values - labels),
            "weight": weights,
        }
    )
    clustered = calibration.groupby("trade_date", sort=True).agg(
        residual_sum=("weighted_residual", "sum"),
        weight_sum=("weight", "sum"),
    )
    day_residual = (
        clustered["residual_sum"]
        / clustered["weight_sum"].replace(0.0, np.nan)
    ).dropna()
    if day_residual.empty:
        return 0.0
    rng = np.random.default_rng(seed)
    choices = rng.integers(
        0,
        len(day_residual),
        size=(samples, len(day_residual)),
        endpoint=False,
    )
    bootstrap_mean = day_residual.to_numpy(dtype=float)[choices].mean(axis=1)
    return float(max(np.quantile(bootstrap_mean, 0.95), 0.0))


def _day_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    keys = frame["trade_date"].astype(str) + "|" + frame["signal_slot"].astype(str)
    counts = keys.groupby(keys, sort=False).transform("size").astype(float)
    weights = 1.0 / counts.clip(lower=1.0)
    return (weights / weights.mean()).to_numpy(dtype=float)


def _severe_target(frame: pd.DataFrame) -> np.ndarray:
    if "target_severe_loss" in frame:
        values = pd.to_numeric(
            frame["target_severe_loss"],
            errors="coerce",
        )
        if values.notna().any():
            return values.fillna(0).astype(int).to_numpy()
    return (
        _numeric(frame, "net_return_pct").le(-2.0).astype(int).to_numpy()
    )


def _slot_minute(values: pd.Series) -> pd.Series:
    parsed = values.astype(str).str.extract(
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2})"
    )
    return (
        pd.to_numeric(parsed["hour"], errors="coerce") * 60
        + pd.to_numeric(parsed["minute"], errors="coerce")
        - (14 * 60 + 20)
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean(values: pd.Series | bool) -> pd.Series:
    if isinstance(values, bool):
        return pd.Series(dtype=bool)
    if values.dtype == bool:
        return values.fillna(False)
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y", "qualified", "pass"})
    )
