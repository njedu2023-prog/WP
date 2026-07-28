from __future__ import annotations

import gc
from dataclasses import asdict, dataclass, field
from collections.abc import Collection
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from .contracts import V3Config
from .dataset import first_crossing_candidates
from .diagnostics import build_prediction_diagnostics
from .model import ModelBundle, predict_bundle, train_bundle
from .policy import apply_nested_oos_policies
from .statistics import day_clustered_intervals, wilson_interval


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_days: int
    test_days: int


@dataclass(frozen=True)
class BacktestResult:
    folds: list[WalkForwardFold]
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    candidates: pd.DataFrame
    policy_audit: list[dict[str, Any]] = field(default_factory=list)
    final_policy: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "folds": [asdict(fold) for fold in self.folds],
            "metrics": self.metrics,
            "policy_audit": self.policy_audit,
            "final_policy": self.final_policy,
        }


def walk_forward_backtest(
    panel: pd.DataFrame,
    config: V3Config,
    *,
    max_folds: int | None = None,
    fold_numbers: Collection[int] | None = None,
    calendar_dates: Collection[str] | None = None,
    evaluate: bool = True,
) -> BacktestResult:
    ordered = panel
    row_dates = ordered["trade_date"].astype(str)
    dates = np.array(
        sorted(
            set(str(value) for value in calendar_dates)
            if calendar_dates is not None
            else row_dates.unique()
        )
    )
    test_starts = _walk_forward_test_starts(dates, config)
    schedule = list(enumerate(test_starts, start=1))
    total_folds = len(schedule)
    if max_folds is not None and fold_numbers is not None:
        raise ValueError("max_folds and fold_numbers are mutually exclusive")
    if max_folds is not None:
        if max_folds < 1:
            raise ValueError("max_folds must be positive")
        schedule = schedule[-max_folds:]
    if fold_numbers is not None:
        selected = {int(number) for number in fold_numbers}
        invalid = sorted(number for number in selected if not 1 <= number <= total_folds)
        if invalid:
            raise ValueError(
                f"fold_numbers contains invalid folds {invalid}; "
                f"valid range is 1..{total_folds}"
            )
        schedule = [
            (fold_number, test_start_index)
            for fold_number, test_start_index in schedule
            if fold_number in selected
        ]
        if not schedule:
            raise ValueError("fold_numbers selected no walk-forward folds")

    fold_rows: list[pd.DataFrame] = []
    folds: list[WalkForwardFold] = []
    for fold_number, test_start_index in schedule:
        test_dates = dates[test_start_index : test_start_index + config.model.test_days]
        if len(test_dates) == 0:
            continue
        train_dates = dates[: test_start_index - config.model.purge_days]
        retained_train_days = (
            max(config.model.ensemble_windows_days)
            + config.model.calibration_days
            + config.model.purge_days
        )
        model_train_dates = train_dates[-retained_train_days:]
        training = ordered.loc[row_dates.isin(model_train_dates)]
        testing = ordered.loc[row_dates.isin(test_dates)]
        print(
            f"[wp-v6] walk-forward fold {fold_number}/{total_folds} "
            f"train={model_train_dates[0]}..{model_train_dates[-1]} "
            f"test={test_dates[0]}..{test_dates[-1]}",
            flush=True,
        )
        bundle = train_bundle(
            training,
            config,
            allow_below_minimum=False,
            model_version=f"wpv6-wf-{test_dates[0]}",
        )
        prediction = predict_bundle(bundle, testing)
        prediction["fold"] = fold_number
        prediction["test_start"] = str(test_dates[0])
        prediction["test_end"] = str(test_dates[-1])
        fold_rows.append(prediction)
        print(
            f"[wp-v6] completed fold {fold_number}/{total_folds} "
            f"prediction_rows={len(prediction):,}",
            flush=True,
        )
        folds.append(
            WalkForwardFold(
                fold=fold_number,
                train_start=bundle.train_start,
                train_end=bundle.train_end,
                test_start=str(test_dates[0]),
                test_end=str(test_dates[-1]),
                train_days=int(len(model_train_dates)),
                test_days=int(len(test_dates)),
            )
        )
        del bundle
        del testing
        del training
        gc.collect()

    predictions = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()
    policy_audit: list[dict[str, Any]] = []
    final_policy: dict[str, Any] = {}
    if evaluate:
        predictions, policy_audit, final_selection = apply_nested_oos_policies(
            predictions,
            config,
        )
        final_policy = final_selection.as_dict()
        all_oos_predictions = predictions
        predictions = evaluation_window(all_oos_predictions, config)
        candidates = first_crossing_candidates(predictions, config)
        metrics = evaluate_predictions(predictions, candidates, config)
        metrics["evaluation_contract"] = evaluation_contract_summary(
            all_oos_predictions,
            predictions,
            config,
        )
        metrics["nested_policy"] = {
            "folds": policy_audit,
            "final": final_policy,
        }
    else:
        candidates = pd.DataFrame()
        metrics = {}
    return BacktestResult(
        folds=folds,
        metrics=metrics,
        predictions=predictions,
        candidates=candidates,
        policy_audit=policy_audit,
        final_policy=final_policy,
    )


def walk_forward_fold_count(panel: pd.DataFrame, config: V3Config) -> int:
    dates = np.array(sorted(panel["trade_date"].astype(str).unique()))
    return len(_walk_forward_test_starts(dates, config))


def evaluation_window(
    predictions: pd.DataFrame,
    config: V3Config,
) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    dates = predictions["trade_date"].astype(str)
    selected = predictions.loc[
        dates.between(
            config.history.evaluation_start_date,
            config.history.evaluation_end_date,
            inclusive="both",
        )
    ].copy()
    if selected.empty:
        raise RuntimeError(
            "walk-forward predictions do not cover the declared evaluation window "
            f"{config.history.evaluation_start_date}-"
            f"{config.history.evaluation_end_date}"
        )
    observed_start = str(selected["trade_date"].astype(str).min())
    observed_end = str(selected["trade_date"].astype(str).max())
    if (
        observed_start != config.history.evaluation_start_date
        or observed_end != config.history.evaluation_end_date
    ):
        raise RuntimeError(
            "walk-forward predictions cover "
            f"{observed_start}-{observed_end}, expected the complete boundary "
            f"{config.history.evaluation_start_date}-"
            f"{config.history.evaluation_end_date}"
        )
    return selected.reset_index(drop=True)


def evaluation_contract_summary(
    all_oos_predictions: pd.DataFrame,
    evaluation_predictions: pd.DataFrame,
    config: V3Config,
) -> dict[str, Any]:
    evaluation_dates = evaluation_predictions["trade_date"].astype(str)
    all_dates = all_oos_predictions["trade_date"].astype(str)
    return {
        "evaluation_start_date": config.history.evaluation_start_date,
        "evaluation_end_date": config.history.evaluation_end_date,
        "observed_evaluation_start_date": str(evaluation_dates.min()),
        "observed_evaluation_end_date": str(evaluation_dates.max()),
        "evaluation_trade_days": int(evaluation_dates.nunique()),
        "evaluation_slot_rows": int(len(evaluation_predictions)),
        "prior_oos_policy_warmup_trade_days": int(
            all_dates.loc[
                all_dates.lt(config.history.evaluation_start_date)
            ].nunique()
        ),
        "prior_oos_policy_warmup_slot_rows": int(
            all_dates.lt(config.history.evaluation_start_date).sum()
        ),
        "all_oos_trade_days": int(all_dates.nunique()),
        "all_oos_slot_rows": int(len(all_oos_predictions)),
    }


def walk_forward_fold_dates(
    calendar_dates: Collection[str],
    config: V3Config,
    fold_number: int,
) -> tuple[np.ndarray, np.ndarray]:
    dates = np.array(sorted(set(str(value) for value in calendar_dates)))
    starts = _walk_forward_test_starts(dates, config)
    if not 1 <= int(fold_number) <= len(starts):
        raise ValueError(
            f"fold_number must be in 1..{len(starts)}; received {fold_number}"
        )
    test_start = starts[int(fold_number) - 1]
    test_dates = dates[test_start : test_start + config.model.test_days]
    train_dates = dates[: test_start - config.model.purge_days]
    retained_train_days = (
        max(config.model.ensemble_windows_days)
        + config.model.calibration_days
        + config.model.purge_days
    )
    return train_dates[-retained_train_days:], test_dates


def _walk_forward_test_starts(
    dates: np.ndarray,
    config: V3Config,
) -> list[int]:
    start = (
        config.model.minimum_train_days
        + config.model.calibration_days
        + 2 * config.model.purge_days
    )
    if len(dates) < start + config.model.test_days:
        raise ValueError(
            f"walk-forward requires at least {start + config.model.test_days} trade days; "
            f"received {len(dates)}"
        )
    return list(range(start, len(dates), config.model.test_days))


def evaluate_predictions(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    config: V3Config,
) -> dict[str, Any]:
    labelled = predictions.loc[
        pd.to_numeric(predictions.get("target_net_positive"), errors="coerce").notna()
    ].copy()
    target = pd.to_numeric(labelled.get("target_net_positive"), errors="coerce").astype(int)
    probability = pd.to_numeric(labelled.get("p_net_positive"), errors="coerce").clip(1e-6, 1 - 1e-6)

    candidate_returns = pd.to_numeric(candidates.get("net_return_pct"), errors="coerce").dropna()
    wins = int(candidate_returns.gt(0).sum())
    total = int(len(candidate_returns))
    win_rate = wins / total if total else 0.0
    lower, upper = wilson_interval(wins, total)
    clustered = day_clustered_intervals(candidates)
    profits = float(candidate_returns[candidate_returns > 0].sum())
    losses = float(-candidate_returns[candidate_returns < 0].sum())
    profit_factor = profits / losses if losses > 0 else (float("inf") if profits > 0 else 0.0)
    entry_fillable = candidates.get(
        "entry_fillable",
        pd.Series(False, index=candidates.index),
    ).fillna(False).astype(bool)
    exit_fillable = candidates.get(
        "exit_fillable",
        pd.Series(False, index=candidates.index),
    ).fillna(False).astype(bool)
    entry_fill_count = int(entry_fillable.sum())
    exit_fill_count = int((entry_fillable & exit_fillable).sum())
    entry_fill_rate = entry_fill_count / total if total else 0.0
    exit_fill_rate = (
        exit_fill_count / entry_fill_count if entry_fill_count else 0.0
    )

    daily = (
        candidates.assign(
            net_return_pct=pd.to_numeric(candidates.get("net_return_pct"), errors="coerce")
        )
        .dropna(subset=["net_return_pct"])
        .groupby("trade_date", sort=True)["net_return_pct"]
        .mean()
    )
    cumulative = (1.0 + daily / 100.0).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0

    stress: dict[str, dict[str, float | int]] = {}
    for cost_bps in config.execution.stress_cost_bps:
        extra_cost_pct = (
            cost_bps - config.execution.baseline_all_in_cost_bps
        ) / 100.0
        stressed = candidate_returns - extra_cost_pct
        stress[f"{int(cost_bps)}bps"] = {
            "mean_net_return_pct": _finite_or_none(stressed.mean()),
            "win_rate": _finite_or_none(stressed.gt(0).mean()),
            "positive_total_return": bool(stressed.sum() > 0) if len(stressed) else False,
            "candidates": int(len(stressed)),
        }

    metrics: dict[str, Any] = {
        "slot_rows": int(len(predictions)),
        "labelled_slot_rows": int(len(labelled)),
        "candidate_events": total,
        "candidate_days": int(candidates.get("trade_date", pd.Series(dtype=str)).nunique()),
        "entry_fillable_events": entry_fill_count,
        "entry_fill_rate": entry_fill_rate,
        "exit_fillable_events": exit_fill_count,
        "exit_fill_rate": exit_fill_rate,
        "round_trip_fill_rate": exit_fill_count / total if total else 0.0,
        "win_count": wins,
        "win_rate": win_rate,
        "win_rate_wilson_lower": lower,
        "win_rate_wilson_upper": upper,
        "win_rate_day_clustered_lower": clustered.win_rate_lower,
        "win_rate_day_clustered_upper": clustered.win_rate_upper,
        "mean_net_return_pct": _finite_or_none(candidate_returns.mean()),
        "mean_net_return_day_clustered_lower_pct": _finite_or_none(
            clustered.mean_return_lower_pct
        ),
        "mean_net_return_day_clustered_upper_pct": _finite_or_none(
            clustered.mean_return_upper_pct
        ),
        "median_net_return_pct": _finite_or_none(candidate_returns.median()),
        "net_return_q10_pct": _finite_or_none(candidate_returns.quantile(0.10)),
        "net_return_q90_pct": _finite_or_none(candidate_returns.quantile(0.90)),
        "profit_factor": _finite_or_none(profit_factor),
        "maximum_day_equal_weight_drawdown_pct": _finite_or_none(drawdown.min() * 100.0),
        "day_equal_weight_mean_net_return_pct": _finite_or_none(daily.mean()),
        "day_equal_weight_win_rate": _finite_or_none(daily.gt(0).mean()),
        "brier_score": _finite_or_none(
            brier_score_loss(target, probability) if len(target) else np.nan
        ),
        "log_loss": _finite_or_none(
            log_loss(target, probability, labels=[0, 1]) if len(target) else np.nan
        ),
        "ece": _finite_or_none(expected_calibration_error(target, probability)),
        "stress": stress,
        "pbo_estimate": _finite_or_none(estimate_policy_pbo(predictions, config)),
        "benchmarks": _benchmark_metrics(predictions, config),
        "diagnostics": build_prediction_diagnostics(predictions, config),
    }
    metrics["backtest_gate"] = evaluate_backtest_gate(metrics, config)
    return metrics


def evaluate_backtest_gate(metrics: dict[str, Any], config: V3Config) -> dict[str, Any]:
    promotion = config.promotion
    stress_50 = metrics.get("stress", {}).get("50bps", {})
    checks = {
        "minimum_oos_candidates": metrics.get("candidate_events", 0)
        >= promotion.minimum_oos_candidates,
        "minimum_oos_win_rate": _number_or_default(metrics.get("win_rate"), 0.0)
        >= promotion.minimum_oos_win_rate,
        "minimum_oos_win_rate_lower": _number_or_default(
            metrics.get("win_rate_wilson_lower"),
            0.0,
        )
        >= promotion.minimum_oos_win_rate_lower,
        "minimum_clustered_win_rate_lower": _number_or_default(
            metrics.get("win_rate_day_clustered_lower"),
            0.0,
        )
        >= promotion.minimum_clustered_win_rate_lower,
        "minimum_mean_net_return": _number_or_default(
            metrics.get("mean_net_return_pct"),
            -999.0,
        )
        >= promotion.minimum_mean_net_return_pct,
        "minimum_clustered_mean_return_lower": _number_or_default(
            metrics.get("mean_net_return_day_clustered_lower_pct"),
            -999.0,
        )
        >= promotion.minimum_clustered_mean_return_lower_pct,
        "minimum_median_net_return": _number_or_default(
            metrics.get("median_net_return_pct"),
            -999.0,
        )
        >= promotion.minimum_median_net_return_pct,
        "minimum_profit_factor": _number_or_default(
            metrics.get("profit_factor"),
            0.0,
        )
        >= promotion.minimum_profit_factor,
        "minimum_entry_fill_rate": _number_or_default(
            metrics.get("entry_fill_rate"),
            0.0,
        )
        >= promotion.minimum_entry_fill_rate,
        "minimum_exit_fill_rate": _number_or_default(
            metrics.get("exit_fill_rate"),
            0.0,
        )
        >= promotion.minimum_exit_fill_rate,
        "maximum_ece": _number_or_default(metrics.get("ece"), 999.0)
        <= promotion.maximum_ece,
        "stress_50bps_nonnegative": (
            not promotion.require_50bps_stress_nonnegative
            or bool(stress_50.get("positive_total_return", False))
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def expected_calibration_error(
    target: pd.Series | np.ndarray,
    probability: pd.Series | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    y = np.asarray(target, dtype=float)
    p = np.asarray(probability, dtype=float)
    if len(y) == 0:
        return float("nan")
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        left, right = boundaries[index], boundaries[index + 1]
        mask = (p >= left) & (p < right if index < bins - 1 else p <= right)
        if not np.any(mask):
            continue
        total += float(np.mean(mask)) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return total


def estimate_policy_pbo(predictions: pd.DataFrame, config: V3Config) -> float:
    """Approximate policy-selection overfit across chronological fold pairs."""
    if "fold" not in predictions or predictions["fold"].nunique() < 4:
        return float("nan")
    thresholds = np.array(config.model.probability_grid, dtype=float)
    fold_ids = sorted(predictions["fold"].dropna().unique())
    overfit = 0
    trials = 0
    for split in range(2, len(fold_ids)):
        in_sample = predictions[predictions["fold"].isin(fold_ids[:split])]
        out_sample = predictions[predictions["fold"].isin(fold_ids[split:])]
        if out_sample.empty:
            continue
        in_scores = np.array([_policy_score(in_sample, threshold, config) for threshold in thresholds])
        out_scores = np.array([_policy_score(out_sample, threshold, config) for threshold in thresholds])
        best = int(np.nanargmax(in_scores))
        if np.isfinite(out_scores[best]):
            overfit += int(out_scores[best] < np.nanmedian(out_scores))
            trials += 1
    return overfit / trials if trials else float("nan")


def _policy_score(frame: pd.DataFrame, threshold: float, config: V3Config) -> float:
    candidate = frame.copy()
    candidate["variant_pass"] = (
        pd.to_numeric(candidate["p_net_positive"], errors="coerce").ge(threshold)
        & pd.to_numeric(candidate["p_market_positive"], errors="coerce").ge(0.45)
        & pd.to_numeric(candidate["p_cross_section_top"], errors="coerce").ge(0.45)
        & pd.to_numeric(candidate["p_severe_loss"], errors="coerce").le(0.45)
        & pd.to_numeric(candidate["expected_net_return_pct"], errors="coerce").ge(
            -0.25
        )
        & pd.to_numeric(candidate["selection_rank_pct"], errors="coerce").ge(
            0.99
        )
        & candidate["execution_eligible"].fillna(False)
    )
    selected = first_crossing_candidates(candidate, config, status_column="variant_pass")
    returns = pd.to_numeric(selected.get("net_return_pct"), errors="coerce").dropna()
    if len(returns) < 5:
        return -999.0
    downside = abs(float(returns[returns < 0].mean())) if (returns < 0).any() else 0.01
    return float(returns.mean()) / max(downside, 0.01)


def _benchmark_metrics(
    predictions: pd.DataFrame,
    config: V3Config,
) -> dict[str, dict[str, Any]]:
    if predictions.empty:
        return {}
    eligible = predictions.loc[
        predictions.get(
            "execution_eligible",
            pd.Series(False, index=predictions.index),
        ).fillna(False)
        & predictions["signal_slot"].astype(str).eq("14:50")
    ].copy()
    legacy = eligible.loc[
        pd.to_numeric(eligible.get("ret_from_prev_close_pct"), errors="coerce").between(
            8.0,
            12.0,
            inclusive="both",
        )
    ].copy()
    return {
        "all_executable_at_1450": _return_summary(eligible),
        "retired_8_to_12_pct_rule_at_1450": _return_summary(legacy),
    }


def _return_summary(frame: pd.DataFrame) -> dict[str, Any]:
    returns = pd.to_numeric(frame.get("net_return_pct"), errors="coerce").dropna()
    wins = int(returns.gt(0).sum())
    lower, _ = wilson_interval(wins, len(returns))
    return {
        "events": int(len(returns)),
        "win_rate": _finite_or_none(returns.gt(0).mean()),
        "win_rate_wilson_lower": lower,
        "mean_net_return_pct": _finite_or_none(returns.mean()),
        "median_net_return_pct": _finite_or_none(returns.median()),
    }


def _finite_or_none(value: Any) -> Any:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _number_or_default(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default
