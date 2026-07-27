from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from .contracts import V3Config
from .dataset import first_crossing_candidates
from .diagnostics import build_prediction_diagnostics
from .model import ModelBundle, predict_bundle, train_bundle
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

    def summary(self) -> dict[str, Any]:
        return {
            "folds": [asdict(fold) for fold in self.folds],
            "metrics": self.metrics,
        }


def walk_forward_backtest(
    panel: pd.DataFrame,
    config: V3Config,
    *,
    max_folds: int | None = None,
) -> BacktestResult:
    ordered = panel.copy()
    ordered["trade_date"] = ordered["trade_date"].astype(str)
    ordered = ordered.sort_values(["trade_date", "signal_slot", "ts_code"], kind="stable")
    dates = np.array(sorted(ordered["trade_date"].unique()))
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

    test_starts = list(range(start, len(dates), config.model.test_days))
    if max_folds is not None:
        test_starts = test_starts[-max_folds:]

    fold_rows: list[pd.DataFrame] = []
    folds: list[WalkForwardFold] = []
    for fold_number, test_start_index in enumerate(test_starts, start=1):
        test_dates = dates[test_start_index : test_start_index + config.model.test_days]
        if len(test_dates) == 0:
            continue
        train_dates = dates[: test_start_index - config.model.purge_days]
        training = ordered.loc[ordered["trade_date"].isin(train_dates)]
        testing = ordered.loc[ordered["trade_date"].isin(test_dates)]
        bundle = train_bundle(
            training,
            config,
            allow_below_minimum=False,
            model_version=f"wpv3-wf-{test_dates[0]}",
        )
        prediction = predict_bundle(bundle, testing)
        prediction["fold"] = fold_number
        prediction["test_start"] = str(test_dates[0])
        prediction["test_end"] = str(test_dates[-1])
        fold_rows.append(prediction)
        folds.append(
            WalkForwardFold(
                fold=fold_number,
                train_start=bundle.train_start,
                train_end=bundle.calibration_end,
                test_start=str(test_dates[0]),
                test_end=str(test_dates[-1]),
                train_days=int(len(train_dates)),
                test_days=int(len(test_dates)),
            )
        )

    predictions = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()
    candidates = first_crossing_candidates(predictions, config)
    metrics = evaluate_predictions(predictions, candidates, config)
    return BacktestResult(
        folds=folds,
        metrics=metrics,
        predictions=predictions,
        candidates=candidates,
    )


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
        "minimum_oos_win_rate": (metrics.get("win_rate") or 0)
        >= promotion.minimum_oos_win_rate,
        "minimum_oos_win_rate_lower": (metrics.get("win_rate_wilson_lower") or 0)
        >= promotion.minimum_oos_win_rate_lower,
        "minimum_clustered_win_rate_lower": (
            metrics.get("win_rate_day_clustered_lower") or 0
        )
        >= promotion.minimum_clustered_win_rate_lower,
        "minimum_mean_net_return": (metrics.get("mean_net_return_pct") or -999)
        >= promotion.minimum_mean_net_return_pct,
        "minimum_clustered_mean_return_lower": (
            metrics.get("mean_net_return_day_clustered_lower_pct")
            if metrics.get("mean_net_return_day_clustered_lower_pct") is not None
            else -999
        )
        >= promotion.minimum_clustered_mean_return_lower_pct,
        "minimum_median_net_return": (metrics.get("median_net_return_pct") or -999)
        >= promotion.minimum_median_net_return_pct,
        "minimum_profit_factor": (metrics.get("profit_factor") or 0)
        >= promotion.minimum_profit_factor,
        "maximum_ece": (metrics.get("ece") if metrics.get("ece") is not None else 999)
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
    thresholds = np.array([0.56, 0.58, 0.60, 0.62, 0.65, 0.68])
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
        & pd.to_numeric(candidate["p_net_positive_lower"], errors="coerce").ge(
            config.model.probability_lower_threshold
        )
        & pd.to_numeric(candidate["expected_net_return_pct"], errors="coerce").ge(
            config.model.min_expected_net_return_pct
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
        "legacy_8_to_12_pct_at_1450": _return_summary(legacy),
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
