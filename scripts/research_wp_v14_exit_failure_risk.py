from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from wp.v3.contracts import load_v3_config
from wp.v3.exit_risk import exit_failure_target, fit_exit_failure_risk
from wp.v3.features import FEATURE_COLUMNS
from wp.v3.io import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
)
from wp.v3.meta_alpha import IDENTITY_COLUMNS
from wp.v3.overlay import performance_summary


RISK_TRAIN_DAYS = 126
RISK_CALIBRATION_DAYS = 42
PURGE_DAYS = 2
ABSOLUTE_RISK_THRESHOLDS = (0.005, 0.010, 0.020, 0.030, 0.050)
SAFE_RANK_THRESHOLDS = (0.10, 0.20, 0.50)
MINIMUM_DIRECTION_EVENTS = 10
MINIMUM_DIRECTION_DAYS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a causal T-day feature model for T+1 close-auction exit "
            "failure and audit fixed risk overlays on immutable V10 candidates."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--v11-source-dir", required=True)
    parser.add_argument("--v10-source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_v3_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frontier, source_audit = load_v11_frontier(args.v11_source_dir)
    identities = load_v10_identities(args.v10_source_dir)
    frontier = attach_original_features(
        frontier,
        args.panel_dir,
        start_date=config.history.evaluation_start_date,
        end_date=config.history.evaluation_end_date,
    )
    frontier = materialize_close_baseline(frontier)
    frontier = mark_exact_v10(frontier, identities)
    exact_all = frontier.loc[frontier["is_exact_v10"]].copy()
    if len(exact_all) != len(identities):
        raise RuntimeError(
            f"matched {len(exact_all)} of {len(identities)} V10 identities"
        )

    scored_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    relevant_folds = sorted(
        pd.to_numeric(exact_all["fold"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    numeric_fold = pd.to_numeric(frontier["fold"], errors="coerce")
    for fold in relevant_folds:
        test = frontier.loc[numeric_fold.eq(fold)].copy()
        test_start = str(test["trade_date"].min())
        history = frontier.loc[
            ~numeric_fold.eq(fold)
            & frontier["trade_date"].astype(str).lt(test_start)
        ].copy()
        segments = rolling_segments(
            sorted(history["trade_date"].astype(str).unique())
        )
        base_row: dict[str, Any] = {
            "fold": fold,
            "test_start": test_start,
            "test_end": str(test["trade_date"].max()),
            "test_rows": int(len(test)),
            "exact_v10_test_rows": int(test["is_exact_v10"].sum()),
        }
        if segments is None:
            fold_rows.append(
                {
                    **base_row,
                    "scored": False,
                    "reason": "insufficient_prior_oos_history",
                }
            )
            continue
        train_dates, calibration_dates = segments
        train = eligible_training_rows(
            history.loc[
                history["trade_date"].astype(str).isin(train_dates)
            ]
        )
        calibration = eligible_training_rows(
            history.loc[
                history["trade_date"].astype(str).isin(calibration_dates)
            ]
        )
        print(
            f"[wp-v14] fold={fold} "
            f"train={train_dates[0]}..{train_dates[-1]} rows={len(train):,} "
            f"failures={int(exit_failure_target(train).sum()):,} "
            f"calibration={calibration_dates[0]}.."
            f"{calibration_dates[-1]} rows={len(calibration):,} "
            f"failures={int(exit_failure_target(calibration).sum()):,} "
            f"test={test_start}..{str(test['trade_date'].max())} "
            f"rows={len(test):,}",
            flush=True,
        )
        try:
            bundle = fit_exit_failure_risk(
                train,
                calibration,
                random_seed=config.model.random_seed + fold * 139,
            )
        except ValueError as error:
            fold_rows.append(
                {
                    **base_row,
                    "scored": False,
                    "reason": str(error),
                    "train_rows": int(len(train)),
                    "calibration_rows": int(len(calibration)),
                }
            )
            continue
        scored = bundle.predict(test)
        scored["risk_fold"] = fold
        scored_frames.append(scored)
        diagnostic = classifier_diagnostics(
            eligible_training_rows(scored)
        )
        fold_rows.append(
            {
                **base_row,
                "scored": True,
                "reason": "causal_model_scored_frozen_outer_fold",
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "train_rows": int(len(train)),
                "train_failures": int(exit_failure_target(train).sum()),
                "calibration_start": calibration_dates[0],
                "calibration_end": calibration_dates[-1],
                "calibration_rows": int(len(calibration)),
                "calibration_failures": int(
                    exit_failure_target(calibration).sum()
                ),
                "failure_class_weight": bundle.failure_weight,
                "model_feature_count": len(bundle.feature_columns),
                "test_diagnostic": diagnostic,
            }
        )
        print(
            f"[wp-v14] fold={fold} exact={int(scored['is_exact_v10'].sum())} "
            f"test_failures={diagnostic['failures']} "
            f"auc={diagnostic['roc_auc']} "
            f"average_precision={diagnostic['average_precision']}",
            flush=True,
        )

    scored_frontier = (
        pd.concat(scored_frames, ignore_index=True)
        if scored_frames
        else frontier.head(0).copy()
    )
    scored_exact = scored_frontier.loc[
        scored_frontier["is_exact_v10"]
    ].copy()
    policy_rows = evaluate_fixed_policies(
        scored_exact,
        config,
    )
    full_baseline = performance_summary(
        exact_all,
        config,
        bootstrap_samples=4_000,
        seed=config.model.random_seed + 14_001,
    )
    best_direction = rank_research_direction(policy_rows)
    summary = {
        "schema_version": "wp_v14_exit_failure_risk_research_1",
        "research_only": True,
        "production_model_changed": False,
        "objective": (
            "remove T-day candidates with high causal probability of being "
            "unable to execute the fixed T+1 close-auction exit"
        ),
        "source": source_audit,
        "evaluation_start": config.history.evaluation_start_date,
        "evaluation_end": config.history.evaluation_end_date,
        "protocol": {
            "risk_train_days": RISK_TRAIN_DAYS,
            "risk_calibration_days": RISK_CALIBRATION_DAYS,
            "purge_days_between_train_calibration_and_test": PURGE_DAYS,
            "model": (
                "70% histogram gradient boosting plus 30% logistic model, "
                "independently isotonic-calibrated"
            ),
            "features": (
                "only the frozen causal T-day FEATURE_COLUMNS available at "
                "the candidate signal"
            ),
            "target": (
                "T+1 close-auction exit non-fill under the immutable V10 "
                "execution contract"
            ),
            "fixed_absolute_risk_thresholds": list(
                ABSOLUTE_RISK_THRESHOLDS
            ),
            "fixed_cross_section_safe_rank_thresholds": list(
                SAFE_RANK_THRESHOLDS
            ),
            "no_post_test_threshold_tuning": True,
            "multiple_testing": (
                "all fixed overlays are diagnostics; none can authorize "
                "production from this historical run"
            ),
            "production_requires_new_150_day_shadow": True,
        },
        "candidate_frontier_rows": int(len(frontier)),
        "exact_v10_rows": int(len(exact_all)),
        "relevant_folds": relevant_folds,
        "folds_scored": int(
            sum(bool(row.get("scored")) for row in fold_rows)
        ),
        "exact_v10_scored_rows": int(len(scored_exact)),
        "exact_v10_unscored_rows": int(len(exact_all) - len(scored_exact)),
        "full_exact_v10_baseline": full_baseline,
        "folds": fold_rows,
        "fixed_policy_diagnostics": policy_rows,
        "best_research_direction": best_direction,
    }

    atomic_write_json(
        output / "wp_v14_exit_failure_risk_summary.json",
        summary,
    )
    atomic_write_csv(
        pd.json_normalize(fold_rows, sep="."),
        output / "wp_v14_exit_failure_risk_folds.csv",
    )
    atomic_write_csv(
        pd.json_normalize(policy_rows, sep="."),
        output / "wp_v14_exit_failure_risk_policy_diagnostics.csv",
    )
    exact_columns = [
        column
        for column in (
            *IDENTITY_COLUMNS,
            "name",
            "fold",
            "net_return_pct",
            "entry_fillable",
            "exit_fillable",
            "p_entry_fill",
            "p_exit_fill_given_entry",
            "p_round_trip_fill_lower",
            "risk_fold",
            "risk_p_exit_failure",
            "risk_p_exit_safe",
            "risk_failure_rank_pct",
        )
        if column in scored_exact
    ]
    atomic_write_csv(
        scored_exact.loc[:, exact_columns],
        output / "wp_v14_exact_v10_risk_scores.csv",
    )
    frontier_columns = [
        column
        for column in (
            *IDENTITY_COLUMNS,
            "name",
            "fold",
            "net_return_pct",
            "entry_fillable",
            "exit_fillable",
            "is_exact_v10",
            "risk_fold",
            "risk_p_exit_failure",
            "risk_p_exit_safe",
            "risk_failure_rank_pct",
        )
        if column in scored_frontier
    ]
    scored_path = atomic_write_parquet(
        scored_frontier.loc[:, frontier_columns],
        output / "wp_v14_frontier_risk_scores.parquet",
    )
    scored_sha = sha256(scored_path)
    summary["frontier_risk_scores_sha256"] = scored_sha
    atomic_write_json(
        output / "wp_v14_exit_failure_risk_summary.json",
        summary,
    )
    marker = {
        "schema_version": summary["schema_version"],
        "source": source_audit,
        "candidate_frontier_rows": summary["candidate_frontier_rows"],
        "exact_v10_rows": summary["exact_v10_rows"],
        "exact_v10_scored_rows": summary["exact_v10_scored_rows"],
        "exact_v10_unscored_rows": summary["exact_v10_unscored_rows"],
        "full_exact_v10_baseline": full_baseline,
        "folds": [
            {
                "fold": row["fold"],
                "scored": row["scored"],
                "reason": row["reason"],
                "exact_v10_test_rows": row["exact_v10_test_rows"],
                "test_diagnostic": row.get("test_diagnostic"),
            }
            for row in fold_rows
        ],
        "fixed_policy_diagnostics": policy_rows,
        "best_research_direction": best_direction,
        "frontier_risk_scores_sha256": scored_sha,
    }
    print(
        "WP_V14_EXIT_FAILURE_RISK_RESULT="
        + json.dumps(
            marker,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
            default=json_default,
        ),
        flush=True,
    )
    return 0


def load_v11_frontier(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(path)
    frontier_paths = sorted(root.rglob("wp_v11_exit_frontier.parquet"))
    summary_paths = sorted(root.rglob("wp_v11_exit_summary.json"))
    if len(frontier_paths) != 1 or len(summary_paths) != 1:
        raise FileNotFoundError(
            "V11 source must contain exactly one frontier and one summary"
        )
    frontier_path = frontier_paths[0]
    summary = json.loads(summary_paths[0].read_text(encoding="utf-8"))
    expected_sha = str(summary.get("candidate_frontier_sha256") or "")
    actual_sha = sha256(frontier_path)
    if not expected_sha or actual_sha != expected_sha:
        raise RuntimeError(
            f"V11 frontier digest mismatch: {actual_sha} != {expected_sha}"
        )
    frame = pd.read_parquet(frontier_path)
    missing = sorted(
        {
            *IDENTITY_COLUMNS,
            "fold",
            "entry_fillable",
            "net_t1_close_auction_pct",
            "exit_t1_close_auction_fillable",
        }
        - set(frame.columns)
    )
    if missing:
        raise ValueError(f"V11 frontier missing columns: {missing}")
    return frame, {
        "v11_schema_version": summary.get("schema_version"),
        "v11_frontier_sha256": actual_sha,
        "v11_frontier_rows": int(len(frame)),
    }


def load_v10_identities(path: str | Path) -> pd.DataFrame:
    matches = sorted(Path(path).rglob("wp_v10_meta_oos_candidates.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            "V10 source must contain exactly one selected-candidate CSV"
        )
    frame = pd.read_csv(matches[0], dtype=str)
    missing = sorted(set(IDENTITY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"V10 selected candidates missing {missing}")
    identities = frame.loc[:, IDENTITY_COLUMNS].drop_duplicates()
    if len(identities) != len(frame):
        raise RuntimeError("V10 selected-candidate identities are not unique")
    return identities.reset_index(drop=True)


def attach_original_features(
    frontier: pd.DataFrame,
    panel_dir: str | Path,
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    result = frontier.copy()
    for column in IDENTITY_COLUMNS:
        result[column] = result[column].astype(str)
    missing_features = [
        column for column in FEATURE_COLUMNS if column not in result
    ]
    if not missing_features:
        return result
    panel_columns = list(
        dict.fromkeys([*IDENTITY_COLUMNS, *missing_features])
    )
    requested = result.loc[:, IDENTITY_COLUMNS].drop_duplicates().copy()
    requested["month"] = requested["trade_date"].str[:6]
    paths = {
        path.stem.rsplit("_", 1)[-1]: path
        for path in Path(panel_dir).glob("wp_v3_panel_*.parquet")
        if start_date[:6]
        <= path.stem.rsplit("_", 1)[-1]
        <= end_date[:6]
    }
    feature_frames: list[pd.DataFrame] = []
    for month, month_requested in requested.groupby("month", sort=True):
        path = paths.get(str(month))
        if path is None:
            raise FileNotFoundError(
                f"missing panel feature partition for {month}"
            )
        panel = pd.read_parquet(path, columns=panel_columns)
        for column in IDENTITY_COLUMNS:
            panel[column] = panel[column].astype(str)
        month_identities = month_requested.loc[:, IDENTITY_COLUMNS]
        matched = panel.merge(
            month_identities,
            on=list(IDENTITY_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        if len(matched) != len(month_identities):
            raise RuntimeError(
                f"matched {len(matched)} of {len(month_identities)} "
                f"frontier feature identities for {month}"
            )
        feature_frames.append(matched)
    panel_features = pd.concat(feature_frames, ignore_index=True)
    duplicates = panel_features.duplicated(
        list(IDENTITY_COLUMNS),
        keep=False,
    )
    if duplicates.any():
        raise RuntimeError(
            "matched panel features contain "
            f"{int(duplicates.sum())} duplicate identities"
        )
    result = result.merge(
        panel_features,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    completely_missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in result or result[column].notna().sum() == 0
    ]
    if len(completely_missing) == len(FEATURE_COLUMNS):
        raise RuntimeError("no causal feature values matched the V11 frontier")
    return result


def materialize_close_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["net_return_pct"] = pd.to_numeric(
        result["net_t1_close_auction_pct"],
        errors="coerce",
    )
    result["entry_fillable"] = boolean(result["entry_fillable"])
    result["exit_fillable"] = boolean(
        result["exit_t1_close_auction_fillable"]
    )
    result["label_available"] = result["net_return_pct"].notna()
    result["exit_contract_id"] = "t1_close_auction"
    return result


def mark_exact_v10(
    frontier: pd.DataFrame,
    identities: pd.DataFrame,
) -> pd.DataFrame:
    marker = identities.copy()
    for column in IDENTITY_COLUMNS:
        marker[column] = marker[column].astype(str)
    marker["is_exact_v10"] = True
    result = frontier.merge(
        marker,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    result["is_exact_v10"] = (
        result["is_exact_v10"].fillna(False).astype(bool)
    )
    return result


def rolling_segments(
    prior_dates: list[str],
) -> tuple[list[str], list[str]] | None:
    needed = (
        RISK_TRAIN_DAYS
        + RISK_CALIBRATION_DAYS
        + 2 * PURGE_DAYS
    )
    if len(prior_dates) < needed:
        return None
    selected = prior_dates[-needed:]
    train = selected[:RISK_TRAIN_DAYS]
    calibration_start = RISK_TRAIN_DAYS + PURGE_DAYS
    calibration = selected[
        calibration_start : calibration_start + RISK_CALIBRATION_DAYS
    ]
    if selected[-PURGE_DAYS:] == calibration[-PURGE_DAYS:]:
        raise AssertionError("test purge days leaked into calibration")
    return train, calibration


def eligible_training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        frame["label_available"].fillna(False).astype(bool)
        & boolean(frame["entry_fillable"])
    ].copy()


def classifier_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "events": 0,
            "failures": 0,
            "failure_rate": None,
            "roc_auc": None,
            "average_precision": None,
            "brier_score": None,
        }
    target = exit_failure_target(frame).to_numpy(dtype=int)
    probability = pd.to_numeric(
        frame["risk_p_exit_failure"],
        errors="coerce",
    ).to_numpy(dtype=float)
    valid = np.isfinite(probability)
    target = target[valid]
    probability = probability[valid]
    both_classes = len(np.unique(target)) == 2
    return {
        "events": int(len(target)),
        "failures": int(target.sum()),
        "failure_rate": finite(target.mean()),
        "roc_auc": finite(
            roc_auc_score(target, probability)
            if both_classes
            else np.nan
        ),
        "average_precision": finite(
            average_precision_score(target, probability)
            if both_classes
            else np.nan
        ),
        "brier_score": finite(
            brier_score_loss(target, probability)
            if len(target)
            else np.nan
        ),
    }


def evaluate_fixed_policies(
    scored_exact: pd.DataFrame,
    config: Any,
) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = [
        {
            "policy_id": "scored_baseline",
            "rule": "all causally scored exact V10 candidates",
            "mask": pd.Series(True, index=scored_exact.index),
        }
    ]
    risk = pd.to_numeric(
        scored_exact.get("risk_p_exit_failure"),
        errors="coerce",
    )
    rank = pd.to_numeric(
        scored_exact.get("risk_failure_rank_pct"),
        errors="coerce",
    )
    for threshold in ABSOLUTE_RISK_THRESHOLDS:
        policies.append(
            {
                "policy_id": f"risk_abs_le_{threshold:.3f}",
                "rule": (
                    "risk_p_exit_failure <= "
                    f"{threshold:.3%}"
                ),
                "mask": risk.le(threshold),
            }
        )
    for threshold in SAFE_RANK_THRESHOLDS:
        policies.append(
            {
                "policy_id": f"risk_rank_safest_{int(threshold * 100):02d}pct",
                "rule": (
                    "within-slot exit-failure risk percentile <= "
                    f"{threshold:.0%}"
                ),
                "mask": rank.le(threshold),
            }
        )
    fill_gate = (
        pd.to_numeric(
            scored_exact.get("p_entry_fill"),
            errors="coerce",
        ).ge(0.985)
        & pd.to_numeric(
            scored_exact.get("p_exit_fill_given_entry"),
            errors="coerce",
        ).ge(0.995)
        & pd.to_numeric(
            scored_exact.get("p_round_trip_fill_lower"),
            errors="coerce",
        ).ge(0.985 * 0.995)
    )
    policies.extend(
        [
            {
                "policy_id": "existing_fill_probability_gate",
                "rule": (
                    "p_entry_fill >= 98.5%, p_exit_given_entry >= 99.5%, "
                    "round_trip_lower >= 98.0075%"
                ),
                "mask": fill_gate,
            },
            {
                "policy_id": "hybrid_risk_2pct_and_fill_gate",
                "rule": (
                    "risk_p_exit_failure <= 2% plus existing fixed fill gate"
                ),
                "mask": risk.le(0.020) & fill_gate,
            },
        ]
    )

    baseline_failures = int(
        (~boolean(scored_exact["exit_fillable"])).sum()
    )
    rows: list[dict[str, Any]] = []
    for index, policy in enumerate(policies):
        mask = policy["mask"].fillna(False).astype(bool)
        selected = scored_exact.loc[mask].copy()
        remaining_failures = int(
            (~boolean(selected["exit_fillable"])).sum()
        )
        rows.append(
            {
                "policy_id": policy["policy_id"],
                "rule": policy["rule"],
                "retained_events": int(len(selected)),
                "excluded_events": int(len(scored_exact) - len(selected)),
                "retention_rate": (
                    len(selected) / len(scored_exact)
                    if len(scored_exact)
                    else 0.0
                ),
                "baseline_exit_failures": baseline_failures,
                "remaining_exit_failures": remaining_failures,
                "removed_exit_failures": (
                    baseline_failures - remaining_failures
                ),
                "metrics": performance_summary(
                    selected,
                    config,
                    bootstrap_samples=4_000,
                    seed=config.model.random_seed + 14_100 + index,
                ),
            }
        )
    return rows


def rank_research_direction(
    policy_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for row in policy_rows:
        if row["policy_id"] == "scored_baseline":
            continue
        metrics = row["metrics"]
        stress = metrics.get("stress", {}).get("50bps", {})
        basic_direction = bool(
            int(metrics.get("events", 0) or 0) >= MINIMUM_DIRECTION_EVENTS
            and int(metrics.get("trade_days", 0) or 0)
            >= MINIMUM_DIRECTION_DAYS
            and (metrics.get("mean_net_return_pct") or -999.0) > 0.0
            and (metrics.get("profit_factor") or 0.0) > 1.0
            and bool(stress.get("positive_total_return", False))
        )
        if basic_direction:
            candidates.append(row)
    if not candidates:
        return {
            "status": "no_positive_exit_failure_risk_direction",
            "policy_id": None,
            "production_authorized": False,
            "reason": (
                "no predeclared overlay retained at least 10 events over "
                "5 days with positive mean, profit factor above 1, and "
                "positive total return at 50bp"
            ),
        }
    candidates.sort(
        key=lambda row: (
            row["metrics"]
            .get("stress", {})
            .get("50bps", {})
            .get("mean_net_return_pct")
            or -999.0,
            row["metrics"].get("mean_net_return_pct") or -999.0,
            row["metrics"].get("events") or 0,
        ),
        reverse=True,
    )
    best = candidates[0]
    metrics = best["metrics"]
    confirmed = bool(
        int(metrics.get("events", 0) or 0) >= 250
        and (
            metrics.get("mean_net_return_day_clustered_lower_pct")
            or -999.0
        )
        > 0.0
        and (metrics.get("win_rate_day_clustered_lower") or 0.0) >= 0.52
        and (metrics.get("profit_factor") or 0.0) >= 1.20
        and (metrics.get("exit_fill_rate_given_entry") or 0.0) >= 0.98
    )
    return {
        "status": (
            "positive_and_statistically_confirmed"
            if confirmed
            else "positive_historical_direction_unconfirmed"
        ),
        "policy_id": best["policy_id"],
        "rule": best["rule"],
        "retained_events": best["retained_events"],
        "removed_exit_failures": best["removed_exit_failures"],
        "metrics": metrics,
        "production_authorized": False,
        "reason": (
            "new_150_day_shadow_required"
            if confirmed
            else "historical_sample_below_promotion_standard"
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boolean(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "qualified", "pass"}
    )


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return finite(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


if __name__ == "__main__":
    raise SystemExit(main())
