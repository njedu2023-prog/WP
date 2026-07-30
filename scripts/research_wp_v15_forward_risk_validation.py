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
from wp.v3.forward_risk import (
    DISCOVERY_END_DATE,
    DISCOVERY_FOLD,
    FORWARD_VALIDATION_FOLDS,
    SAFE_RISK_RANK_MAX,
    assert_strictly_forward,
    frozen_meta_policy,
    select_forward_candidates,
)
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
MINIMUM_CONFIRMATION_EVENTS = 250
MINIMUM_CONFIRMATION_DAYS = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the frozen V14 safe-half exit-risk gate on V10 folds "
            "strictly after the discovery fold, without retuning any rule."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--v11-source-dir", required=True)
    parser.add_argument("--v10-source-dir", required=True)
    parser.add_argument("--v14-source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_v3_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frontier, v11_audit = load_v11_frontier(args.v11_source_dir)
    meta_scores, v10_audit = load_v10_scores(args.v10_source_dir)
    v14_audit = load_v14_discovery(args.v14_source_dir)
    frontier = attach_original_features(
        frontier,
        args.panel_dir,
        start_date=config.history.evaluation_start_date,
        end_date=config.history.evaluation_end_date,
    )
    frontier = materialize_close_baseline(frontier)

    fold_rows: list[dict[str, Any]] = []
    baseline_frames: list[pd.DataFrame] = []
    challenger_frames: list[pd.DataFrame] = []
    scored_meta_frames: list[pd.DataFrame] = []
    numeric_fold = pd.to_numeric(frontier["fold"], errors="coerce")
    for fold in FORWARD_VALIDATION_FOLDS:
        test = frontier.loc[numeric_fold.eq(fold)].copy()
        source_meta = meta_scores.loc[
            meta_scores["meta_source_fold"].eq(fold)
        ].copy()
        if test.empty or source_meta.empty:
            fold_rows.append(
                {
                    "fold": fold,
                    "scored": False,
                    "reason": "missing_test_frontier_or_meta_scores",
                    "frontier_rows": int(len(test)),
                    "meta_rows": int(len(source_meta)),
                }
            )
            continue
        assert_strictly_forward(test)
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
            "frontier_rows": int(len(test)),
            "meta_rows": int(len(source_meta)),
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
            f"[wp-v15] fold={fold} "
            f"train={train_dates[0]}..{train_dates[-1]} rows={len(train):,} "
            f"failures={int(exit_failure_target(train).sum()):,} "
            f"calibration={calibration_dates[0]}.."
            f"{calibration_dates[-1]} rows={len(calibration):,} "
            f"failures={int(exit_failure_target(calibration).sum()):,} "
            f"test={test_start}..{str(test['trade_date'].max())} "
            f"frontier={len(test):,} meta={len(source_meta):,}",
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

        scored_test = bundle.predict(test)
        scored_meta = merge_meta_scores(
            scored_test,
            source_meta,
            fold=fold,
        )
        assert_strictly_forward(scored_meta)
        baseline = select_forward_candidates(
            scored_meta,
            apply_exit_risk_gate=False,
        )
        challenger = select_forward_candidates(
            scored_meta,
            apply_exit_risk_gate=True,
        )
        baseline["validation_variant"] = "frozen_v10_policy"
        challenger["validation_variant"] = (
            "frozen_v10_policy_plus_v14_safe_half"
        )
        baseline_frames.append(baseline)
        challenger_frames.append(challenger)

        scored_meta = selection_markers(
            scored_meta,
            baseline,
            challenger,
        )
        scored_meta_frames.append(scored_meta)
        diagnostic = classifier_diagnostics(
            eligible_training_rows(scored_test)
        )
        comparison = candidate_comparison(baseline, challenger)
        fold_rows.append(
            {
                **base_row,
                "scored": True,
                "reason": "strict_forward_frozen_rules",
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
                "risk_model_diagnostic": diagnostic,
                "candidate_comparison": comparison,
                "baseline_metrics": performance_summary(
                    baseline,
                    config,
                    bootstrap_samples=2_000,
                    seed=config.model.random_seed + fold * 17,
                ),
                "challenger_metrics": performance_summary(
                    challenger,
                    config,
                    bootstrap_samples=2_000,
                    seed=config.model.random_seed + fold * 19,
                ),
            }
        )
        print(
            f"[wp-v15] fold={fold} "
            f"baseline={len(baseline)} challenger={len(challenger)} "
            f"baseline_failures={comparison['baseline_exit_failures']} "
            f"challenger_failures={comparison['challenger_exit_failures']}",
            flush=True,
        )

    baseline_all = concat_or_empty(baseline_frames, frontier)
    challenger_all = concat_or_empty(challenger_frames, frontier)
    scored_meta_all = concat_or_empty(scored_meta_frames, frontier)
    if not baseline_all.empty:
        assert_strictly_forward(baseline_all)
    if not challenger_all.empty:
        assert_strictly_forward(challenger_all)
    baseline_metrics = performance_summary(
        baseline_all,
        config,
        bootstrap_samples=4_000,
        seed=config.model.random_seed + 15_001,
    )
    challenger_metrics = performance_summary(
        challenger_all,
        config,
        bootstrap_samples=4_000,
        seed=config.model.random_seed + 15_002,
    )
    aggregate_comparison = candidate_comparison(
        baseline_all,
        challenger_all,
    )
    yearly_rows = yearly_metrics(
        baseline_all,
        challenger_all,
        config,
    )
    validation = evaluate_forward_evidence(
        fold_rows=fold_rows,
        metrics=challenger_metrics,
    )
    source = {
        **v11_audit,
        **v10_audit,
        **v14_audit,
    }
    summary = {
        "schema_version": "wp_v15_forward_risk_validation_1",
        "research_only": True,
        "production_model_changed": False,
        "production_authorized": False,
        "objective": (
            "strictly forward-validate whether the frozen V14 exit-risk "
            "safe-half gate improves the frozen V10 executable entry policy"
        ),
        "source": source,
        "discovery_period": {
            "fold": DISCOVERY_FOLD,
            "end_date": DISCOVERY_END_DATE,
            "used_for_v15_rule_selection": False,
        },
        "forward_validation_folds": list(FORWARD_VALIDATION_FOLDS),
        "protocol": {
            "frozen_meta_policy": frozen_meta_policy().as_dict(),
            "frozen_exit_risk_gate": {
                "metric": "within_trade_date_and_slot_failure_risk_rank_pct",
                "maximum": SAFE_RISK_RANK_MAX,
                "order": "risk_gate_before_daily_top_three_selection",
            },
            "risk_train_days": RISK_TRAIN_DAYS,
            "risk_calibration_days": RISK_CALIBRATION_DAYS,
            "purge_days_between_segments": PURGE_DAYS,
            "risk_model": (
                "70% histogram gradient boosting plus 30% logistic model, "
                "independently isotonic-calibrated"
            ),
            "features": "frozen causal T-day FEATURE_COLUMNS only",
            "exit_contract": "immutable T+1 close auction",
            "threshold_or_policy_retuning_on_forward_folds": False,
            "comparison": (
                "frozen V10 policy alone versus the same policy after the "
                "frozen V14 safe-half risk gate"
            ),
        },
        "folds": fold_rows,
        "aggregate_candidate_comparison": aggregate_comparison,
        "baseline_metrics": baseline_metrics,
        "challenger_metrics": challenger_metrics,
        "yearly": yearly_rows,
        "forward_evidence": validation,
        "shadow_requirement": {
            "minimum_trading_days": (
                config.promotion.minimum_shadow_trading_days
            ),
            "status": "not_started_for_v15",
        },
    }

    atomic_write_csv(
        baseline_all,
        output / "wp_v15_forward_baseline_candidates.csv",
    )
    atomic_write_csv(
        challenger_all,
        output / "wp_v15_forward_challenger_candidates.csv",
    )
    atomic_write_csv(
        pd.json_normalize(fold_rows, sep="."),
        output / "wp_v15_forward_folds.csv",
    )
    atomic_write_csv(
        pd.json_normalize(yearly_rows, sep="."),
        output / "wp_v15_forward_yearly.csv",
    )
    scored_path = atomic_write_parquet(
        scored_meta_all,
        output / "wp_v15_forward_scored_meta_frontier.parquet",
    )
    summary["scored_meta_frontier_sha256"] = sha256(scored_path)
    atomic_write_json(
        output / "wp_v15_forward_risk_validation_summary.json",
        summary,
    )
    marker = {
        "schema_version": summary["schema_version"],
        "source": source,
        "forward_validation_folds": list(FORWARD_VALIDATION_FOLDS),
        "folds_scored": validation["folds_scored"],
        "folds": [
            {
                "fold": row["fold"],
                "scored": row["scored"],
                "reason": row["reason"],
                "candidate_comparison": row.get(
                    "candidate_comparison"
                ),
                "baseline_metrics": compact_metrics(
                    row.get("baseline_metrics")
                ),
                "challenger_metrics": compact_metrics(
                    row.get("challenger_metrics")
                ),
            }
            for row in fold_rows
        ],
        "yearly": [
            {
                "year": row["year"],
                "baseline": compact_metrics(row["baseline"]),
                "challenger": compact_metrics(row["challenger"]),
            }
            for row in yearly_rows
        ],
        "aggregate_candidate_comparison": aggregate_comparison,
        "baseline_metrics": baseline_metrics,
        "challenger_metrics": challenger_metrics,
        "forward_evidence": validation,
        "scored_meta_frontier_sha256": (
            summary["scored_meta_frontier_sha256"]
        ),
    }
    print(
        "WP_V15_FORWARD_RISK_VALIDATION_RESULT="
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
            "p_round_trip_fill_lower",
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


def load_v10_scores(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(path)
    score_paths = sorted(
        root.rglob("wp_v10_meta_scored_test_frontier.csv")
    )
    summary_paths = sorted(root.rglob("wp_v10_meta_summary.json"))
    if len(score_paths) != 1 or len(summary_paths) != 1:
        raise FileNotFoundError(
            "V10 source must contain exactly one scored frontier and summary"
        )
    score_path = score_paths[0]
    summary = json.loads(summary_paths[0].read_text(encoding="utf-8"))
    discovery_rows = [
        row
        for row in summary.get("folds", [])
        if int(row.get("fold", -1)) == DISCOVERY_FOLD
    ]
    if len(discovery_rows) != 1:
        raise RuntimeError("V10 summary has no unique discovery fold")
    discovery = discovery_rows[0]
    policy = discovery.get("policy") or {}
    frozen_id = frozen_meta_policy().policy_id
    if (
        not bool(discovery.get("authorized"))
        or str(policy.get("policy_id")) != frozen_id
    ):
        raise RuntimeError(
            "V10 discovery policy does not match frozen V15 policy: "
            f"{policy.get('policy_id')} != {frozen_id}"
        )

    frame = pd.read_csv(score_path, dtype=str)
    required = {
        *IDENTITY_COLUMNS,
        "fold",
        "meta_fold",
        "meta_p_positive",
        "meta_expected_net_return_pct",
        "meta_p_severe_loss",
        "meta_score",
        "meta_rank_pct",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V10 scored frontier missing columns: {missing}")
    duplicates = frame.duplicated(list(IDENTITY_COLUMNS), keep=False)
    if duplicates.any():
        raise RuntimeError(
            "V10 scored frontier contains "
            f"{int(duplicates.sum())} duplicate identities"
        )
    frame["meta_source_fold"] = pd.to_numeric(
        frame["meta_fold"],
        errors="raise",
    ).astype(int)
    columns = [
        *IDENTITY_COLUMNS,
        "meta_source_fold",
        "meta_p_positive",
        "meta_expected_net_return_pct",
        "meta_p_severe_loss",
        "meta_score",
        "meta_rank_pct",
    ]
    frame = frame.loc[:, columns].copy()
    for column in IDENTITY_COLUMNS:
        frame[column] = frame[column].astype(str)
    return frame, {
        "v10_schema_version": summary.get("schema_version"),
        "v10_scored_frontier_sha256": sha256(score_path),
        "v10_scored_frontier_rows": int(len(frame)),
        "v10_discovery_fold": DISCOVERY_FOLD,
        "v10_discovery_policy_id": frozen_id,
    }


def load_v14_discovery(path: str | Path) -> dict[str, Any]:
    matches = sorted(
        Path(path).rglob("wp_v14_exit_failure_risk_summary.json")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            "V14 source must contain exactly one risk summary"
        )
    summary_path = matches[0]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best = summary.get("best_research_direction") or {}
    if str(best.get("policy_id")) != "risk_rank_safest_50pct":
        raise RuntimeError(
            "V14 discovery direction is not the frozen safe-half gate"
        )
    return {
        "v14_schema_version": summary.get("schema_version"),
        "v14_summary_sha256": sha256(summary_path),
        "v14_discovery_policy_id": best.get("policy_id"),
        "v14_discovery_events": best.get("retained_events"),
        "v14_production_authorized": bool(
            best.get("production_authorized")
        ),
    }


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
    requested["month"] = (
        requested["trade_date"]
        .astype(str)
        .str.replace("-", "", regex=False)
        .str[:6]
    )
    paths = {
        path.stem.rsplit("_", 1)[-1]: path
        for path in Path(panel_dir).glob("wp_v3_panel_*.parquet")
        if start_date.replace("-", "")[:6]
        <= path.stem.rsplit("_", 1)[-1]
        <= end_date.replace("-", "")[:6]
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
    final_purge = selected[
        calibration_start + RISK_CALIBRATION_DAYS :
    ]
    if len(final_purge) != PURGE_DAYS:
        raise AssertionError("invalid final purge length")
    return train, calibration


def eligible_training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        frame["label_available"].fillna(False).astype(bool)
        & boolean(frame["entry_fillable"])
    ].copy()


def merge_meta_scores(
    scored_test: pd.DataFrame,
    source_meta: pd.DataFrame,
    *,
    fold: int,
) -> pd.DataFrame:
    result = scored_test.merge(
        source_meta,
        on=list(IDENTITY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    if len(result) != len(source_meta):
        raise RuntimeError(
            f"fold {fold} matched {len(result)} of "
            f"{len(source_meta)} V10 meta-score identities"
        )
    if not result["meta_source_fold"].eq(fold).all():
        raise RuntimeError(f"fold {fold} contains mismatched meta scores")
    result["meta_fold"] = result["meta_source_fold"]
    for column in (
        "meta_p_positive",
        "meta_expected_net_return_pct",
        "meta_p_severe_loss",
        "meta_score",
        "meta_rank_pct",
        "p_round_trip_fill_lower",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


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


def selection_markers(
    scored_meta: pd.DataFrame,
    baseline: pd.DataFrame,
    challenger: pd.DataFrame,
) -> pd.DataFrame:
    result = scored_meta.copy()
    baseline_keys = identity_keys(baseline)
    challenger_keys = identity_keys(challenger)
    keys = result.loc[:, IDENTITY_COLUMNS].astype(str).agg("|".join, axis=1)
    result["selected_by_frozen_v10_policy"] = keys.isin(baseline_keys)
    result["selected_by_frozen_v15_challenger"] = keys.isin(
        challenger_keys
    )
    return result


def candidate_comparison(
    baseline: pd.DataFrame,
    challenger: pd.DataFrame,
) -> dict[str, Any]:
    baseline_keys = identity_keys(baseline)
    challenger_keys = identity_keys(challenger)
    return {
        "baseline_events": int(len(baseline)),
        "challenger_events": int(len(challenger)),
        "overlap_events": int(len(baseline_keys & challenger_keys)),
        "baseline_only_events": int(len(baseline_keys - challenger_keys)),
        "safe_replacement_events": int(
            len(challenger_keys - baseline_keys)
        ),
        "baseline_exit_failures": int(
            (~boolean(baseline.get("exit_fillable"))).sum()
            if "exit_fillable" in baseline
            else 0
        ),
        "challenger_exit_failures": int(
            (~boolean(challenger.get("exit_fillable"))).sum()
            if "exit_fillable" in challenger
            else 0
        ),
    }


def identity_keys(frame: pd.DataFrame) -> set[str]:
    if frame.empty:
        return set()
    return set(
        frame.loc[:, IDENTITY_COLUMNS]
        .astype(str)
        .agg("|".join, axis=1)
        .tolist()
    )


def concat_or_empty(
    frames: list[pd.DataFrame],
    template: pd.DataFrame,
) -> pd.DataFrame:
    if frames:
        return pd.concat(frames, ignore_index=True)
    return template.head(0).copy()


def yearly_metrics(
    baseline: pd.DataFrame,
    challenger: pd.DataFrame,
    config: Any,
) -> list[dict[str, Any]]:
    years = sorted(
        set(baseline.get("trade_date", pd.Series(dtype=str)).astype(str).str[:4])
        | set(
            challenger.get(
                "trade_date",
                pd.Series(dtype=str),
            ).astype(str).str[:4]
        )
    )
    rows: list[dict[str, Any]] = []
    for year in years:
        if not year:
            continue
        baseline_year = baseline.loc[
            baseline["trade_date"].astype(str).str[:4].eq(year)
        ]
        challenger_year = challenger.loc[
            challenger["trade_date"].astype(str).str[:4].eq(year)
        ]
        rows.append(
            {
                "year": year,
                "baseline": performance_summary(
                    baseline_year,
                    config,
                    bootstrap_samples=2_000,
                    seed=config.model.random_seed + int(year) + 15_000,
                ),
                "challenger": performance_summary(
                    challenger_year,
                    config,
                    bootstrap_samples=2_000,
                    seed=config.model.random_seed + int(year) + 15_500,
                ),
            }
        )
    return rows


def evaluate_forward_evidence(
    *,
    fold_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    folds_scored = int(
        sum(bool(row.get("scored")) for row in fold_rows)
    )
    stress = metrics.get("stress", {}).get("50bps", {})
    gates = {
        "all_forward_folds_scored": (
            folds_scored == len(FORWARD_VALIDATION_FOLDS)
        ),
        "minimum_250_events": (
            int(metrics.get("events", 0) or 0)
            >= MINIMUM_CONFIRMATION_EVENTS
        ),
        "minimum_60_trade_days": (
            int(metrics.get("trade_days", 0) or 0)
            >= MINIMUM_CONFIRMATION_DAYS
        ),
        "positive_mean_net_return": (
            (metrics.get("mean_net_return_pct") or -999.0) > 0.0
        ),
        "positive_day_clustered_mean_lower": (
            (
                metrics.get("mean_net_return_day_clustered_lower_pct")
                or -999.0
            )
            > 0.0
        ),
        "day_clustered_win_lower_at_least_52pct": (
            (metrics.get("win_rate_day_clustered_lower") or 0.0) >= 0.52
        ),
        "profit_factor_at_least_1_20": (
            (metrics.get("profit_factor") or 0.0) >= 1.20
        ),
        "exit_fill_rate_at_least_98pct": (
            (metrics.get("exit_fill_rate_given_entry") or 0.0) >= 0.98
        ),
        "positive_total_return_at_50bp": bool(
            stress.get("positive_total_return", False)
        ),
    }
    all_passed = all(gates.values())
    positive_direction = bool(
        gates["all_forward_folds_scored"]
        and gates["positive_mean_net_return"]
        and gates["profit_factor_at_least_1_20"]
        and gates["positive_total_return_at_50bp"]
    )
    if all_passed:
        status = "positive_forward_evidence_requires_150_day_shadow"
        reason = "all_predeclared_forward_confirmation_gates_passed"
    elif positive_direction:
        status = "positive_forward_direction_unconfirmed"
        reason = "one_or_more_statistical_or_sample_gates_failed"
    else:
        status = "forward_holdout_not_profitable"
        reason = "predeclared_positive_forward_direction_gates_failed"
    return {
        "status": status,
        "reason": reason,
        "folds_scored": folds_scored,
        "folds_required": len(FORWARD_VALIDATION_FOLDS),
        "gates": gates,
        "all_gates_passed": all_passed,
        "production_authorized": False,
    }


def compact_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metrics:
        return None
    return {
        key: metrics.get(key)
        for key in (
            "events",
            "trade_days",
            "win_rate",
            "win_rate_day_clustered_lower",
            "mean_net_return_pct",
            "mean_net_return_day_clustered_lower_pct",
            "profit_factor",
            "exit_fill_rate_given_entry",
            "day_equal_weight_mean_net_return_pct",
            "day_equal_weight_cumulative_return_pct",
            "maximum_day_equal_weight_drawdown_pct",
            "stress",
        )
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boolean(values: pd.Series | None) -> pd.Series:
    if values is None:
        return pd.Series(dtype=bool)
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
