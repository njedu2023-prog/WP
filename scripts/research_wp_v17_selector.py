from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from wp.v3.contracts import load_v3_config
from wp.v3.io import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    file_sha256,
)
from wp.v3.meta_alpha import IDENTITY_COLUMNS
from wp.v3.v16_policy import (
    benjamini_hochberg,
    policy_metrics,
)
from wp.v3.v16_provenance import bind_base_model
from wp.v3.v16_research import (
    EXIT_CONTRACT_ID,
    MODEL_CALIBRATION_DAYS,
    MODEL_MAX_TRAIN_DAYS,
    MODEL_MIN_TRAIN_DAYS,
    POLICY_CONFIRMATION_DAYS,
    POLICY_DESIGN_DAYS,
    PURGE_DAYS,
    eligible_labeled_rows,
    load_v11_frontier,
    materialize_close_labels,
    research_readiness,
    rolling_model_segments,
    rolling_policy_segments,
)
from wp.v3.v17_selector import (
    SelectorPolicySelection,
    apply_selector_policy,
    fit_selector,
    select_selector_policy,
    selector_policy_grid,
)


SCHEMA_VERSION = "wp_v17_evidence_calibrated_selector_1"
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the V17 low-dimensional, frequency-constrained selector over "
            "the immutable V9/V11 causal OOS frontier."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--v11-source-dir", required=True)
    parser.add_argument("--v15-source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--base-model-registry",
        default=str(ROOT / "outputs/json/wp_model_registry_v3.json"),
    )
    parser.add_argument("--expected-base-model-fingerprint", required=True)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_v3_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    base_artifact, _, base_contract = bind_base_model(
        args.base_model_registry,
        expected_fingerprint=args.expected_base_model_fingerprint,
        repository_root=args.repository_root,
    )
    frontier, source = load_v11_frontier(args.v11_source_dir)
    frontier = materialize_close_labels(
        frontier,
        severe_loss_threshold_pct=config.model.severe_loss_threshold_pct,
    )
    source["base_v9"] = base_contract
    v15 = load_v15_evidence(args.v15_source_dir)
    source["v15"] = v15["source"]

    fold_rows: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    evaluated_dates: set[str] = set()
    temporal_integrity = True
    numeric_fold = pd.to_numeric(frontier["fold"], errors="coerce")
    folds = sorted(numeric_fold.dropna().astype(int).unique())

    for fold in folds:
        test = frontier.loc[numeric_fold.eq(fold)].copy()
        if test.empty:
            continue
        test_dates = sorted(test["trade_date"].astype(str).unique())
        overlap = evaluated_dates.intersection(test_dates)
        if overlap:
            raise RuntimeError(
                f"outer test dates overlap: {sorted(overlap)[:5]}"
            )
        test_start = test_dates[0]
        test_end = test_dates[-1]
        history = frontier.loc[
            frontier["trade_date"].astype(str).lt(test_start)
        ].copy()
        model_segments = rolling_model_segments(
            history["trade_date"].astype(str).unique()
        )
        base = {
            "fold": int(fold),
            "test_start": test_start,
            "test_end": test_end,
            "test_rows": int(len(test)),
            "test_days": int(len(test_dates)),
        }
        if model_segments is None:
            fold_rows.append(
                {
                    **base,
                    "scored": False,
                    "reason": "insufficient_prior_model_history",
                    "policy_authorized": False,
                }
            )
            continue
        train_dates, calibration_dates = model_segments
        train = eligible_labeled_rows(
            history.loc[
                history["trade_date"].astype(str).isin(train_dates)
            ]
        )
        calibration = eligible_labeled_rows(
            history.loc[
                history["trade_date"].astype(str).isin(calibration_dates)
            ]
        )
        model_temporal_ok = bool(
            train_dates[-1] < calibration_dates[0]
            and calibration_dates[-1] < test_start
        )
        temporal_integrity &= model_temporal_ok
        print(
            f"[wp-v17] fold={fold} "
            f"train={train_dates[0]}..{train_dates[-1]} rows={len(train):,} "
            f"calibration={calibration_dates[0]}.."
            f"{calibration_dates[-1]} rows={len(calibration):,} "
            f"test={test_start}..{test_end} rows={len(test):,}",
            flush=True,
        )
        bundle = fit_selector(
            train,
            calibration,
            random_seed=config.model.random_seed + int(fold) * 1_301,
        )
        scored_test = bundle.predict(test)
        scored_test["selector_source_fold"] = int(fold)

        prior_scored = concat_or_empty(scored_frames, scored_test)
        policy_segments = rolling_policy_segments(
            prior_scored["trade_date"].astype(str).unique()
            if not prior_scored.empty
            else []
        )
        selection: SelectorPolicySelection | None = None
        selected = scored_test.head(0).copy()
        policy_temporal_ok = True
        reason = "insufficient_prior_oos_policy_history"
        if policy_segments is not None:
            design_dates, confirmation_dates = policy_segments
            design = prior_scored.loc[
                prior_scored["trade_date"].astype(str).isin(design_dates)
            ].copy()
            confirmation = prior_scored.loc[
                prior_scored["trade_date"]
                .astype(str)
                .isin(confirmation_dates)
            ].copy()
            policy_temporal_ok = bool(
                design_dates[-1] < confirmation_dates[0]
                and confirmation_dates[-1] < test_start
            )
            temporal_integrity &= policy_temporal_ok
            selection = select_selector_policy(
                design,
                confirmation,
                design_total_days=len(design_dates),
                confirmation_total_days=len(confirmation_dates),
                seed=config.model.random_seed + int(fold) * 13_007,
                bootstrap_samples=args.bootstrap_samples,
            )
            if selection.policy is not None:
                selected = apply_selector_policy(
                    scored_test,
                    selection.policy,
                )
                selected["selector_source_fold"] = int(fold)
                selected["nested_policy_id"] = selection.policy.policy_id
                selected_frames.append(selected)
                reason = "prior_oos_policy_passed_design_and_confirmation"
            else:
                reason = "prior_oos_policy_not_confirmed"

        scored_frames.append(scored_test)
        evaluated_dates.update(test_dates)
        test_metrics = policy_metrics(
            selected,
            total_days=len(test_dates),
            seed=config.model.random_seed + int(fold) * 37,
            bootstrap_samples=args.bootstrap_samples,
        )
        fold_rows.append(
            {
                **base,
                "scored": True,
                "reason": reason,
                "model_temporal_integrity": model_temporal_ok,
                "policy_temporal_integrity": policy_temporal_ok,
                "model": {
                    "train_start": train_dates[0],
                    "train_end": train_dates[-1],
                    "train_days": len(train_dates),
                    "train_rows": bundle.train_rows,
                    "calibration_start": calibration_dates[0],
                    "calibration_end": calibration_dates[-1],
                    "calibration_days": len(calibration_dates),
                    "calibration_rows": bundle.calibration_rows,
                    "feature_count": len(bundle.feature_columns),
                    "features": list(bundle.feature_columns),
                },
                "policy_authorized": bool(
                    selection is not None and selection.policy is not None
                ),
                "policy_selection": compact_selection(selection),
                "test": test_metrics,
            }
        )
        print(
            f"[wp-v17] fold={fold} policy="
            f"{selection.policy.policy_id if selection and selection.policy else 'NO_SIGNAL'} "
            f"events={test_metrics['events']} "
            f"days={test_metrics['candidate_days']} "
            f"win={test_metrics['win_rate']:.4f} "
            f"mean={test_metrics['mean_net_return_pct']}",
            flush=True,
        )

    scored_all = concat_or_empty(scored_frames, frontier)
    selected_all = concat_or_empty(selected_frames, scored_all)
    assert_unique(scored_all, "V17 scored OOS frontier")
    assert_unique(selected_all, "V17 nested OOS candidates")
    total_oos_days = len(evaluated_dates)
    nested_metrics = policy_metrics(
        selected_all,
        total_days=total_oos_days,
        seed=config.model.random_seed + 17_000,
        bootstrap_samples=max(4_000, args.bootstrap_samples),
    )
    readiness = research_readiness(
        nested_metrics,
        temporal_integrity=temporal_integrity,
    )
    frontier_table = descriptive_frontier(
        scored_all,
        total_days=total_oos_days,
        seed=config.model.random_seed + 170_000,
        bootstrap_samples=args.bootstrap_samples,
    )
    diagnostics = frontier_diagnostics(frontier_table)
    final_selection = select_final_policy(
        scored_all,
        config=config,
        bootstrap_samples=args.bootstrap_samples,
    )
    final_selection_summary = compact_selection(final_selection)
    final_bundle, final_period = fit_final_selector(
        frontier,
        config=config,
    )
    model_payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_authorized": False,
        "exit_contract_id": EXIT_CONTRACT_ID,
        "candidate_policy": (
            final_selection.policy if final_selection else None
        ),
        "selector": final_bundle,
        "model_period": final_period,
        "base_model_contract": base_contract,
        "base_model_artifact": base_artifact,
        "source": source,
    }
    model_path = output / "wp_v17_shadow_model.joblib"
    joblib.dump(model_payload, model_path, compress=3)

    scored_path = atomic_write_parquet(
        scored_all,
        output / "wp_v17_selector_scored_oos.parquet",
    )
    candidate_path = atomic_write_csv(
        selected_all,
        output / "wp_v17_nested_oos_candidates.csv",
    )
    fold_path = atomic_write_csv(
        pd.json_normalize(fold_rows, sep="."),
        output / "wp_v17_nested_folds.csv",
    )
    frontier_path = atomic_write_csv(
        frontier_table,
        output / "wp_v17_frequency_profit_frontier.csv",
    )
    yearly = yearly_metrics(
        selected_all,
        total_dates=sorted(evaluated_dates),
        seed=config.model.random_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_authorized": False,
        "v15_frozen_candidate_changed": False,
        "v16_failed_specialist_layer_reused": False,
        "objective": (
            "maximize the probability that every executable 14:20-14:50 "
            "candidate earns positive net return at immutable T+1 close after "
            "costs, subject to a useful candidate-day frequency"
        ),
        "source": source,
        "evaluation": {
            "requested_start": config.history.evaluation_start_date,
            "requested_end": config.history.evaluation_end_date,
            "source_start": str(frontier["trade_date"].astype(str).min()),
            "source_end": str(frontier["trade_date"].astype(str).max()),
            "source_rows": int(len(frontier)),
            "source_days": int(
                frontier["trade_date"].astype(str).nunique()
            ),
            "folds_total": len(folds),
            "folds_scored": int(
                sum(bool(row.get("scored")) for row in fold_rows)
            ),
            "nested_oos_days": total_oos_days,
            "selector_scored_rows": int(len(scored_all)),
            "nested_selected_rows": int(len(selected_all)),
        },
        "protocol": {
            "candidate_source": (
                "immutable V9 OOS scores with V11 executable T+1 truth"
            ),
            "selector": (
                "single pooled regularized tree-linear positive-return model "
                "plus robust location and 25th-percentile return models"
            ),
            "feature_count_maximum": len(final_bundle.feature_columns),
            "model_max_train_days": MODEL_MAX_TRAIN_DAYS,
            "model_min_train_days": MODEL_MIN_TRAIN_DAYS,
            "model_calibration_days": MODEL_CALIBRATION_DAYS,
            "policy_design_days": POLICY_DESIGN_DAYS,
            "policy_confirmation_days": POLICY_CONFIRMATION_DAYS,
            "purge_days_between_segments": PURGE_DAYS,
            "predeclared_policy_count": len(selector_policy_grid()),
            "policy_family_control": "Benjamini-Hochberg q<=0.10",
            "frequency_constraint": (
                "15%-60% candidate days in design and at least 12% in "
                "independent confirmation"
            ),
            "stress_test": "subtract an additional real 50bps per trade",
            "exit_contract_id": EXIT_CONTRACT_ID,
            "no_signal_allowed": True,
            "future_shadow_days_required": 150,
        },
        "v15_frozen_reference": v15["reference"],
        "nested_oos_metrics": nested_metrics,
        "yearly": yearly,
        "descriptive_frontier_diagnostics": diagnostics,
        "historical_readiness": readiness,
        "final_shadow_candidate": {
            "selection": final_selection_summary,
            "model_period": final_period,
            "status": (
                "READY_FOR_150_DAY_FUTURE_SHADOW"
                if (
                    final_selection is not None
                    and final_selection.policy is not None
                    and readiness["all_historical_gates_passed"]
                )
                else "NOT_READY_FOR_SHADOW"
            ),
            "production_authorized": False,
        },
        "artifacts": {
            "selector_scored_oos": artifact(scored_path),
            "nested_oos_candidates": artifact(candidate_path),
            "nested_folds": artifact(fold_path),
            "frequency_profit_frontier": artifact(frontier_path),
            "shadow_model": artifact(model_path),
        },
        "folds": fold_rows,
    }
    atomic_write_json(output / "wp_v17_research_summary.json", summary)
    print(
        "WP_V17_RESEARCH_RESULT="
        + json.dumps(
            json_safe(
                {
                    "schema_version": SCHEMA_VERSION,
                    "nested_oos_metrics": nested_metrics,
                    "historical_readiness": readiness,
                    "shadow_status": summary[
                        "final_shadow_candidate"
                    ]["status"],
                    "shadow_policy_id": (
                        final_selection.policy.policy_id
                        if final_selection and final_selection.policy
                        else None
                    ),
                    "descriptive_frontier_diagnostics": diagnostics,
                    "final_policy_selection": final_selection_summary,
                    "v15_reference": v15["reference"],
                }
            ),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def load_v15_evidence(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    summaries = sorted(
        root.rglob("wp_v15_forward_risk_validation_summary.json")
    )
    candidates = sorted(
        root.rglob("wp_v15_forward_challenger_candidates.csv")
    )
    if len(summaries) != 1 or len(candidates) != 1:
        raise FileNotFoundError(
            "V15 source must contain exactly one summary and challenger CSV"
        )
    summary_path = summaries[0]
    candidate_path = candidates[0]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary.get("challenger_metrics") or {}
    evidence = summary.get("forward_evidence") or {}
    if bool(evidence.get("production_authorized")):
        raise RuntimeError("V15 research evidence cannot authorize production")
    return {
        "source": {
            "schema_version": summary.get("schema_version"),
            "summary_sha256": file_sha256(summary_path),
            "challenger_sha256": file_sha256(candidate_path),
        },
        "reference": {
            "status": evidence.get("status"),
            "events": int(metrics.get("events", 0)),
            "trade_days": int(metrics.get("trade_days", 0)),
            "win_rate": metrics.get("win_rate"),
            "mean_net_return_pct": metrics.get("mean_net_return_pct"),
            "mean_net_return_day_clustered_lower_pct": metrics.get(
                "mean_net_return_day_clustered_lower_pct"
            ),
            "profit_factor": metrics.get("profit_factor"),
            "stress_50bps_mean_net_return_pct": (
                (metrics.get("stress") or {})
                .get("50bps", {})
                .get("mean_net_return_pct")
            ),
            "production_authorized": False,
        },
    }


def descriptive_frontier(
    scored: pd.DataFrame,
    *,
    total_days: int,
    seed: int,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows = []
    for offset, policy in enumerate(selector_policy_grid()):
        metrics = policy_metrics(
            apply_selector_policy(scored, policy),
            total_days=total_days,
            seed=seed + offset,
            bootstrap_samples=bootstrap_samples,
        )
        rows.append({**policy.as_dict(), **metrics})
    q_values = benjamini_hochberg(
        float(row["mean_return_p_value"]) for row in rows
    )
    for row, q_value in zip(rows, q_values, strict=True):
        row["mean_return_q_value"] = q_value
    return pd.DataFrame(rows).sort_values(
        [
            "clustered_mean_lower_pct",
            "candidate_day_rate",
            "mean_net_return_pct",
        ],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def frontier_diagnostics(frontier: pd.DataFrame) -> dict[str, Any]:
    columns = [
        "policy_id",
        "probability_lower_min",
        "expected_return_min_pct",
        "score_rank_min",
        "max_candidates_per_day",
        "events",
        "candidate_days",
        "candidate_day_rate",
        "win_rate",
        "win_rate_wilson_lower",
        "mean_net_return_pct",
        "clustered_mean_lower_pct",
        "profit_factor",
        "stress_50bps_mean_net_return_pct",
        "return_p10_pct",
        "mean_return_q_value",
    ]
    available = [column for column in columns if column in frontier.columns]
    if frontier.empty:
        return {
            "policies_evaluated": 0,
            "positive_mean_policy_count": 0,
            "positive_clustered_lower_policy_count": 0,
            "bh_q_le_0_10_policy_count": 0,
            "practical_frequency_policy_count": 0,
            "best_clustered_lower": [],
            "best_mean_return": [],
            "best_practical_frequency": [],
        }

    mean_return = pd.to_numeric(
        frontier["mean_net_return_pct"], errors="coerce"
    )
    clustered_lower = pd.to_numeric(
        frontier["clustered_mean_lower_pct"], errors="coerce"
    )
    q_value = pd.to_numeric(
        frontier["mean_return_q_value"], errors="coerce"
    )
    frequency = pd.to_numeric(
        frontier["candidate_day_rate"], errors="coerce"
    )
    practical = frontier.loc[frequency.between(0.12, 0.60)].copy()
    by_mean = frontier.assign(_metric=mean_return).sort_values(
        ["_metric", "candidate_day_rate"],
        ascending=[False, False],
        kind="stable",
    )
    practical_by_mean = practical.sort_values(
        ["mean_net_return_pct", "clustered_mean_lower_pct"],
        ascending=[False, False],
        kind="stable",
    )

    def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return frame.loc[:, available].head(3).to_dict(orient="records")

    return {
        "policies_evaluated": int(len(frontier)),
        "positive_mean_policy_count": int(mean_return.gt(0.0).sum()),
        "positive_clustered_lower_policy_count": int(
            clustered_lower.gt(0.0).sum()
        ),
        "bh_q_le_0_10_policy_count": int(q_value.le(0.10).sum()),
        "practical_frequency_policy_count": int(len(practical)),
        "best_clustered_lower": records(frontier),
        "best_mean_return": records(by_mean),
        "best_practical_frequency": records(practical_by_mean),
    }


def select_final_policy(
    scored: pd.DataFrame,
    *,
    config: Any,
    bootstrap_samples: int,
) -> SelectorPolicySelection | None:
    if scored.empty:
        return None
    segments = rolling_policy_segments(
        scored["trade_date"].astype(str).unique(),
        reserve_final_purge=False,
    )
    if segments is None:
        return None
    design_dates, confirmation_dates = segments
    return select_selector_policy(
        scored.loc[
            scored["trade_date"].astype(str).isin(design_dates)
        ],
        scored.loc[
            scored["trade_date"].astype(str).isin(confirmation_dates)
        ],
        design_total_days=len(design_dates),
        confirmation_total_days=len(confirmation_dates),
        seed=config.model.random_seed + 1_700_000,
        bootstrap_samples=bootstrap_samples,
    )


def fit_final_selector(
    frontier: pd.DataFrame,
    *,
    config: Any,
) -> tuple[Any, dict[str, Any]]:
    segments = rolling_model_segments(
        frontier["trade_date"].astype(str).unique(),
        reserve_final_purge=False,
    )
    if segments is None:
        raise RuntimeError("insufficient history for final V17 selector")
    train_dates, calibration_dates = segments
    bundle = fit_selector(
        eligible_labeled_rows(
            frontier.loc[
                frontier["trade_date"].astype(str).isin(train_dates)
            ]
        ),
        eligible_labeled_rows(
            frontier.loc[
                frontier["trade_date"].astype(str).isin(calibration_dates)
            ]
        ),
        random_seed=config.model.random_seed + 17_000_000,
    )
    return bundle, {
        "train_start": train_dates[0],
        "train_end": train_dates[-1],
        "train_days": len(train_dates),
        "train_rows": bundle.train_rows,
        "calibration_start": calibration_dates[0],
        "calibration_end": calibration_dates[-1],
        "calibration_days": len(calibration_dates),
        "calibration_rows": bundle.calibration_rows,
        "feature_count": len(bundle.feature_columns),
        "features": list(bundle.feature_columns),
    }


def yearly_metrics(
    selected: pd.DataFrame,
    *,
    total_dates: list[str],
    seed: int,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    rows = []
    for year in sorted({str(value)[:4] for value in total_dates}):
        dates = [value for value in total_dates if str(value).startswith(year)]
        group = selected.loc[
            selected["trade_date"].astype(str).str.startswith(year)
        ]
        rows.append(
            {
                "year": year,
                **policy_metrics(
                    group,
                    total_days=len(dates),
                    seed=seed + int(year),
                    bootstrap_samples=bootstrap_samples,
                ),
            }
        )
    return rows


def compact_selection(
    selection: SelectorPolicySelection | None,
) -> dict[str, Any]:
    if selection is None:
        return {"policy": None, "reason": "not_run"}
    payload = selection.as_dict()
    design = payload["design"]
    if "policies" in design:
        policies = sorted(
            design["policies"],
            key=lambda row: (
                float(row.get("mean_return_q_value") or 1.0),
                -float(row.get("mean_net_return_pct") or -999.0),
            ),
        )
        payload["design"] = {
            "reason": design["reason"],
            "best_diagnostics": policies[:5],
        }
    return payload


def concat_or_empty(
    frames: list[pd.DataFrame],
    template: pd.DataFrame,
) -> pd.DataFrame:
    if not frames:
        return template.head(0).copy()
    return pd.concat(frames, ignore_index=True)


def assert_unique(frame: pd.DataFrame, label: str) -> None:
    if frame.empty:
        return
    duplicates = frame.duplicated(list(IDENTITY_COLUMNS), keep=False)
    if duplicates.any():
        raise RuntimeError(
            f"{label} has {int(duplicates.sum())} duplicate identities"
        )


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
