from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from wp.v3.contracts import load_v3_config
from wp.v3.io import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    file_sha256,
)
from wp.v3.v16_policy import (
    PolicySelection,
    apply_expert_policy,
    policy_metrics,
    select_nested_policy,
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
    SCHEMA_VERSION,
    attach_original_features,
    descriptive_policy_frontier,
    eligible_labeled_rows,
    load_v11_frontier,
    materialize_close_labels,
    research_readiness,
    rolling_model_segments,
    rolling_policy_segments,
)
from wp.v3.v16_specialists import (
    fit_and_score_specialists,
    fit_specialists,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run V16 multi-specialist nested walk-forward research over the "
            "immutable V11 causal candidate frontier."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--v11-source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--base-model-registry",
        default=str(ROOT / "outputs/json/wp_model_registry_v3.json"),
    )
    parser.add_argument(
        "--expected-base-model-fingerprint",
        required=True,
    )
    parser.add_argument(
        "--repository-root",
        default=str(ROOT),
    )
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
    frontier, source_audit = load_v11_frontier(args.v11_source_dir)
    source_audit = {**source_audit, "base_v9": base_contract}
    frontier = attach_original_features(
        frontier,
        args.panel_dir,
        start_date=config.history.evaluation_start_date,
        end_date=config.history.evaluation_end_date,
    )
    frontier = materialize_close_labels(
        frontier,
        severe_loss_threshold_pct=(
            config.model.severe_loss_threshold_pct
        ),
    )

    fold_rows: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    evaluated_test_dates: set[str] = set()
    temporal_integrity = True
    numeric_fold = pd.to_numeric(frontier["fold"], errors="coerce")
    folds = sorted(numeric_fold.dropna().astype(int).unique())

    for fold in folds:
        test = frontier.loc[numeric_fold.eq(fold)].copy()
        if test.empty:
            continue
        test_dates = sorted(test["trade_date"].astype(str).unique())
        overlapping_dates = evaluated_test_dates.intersection(test_dates)
        if overlapping_dates:
            raise RuntimeError(
                "outer test folds overlap on trade dates: "
                f"{sorted(overlapping_dates)[:5]}"
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
                    "policy_authorized": False,
                    "reason": "insufficient_prior_model_history",
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
            f"[wp-v16] fold={fold} "
            f"train={train_dates[0]}..{train_dates[-1]} "
            f"rows={len(train):,} calibration={calibration_dates[0]}.."
            f"{calibration_dates[-1]} rows={len(calibration):,} "
            f"test={test_start}..{test_end} rows={len(test):,}",
            flush=True,
        )
        scored_test, expert_audit = fit_and_score_specialists(
            train,
            calibration,
            test,
            random_seed=config.model.random_seed + int(fold) * 1_009,
        )
        scored_test["expert_source_fold"] = int(fold)

        prior_scored = concat_or_empty(scored_frames, scored_test)
        policy_segments = rolling_policy_segments(
            prior_scored["trade_date"].astype(str).unique()
            if not prior_scored.empty
            else []
        )
        selection: PolicySelection | None = None
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
            selection = select_nested_policy(
                design,
                confirmation,
                design_total_days=len(design_dates),
                confirmation_total_days=len(confirmation_dates),
                seed=config.model.random_seed + int(fold) * 10_007,
                bootstrap_samples=args.bootstrap_samples,
            )
            if selection.policy is not None:
                selected = apply_expert_policy(
                    scored_test,
                    selection.policy,
                )
                selected["expert_source_fold"] = int(fold)
                selected["nested_policy_id"] = selection.policy.policy_id
                selected_frames.append(selected)
                reason = "prior_oos_policy_passed_design_and_confirmation"
            else:
                reason = "prior_oos_policy_not_confirmed"

        scored_frames.append(scored_test)
        evaluated_test_dates.update(test_dates)
        test_metrics = policy_metrics(
            selected,
            total_days=len(test_dates),
            seed=config.model.random_seed + int(fold) * 31,
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
                    "train_rows": int(len(train)),
                    "calibration_start": calibration_dates[0],
                    "calibration_end": calibration_dates[-1],
                    "calibration_days": len(calibration_dates),
                    "calibration_rows": int(len(calibration)),
                    "experts": expert_audit,
                },
                "policy_authorized": bool(
                    selection is not None and selection.policy is not None
                ),
                "policy_selection": compact_selection(selection),
                "test": test_metrics,
            }
        )
        print(
            f"[wp-v16] fold={fold} policy="
            f"{selection.policy.policy_id if selection and selection.policy else 'NO_SIGNAL'} "
            f"events={test_metrics['events']} "
            f"win={test_metrics['win_rate']:.4f} "
            f"mean={test_metrics['mean_net_return_pct']}",
            flush=True,
        )

    scored_all = concat_or_empty(scored_frames, frontier)
    selected_all = concat_or_empty(selected_frames, scored_all)
    assert_unique_identities(scored_all, label="expert-scored OOS frontier")
    assert_unique_identities(selected_all, label="nested OOS candidates")
    total_oos_days = len(evaluated_test_dates)
    nested_metrics = policy_metrics(
        selected_all,
        total_days=total_oos_days,
        seed=config.model.random_seed + 16_000,
        bootstrap_samples=max(args.bootstrap_samples, 4_000),
    )
    readiness = research_readiness(
        nested_metrics,
        temporal_integrity=temporal_integrity,
    )
    frontier_table = descriptive_policy_frontier(
        scored_all,
        total_days=total_oos_days,
        seed=config.model.random_seed + 160_000,
        bootstrap_samples=args.bootstrap_samples,
    )

    final_selection = select_final_shadow_policy(
        scored_all,
        config=config,
        bootstrap_samples=args.bootstrap_samples,
    )
    final_bundles, final_model_audit, final_model_period = fit_final_models(
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
            final_selection.policy
            if final_selection is not None
            else None
        ),
        "base_model_contract": base_contract,
        "base_model_artifact": base_artifact,
        "specialists": final_bundles,
        "model_period": final_model_period,
        "source": source_audit,
    }
    model_path = output / "wp_v16_shadow_model.joblib"
    joblib.dump(model_payload, model_path, compress=3)

    scored_path = atomic_write_parquet(
        scored_all,
        output / "wp_v16_expert_scored_oos.parquet",
    )
    selected_path = atomic_write_csv(
        selected_all,
        output / "wp_v16_nested_oos_candidates.csv",
    )
    fold_path = atomic_write_csv(
        pd.json_normalize(fold_rows, sep="."),
        output / "wp_v16_nested_folds.csv",
    )
    frontier_path = atomic_write_csv(
        frontier_table,
        output / "wp_v16_frequency_profit_frontier.csv",
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_authorized": False,
        "v15_frozen_candidate_changed": False,
        "objective": (
            "maximize the probability that every executable 14:20-14:50 "
            "candidate earns positive net return at fixed T+1 close after "
            "costs, while allowing NO_SIGNAL"
        ),
        "source": source_audit,
        "evaluation": {
            "start": config.history.evaluation_start_date,
            "end": config.history.evaluation_end_date,
            "frontier_rows": int(len(frontier)),
            "frontier_days": int(
                frontier["trade_date"].astype(str).nunique()
            ),
            "folds_total": len(folds),
            "folds_scored": int(
                sum(bool(row.get("scored")) for row in fold_rows)
            ),
            "nested_oos_days": total_oos_days,
            "expert_scored_rows": int(len(scored_all)),
            "nested_selected_rows": int(len(selected_all)),
        },
        "protocol": {
            "exit_contract_id": EXIT_CONTRACT_ID,
            "model_max_train_days": MODEL_MAX_TRAIN_DAYS,
            "model_min_train_days": MODEL_MIN_TRAIN_DAYS,
            "model_calibration_days": MODEL_CALIBRATION_DAYS,
            "policy_design_days": POLICY_DESIGN_DAYS,
            "policy_confirmation_days": POLICY_CONFIRMATION_DAYS,
            "purge_days_between_segments": PURGE_DAYS,
            "policy_family_control": "Benjamini-Hochberg q<=0.10",
            "uncertainty": (
                "trade-day mean one-sided 5-lag HAC test plus 5-day circular "
                "block-bootstrap confidence intervals"
            ),
            "stress_test": "subtract an additional real 50bps per trade",
            "no_trade_allowed": True,
            "all_historical_results_are_research_only": True,
            "future_shadow_days_required": 150,
        },
        "nested_oos_metrics": nested_metrics,
        "historical_readiness": readiness,
        "final_shadow_candidate": {
            "selection": compact_selection(final_selection),
            "model_period": final_model_period,
            "model_experts": final_model_audit,
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
            "expert_scored_oos": artifact(scored_path),
            "nested_oos_candidates": artifact(selected_path),
            "nested_folds": artifact(fold_path),
            "frequency_profit_frontier": artifact(frontier_path),
            "shadow_model": artifact(model_path),
        },
        "folds": fold_rows,
    }
    atomic_write_json(output / "wp_v16_research_summary.json", summary)
    print(
        "WP_V16_RESEARCH_RESULT="
        + json.dumps(
            json_safe(
                {
                "schema_version": SCHEMA_VERSION,
                "nested_oos_metrics": nested_metrics,
                "historical_readiness": readiness,
                "shadow_status": summary["final_shadow_candidate"]["status"],
                "shadow_policy_id": (
                    final_selection.policy.policy_id
                    if final_selection and final_selection.policy
                    else None
                ),
                }
            ),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        ),
        flush=True,
    )
    return 0


def select_final_shadow_policy(
    scored: pd.DataFrame,
    *,
    config: Any,
    bootstrap_samples: int,
) -> PolicySelection | None:
    if scored.empty:
        return None
    segments = rolling_policy_segments(
        scored["trade_date"].astype(str).unique(),
        reserve_final_purge=False,
    )
    if segments is None:
        return None
    design_dates, confirmation_dates = segments
    design = scored.loc[
        scored["trade_date"].astype(str).isin(design_dates)
    ]
    confirmation = scored.loc[
        scored["trade_date"].astype(str).isin(confirmation_dates)
    ]
    return select_nested_policy(
        design,
        confirmation,
        design_total_days=len(design_dates),
        confirmation_total_days=len(confirmation_dates),
        seed=config.model.random_seed + 1_600_000,
        bootstrap_samples=bootstrap_samples,
    )


def fit_final_models(
    frontier: pd.DataFrame,
    *,
    config: Any,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    segments = rolling_model_segments(
        frontier["trade_date"].astype(str).unique(),
        reserve_final_purge=False,
    )
    if segments is None:
        raise RuntimeError("insufficient history for final V16 shadow models")
    train_dates, calibration_dates = segments
    train = eligible_labeled_rows(
        frontier.loc[
            frontier["trade_date"].astype(str).isin(train_dates)
        ]
    )
    calibration = eligible_labeled_rows(
        frontier.loc[
            frontier["trade_date"].astype(str).isin(calibration_dates)
        ]
    )
    bundles, audit = fit_specialists(
        train,
        calibration,
        random_seed=config.model.random_seed + 16_000_000,
    )
    if len(bundles) < 4:
        raise RuntimeError(
            "final V16 shadow package requires at least four independent "
            f"specialists; fitted {len(bundles)}"
        )
    period = {
        "train_start": train_dates[0],
        "train_end": train_dates[-1],
        "train_days": len(train_dates),
        "train_rows": int(len(train)),
        "calibration_start": calibration_dates[0],
        "calibration_end": calibration_dates[-1],
        "calibration_days": len(calibration_dates),
        "calibration_rows": int(len(calibration)),
    }
    return bundles, audit, period


def compact_selection(
    selection: PolicySelection | None,
) -> dict[str, Any]:
    if selection is None:
        return {"policy": None, "reason": "not_run"}
    design = selection.design
    if "policies" in design:
        policies = design.get("policies") or []
        ranked = sorted(
            policies,
            key=lambda row: (
                float(row.get("mean_return_q_value") or 1.0),
                -float(row.get("clustered_mean_lower_pct") or -999.0),
            ),
        )[:5]
        compact_design = {
            "reason": design.get("reason"),
            "best_five_by_q_then_clustered_lower": ranked,
        }
    else:
        compact_design = design
    return {
        "policy": selection.policy.as_dict() if selection.policy else None,
        "design": compact_design,
        "confirmation": selection.confirmation,
        "search": {
            "design_evaluated": selection.design_evaluated,
            "design_gate_passed": selection.design_gate_passed,
            "confirmation_passed": selection.confirmation_passed,
        },
    }


def concat_or_empty(
    frames: list[pd.DataFrame],
    template: pd.DataFrame,
) -> pd.DataFrame:
    if not frames:
        return template.head(0).copy()
    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def assert_unique_identities(frame: pd.DataFrame, *, label: str) -> None:
    identity = ["trade_date", "signal_slot", "ts_code"]
    missing = [column for column in identity if column not in frame]
    if missing:
        raise RuntimeError(f"{label} missing identity columns: {missing}")
    duplicates = frame.duplicated(identity, keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, identity].head(5).to_dict(
            orient="records"
        )
        raise RuntimeError(
            f"{label} has {int(duplicates.sum())} duplicate identities: "
            f"{examples}"
        )


def artifact(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    return {
        "path": target.name,
        "sha256": file_sha256(target),
        "bytes": target.stat().st_size,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not pd.notna(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
