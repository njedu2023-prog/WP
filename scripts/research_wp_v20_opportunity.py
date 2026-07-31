from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from research_wp_v19_recall import (
    artifact,
    assert_unique,
    json_safe,
    load_recall_frontier,
    yearly_metrics,
)
from wp.v3.contracts import load_v3_config
from wp.v3.io import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
)
from wp.v3.v16_policy import policy_metrics
from wp.v3.v19_recall import (
    DEFAULT_EXPLORATION_PER_SLOT,
    DEFAULT_TOP_PER_SOURCE,
)
from wp.v3.v20_opportunity import (
    DEFAULT_LEADERS_PER_SLOT,
    FIXED_MAX_CANDIDATES_PER_DAY,
    FIXED_TARGET_CANDIDATE_DAY_RATE,
    GATE_CALIBRATION_DAYS,
    GATE_PURGE_DAYS,
    GATE_TRAIN_DAYS,
    SCHEMA_VERSION,
    FrozenOpportunityPolicy,
    OpportunityGateBundle,
    OpportunityPolicySpec,
    apply_opportunity_policy,
    build_opportunity_leaders,
    calibrate_opportunity_policy,
    fit_opportunity_gate,
    rolling_opportunity_model_segments,
    v20_research_readiness,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered V20 hierarchical opportunity gate over "
            "immutable V9 out-of-sample predictions."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--top-per-source",
        type=int,
        default=DEFAULT_TOP_PER_SOURCE,
    )
    parser.add_argument(
        "--exploration-per-slot",
        type=int,
        default=DEFAULT_EXPLORATION_PER_SLOT,
    )
    parser.add_argument(
        "--leaders-per-slot",
        type=int,
        default=DEFAULT_LEADERS_PER_SLOT,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_v3_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frontier, source = load_recall_frontier(
        args.shard_dir,
        evaluation_start="00000000",
        evaluation_end=config.history.evaluation_end_date,
        top_per_source=args.top_per_source,
        exploration_per_slot=args.exploration_per_slot,
    )
    leaders = build_opportunity_leaders(
        frontier,
        leaders_per_slot=args.leaders_per_slot,
    )
    assert_unique(leaders, "V20 V9 OOS leaders")
    evaluation_mask = leaders["trade_date"].astype(str).between(
        config.history.evaluation_start_date,
        config.history.evaluation_end_date,
    )
    evaluation_dates = sorted(
        leaders.loc[evaluation_mask, "trade_date"].astype(str).unique()
    )
    if not evaluation_dates:
        raise RuntimeError("V20 evaluation window has no leader rows")

    folds = sorted(
        pd.to_numeric(leaders.loc[evaluation_mask, "fold"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    numeric_fold = pd.to_numeric(leaders["fold"], errors="coerce")
    fold_rows: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    model_covered_dates: set[str] = set()
    temporal_integrity = True
    policy_spec = OpportunityPolicySpec(
        target_candidate_day_rate=FIXED_TARGET_CANDIDATE_DAY_RATE,
        max_candidates_per_day=FIXED_MAX_CANDIDATES_PER_DAY,
    )

    for fold in folds:
        fold_test = leaders.loc[numeric_fold.eq(fold)].copy()
        if fold_test.empty:
            continue
        fold_dates = sorted(fold_test["trade_date"].astype(str).unique())
        test_start = fold_dates[0]
        test_end = fold_dates[-1]
        test = fold_test.loc[
            fold_test["trade_date"].astype(str).between(
                config.history.evaluation_start_date,
                config.history.evaluation_end_date,
            )
        ].copy()
        test_dates = sorted(test["trade_date"].astype(str).unique())
        if not test_dates:
            continue
        overlap = model_covered_dates.intersection(test_dates)
        if overlap:
            raise RuntimeError(
                f"V20 outer test dates overlap: {sorted(overlap)[:5]}"
            )

        history = leaders.loc[
            leaders["trade_date"].astype(str).lt(test_start)
        ].copy()
        segments = rolling_opportunity_model_segments(
            history["trade_date"].astype(str).unique()
        )
        base = {
            "fold": int(fold),
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "test_rows": int(len(test)),
            "test_days": int(len(test_dates)),
        }
        if segments is None:
            fold_rows.append(
                {
                    **base,
                    "scored": False,
                    "reason": "insufficient_prior_oos_gate_history",
                    "selected": policy_metrics(
                        test.head(0),
                        total_days=len(test_dates),
                        seed=config.model.random_seed + int(fold),
                        bootstrap_samples=args.bootstrap_samples,
                    ),
                }
            )
            continue

        train_dates, calibration_dates = segments
        train = labeled_rows(
            history.loc[
                history["trade_date"].astype(str).isin(train_dates)
            ]
        )
        calibration = labeled_rows(
            history.loc[
                history["trade_date"].astype(str).isin(calibration_dates)
            ]
        )
        fold_temporal_integrity = bool(
            train_dates[-1] < calibration_dates[0]
            and calibration_dates[-1] < test_start
        )
        temporal_integrity &= fold_temporal_integrity
        if not fold_temporal_integrity:
            raise RuntimeError(
                f"V20 fold {fold} historical evidence crosses test start"
            )

        print(
            f"[wp-v20] fold={fold} "
            f"train={train_dates[0]}..{train_dates[-1]} rows={len(train):,} "
            f"calibration={calibration_dates[0]}.."
            f"{calibration_dates[-1]} rows={len(calibration):,} "
            f"test={test_dates[0]}..{test_dates[-1]} rows={len(test):,}",
            flush=True,
        )
        bundle = fit_opportunity_gate(
            train,
            calibration,
            random_seed=config.model.random_seed + int(fold) * 20_003,
        )
        scored_calibration = bundle.predict(calibration)
        policy = calibrate_opportunity_policy(
            scored_calibration,
            calibration_dates=calibration_dates,
            spec=policy_spec,
        )
        scored_test = bundle.predict(test)
        scored_test["v20_source_fold"] = int(fold)
        selected = apply_opportunity_policy(scored_test, policy)
        selected["v20_source_fold"] = int(fold)
        validate_selected_contract(selected, policy)

        scored_frames.append(scored_test)
        if not selected.empty:
            selected_frames.append(selected)
        model_covered_dates.update(test_dates)
        selected_metrics = policy_metrics(
            selected,
            total_days=len(test_dates),
            seed=config.model.random_seed + int(fold) * 59,
            bootstrap_samples=args.bootstrap_samples,
        )
        fold_rows.append(
            {
                **base,
                "scored": True,
                "reason": "fixed_prior_oos_gate_applied",
                "temporal_integrity": fold_temporal_integrity,
                "model": {
                    "train_start": train_dates[0],
                    "train_end": train_dates[-1],
                    "train_days": len(train_dates),
                    "train_rows": len(train),
                    "calibration_start": calibration_dates[0],
                    "calibration_end": calibration_dates[-1],
                    "calibration_days": len(calibration_dates),
                    "calibration_rows": len(calibration),
                    "feature_count": len(bundle.feature_columns),
                    "features": list(bundle.feature_columns),
                },
                "policy": policy.as_dict(),
                "selected": selected_metrics,
            }
        )
        print(
            f"[wp-v20] fold={fold} "
            f"threshold={policy.score_threshold:.6f} "
            f"events={selected_metrics['events']} "
            f"days={selected_metrics['candidate_days']} "
            f"win={selected_metrics['win_rate']:.4f} "
            f"mean={selected_metrics['mean_net_return_pct']}",
            flush=True,
        )

    scored_all = concat_or_empty(scored_frames, leaders)
    selected_all = concat_or_empty(selected_frames, scored_all)
    assert_unique(scored_all, "V20 nested OOS scored leaders")
    assert_unique(selected_all, "V20 nested OOS candidates")
    validate_selected_contract(selected_all, None)
    nested_metrics = policy_metrics(
        selected_all,
        total_days=len(evaluation_dates),
        seed=config.model.random_seed + 20_000,
        bootstrap_samples=max(4_000, args.bootstrap_samples),
    )
    yearly = yearly_metrics(
        selected_all,
        total_dates=evaluation_dates,
        seed=config.model.random_seed + 20,
        bootstrap_samples=args.bootstrap_samples,
    )
    readiness = v20_research_readiness(
        nested_metrics,
        yearly=yearly,
        temporal_integrity=temporal_integrity,
        source_integrity=bool(source["source_integrity"]),
    )
    final_bundle, final_policy, final_model = fit_final_gate(
        leaders,
        random_seed=config.model.random_seed,
        spec=policy_spec,
    )
    final_bundle_path = output / "wp_v20_frozen_research_bundle.joblib"
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "research_only": True,
            "production_authorized": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "gate": final_bundle,
            "policy": final_policy,
            "source": source,
            "protocol": {
                "leaders_per_slot": args.leaders_per_slot,
                "target_candidate_day_rate": (
                    policy_spec.target_candidate_day_rate
                ),
                "max_candidates_per_day": (
                    policy_spec.max_candidates_per_day
                ),
            },
        },
        final_bundle_path,
        compress=3,
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "production_authorized": False,
        "objective": (
            "Maximize positive net return probability for executable "
            "14:20-14:50 candidates under the fixed T+1 close exit after "
            "all established costs."
        ),
        "evaluation_start": config.history.evaluation_start_date,
        "evaluation_end": config.history.evaluation_end_date,
        "evaluation_days": len(evaluation_dates),
        "model_covered_days": len(model_covered_dates),
        "model_coverage_rate": (
            len(model_covered_dates) / max(len(evaluation_dates), 1)
        ),
        "same_historical_window_already_explored": True,
        "historical_result_role": (
            "research_screen_only; cannot replace future 150-day shadow"
        ),
        "execution_contract": {
            "entry": config.execution.entry_price_contract,
            "exit": config.execution.exit_order_contract,
            "baseline_all_in_cost_bps": (
                config.execution.baseline_all_in_cost_bps
            ),
            "stress_cost_bps": list(config.execution.stress_cost_bps),
            "failed_exit_penalty_pct": (
                config.execution.non_fill_penalty_pct
            ),
        },
        "protocol": {
            "mechanism": (
                "V9 stock ranking followed by a prior-OOS hierarchical "
                "opportunity gate"
            ),
            "leaders_per_slot": args.leaders_per_slot,
            "gate_train_days": GATE_TRAIN_DAYS,
            "gate_calibration_days": GATE_CALIBRATION_DAYS,
            "purge_days": GATE_PURGE_DAYS,
            "policy_family_size": 1,
            "fixed_target_candidate_day_rate": (
                policy_spec.target_candidate_day_rate
            ),
            "fixed_max_candidates_per_day": (
                policy_spec.max_candidates_per_day
            ),
            "first_qualifying_signal_is_immutable": True,
            "no_trade_allowed": True,
            "future_information_allowed": False,
            "post_result_threshold_search_allowed": False,
        },
        "source": source,
        "frontier_rows": int(len(frontier)),
        "leader_rows": int(len(leaders)),
        "folds": fold_rows,
        "nested_oos_metrics": nested_metrics,
        "yearly": yearly,
        "temporal_integrity": temporal_integrity,
        "research_readiness": readiness,
        "final_model": final_model,
        "final_policy": final_policy.as_dict(),
        "frozen_bundle": artifact(final_bundle_path),
    }
    atomic_write_json(output / "wp_v20_research_summary.json", summary)
    atomic_write_csv(
        pd.json_normalize(fold_rows, sep="."),
        output / "wp_v20_folds.csv",
    )
    atomic_write_parquet(
        selected_all,
        output / "wp_v20_nested_oos_candidates.parquet",
    )
    atomic_write_csv(
        selected_all,
        output / "wp_v20_nested_oos_candidates.csv",
    )
    atomic_write_csv(
        pd.json_normalize(yearly, sep="."),
        output / "wp_v20_yearly.csv",
    )
    atomic_write_csv(
        pd.DataFrame(source["shards"]),
        output / "wp_v20_source_shards.csv",
    )
    print(
        "WP_V20_RESULT="
        + json.dumps(
            json_safe(
                {
                    "evaluation_days": len(evaluation_dates),
                    "model_covered_days": len(model_covered_dates),
                    "leader_rows": int(len(leaders)),
                    "nested_oos_metrics": nested_metrics,
                    "yearly": yearly,
                    "research_readiness": readiness,
                    "final_policy": final_policy.as_dict(),
                }
            ),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def labeled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    net = pd.to_numeric(frame.get("net_return_pct"), errors="coerce")
    positive = pd.to_numeric(
        frame.get("target_net_positive"),
        errors="coerce",
    )
    return frame.loc[net.notna() & positive.notna()].copy()


def fit_final_gate(
    leaders: pd.DataFrame,
    *,
    random_seed: int,
    spec: OpportunityPolicySpec,
) -> tuple[
    OpportunityGateBundle,
    FrozenOpportunityPolicy,
    dict[str, Any],
]:
    segments = rolling_opportunity_model_segments(
        leaders["trade_date"].astype(str).unique(),
        reserve_final_purge=False,
    )
    if segments is None:
        raise RuntimeError("V20 final gate has insufficient OOS history")
    train_dates, calibration_dates = segments
    train = labeled_rows(
        leaders.loc[
            leaders["trade_date"].astype(str).isin(train_dates)
        ]
    )
    calibration = labeled_rows(
        leaders.loc[
            leaders["trade_date"].astype(str).isin(calibration_dates)
        ]
    )
    bundle = fit_opportunity_gate(
        train,
        calibration,
        random_seed=random_seed + 200_020,
    )
    scored_calibration = bundle.predict(calibration)
    policy = calibrate_opportunity_policy(
        scored_calibration,
        calibration_dates=calibration_dates,
        spec=spec,
    )
    return bundle, policy, {
        "train_start": train_dates[0],
        "train_end": train_dates[-1],
        "train_days": len(train_dates),
        "train_rows": len(train),
        "calibration_start": calibration_dates[0],
        "calibration_end": calibration_dates[-1],
        "calibration_days": len(calibration_dates),
        "calibration_rows": len(calibration),
        "feature_count": len(bundle.feature_columns),
        "features": list(bundle.feature_columns),
        "research_only": True,
        "production_authorized": False,
    }


def validate_selected_contract(
    selected: pd.DataFrame,
    policy: FrozenOpportunityPolicy | None,
) -> None:
    if selected.empty:
        return
    stock_day_duplicates = selected.duplicated(
        ["trade_date", "ts_code"],
        keep=False,
    )
    if stock_day_duplicates.any():
        raise RuntimeError(
            "V20 selected output rewrote a first qualifying signal"
        )
    maximum = (
        policy.spec.max_candidates_per_day
        if policy is not None
        else FIXED_MAX_CANDIDATES_PER_DAY
    )
    per_day = selected.groupby("trade_date", sort=False).size()
    if int(per_day.max()) > maximum:
        raise RuntimeError("V20 selected output exceeds fixed daily maximum")
    slot_minute = selected["signal_slot"].astype(str).str.replace(":", "")
    if not slot_minute.between("1420", "1450").all():
        raise RuntimeError("V20 selected output contains an illegal signal slot")


def concat_or_empty(
    frames: list[pd.DataFrame],
    template: pd.DataFrame,
) -> pd.DataFrame:
    if not frames:
        return template.head(0).copy()
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    raise SystemExit(main())
