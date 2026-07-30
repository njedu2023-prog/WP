from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wp.v3.io import atomic_write_csv, atomic_write_json, file_sha256
from wp.v3.v16_policy import ExpertPolicy, apply_expert_policy


DIAGNOSIS_SCHEMA_VERSION = "wp_v16_evidence_diagnosis_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose a frozen V16 nested-OOS evidence package."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    summary_path = find_one(source, "wp_v16_research_summary.json")
    scored_path = find_one(source, "wp_v16_expert_scored_oos.parquet")
    frontier_path = find_one(
        source,
        "wp_v16_frequency_profit_frontier.csv",
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scored = pd.read_parquet(scored_path)
    frontier = pd.read_csv(frontier_path)
    diagnosis, ranked = diagnose_evidence(summary, scored, frontier)
    ranked_path = atomic_write_csv(
        ranked,
        output / "wp_v16_policy_near_misses.csv",
    )
    diagnosis["artifacts"] = {
        "source_summary_sha256": file_sha256(summary_path),
        "source_scored_sha256": file_sha256(scored_path),
        "source_frontier_sha256": file_sha256(frontier_path),
        "near_misses": {
            "path": ranked_path.name,
            "sha256": file_sha256(ranked_path),
        },
    }
    atomic_write_json(output / "wp_v16_evidence_diagnosis.json", diagnosis)
    print(
        "WP_V16_DIAGNOSIS_RESULT="
        + json.dumps(
            diagnosis,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def diagnose_evidence(
    summary: dict[str, Any],
    scored: pd.DataFrame,
    frontier: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    permissive = ExpertPolicy(
        probability_lower_min=0.52,
        expected_return_lower_min_pct=0.00,
        severe_loss_max=0.35,
        round_trip_fill_min=0.95,
        minimum_experts=1,
        probability_spread_max=0.18,
        score_rank_min=0.90,
        max_candidates_per_day=3,
        slot_group="all",
    )
    attrition = policy_attrition(scored, permissive)
    ranked = rank_frontier(frontier)
    finite = frontier.replace([np.inf, -np.inf], np.nan).copy()
    positive_mean = _numeric(finite, "mean_net_return_pct").gt(0)
    stress_nonnegative = _numeric(
        finite,
        "stress_50bps_mean_net_return_pct",
    ).ge(0)
    q_significant = _numeric(
        finite,
        "mean_return_q_value",
    ).le(0.10)
    enough_events = _numeric(finite, "events").ge(20)
    folds = summary.get("folds") or []
    fold_diagnostics = []
    for row in folds:
        if not row.get("scored"):
            continue
        selection = row.get("policy_selection") or {}
        search = selection.get("search") or {}
        fold_diagnostics.append(
            {
                "fold": row.get("fold"),
                "test_start": row.get("test_start"),
                "test_end": row.get("test_end"),
                "reason": row.get("reason"),
                "design_evaluated": int(
                    search.get("design_evaluated") or 0
                ),
                "design_gate_passed": int(
                    search.get("design_gate_passed") or 0
                ),
                "confirmation_passed": bool(
                    search.get("confirmation_passed")
                ),
                "test_events": int((row.get("test") or {}).get("events") or 0),
            }
        )
    diagnosis = {
        "schema_version": DIAGNOSIS_SCHEMA_VERSION,
        "research_only": True,
        "source_schema_version": summary.get("schema_version"),
        "source_production_authorized": bool(
            summary.get("production_authorized")
        ),
        "scored": {
            "rows": int(len(scored)),
            "trade_days": int(scored["trade_date"].astype(str).nunique()),
            "folds": int(
                pd.to_numeric(
                    scored.get("expert_source_fold"),
                    errors="coerce",
                ).nunique()
            ),
            "feature_quantiles": feature_quantiles(scored),
        },
        "most_permissive_grid_policy": {
            "policy": permissive.as_dict(),
            "attrition": attrition,
        },
        "policy_family": {
            "evaluated": int(len(finite)),
            "with_at_least_20_events": int(enough_events.sum()),
            "positive_mean": int(positive_mean.sum()),
            "positive_mean_with_20_events": int(
                (positive_mean & enough_events).sum()
            ),
            "stress_50bps_nonnegative": int(stress_nonnegative.sum()),
            "bh_q_le_10pct": int(q_significant.sum()),
            "positive_mean_and_bh_q_le_10pct": int(
                (positive_mean & q_significant).sum()
            ),
            "maximum_events": _finite_max(finite, "events"),
            "maximum_candidate_days": _finite_max(
                finite,
                "candidate_days",
            ),
            "best_mean_net_return_pct_with_20_events": _conditional_max(
                finite,
                "mean_net_return_pct",
                enough_events,
            ),
            "best_win_rate_with_20_events": _conditional_max(
                finite,
                "win_rate",
                enough_events,
            ),
            "best_stress_50bps_mean_with_20_events": _conditional_max(
                finite,
                "stress_50bps_mean_net_return_pct",
                enough_events,
            ),
        },
        "nested_policy_folds": fold_diagnostics,
        "bottleneck": classify_bottleneck(
            attrition,
            positive_mean_with_sample=int(
                (positive_mean & enough_events).sum()
            ),
            stress_nonnegative=int(stress_nonnegative.sum()),
            q_significant=int(q_significant.sum()),
        ),
        "next_action_contract": (
            "Do not retrain experts or authorize production. Redesign only "
            "the policy-selection layer over immutable nested-OOS scores, "
            "then repeat untouched confirmation and final readiness gates."
        ),
    }
    return diagnosis, ranked


def policy_attrition(
    frame: pd.DataFrame,
    policy: ExpertPolicy,
) -> list[dict[str, Any]]:
    working = frame.copy()
    if "expert_score_rank_pct" not in working:
        working["expert_score_rank_pct"] = working.groupby(
            ["trade_date", "signal_slot"],
            sort=False,
        )["expert_score"].rank(method="average", pct=True)
    stages = [
        ("source", pd.Series(True, index=working.index)),
        (
            "minimum_experts",
            _numeric(working, "expert_count").ge(policy.minimum_experts),
        ),
        (
            "probability_lower",
            _numeric(working, "expert_p_positive_lower").ge(
                policy.probability_lower_min
            ),
        ),
        (
            "expected_return_lower",
            _numeric(working, "expert_expected_return_lower_pct").ge(
                policy.expected_return_lower_min_pct
            ),
        ),
        (
            "severe_loss",
            _numeric(working, "expert_p_severe").le(
                policy.severe_loss_max
            ),
        ),
        (
            "round_trip_fill",
            _numeric(working, "p_round_trip_fill_lower").ge(
                policy.round_trip_fill_min
            ),
        ),
        (
            "expert_agreement",
            _numeric(working, "expert_probability_spread").le(
                policy.probability_spread_max
            ),
        ),
        (
            "score_rank",
            _numeric(working, "expert_score_rank_pct").ge(
                policy.score_rank_min
            ),
        ),
    ]
    mask = pd.Series(True, index=working.index, dtype=bool)
    rows = []
    for stage, condition in stages:
        mask &= condition.fillna(False)
        kept = working.loc[mask]
        rows.append(
            {
                "stage": stage,
                "rows": int(len(kept)),
                "trade_days": int(
                    kept["trade_date"].astype(str).nunique()
                ),
            }
        )
    selected = apply_expert_policy(frame, policy)
    rows.append(
        {
            "stage": "first_signal_and_daily_top_k",
            "rows": int(len(selected)),
            "trade_days": int(
                selected["trade_date"].astype(str).nunique()
            ),
        }
    )
    return rows


def feature_quantiles(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    result = {}
    for column in (
        "expert_count",
        "expert_p_positive_lower",
        "expert_expected_return_lower_pct",
        "expert_p_severe",
        "p_round_trip_fill_lower",
        "expert_probability_spread",
        "expert_score",
    ):
        values = _numeric(frame, column).dropna()
        result[column] = {
            str(quantile): (
                float(values.quantile(quantile))
                if not values.empty
                else None
            )
            for quantile in (0.10, 0.50, 0.90, 0.99)
        }
    return result


def rank_frontier(frontier: pd.DataFrame) -> pd.DataFrame:
    ranked = frontier.replace([np.inf, -np.inf], np.nan).copy()
    for column in (
        "events",
        "candidate_days",
        "candidate_day_rate",
        "win_rate",
        "win_rate_wilson_lower",
        "mean_net_return_pct",
        "profit_factor",
        "stress_50bps_mean_net_return_pct",
        "mean_return_p_value",
        "clustered_mean_lower_pct",
    ):
        if column in ranked:
            ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    ranked["_sample_ok"] = ranked["events"].ge(20)
    ranked.sort_values(
        [
            "_sample_ok",
            "clustered_mean_lower_pct",
            "win_rate_wilson_lower",
            "mean_net_return_pct",
            "candidate_days",
        ],
        ascending=[False, False, False, False, False],
        kind="stable",
        inplace=True,
    )
    return ranked.drop(columns=["_sample_ok"]).head(100).reset_index(drop=True)


def classify_bottleneck(
    attrition: list[dict[str, Any]],
    *,
    positive_mean_with_sample: int,
    stress_nonnegative: int,
    q_significant: int,
) -> str:
    if attrition[-1]["rows"] == 0:
        return "fixed_thresholds_eliminate_all_candidates"
    if positive_mean_with_sample == 0:
        return "no_positive_mean_policy_with_minimum_sample"
    if stress_nonnegative == 0:
        return "edge_smaller_than_additional_50bps_stress"
    if q_significant == 0:
        return "multiple_testing_gate_dominates_short_design_window"
    return "nested_design_confirmation_gate_is_primary_bottleneck"


def find_one(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one {name} under {root}, found {len(matches)}"
        )
    return matches[0]


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _finite_max(frame: pd.DataFrame, column: str) -> float | None:
    values = _numeric(frame, column).dropna()
    return float(values.max()) if not values.empty else None


def _conditional_max(
    frame: pd.DataFrame,
    column: str,
    mask: pd.Series,
) -> float | None:
    values = _numeric(frame.loc[mask], column).dropna()
    return float(values.max()) if not values.empty else None


if __name__ == "__main__":
    raise SystemExit(main())
