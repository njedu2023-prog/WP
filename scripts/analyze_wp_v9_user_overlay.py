from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from wp.v3.contracts import load_v3_config
from wp.v3.history import load_panel_partitions
from wp.v3.io import atomic_write_csv, atomic_write_json
from wp.v3.overlay import (
    attach_previous_limit_flags,
    build_limit_up_flags,
    overlay_mask,
    performance_summary,
    top_n_per_day,
)
from wp.v3.policy import apply_candidate_policy, select_candidate_policy
from wp.v3.sharding import (
    SHARD_MANIFEST_NAME,
    SHARD_PREDICTIONS_NAME,
)


ROOT = Path(__file__).resolve().parents[1]
PREDICTION_COLUMNS = (
    "trade_date",
    "signal_slot",
    "ts_code",
    "name",
    "fold",
    "net_return_pct",
    "ret_from_prev_close_pct",
    "execution_eligible",
    "entry_fillable",
    "exit_fillable",
    "target_net_positive",
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
OPTIONAL_PREDICTION_COLUMNS = ("data_age_seconds",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a user-specified overlay on immutable WP V9 OOS predictions."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--raw-cache-dir", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--signal-slot", default="14:20")
    parser.add_argument("--intraday-return-min-pct", type=float, default=7.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_v3_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    calendar = load_panel_partitions(
        args.panel_dir,
        columns=["trade_date"],
    )
    trade_dates = sorted(calendar["trade_date"].astype(str).unique())
    del calendar
    predictions, shard_audit = _load_predictions(
        args.shard_dir,
        signal_slot=args.signal_slot,
        evaluation_start=config.history.evaluation_start_date,
        evaluation_end=config.history.evaluation_end_date,
    )
    previous_dates = _previous_dates_needed(predictions, trade_dates)
    flags, raw_audit = _load_raw_limit_flags(
        args.raw_cache_dir,
        previous_dates,
    )
    predictions = attach_previous_limit_flags(
        predictions,
        flags,
        trade_dates=trade_dates,
    )

    ret = pd.to_numeric(
        predictions["ret_from_prev_close_pct"],
        errors="coerce",
    )
    execution = predictions["execution_eligible"].fillna(False).astype(bool)
    base = predictions.loc[
        ret.gt(args.intraday_return_min_pct) & execution
    ].copy()
    primary = predictions.loc[
        overlay_mask(
            predictions,
            signal_slot=args.signal_slot,
            intraday_return_min_pct=args.intraday_return_min_pct,
            previous_limit_mode="closed",
        )
    ].copy()
    strict = predictions.loc[
        overlay_mask(
            predictions,
            signal_slot=args.signal_slot,
            intraday_return_min_pct=args.intraday_return_min_pct,
            previous_limit_mode="touched",
        )
    ].copy()

    cohorts: list[dict[str, Any]] = []
    cohort_frames: dict[str, pd.DataFrame] = {
        "return_gt_7_all": base,
        "primary_prev_not_closed_limit_all": primary,
        "strict_prev_not_touched_limit_all": strict,
    }
    score_contracts = {
        "p_net_positive": "probability",
        "expected_utility_pct": "expected_utility",
        "selection_score": "selection",
    }
    for score_column, label in score_contracts.items():
        for count in (1, 3, 5):
            cohort_frames[f"primary_{label}_top{count}"] = top_n_per_day(
                primary,
                score_column=score_column,
                count=count,
            )
    for index, (name, frame) in enumerate(cohort_frames.items(), start=1):
        cohorts.append(
            {
                "cohort": name,
                **performance_summary(
                    frame,
                    config,
                    seed=config.model.random_seed + index,
                ),
            }
        )

    nested_candidates, nested_audit = _nested_overlay_candidates(
        predictions,
        config,
    )
    nested_metrics = performance_summary(
        nested_candidates,
        config,
        seed=config.model.random_seed + 10_000,
    )
    cohorts.append(
        {
            "cohort": "primary_nested_original_policy_grid",
            **nested_metrics,
        }
    )

    yearly_rows: list[dict[str, Any]] = []
    for cohort_name in (
        "primary_prev_not_closed_limit_all",
        "primary_probability_top1",
        "primary_expected_utility_top1",
        "primary_selection_top1",
    ):
        frame = cohort_frames[cohort_name].copy()
        frame["year"] = frame["trade_date"].astype(str).str[:4]
        for year, group in frame.groupby("year", sort=True):
            yearly_rows.append(
                {
                    "cohort": cohort_name,
                    "year": str(year),
                    **performance_summary(
                        group,
                        config,
                        bootstrap_samples=1_000,
                        seed=config.model.random_seed + int(year),
                    ),
                }
            )

    summary = {
        "schema_version": "wp_v9_user_overlay_1",
        "research_only": True,
        "model_retrained": False,
        "prediction_source": "immutable_wp_v9_walk_forward_oos_shards",
        "evaluation_start": config.history.evaluation_start_date,
        "evaluation_end": config.history.evaluation_end_date,
        "contract": {
            "signal_slot": args.signal_slot,
            "intraday_return_operator": ">",
            "intraday_return_min_pct": args.intraday_return_min_pct,
            "primary_previous_day_rule": "close_below_exact_up_limit",
            "sensitivity_previous_day_rule": "high_below_exact_up_limit",
            "entry_contract": config.execution.entry_price_contract,
            "exit_contract": config.execution.exit_order_contract,
            "baseline_cost_bps": config.execution.baseline_all_in_cost_bps,
            "stress_cost_bps": list(config.execution.stress_cost_bps),
        },
        "shards": shard_audit,
        "raw_truth": raw_audit,
        "cohorts": cohorts,
        "yearly": yearly_rows,
        "nested_policy": nested_audit,
        "conclusion": _conclusion(cohorts),
    }
    atomic_write_json(output / "wp_v9_user_overlay_summary.json", summary)
    atomic_write_csv(
        pd.json_normalize(cohorts, sep="."),
        output / "wp_v9_user_overlay_cohorts.csv",
    )
    atomic_write_csv(
        pd.json_normalize(yearly_rows, sep="."),
        output / "wp_v9_user_overlay_yearly.csv",
    )
    atomic_write_csv(
        nested_candidates,
        output / "wp_v9_user_overlay_nested_candidates.csv",
    )
    print(
        "WP_V9_USER_OVERLAY_RESULT="
        + json.dumps(
            {
                "contract": summary["contract"],
                "cohorts": cohorts,
                "nested_policy": {
                    "candidate_events": nested_metrics["events"],
                    "final": nested_audit["final"],
                },
                "conclusion": summary["conclusion"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def _load_predictions(
    shard_dir: str | Path,
    *,
    signal_slot: str,
    evaluation_start: str,
    evaluation_end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(shard_dir)
    manifests = sorted(root.rglob(SHARD_MANIFEST_NAME))
    if not manifests:
        raise FileNotFoundError(f"no shard manifests under {root}")
    frames: list[pd.DataFrame] = []
    folds: set[int] = set()
    rows = 0
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prediction_path = manifest_path.parent / SHARD_PREDICTIONS_NAME
        if _sha256(prediction_path) != manifest["prediction_sha256"]:
            raise RuntimeError(f"prediction digest mismatch: {prediction_path}")
        available = set(pq.ParquetFile(prediction_path).schema_arrow.names)
        missing = sorted(set(PREDICTION_COLUMNS) - available)
        if missing:
            raise RuntimeError(
                f"prediction shard missing overlay columns {missing}: "
                f"{prediction_path}"
            )
        retained_columns = [
            *PREDICTION_COLUMNS,
            *[
                column
                for column in OPTIONAL_PREDICTION_COLUMNS
                if column in available
            ],
        ]
        frame = pd.read_parquet(
            prediction_path,
            columns=retained_columns,
            filters=[
                ("signal_slot", "==", signal_slot),
                ("trade_date", ">=", evaluation_start),
                ("trade_date", "<=", evaluation_end),
            ],
        )
        frames.append(frame)
        rows += len(frame)
        folds.update(int(value) for value in manifest["produced_folds"])
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["fold", "trade_date", "ts_code"],
        kind="stable",
    )
    if combined.duplicated(["trade_date", "signal_slot", "ts_code"]).any():
        raise RuntimeError("overlay predictions contain duplicate identities")
    return combined.reset_index(drop=True), {
        "manifest_count": len(manifests),
        "fold_count": len(folds),
        "prediction_rows_at_slot": rows,
        "folds": sorted(folds),
    }


def _previous_dates_needed(
    predictions: pd.DataFrame,
    trade_dates: list[str],
) -> list[str]:
    mapping = {
        current: previous
        for previous, current in zip(
            trade_dates,
            trade_dates[1:],
            strict=False,
        )
    }
    return sorted(
        {
            mapping[date]
            for date in predictions["trade_date"].astype(str).unique()
            if date in mapping
        }
    )


def _load_raw_limit_flags(
    cache_dir: str | Path,
    trade_dates: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(cache_dir)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for trade_date in trade_dates:
        daily_path = _cache_file(
            root / "daily",
            trade_date,
            {"trade_date", "ts_code", "close", "high"},
        )
        limit_path = _cache_file(
            root / "stk_limit",
            trade_date,
            {"trade_date", "ts_code", "up_limit"},
        )
        if daily_path is None or limit_path is None:
            missing.append(trade_date)
            continue
        daily = pd.read_parquet(
            daily_path,
            columns=["trade_date", "ts_code", "close", "high"],
        )
        limits = pd.read_parquet(
            limit_path,
            columns=["trade_date", "ts_code", "up_limit"],
        )
        frames.append(build_limit_up_flags(daily, limits))
    coverage = len(frames) / len(trade_dates) if trade_dates else 0.0
    if coverage < 0.98:
        raise RuntimeError(
            f"exact prior-day limit truth coverage {coverage:.2%} is below 98%; "
            f"missing={missing[:20]}"
        )
    flags = pd.concat(frames, ignore_index=True)
    return flags, {
        "requested_trade_days": len(trade_dates),
        "covered_trade_days": len(frames),
        "coverage": coverage,
        "missing_trade_days": missing,
        "closed_limit_rows": int(flags["closed_up_limit"].sum()),
        "touched_limit_rows": int(flags["touched_up_limit"].sum()),
    }


def _cache_file(
    directory: Path,
    cache_key: str,
    required_columns: set[str],
) -> Path | None:
    for path in sorted(directory.glob(f"{cache_key}__*.parquet"), reverse=True):
        if required_columns.issubset(
            pq.ParquetFile(path).schema_arrow.names
        ):
            return path
    return None


def _nested_overlay_candidates(
    predictions: pd.DataFrame,
    config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ordered = predictions.sort_values(
        ["fold", "trade_date", "ts_code"],
        kind="stable",
    ).reset_index(drop=True)
    overlay = overlay_mask(
        ordered,
        signal_slot="14:20",
        intraday_return_min_pct=7.0,
        previous_limit_mode="closed",
    )
    total_policy_days = (
        config.model.policy_design_days
        + config.model.policy_confirmation_days
    )
    candidates: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    folds = sorted(
        pd.to_numeric(ordered["fold"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    for fold in folds:
        current = pd.to_numeric(ordered["fold"], errors="coerce").eq(fold)
        current_start = str(ordered.loc[current, "trade_date"].min())
        history = ordered.loc[
            pd.to_numeric(ordered["fold"], errors="coerce").lt(fold)
            & ordered["trade_date"].astype(str).lt(current_start)
        ]
        history_dates = sorted(history["trade_date"].astype(str).unique())
        if len(history_dates) < total_policy_days:
            audits.append(
                {
                    "fold": fold,
                    "reason": "insufficient_prior_oos_policy_days",
                    "authorized": False,
                }
            )
            continue
        selected_dates = history_dates[-total_policy_days:]
        split = config.model.policy_design_days
        design_mask = (
            history["trade_date"].astype(str).isin(selected_dates[:split])
            & overlay.reindex(history.index, fill_value=False)
        )
        confirmation_mask = (
            history["trade_date"].astype(str).isin(selected_dates[split:])
            & overlay.reindex(history.index, fill_value=False)
        )
        selection = select_candidate_policy(
            history.loc[design_mask],
            history.loc[confirmation_mask],
            config,
        )
        current_overlay = ordered.loc[
            current & overlay
        ].copy()
        passed = apply_candidate_policy(
            current_overlay,
            selection.policy,
            config,
        )
        if passed.any():
            candidates.append(current_overlay.loc[passed].copy())
        audits.append(
            {
                "fold": fold,
                "reason": selection.policy.reason,
                "authorized": selection.policy.authorized,
                "policy_id": selection.policy.policy_id,
                "design_events": selection.design.get("events", 0),
                "confirmation_events": selection.confirmation.get(
                    "events",
                    0,
                ),
            }
        )

    dates = sorted(ordered["trade_date"].astype(str).unique())
    if len(dates) >= total_policy_days:
        selected_dates = dates[-total_policy_days:]
        split = config.model.policy_design_days
        final = select_candidate_policy(
            ordered.loc[
                ordered["trade_date"].astype(str).isin(selected_dates[:split])
                & overlay
            ],
            ordered.loc[
                ordered["trade_date"].astype(str).isin(selected_dates[split:])
                & overlay
            ],
            config,
        )
        final_payload = final.as_dict()
    else:
        final_payload = {
            "policy": {
                "authorized": False,
                "reason": "insufficient_final_oos_policy_days",
            }
        }
    result = (
        pd.concat(candidates, ignore_index=True)
        if candidates
        else ordered.head(0).copy()
    )
    result = result.drop_duplicates(
        ["trade_date", "ts_code"],
        keep="first",
    )
    return result, {
        "folds": audits,
        "authorized_fold_count": sum(
            bool(item["authorized"]) for item in audits
        ),
        "final": final_payload,
    }


def _conclusion(cohorts: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {item["cohort"]: item for item in cohorts}
    primary = by_name["primary_prev_not_closed_limit_all"]
    nested = by_name["primary_nested_original_policy_grid"]
    return {
        "raw_overlay_positive_mean": (
            (primary.get("mean_net_return_pct") or 0.0) > 0
        ),
        "raw_overlay_clustered_lower_positive": (
            (
                primary.get(
                    "mean_net_return_day_clustered_lower_pct"
                )
                or -999.0
            )
            > 0
        ),
        "nested_policy_found_candidates": nested.get("events", 0) > 0,
        "economically_validated": (
            (primary.get("mean_net_return_pct") or 0.0) > 0
            and (
                primary.get(
                    "mean_net_return_day_clustered_lower_pct"
                )
                or -999.0
            )
            > 0
            and nested.get("events", 0) >= 250
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
