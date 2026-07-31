from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from wp.v3.io import atomic_write_csv, atomic_write_json, file_sha256
from wp.v3.v24_cross_section import MODEL_FEATURES


SCHEMA_VERSION = "wp_v24_failure_attribution_1"
STRESS_BPS = 50
MIN_FEATURE_ROWS = 500
MIN_CROSS_SECTIONS = 100

PREDICTION_TARGETS = (
    ("v23_p_positive", 0.0, "above", "positive"),
    ("v23_p_margin", 0.5, "above", "margin"),
    ("v23_p_severe_loss", -2.0, "below", "severe_loss"),
)
PREDICTION_RETURN_COLUMNS = (
    "v23_expected_net_return_pct",
    "v23_expected_net_return_lower_pct",
    "v23_economic_score",
    "v24_cross_section_score",
)
CATEGORICAL_DIMENSIONS = (
    "year",
    "signal_slot",
    "v24_source_fold",
    "v20_stock_rank_in_slot",
    "v20_leader_appearances_so_far",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute the frozen V24 failure without selecting or "
            "authorizing a successor policy."
        )
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    summary_path = find_one(source, "wp_v24_research_summary.json")
    scored_path = find_one(
        source,
        "wp_v24_nested_oos_scored_candidates.parquet",
    )
    selected_path = find_one(
        source,
        "wp_v24_nested_oos_candidates.csv",
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scored = pd.read_parquet(scored_path)
    selected = pd.read_csv(selected_path)
    diagnosis, feature_table, prediction_table, group_table = diagnose(
        summary,
        scored,
        selected,
    )

    feature_path = atomic_write_csv(
        feature_table,
        output / "wp_v24_causal_feature_diagnostics.csv",
    )
    prediction_path = atomic_write_csv(
        prediction_table,
        output / "wp_v24_prediction_diagnostics.csv",
    )
    group_path = atomic_write_csv(
        group_table,
        output / "wp_v24_selected_group_attribution.csv",
    )
    diagnosis["artifacts"] = {
        "source_summary_sha256": file_sha256(summary_path),
        "source_scored_sha256": file_sha256(scored_path),
        "source_selected_sha256": file_sha256(selected_path),
        "causal_feature_diagnostics_sha256": file_sha256(feature_path),
        "prediction_diagnostics_sha256": file_sha256(prediction_path),
        "selected_group_attribution_sha256": file_sha256(group_path),
    }
    atomic_write_json(
        output / "wp_v24_failure_attribution.json",
        diagnosis,
    )
    print(
        "WP_V24_DIAGNOSIS_RESULT="
        + json.dumps(
            json_safe(
                {
                    "overall": diagnosis["overall"],
                    "model_discrimination": diagnosis[
                        "model_discrimination"
                    ],
                    "threshold_drift": diagnosis["threshold_drift"],
                    "stable_causal_features": diagnosis[
                        "stable_causal_features"
                    ],
                    "feature_family_summary": diagnosis[
                        "feature_family_summary"
                    ],
                    "exploratory_positive_selected_groups": diagnosis[
                        "exploratory_positive_selected_groups"
                    ],
                    "next_research_decision": diagnosis[
                        "next_research_decision"
                    ],
                }
            ),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def diagnose(
    summary: dict[str, Any],
    scored: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame | dict[str, Any], ...]:
    required = {
        "trade_date",
        "signal_slot",
        "ts_code",
        "net_return_pct",
    }
    missing_scored = sorted(required - set(scored.columns))
    missing_selected = sorted(required - set(selected.columns))
    if missing_scored or missing_selected:
        raise ValueError(
            "V24 diagnosis inputs missing columns: "
            f"scored={missing_scored} selected={missing_selected}"
        )

    scored_frame = prepare_frame(scored)
    selected_frame = prepare_frame(selected)
    expected_selected = int(
        (summary.get("nested_oos_metrics") or {}).get("events") or 0
    )
    if len(selected_frame) != expected_selected:
        raise RuntimeError(
            "V24 selected count mismatch: "
            f"{len(selected_frame)} != {expected_selected}"
        )
    if selected_frame.duplicated(
        ["trade_date", "signal_slot", "ts_code"],
        keep=False,
    ).any():
        raise RuntimeError("V24 selected evidence has duplicate identities")

    predictions = prediction_diagnostics(scored_frame)
    features = causal_feature_diagnostics(scored_frame)
    groups = selected_group_attribution(selected_frame)
    stable = features.loc[features["stable_exploratory_signal"]].copy()
    stable.sort_values(
        [
            "daily_slot_ic_bh_q",
            "mean_daily_slot_ic",
            "mean_daily_top_minus_bottom_return_pct",
        ],
        ascending=[True, False, False],
        kind="stable",
        inplace=True,
    )
    family = feature_family_summary(features)
    positive_groups = groups.loc[
        groups["events"].ge(20)
        & groups["stress_50bps_mean_net_return_pct"].ge(0.0)
    ].sort_values(
        ["stress_50bps_mean_net_return_pct", "events"],
        ascending=[False, False],
        kind="stable",
    )
    thresholds = threshold_drift(summary)
    overall = {
        "scored_rows": int(len(scored_frame)),
        "scored_days": int(scored_frame["trade_date"].nunique()),
        "selected": simple_metrics(selected_frame),
        "selected_loss_concentration": loss_concentration(selected_frame),
        "causal_features_evaluated": int(len(features)),
        "prediction_outputs_evaluated": int(len(predictions)),
        "selected_groups_evaluated": int(len(groups)),
    }
    predictive_rows = predictions.loc[
        predictions["metric"].isin(
            ["roc_auc", "spearman_return_correlation"]
        )
    ].copy()
    model_discrimination = records(
        predictive_rows.sort_values(
            ["metric", "value"],
            ascending=[True, False],
            kind="stable",
        )
    )
    next_decision = (
        "Preregister a new parsimonious ranking hypothesis only after "
        "separating discovery from confirmation; the listed causal features "
        "remain exploratory and cannot authorize a policy."
        if len(stable)
        else (
            "Do not build another selector from the existing V9-V24 feature "
            "family. No stable within-slot causal rank signal survived the "
            "diagnostic contract; acquire an independent point-in-time "
            "information family and use future shadow evidence."
        )
    )
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "research_only": True,
            "production_authorized": False,
            "shadow_authorized": False,
            "successor_policy_authorized": False,
            "outcome_driven_threshold_selection_allowed": False,
            "source_schema_version": summary.get("schema_version"),
            "overall": overall,
            "model_discrimination": model_discrimination,
            "threshold_drift": thresholds,
            "stable_causal_features": records(stable.head(20)),
            "feature_family_summary": records(family),
            "exploratory_positive_selected_groups": records(
                positive_groups.head(20)
            ),
            "next_research_decision": next_decision,
            "interpretation_contract": (
                "All feature and subgroup findings are post-result failure "
                "attribution. They may formulate a separately preregistered "
                "hypothesis but cannot authorize, tune, or rescue V24."
            ),
        },
        features,
        predictions,
        groups,
    )


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["trade_date"] = result["trade_date"].astype(str)
    result["signal_slot"] = result["signal_slot"].astype(str)
    result["ts_code"] = result["ts_code"].astype(str)
    result["year"] = result["trade_date"].str[:4]
    target = numeric(result, "net_return_pct")
    return result.loc[target.notna()].copy()


def prediction_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    returns = numeric(frame, "net_return_pct")
    for column, threshold, direction, label in PREDICTION_TARGETS:
        if column not in frame:
            continue
        score = numeric(frame, column)
        clean = score.notna() & returns.notna()
        target = (
            returns.le(threshold)
            if direction == "below"
            else returns.gt(threshold)
        ).astype(int)
        if clean.sum() < MIN_FEATURE_ROWS or target.loc[clean].nunique() < 2:
            continue
        rows.extend(
            [
                {
                    "output": column,
                    "target": label,
                    "scope": "overall",
                    "metric": "roc_auc",
                    "events": int(clean.sum()),
                    "value": binary_auc(
                        target.loc[clean],
                        score.loc[clean],
                    ),
                },
                {
                    "output": column,
                    "target": label,
                    "scope": "overall",
                    "metric": "brier_score",
                    "events": int(clean.sum()),
                    "value": float(
                        np.mean(
                            (
                                score.loc[clean].clip(0.0, 1.0)
                                - target.loc[clean]
                            )
                            ** 2
                        )
                    ),
                },
            ]
        )
        for year, indexes in frame.loc[clean].groupby("year").groups.items():
            year_target = target.loc[indexes]
            if len(indexes) < 100 or year_target.nunique() < 2:
                continue
            rows.append(
                {
                    "output": column,
                    "target": label,
                    "scope": str(year),
                    "metric": "roc_auc",
                    "events": int(len(indexes)),
                    "value": binary_auc(
                        year_target,
                        score.loc[indexes],
                    ),
                }
            )
    for column in PREDICTION_RETURN_COLUMNS:
        if column not in frame:
            continue
        values = numeric(frame, column)
        clean = values.notna() & returns.notna()
        if clean.sum() < MIN_FEATURE_ROWS or values.loc[clean].nunique() < 2:
            continue
        rows.append(
            {
                "output": column,
                "target": "net_return_pct",
                "scope": "overall",
                "metric": "spearman_return_correlation",
                "events": int(clean.sum()),
                "value": float(
                    values.loc[clean].corr(
                        returns.loc[clean],
                        method="spearman",
                    )
                ),
            }
        )
        for year, indexes in frame.loc[clean].groupby("year").groups.items():
            if len(indexes) < 100:
                continue
            rows.append(
                {
                    "output": column,
                    "target": "net_return_pct",
                    "scope": str(year),
                    "metric": "spearman_return_correlation",
                    "events": int(len(indexes)),
                    "value": float(
                        values.loc[indexes].corr(
                            returns.loc[indexes],
                            method="spearman",
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def causal_feature_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target = numeric(frame, "net_return_pct")
    years = (
        frame["year"].astype(str)
        if "year" in frame
        else frame["trade_date"].astype(str).str[:4]
    )
    for feature in MODEL_FEATURES:
        if feature not in frame:
            continue
        values = numeric(frame, feature)
        clean = values.notna() & target.notna()
        if (
            clean.sum() < MIN_FEATURE_ROWS
            or values.loc[clean].nunique() < 5
        ):
            continue
        section_rows: list[dict[str, Any]] = []
        subset = frame.loc[clean, ["trade_date", "signal_slot"]].copy()
        subset["year"] = years.loc[clean]
        subset["_feature"] = values.loc[clean]
        subset["_target"] = target.loc[clean]
        for (trade_date, signal_slot), group in subset.groupby(
            ["trade_date", "signal_slot"],
            sort=False,
        ):
            if len(group) < 4 or group["_feature"].nunique() < 2:
                continue
            correlation = group["_feature"].corr(
                group["_target"],
                method="spearman",
            )
            ordered = group.sort_values(
                ["_feature"],
                kind="stable",
            )
            section_rows.append(
                {
                    "trade_date": str(trade_date),
                    "signal_slot": str(signal_slot),
                    "year": str(group["year"].iloc[0]),
                    "ic": float(correlation),
                    "spread": float(
                        ordered["_target"].iloc[-1]
                        - ordered["_target"].iloc[0]
                    ),
                }
            )
        sections = pd.DataFrame(section_rows)
        if sections.empty:
            continue
        daily = (
            sections.groupby(["trade_date", "year"], as_index=False)
            .agg(ic=("ic", "mean"), spread=("spread", "mean"))
        )
        p_value = mean_zero_p_value(daily["ic"])
        yearly = (
            daily.groupby("year", sort=True)
            .agg(
                mean_ic=("ic", "mean"),
                mean_spread=("spread", "mean"),
                days=("trade_date", "nunique"),
            )
            .reset_index()
        )
        rows.append(
            {
                "feature": feature,
                "feature_family": feature_family(feature),
                "events": int(clean.sum()),
                "cross_sections": int(len(sections)),
                "days": int(len(daily)),
                "overall_spearman_return_correlation": float(
                    values.loc[clean].corr(
                        target.loc[clean],
                        method="spearman",
                    )
                ),
                "mean_daily_slot_ic": float(daily["ic"].mean()),
                "median_daily_slot_ic": float(daily["ic"].median()),
                "positive_daily_ic_share": float(daily["ic"].gt(0.0).mean()),
                "daily_slot_ic_p_value": p_value,
                "mean_daily_top_minus_bottom_return_pct": float(
                    daily["spread"].mean()
                ),
                "positive_years": int(yearly["mean_ic"].gt(0.0).sum()),
                "negative_years": int(yearly["mean_ic"].lt(0.0).sum()),
                "minimum_year_ic": float(yearly["mean_ic"].min()),
                "minimum_year_spread_pct": float(
                    yearly["mean_spread"].min()
                ),
                "year_metrics": json.dumps(
                    records(yearly),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["daily_slot_ic_bh_q"] = benjamini_hochberg(
        result["daily_slot_ic_p_value"]
    )
    result["stable_exploratory_signal"] = (
        result["cross_sections"].ge(MIN_CROSS_SECTIONS)
        & result["mean_daily_slot_ic"].ge(0.05)
        & result["daily_slot_ic_bh_q"].le(0.10)
        & result["mean_daily_top_minus_bottom_return_pct"].ge(0.20)
        & result["positive_years"].ge(3)
        & result["minimum_year_ic"].gt(-0.02)
        & result["minimum_year_spread_pct"].gt(-0.25)
    )
    result.sort_values(
        [
            "stable_exploratory_signal",
            "daily_slot_ic_bh_q",
            "mean_daily_slot_ic",
        ],
        ascending=[False, True, False],
        kind="stable",
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)
    return result


def selected_group_attribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dimension in CATEGORICAL_DIMENSIONS:
        if dimension not in frame:
            continue
        values = frame[dimension].fillna("MISSING").astype(str)
        for order, value in enumerate(sorted(values.unique())):
            group = frame.loc[values.eq(value)]
            rows.append(
                {
                    "dimension": dimension,
                    "group": value,
                    "group_order": order,
                    **simple_metrics(group),
                }
            )
    return pd.DataFrame(rows)


def threshold_drift(summary: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for fold in summary.get("folds") or []:
        if not fold.get("scored"):
            continue
        policy = fold.get("policy") or {}
        selected = fold.get("selected") or {}
        rows.append(
            {
                "fold": int(fold.get("fold")),
                "threshold": float(policy.get("score_threshold")),
                "events": int(selected.get("events") or 0),
                "candidate_days": int(selected.get("candidate_days") or 0),
                "candidate_day_rate": float(
                    selected.get("candidate_day_rate") or 0.0
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"scored_folds": 0}
    return {
        "scored_folds": int(len(frame)),
        "zero_candidate_folds": int(frame["events"].eq(0).sum()),
        "threshold_min": float(frame["threshold"].min()),
        "threshold_max": float(frame["threshold"].max()),
        "threshold_std": float(frame["threshold"].std(ddof=0)),
        "candidate_day_rate_min": float(
            frame["candidate_day_rate"].min()
        ),
        "candidate_day_rate_max": float(
            frame["candidate_day_rate"].max()
        ),
        "threshold_event_spearman": float(
            frame["threshold"].corr(frame["events"], method="spearman")
        ),
    }


def feature_family_summary(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    rows = []
    for family, group in features.groupby("feature_family", sort=True):
        best = group.sort_values(
            ["daily_slot_ic_bh_q", "mean_daily_slot_ic"],
            ascending=[True, False],
            kind="stable",
        ).iloc[0]
        rows.append(
            {
                "feature_family": family,
                "features_evaluated": int(len(group)),
                "stable_exploratory_features": int(
                    group["stable_exploratory_signal"].sum()
                ),
                "best_feature": str(best["feature"]),
                "best_mean_daily_slot_ic": float(
                    best["mean_daily_slot_ic"]
                ),
                "best_bh_q": float(best["daily_slot_ic_bh_q"]),
                "best_mean_top_minus_bottom_return_pct": float(
                    best["mean_daily_top_minus_bottom_return_pct"]
                ),
            }
        )
    return pd.DataFrame(rows)


def simple_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    returns = numeric(frame, "net_return_pct").dropna()
    events = int(len(returns))
    profits = float(returns.loc[returns > 0.0].sum())
    losses = float(-returns.loc[returns < 0.0].sum())
    profit_factor = (
        profits / losses
        if losses > 0.0
        else (float("inf") if profits > 0.0 else 0.0)
    )
    return {
        "events": events,
        "candidate_days": int(
            frame.loc[returns.index, "trade_date"].nunique()
        ),
        "wins": int(returns.gt(0.0).sum()),
        "win_rate": float(returns.gt(0.0).mean()) if events else 0.0,
        "mean_net_return_pct": (
            float(returns.mean()) if events else float("nan")
        ),
        "median_net_return_pct": (
            float(returns.median()) if events else float("nan")
        ),
        "return_p10_pct": (
            float(returns.quantile(0.10)) if events else float("nan")
        ),
        "profit_factor": profit_factor,
        "stress_50bps_mean_net_return_pct": (
            float((returns - STRESS_BPS / 100.0).mean())
            if events
            else float("nan")
        ),
    }


def loss_concentration(frame: pd.DataFrame) -> dict[str, float]:
    negative = -numeric(frame, "net_return_pct").clip(upper=0.0)
    ordered = negative.sort_values(ascending=False)
    total = float(negative.sum())
    top_ten = float(ordered.head(10).sum())
    top_twenty = float(ordered.head(20).sum())
    return {
        "total_negative_return_points": total,
        "top_10_loss_share": top_ten / total if total else 0.0,
        "top_20_loss_share": top_twenty / total if total else 0.0,
    }


def binary_auc(target: pd.Series, score: pd.Series) -> float:
    y = pd.to_numeric(target, errors="coerce").astype(int)
    s = pd.to_numeric(score, errors="coerce")
    positive = int(y.eq(1).sum())
    negative = int(y.eq(0).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = s.rank(method="average")
    positive_rank_sum = float(ranks.loc[y.eq(1)].sum())
    return (
        positive_rank_sum - positive * (positive + 1) / 2.0
    ) / (positive * negative)


def mean_zero_p_value(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 3:
        return 1.0
    standard_error = float(clean.std(ddof=1)) / math.sqrt(len(clean))
    if standard_error <= 0.0:
        return 0.0 if abs(float(clean.mean())) > 0.0 else 1.0
    z_score = abs(float(clean.mean())) / standard_error
    return float(math.erfc(z_score / math.sqrt(2.0)))


def benjamini_hochberg(values: Iterable[float]) -> np.ndarray:
    p_values = np.asarray(list(values), dtype=float)
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(len(p_values), dtype=float)
    running = 1.0
    total = max(len(p_values), 1)
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = total - reverse_rank + 1
        candidate = p_values[index] * total / max(rank, 1)
        running = min(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def feature_family(feature: str) -> str:
    if feature.startswith("v23_m1_"):
        return "minute_microstructure"
    if feature.startswith("v23_auction_"):
        return "opening_auction"
    if feature.startswith("v23_prev_mf_"):
        return "previous_day_moneyflow"
    if feature.startswith(("v19_", "v20_")):
        return "source_cross_section"
    return "v9_source_prior"


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [json_safe(row) for row in frame.to_dict(orient="records")]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def find_one(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one {name} under {root}, found {len(matches)}"
        )
    return matches[0]


if __name__ == "__main__":
    raise SystemExit(main())
