from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .meta_alpha import IDENTITY_COLUMNS


ADMITTED_SOURCES = (
    "forecast",
    "repurchase",
    "share_float",
    "block_trade",
)
COMMON_FEATURE_COLUMNS = tuple(
    column
    for source in ADMITTED_SOURCES
    for column in (
        f"v32_{source}_event_count_5d",
        f"v32_{source}_active_5d",
        f"v32_{source}_latest_age_td",
    )
)
DETAIL_FEATURES_BY_SOURCE = {
    "forecast": (
        "v32_forecast_p_change_mid_mean",
        "v32_forecast_p_change_mid_min",
        "v32_forecast_p_change_mid_max",
        "v32_forecast_positive_change_share",
    ),
    "repurchase": (
        "v32_repurchase_amount_log_sum",
        "v32_repurchase_vol_log_sum",
        "v32_repurchase_high_limit_to_signal_pct",
        "v32_repurchase_low_limit_to_signal_pct",
    ),
    "share_float": (
        "v32_share_float_share_log_sum",
        "v32_share_float_ratio_sum",
        "v32_share_float_ratio_max",
        "v32_share_float_min_calendar_days",
    ),
    "block_trade": (
        "v32_block_trade_amount_log_sum",
        "v32_block_trade_vol_log_sum",
        "v32_block_trade_weighted_price_to_signal_pct",
        "v32_block_trade_latest_price_to_signal_pct",
    ),
}
V32_EVENT_FEATURE_COLUMNS = (
    *COMMON_FEATURE_COLUMNS,
    *DETAIL_FEATURES_BY_SOURCE["forecast"],
    *DETAIL_FEATURES_BY_SOURCE["repurchase"],
    *DETAIL_FEATURES_BY_SOURCE["share_float"],
    *DETAIL_FEATURES_BY_SOURCE["block_trade"],
)
_HIDDEN_COLUMNS = (
    "_v32_repurchase_high_limit_max",
    "_v32_repurchase_low_limit_mean",
    "_v32_block_trade_weighted_price",
    "_v32_block_trade_latest_price",
)


def build_candidate_event_features(
    candidates: pd.DataFrame,
    events_by_source: Mapping[str, pd.DataFrame],
    lookback_map: Mapping[str, list[str]],
) -> pd.DataFrame:
    required = {*IDENTITY_COLUMNS, "fold", "signal_price"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise RuntimeError(f"V32 candidates missing columns: {missing}")
    if candidates.duplicated(list(IDENTITY_COLUMNS)).any():
        raise RuntimeError("V32 candidates contain duplicate identities")

    base = candidates.loc[
        :,
        [*IDENTITY_COLUMNS, "fold", "signal_price"],
    ].copy()
    base["trade_date"] = base["trade_date"].astype(str)
    base["ts_code"] = base["ts_code"].astype(str)
    base["signal_price"] = pd.to_numeric(
        base["signal_price"],
        errors="coerce",
    )
    event_index = {
        source: {
            str(event_date): group.copy()
            for event_date, group in frame.groupby(
                "event_date",
                sort=False,
            )
        }
        for source, frame in events_by_source.items()
    }
    aggregate_records: list[dict[str, Any]] = []
    unique_targets = base[["trade_date", "ts_code"]].drop_duplicates()
    target_codes = {
        str(trade_date): set(group["ts_code"].astype(str))
        for trade_date, group in unique_targets.groupby(
            "trade_date",
            sort=True,
        )
    }
    for target_date, codes in target_codes.items():
        allowed_dates = list(lookback_map[target_date])
        age_by_date = {
            event_date: len(allowed_dates) - index
            for index, event_date in enumerate(allowed_dates)
        }
        records_by_code: dict[str, dict[str, Any]] = {
            code: {
                "trade_date": target_date,
                "ts_code": code,
            }
            for code in codes
        }
        for source in ADMITTED_SOURCES:
            source_index = event_index.get(source, {})
            parts = [
                source_index[event_date]
                for event_date in allowed_dates
                if event_date in source_index
            ]
            if not parts:
                continue
            window = pd.concat(parts, ignore_index=True)
            window = window.loc[
                window["ts_code"].astype(str).isin(codes)
            ].copy()
            if window.empty:
                continue
            window["ts_code"] = window["ts_code"].astype(str)
            window["_v32_age_td"] = (
                window["event_date"].astype(str).map(age_by_date)
            )
            for code, group in window.groupby("ts_code", sort=False):
                records_by_code[str(code)].update(
                    summarize_source_group(
                        source,
                        group,
                        target_date=target_date,
                    )
                )
        aggregate_records.extend(records_by_code.values())

    aggregates = pd.DataFrame.from_records(aggregate_records)
    result = base.merge(
        aggregates,
        on=["trade_date", "ts_code"],
        how="left",
        validate="many_to_one",
    )
    for source in ADMITTED_SOURCES:
        count_column = f"v32_{source}_event_count_5d"
        active_column = f"v32_{source}_active_5d"
        if count_column not in result:
            result[count_column] = 0.0
        else:
            result[count_column] = pd.to_numeric(
                result[count_column],
                errors="coerce",
            ).fillna(0.0)
        result[active_column] = (
            result[count_column].gt(0).astype(float)
        )

    valid_price = result["signal_price"].gt(0)
    result["v32_repurchase_high_limit_to_signal_pct"] = np.where(
        valid_price,
        (
            pd.to_numeric(
                result.get("_v32_repurchase_high_limit_max"),
                errors="coerce",
            )
            / result["signal_price"]
            - 1.0
        )
        * 100.0,
        np.nan,
    )
    result["v32_repurchase_low_limit_to_signal_pct"] = np.where(
        valid_price,
        (
            pd.to_numeric(
                result.get("_v32_repurchase_low_limit_mean"),
                errors="coerce",
            )
            / result["signal_price"]
            - 1.0
        )
        * 100.0,
        np.nan,
    )
    result["v32_block_trade_weighted_price_to_signal_pct"] = np.where(
        valid_price,
        (
            pd.to_numeric(
                result.get("_v32_block_trade_weighted_price"),
                errors="coerce",
            )
            / result["signal_price"]
            - 1.0
        )
        * 100.0,
        np.nan,
    )
    result["v32_block_trade_latest_price_to_signal_pct"] = np.where(
        valid_price,
        (
            pd.to_numeric(
                result.get("_v32_block_trade_latest_price"),
                errors="coerce",
            )
            / result["signal_price"]
            - 1.0
        )
        * 100.0,
        np.nan,
    )
    result.drop(
        columns=[column for column in _HIDDEN_COLUMNS if column in result],
        inplace=True,
    )
    for column in V32_EVENT_FEATURE_COLUMNS:
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result.sort_values(
        ["fold", *IDENTITY_COLUMNS],
        kind="stable",
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)
    return result[
        [
            *IDENTITY_COLUMNS,
            "fold",
            "signal_price",
            *V32_EVENT_FEATURE_COLUMNS,
        ]
    ]


def summarize_source_group(
    source: str,
    group: pd.DataFrame,
    *,
    target_date: str,
) -> dict[str, float]:
    result = {
        f"v32_{source}_event_count_5d": float(len(group)),
        f"v32_{source}_active_5d": 1.0,
        f"v32_{source}_latest_age_td": float(
            pd.to_numeric(
                group["_v32_age_td"],
                errors="coerce",
            ).min()
        ),
    }
    if source == "forecast":
        low = pd.to_numeric(group["p_change_min"], errors="coerce")
        high = pd.to_numeric(group["p_change_max"], errors="coerce")
        midpoint = pd.concat([low, high], axis=1).mean(
            axis=1,
            skipna=True,
        )
        valid = midpoint.dropna()
        result.update(
            {
                "v32_forecast_p_change_mid_mean": _mean(valid),
                "v32_forecast_p_change_mid_min": _min(valid),
                "v32_forecast_p_change_mid_max": _max(valid),
                "v32_forecast_positive_change_share": (
                    float(valid.gt(0).mean()) if len(valid) else np.nan
                ),
            }
        )
    elif source == "repurchase":
        amount = pd.to_numeric(group["amount"], errors="coerce")
        volume = pd.to_numeric(group["vol"], errors="coerce")
        high_limit = pd.to_numeric(
            group["high_limit"],
            errors="coerce",
        )
        low_limit = pd.to_numeric(
            group["low_limit"],
            errors="coerce",
        )
        result.update(
            {
                "v32_repurchase_amount_log_sum": _log_sum(amount),
                "v32_repurchase_vol_log_sum": _log_sum(volume),
                "_v32_repurchase_high_limit_max": _max(high_limit),
                "_v32_repurchase_low_limit_mean": _mean(low_limit),
            }
        )
    elif source == "share_float":
        shares = pd.to_numeric(group["float_share"], errors="coerce")
        ratio = pd.to_numeric(group["float_ratio"], errors="coerce")
        float_dates = pd.to_datetime(
            group["float_date"].astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
        target = pd.Timestamp(target_date)
        calendar_days = (float_dates - target).dt.days
        result.update(
            {
                "v32_share_float_share_log_sum": _log_sum(shares),
                "v32_share_float_ratio_sum": _sum(ratio),
                "v32_share_float_ratio_max": _max(ratio),
                "v32_share_float_min_calendar_days": _min(
                    calendar_days
                ),
            }
        )
    elif source == "block_trade":
        amount = pd.to_numeric(group["amount"], errors="coerce")
        volume = pd.to_numeric(group["vol"], errors="coerce")
        price = pd.to_numeric(group["price"], errors="coerce")
        event_dates = group["event_date"].astype(str)
        latest = group.loc[event_dates.eq(event_dates.max())]
        latest_amount = pd.to_numeric(
            latest["amount"],
            errors="coerce",
        )
        latest_price = pd.to_numeric(
            latest["price"],
            errors="coerce",
        )
        result.update(
            {
                "v32_block_trade_amount_log_sum": _log_sum(amount),
                "v32_block_trade_vol_log_sum": _log_sum(volume),
                "_v32_block_trade_weighted_price": _weighted_mean(
                    price,
                    amount,
                ),
                "_v32_block_trade_latest_price": _weighted_mean(
                    latest_price,
                    latest_amount,
                ),
            }
        )
    else:
        raise ValueError(f"unsupported V32 event source: {source}")
    return result


def audit_candidate_event_features(
    features: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    identity = list(IDENTITY_COLUMNS)
    left = candidates[identity].astype(str).sort_values(identity)
    right = features[identity].astype(str).sort_values(identity)
    identity_match = bool(
        len(left) == len(right)
        and left.reset_index(drop=True).equals(
            right.reset_index(drop=True)
        )
        and not features.duplicated(identity).any()
    )
    common_complete = True
    active_consistent = True
    detail_coverage: dict[str, float] = {}
    active_rows: dict[str, int] = {}
    for source in ADMITTED_SOURCES:
        count_column = f"v32_{source}_event_count_5d"
        active_column = f"v32_{source}_active_5d"
        common_complete = bool(
            common_complete
            and features[count_column].notna().all()
            and features[active_column].notna().all()
        )
        expected_active = features[count_column].gt(0)
        active_consistent = bool(
            active_consistent
            and features[active_column].eq(
                expected_active.astype(float)
            ).all()
        )
        active_rows[source] = int(expected_active.sum())
        details = list(DETAIL_FEATURES_BY_SOURCE[source])
        if expected_active.any():
            detail_coverage[source] = float(
                features.loc[expected_active, details]
                .notna()
                .any(axis=1)
                .mean()
            )
        else:
            detail_coverage[source] = 0.0
    event_union = pd.concat(
        [
            features[f"v32_{source}_active_5d"].gt(0)
            for source in ADMITTED_SOURCES
        ],
        axis=1,
    ).any(axis=1)
    event_rows = int(event_union.sum())
    event_dates = int(
        features.loc[event_union, "trade_date"].astype(str).nunique()
    )
    candidate_dates = int(features["trade_date"].astype(str).nunique())
    detail_passed = bool(
        all(value >= 0.80 for value in detail_coverage.values())
    )
    return {
        "candidate_rows": int(len(candidates)),
        "feature_rows": int(len(features)),
        "identity_match": identity_match,
        "common_features_complete": common_complete,
        "active_flags_consistent": active_consistent,
        "active_rows_by_source": active_rows,
        "detail_coverage_by_source": detail_coverage,
        "minimum_active_detail_coverage": 0.80,
        "active_detail_coverage_passed": detail_passed,
        "event_union_rows": event_rows,
        "event_union_row_rate": (
            float(event_rows / len(features)) if len(features) else 0.0
        ),
        "event_union_trade_dates": event_dates,
        "event_union_trade_date_rate": (
            float(event_dates / candidate_dates)
            if candidate_dates
            else 0.0
        ),
        "feature_non_null_coverage": {
            column: float(features[column].notna().mean())
            for column in V32_EVENT_FEATURE_COLUMNS
        },
        "coverage_passed": bool(
            identity_match
            and common_complete
            and active_consistent
            and detail_passed
        ),
    }


def _finite(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.loc[np.isfinite(values)]


def _sum(series: pd.Series) -> float:
    values = _finite(series)
    return float(values.sum()) if len(values) else np.nan


def _mean(series: pd.Series) -> float:
    values = _finite(series)
    return float(values.mean()) if len(values) else np.nan


def _min(series: pd.Series) -> float:
    values = _finite(series)
    return float(values.min()) if len(values) else np.nan


def _max(series: pd.Series) -> float:
    values = _finite(series)
    return float(values.max()) if len(values) else np.nan


def _log_sum(series: pd.Series) -> float:
    values = _finite(series).clip(lower=0)
    return float(np.log1p(values.sum())) if len(values) else np.nan


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    frame = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="coerce"),
            "weight": pd.to_numeric(weights, errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["value"])
    if frame.empty:
        return np.nan
    valid_weight = frame["weight"].fillna(0).clip(lower=0)
    if valid_weight.sum() > 0:
        return float(
            np.average(frame["value"], weights=valid_weight)
        )
    return float(frame["value"].mean())
