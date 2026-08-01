from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


SCHEMA_VERSION = "wp_v38_index_futures_data_probe_1"
SIGNAL_SLOTS = (
    "14:20",
    "14:25",
    "14:30",
    "14:35",
    "14:40",
    "14:45",
    "14:50",
)
PAIR_SPECS = (
    {
        "pair_id": "if300",
        "continuous_code": "IF.CFX",
        "etf_code": "510300.SH",
    },
    {
        "pair_id": "ih50",
        "continuous_code": "IH.CFX",
        "etf_code": "510050.SH",
    },
    {
        "pair_id": "ic500",
        "continuous_code": "IC.CFX",
        "etf_code": "510500.SH",
    },
    {
        "pair_id": "im1000",
        "continuous_code": "IM.CFX",
        "etf_code": "512100.SH",
    },
)
ETF_FIELDS = "ts_code,trade_time,open,close,high,low,vol,amount"
FUTURE_FIELDS = (
    "ts_code,trade_time,open,close,high,low,vol,amount,oi"
)
MAPPING_FIELDS = "ts_code,trade_date,mapping_ts_code"
MINIMUM_ROW_COVERAGE = 0.98
MINIMUM_FINITE_FEATURE_RATE = 0.98

PAIR_FEATURE_NAMES = (
    "etf_return_from_open_pct",
    "future_return_from_open_pct",
    "hedge_return_spread_pct",
    "etf_return_20m_pct",
    "future_return_20m_pct",
    "hedge_spread_20m_pct",
    "oi_change_from_open_pct",
    "oi_change_20m_pct",
    "future_last20_amount_share",
    "etf_last20_amount_share",
    "tracking_error_1m_pct",
    "return_correlation",
)
CROSS_FEATURE_NAMES = (
    "v38_etf_small_minus_large_pct",
    "v38_etf_mid_minus_large_pct",
    "v38_future_small_minus_large_pct",
    "v38_future_mid_minus_large_pct",
    "v38_hedge_spread_mean_pct",
    "v38_hedge_spread_std_pct",
    "v38_oi_change_mean_pct",
    "v38_oi_change_dispersion_pct",
)
V38_FEATURE_COLUMNS = tuple(
    f"v38_{spec['pair_id']}_{name}"
    for spec in PAIR_SPECS
    for name in PAIR_FEATURE_NAMES
) + CROSS_FEATURE_NAMES
FORBIDDEN_TOKENS = (
    "target",
    "label",
    "truth",
    "gross_return",
    "net_return",
    "t1_",
    "exit_",
    "next_close",
)


def normalize_mapping(
    frame: pd.DataFrame,
    *,
    required_dates: Iterable[str] | None = None,
) -> pd.DataFrame:
    columns = ("ts_code", "trade_date", "mapping_ts_code")
    result = frame.reindex(columns=columns).copy()
    for column in columns:
        result[column] = result[column].astype(str).str.strip()
    result = result.loc[
        result["ts_code"].isin(
            spec["continuous_code"] for spec in PAIR_SPECS
        )
    ].copy()
    if required_dates is not None:
        dates = {str(value) for value in required_dates}
        result = result.loc[result["trade_date"].isin(dates)].copy()
    result = result.loc[
        result["trade_date"].str.fullmatch(r"\d{8}", na=False)
        & result["mapping_ts_code"].str.fullmatch(
            r"(IF|IH|IC|IM)\d{4}\.CFX",
            na=False,
        )
    ].copy()
    result.sort_values(
        ["trade_date", "ts_code"],
        kind="stable",
        inplace=True,
    )
    result.drop_duplicates(
        ["trade_date", "ts_code", "mapping_ts_code"],
        keep="last",
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)
    return result


def normalize_historical_etf_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    return _normalize_minutes(frame, include_oi=False)


def normalize_realtime_etf_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    return _normalize_minutes(
        frame.rename(columns={"code": "ts_code", "time": "trade_time"}),
        include_oi=False,
    )


def normalize_historical_future_minutes(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return _normalize_minutes(frame, include_oi=True)


def normalize_realtime_future_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    return _normalize_minutes(
        frame.rename(columns={"code": "ts_code", "time": "trade_time"}),
        include_oi=True,
    )


def expected_minute_rows(signal_slot: str) -> int:
    if signal_slot not in SIGNAL_SLOTS:
        raise ValueError(f"unsupported V38 signal slot: {signal_slot}")
    signal = pd.Timestamp(f"2000-01-01 {signal_slot}:00")
    afternoon_start = pd.Timestamp("2000-01-01 13:01:00")
    afternoon_rows = int((signal - afternoon_start).total_seconds() // 60) + 1
    return 121 + afternoon_rows


def build_regime_features(
    probe_dates: Iterable[str],
    mappings: pd.DataFrame,
    etf_minutes: pd.DataFrame,
    future_minutes: pd.DataFrame,
) -> pd.DataFrame:
    dates = tuple(str(value) for value in probe_dates)
    normalized_mapping = normalize_mapping(
        mappings,
        required_dates=dates,
    )
    if normalized_mapping.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("V38 continuous futures mapping is ambiguous")
    etfs = normalize_historical_etf_minutes(etf_minutes)
    futures = normalize_historical_future_minutes(future_minutes)
    etf_groups = {
        (str(code), str(date)): group.reset_index(drop=True)
        for (code, date), group in etfs.groupby(
            ["ts_code", "trade_date"],
            sort=False,
        )
    }
    future_groups = {
        (str(code), str(date)): group.reset_index(drop=True)
        for (code, date), group in futures.groupby(
            ["ts_code", "trade_date"],
            sort=False,
        )
    }
    mapping_lookup = {
        (str(row.trade_date), str(row.ts_code)): str(row.mapping_ts_code)
        for row in normalized_mapping.itertuples(index=False)
    }
    empty_etf = etfs.head(0)
    empty_future = futures.head(0)
    rows: list[dict[str, Any]] = []
    for trade_date in dates:
        date_text = _iso_date(trade_date)
        for signal_slot in SIGNAL_SLOTS:
            signal_time = pd.Timestamp(f"{date_text} {signal_slot}:00")
            values: dict[str, float] = {}
            complete_pairs = 0
            causal_pairs = 0
            latest_times: list[pd.Timestamp] = []
            pair_values: dict[str, dict[str, float]] = {}
            quality: dict[str, Any] = {}
            for spec in PAIR_SPECS:
                pair_id = spec["pair_id"]
                mapped_code = mapping_lookup.get(
                    (trade_date, spec["continuous_code"]),
                    "",
                )
                etf_day = etf_groups.get(
                    (spec["etf_code"], trade_date),
                    empty_etf,
                )
                future_day = future_groups.get(
                    (mapped_code, trade_date),
                    empty_future,
                )
                pair = _build_pair_slot(
                    etf_day,
                    future_day,
                    signal_time=signal_time,
                    signal_slot=signal_slot,
                )
                pair_values[pair_id] = pair["features"]
                values.update(
                    {
                        f"v38_{pair_id}_{name}": pair["features"][name]
                        for name in PAIR_FEATURE_NAMES
                    }
                )
                quality[f"v38_{pair_id}_mapped_code"] = mapped_code or None
                quality[f"v38_{pair_id}_coverage_ratio"] = pair[
                    "coverage_ratio"
                ]
                quality[f"v38_{pair_id}_oi_coverage_ratio"] = pair[
                    "oi_coverage_ratio"
                ]
                quality[f"v38_{pair_id}_complete"] = pair["complete"]
                if pair["complete"]:
                    complete_pairs += 1
                if pair["causal"]:
                    causal_pairs += 1
                if pair["latest_time"] is not None:
                    latest_times.append(pair["latest_time"])
            cross = _cross_pair_features(pair_values)
            values.update(cross)
            finite = bool(
                np.isfinite(
                    np.asarray(
                        [values[column] for column in V38_FEATURE_COLUMNS],
                        dtype=float,
                    )
                ).all()
            )
            latest = max(latest_times) if latest_times else pd.NaT
            rows.append(
                {
                    "trade_date": trade_date,
                    "signal_slot": signal_slot,
                    "v38_complete_pairs": complete_pairs,
                    "v38_all_pairs_complete": complete_pairs
                    == len(PAIR_SPECS),
                    "v38_causal_ok": causal_pairs == len(PAIR_SPECS),
                    "v38_latest_time": (
                        latest.isoformat() if not pd.isna(latest) else None
                    ),
                    "v38_finite_features": finite,
                    **quality,
                    **values,
                }
            )
    result = pd.DataFrame(rows)
    result.sort_values(
        ["trade_date", "signal_slot"],
        kind="stable",
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)
    return result


def audit_probe_contract(
    features: pd.DataFrame,
    mappings: pd.DataFrame,
    *,
    probe_dates: Iterable[str],
    query_failures: int,
) -> dict[str, Any]:
    dates = tuple(str(value) for value in probe_dates)
    normalized_mapping = normalize_mapping(
        mappings,
        required_dates=dates,
    )
    expected_mapping_rows = len(dates) * len(PAIR_SPECS)
    mapping_unique = not normalized_mapping.duplicated(
        ["trade_date", "ts_code"]
    ).any()
    mapping_exact = bool(
        len(normalized_mapping) == expected_mapping_rows
        and mapping_unique
        and set(
            zip(
                normalized_mapping["trade_date"],
                normalized_mapping["ts_code"],
            )
        )
        == {
            (date, spec["continuous_code"])
            for date in dates
            for spec in PAIR_SPECS
        }
    )
    expected_feature_rows = len(dates) * len(SIGNAL_SLOTS)
    identity_unique = not features.duplicated(
        ["trade_date", "signal_slot"]
    ).any()
    identity_exact = bool(
        len(features) == expected_feature_rows
        and identity_unique
        and set(
            zip(features["trade_date"].astype(str), features["signal_slot"])
        )
        == {
            (date, signal_slot)
            for date in dates
            for signal_slot in SIGNAL_SLOTS
        }
    )
    numeric = features.reindex(columns=V38_FEATURE_COLUMNS).apply(
        pd.to_numeric,
        errors="coerce",
    )
    finite_rows = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    finite_rate = float(finite_rows.mean()) if len(features) else 0.0
    complete_rate = (
        float(
            features["v38_all_pairs_complete"]
            .fillna(False)
            .astype(bool)
            .mean()
        )
        if len(features)
        else 0.0
    )
    causal = bool(
        len(features)
        and features["v38_causal_ok"].fillna(False).astype(bool).all()
    )
    diverse_features = int(
        sum(numeric[column].nunique(dropna=True) >= 10 for column in numeric)
    )
    date_coverage = {
        str(date): float(
            group["v38_all_pairs_complete"]
            .fillna(False)
            .astype(bool)
            .mean()
        )
        for date, group in features.groupby("trade_date", sort=True)
    }
    forbidden = sorted(
        column
        for column in features.columns
        if any(token in column.lower() for token in FORBIDDEN_TOKENS)
    )
    coverage_passed = bool(
        query_failures == 0
        and mapping_exact
        and identity_exact
        and complete_rate == 1.0
        and finite_rate >= MINIMUM_FINITE_FEATURE_RATE
        and causal
        and diverse_features
        >= int(np.ceil(len(V38_FEATURE_COLUMNS) * 0.80))
        and len(date_coverage) == len(dates)
        and all(value == 1.0 for value in date_coverage.values())
        and not forbidden
    )
    return {
        "query_failures": int(query_failures),
        "probe_dates": len(dates),
        "expected_mapping_rows": expected_mapping_rows,
        "mapping_rows": int(len(normalized_mapping)),
        "mapping_unique": mapping_unique,
        "mapping_exact": mapping_exact,
        "expected_feature_rows": expected_feature_rows,
        "feature_rows": int(len(features)),
        "identity_unique": identity_unique,
        "identity_exact": identity_exact,
        "complete_pair_row_rate": complete_rate,
        "finite_feature_row_rate": finite_rate,
        "causal_timestamps": causal,
        "diverse_feature_count": diverse_features,
        "feature_count": len(V38_FEATURE_COLUMNS),
        "date_complete_coverage": date_coverage,
        "forbidden_columns": forbidden,
        "coverage_passed": coverage_passed,
    }


def _build_pair_slot(
    etf_day: pd.DataFrame,
    future_day: pd.DataFrame,
    *,
    signal_time: pd.Timestamp,
    signal_slot: str,
) -> dict[str, Any]:
    etf = _causal_bars(etf_day, signal_time)
    future = _causal_bars(future_day, signal_time)
    expected = expected_minute_rows(signal_slot)
    etf_coverage = min(len(etf) / max(expected, 1), 1.0)
    future_coverage = min(len(future) / max(expected, 1), 1.0)
    oi_finite = (
        pd.to_numeric(future["oi"], errors="coerce").notna()
        if "oi" in future
        else pd.Series(dtype=bool)
    )
    oi_coverage = float(oi_finite.mean()) if len(future) else 0.0
    latest_candidates = [
        frame["trade_time"].max()
        for frame in (etf, future)
        if len(frame)
    ]
    latest = max(latest_candidates) if latest_candidates else None
    causal = bool(latest is None or latest <= signal_time)
    aligned = etf[
        ["trade_time", "close"]
    ].rename(columns={"close": "etf_close"}).merge(
        future[["trade_time", "close"]].rename(
            columns={"close": "future_close"}
        ),
        on="trade_time",
        how="inner",
        validate="one_to_one",
    )
    features = _pair_feature_values(etf, future, aligned)
    finite = bool(
        np.isfinite(
            np.asarray(
                [features[name] for name in PAIR_FEATURE_NAMES],
                dtype=float,
            )
        ).all()
    )
    positive_prices = bool(
        len(etf)
        and len(future)
        and pd.to_numeric(etf["close"], errors="coerce").gt(0).all()
        and pd.to_numeric(future["close"], errors="coerce").gt(0).all()
    )
    complete = bool(
        etf_coverage >= MINIMUM_ROW_COVERAGE
        and future_coverage >= MINIMUM_ROW_COVERAGE
        and oi_coverage >= MINIMUM_ROW_COVERAGE
        and len(aligned) >= int(np.floor(expected * MINIMUM_ROW_COVERAGE))
        and causal
        and positive_prices
        and finite
    )
    return {
        "features": features,
        "coverage_ratio": float(min(etf_coverage, future_coverage)),
        "oi_coverage_ratio": oi_coverage,
        "latest_time": latest,
        "causal": causal,
        "complete": complete,
    }


def _pair_feature_values(
    etf: pd.DataFrame,
    future: pd.DataFrame,
    aligned: pd.DataFrame,
) -> dict[str, float]:
    if etf.empty or future.empty or len(aligned) < 2:
        return {name: np.nan for name in PAIR_FEATURE_NAMES}
    etf_return = _return_pct(etf["close"].iloc[0], etf["close"].iloc[-1])
    future_return = _return_pct(
        future["close"].iloc[0],
        future["close"].iloc[-1],
    )
    etf_20m = _window_return_pct(etf, minutes=20)
    future_20m = _window_return_pct(future, minutes=20)
    oi = pd.to_numeric(future["oi"], errors="coerce")
    oi_change = _return_pct(oi.iloc[0], oi.iloc[-1])
    oi_20m = _window_return_pct(
        future.assign(close=oi),
        minutes=20,
    )
    etf_returns = pd.to_numeric(
        aligned["etf_close"],
        errors="coerce",
    ).pct_change()
    future_returns = pd.to_numeric(
        aligned["future_close"],
        errors="coerce",
    ).pct_change()
    valid = etf_returns.notna() & future_returns.notna()
    differences = (
        future_returns.loc[valid] - etf_returns.loc[valid]
    ) * 100.0
    tracking_error = (
        float(differences.std(ddof=0)) if len(differences) else 0.0
    )
    correlation = _safe_correlation(
        etf_returns.loc[valid],
        future_returns.loc[valid],
    )
    return {
        "etf_return_from_open_pct": etf_return,
        "future_return_from_open_pct": future_return,
        "hedge_return_spread_pct": future_return - etf_return,
        "etf_return_20m_pct": etf_20m,
        "future_return_20m_pct": future_20m,
        "hedge_spread_20m_pct": future_20m - etf_20m,
        "oi_change_from_open_pct": oi_change,
        "oi_change_20m_pct": oi_20m,
        "future_last20_amount_share": _last_window_share(
            future["amount"],
            20,
        ),
        "etf_last20_amount_share": _last_window_share(
            etf["amount"],
            20,
        ),
        "tracking_error_1m_pct": tracking_error,
        "return_correlation": correlation,
    }


def _cross_pair_features(
    pair_values: dict[str, dict[str, float]],
) -> dict[str, float]:
    if any(pair_id not in pair_values for pair_id in ("if300", "ic500", "im1000")):
        return {name: np.nan for name in CROSS_FEATURE_NAMES}
    if300 = pair_values["if300"]
    ic500 = pair_values["ic500"]
    im1000 = pair_values["im1000"]
    hedge = np.asarray(
        [
            values["hedge_return_spread_pct"]
            for values in pair_values.values()
        ],
        dtype=float,
    )
    oi_change = np.asarray(
        [
            values["oi_change_from_open_pct"]
            for values in pair_values.values()
        ],
        dtype=float,
    )
    return {
        "v38_etf_small_minus_large_pct": (
            im1000["etf_return_from_open_pct"]
            - if300["etf_return_from_open_pct"]
        ),
        "v38_etf_mid_minus_large_pct": (
            ic500["etf_return_from_open_pct"]
            - if300["etf_return_from_open_pct"]
        ),
        "v38_future_small_minus_large_pct": (
            im1000["future_return_from_open_pct"]
            - if300["future_return_from_open_pct"]
        ),
        "v38_future_mid_minus_large_pct": (
            ic500["future_return_from_open_pct"]
            - if300["future_return_from_open_pct"]
        ),
        "v38_hedge_spread_mean_pct": float(np.mean(hedge)),
        "v38_hedge_spread_std_pct": float(np.std(hedge, ddof=0)),
        "v38_oi_change_mean_pct": float(np.mean(oi_change)),
        "v38_oi_change_dispersion_pct": float(
            np.std(oi_change, ddof=0)
        ),
    }


def _normalize_minutes(
    frame: pd.DataFrame,
    *,
    include_oi: bool,
) -> pd.DataFrame:
    columns = [
        "ts_code",
        "trade_time",
        "open",
        "close",
        "high",
        "low",
        "vol",
        "amount",
    ]
    if include_oi:
        columns.append("oi")
    result = frame.reindex(columns=columns).copy()
    result["ts_code"] = result["ts_code"].astype(str).str.strip()
    result["trade_time"] = pd.to_datetime(
        result["trade_time"],
        errors="coerce",
    )
    result.dropna(subset=["trade_time"], inplace=True)
    result = result.loc[result["ts_code"].ne("")].copy()
    result["trade_date"] = result["trade_time"].dt.strftime("%Y%m%d")
    numeric_columns = [
        "open",
        "close",
        "high",
        "low",
        "vol",
        "amount",
    ]
    if include_oi:
        numeric_columns.append("oi")
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.loc[_continuous_session_mask(result["trade_time"])].copy()
    result.sort_values(
        ["ts_code", "trade_time"],
        kind="stable",
        inplace=True,
    )
    result.drop_duplicates(
        ["ts_code", "trade_time"],
        keep="last",
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)
    return result


def _continuous_session_mask(values: pd.Series) -> pd.Series:
    clock = pd.to_datetime(values, errors="coerce").dt.strftime("%H:%M")
    morning = clock.between("09:30", "11:30", inclusive="both")
    afternoon = clock.between("13:00", "15:00", inclusive="both")
    return morning | afternoon


def _causal_bars(
    frame: pd.DataFrame,
    signal_time: pd.Timestamp,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.loc[frame["trade_time"].le(signal_time)].copy()
    result = result.loc[_continuous_session_mask(result["trade_time"])]
    result.sort_values("trade_time", kind="stable", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def _window_return_pct(frame: pd.DataFrame, *, minutes: int) -> float:
    if frame.empty:
        return np.nan
    latest = frame["trade_time"].iloc[-1]
    start = latest - pd.Timedelta(minutes=minutes)
    window = frame.loc[frame["trade_time"].ge(start)]
    if len(window) < 2:
        return 0.0
    return _return_pct(window["close"].iloc[0], window["close"].iloc[-1])


def _return_pct(start: Any, end: Any) -> float:
    start_value = _finite_float(start)
    end_value = _finite_float(end)
    if not np.isfinite(start_value) or start_value <= 0 or not np.isfinite(end_value):
        return np.nan
    return float((end_value / start_value - 1.0) * 100.0)


def _last_window_share(values: pd.Series, length: int) -> float:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(numeric.sum())
    if total <= 0:
        return 0.0
    return float(numeric.tail(length).sum() / total)


def _safe_correlation(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 3:
        return 0.0
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    if valid.sum() < 3:
        return 0.0
    left_values = left_values[valid]
    right_values = right_values[valid]
    if np.std(left_values) <= 1e-12 or np.std(right_values) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_values, right_values)[0, 1])


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _iso_date(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"
