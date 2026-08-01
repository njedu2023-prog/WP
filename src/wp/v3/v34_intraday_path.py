from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .meta_alpha import IDENTITY_COLUMNS


SCHEMA_VERSION = "wp_v34_full_session_path_probe_1"
SIGNAL_SLOTS = (
    "14:20",
    "14:25",
    "14:30",
    "14:35",
    "14:40",
    "14:45",
    "14:50",
)
MINIMUM_ROW_COVERAGE = 0.98
MINIMUM_DATASET_COVERAGE = 0.98
MAX_SIGNAL_PRICE_ERROR_BPS = 10.0

V34_INTRADAY_PATH_FEATURE_COLUMNS = (
    "v34_session_return_pct",
    "v34_opening_30m_return_pct",
    "v34_morning_return_pct",
    "v34_afternoon_return_pct",
    "v34_lunch_gap_pct",
    "v34_post_1400_return_pct",
    "v34_morning_range_pct",
    "v34_afternoon_range_pct",
    "v34_session_realized_volatility_pct",
    "v34_session_downside_volatility_pct",
    "v34_directional_efficiency",
    "v34_max_drawdown_pct",
    "v34_rebound_from_low_pct",
    "v34_reversal_from_high_pct",
    "v34_session_vwap_gap_pct",
    "v34_above_vwap_share",
    "v34_session_close_position",
    "v34_opening_30m_amount_share",
    "v34_morning_amount_share",
    "v34_last_30m_amount_share",
    "v34_amount_acceleration_15m",
    "v34_signed_amount_imbalance",
    "v34_morning_signed_amount_imbalance",
    "v34_afternoon_signed_amount_imbalance",
    "v34_post_1400_signed_amount_imbalance",
    "v34_flow_regime_agreement",
    "v34_price_amount_correlation",
    "v34_amihud_proxy",
    "v34_high_time_fraction",
    "v34_low_time_fraction",
    "v34_minutes_since_high_fraction",
    "v34_minutes_since_low_fraction",
    "v34_zero_amount_share",
    "v34_return_autocorr1",
    "v34_recent_vs_prior_volatility_ratio",
)

V34_QUALITY_COLUMNS = (
    "v34_observed_rows",
    "v34_expected_rows",
    "v34_coverage_ratio",
    "v34_latest_time",
    "v34_causal_ok",
    "v34_signal_price_error_bps",
    "v34_signal_price_parity_ok",
    "v34_path_complete",
)

FORBIDDEN_TOKENS = (
    "target",
    "label",
    "truth",
    "future",
    "gross_return",
    "net_return",
    "t1_",
    "exit_",
)


def normalize_historical_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    return _normalize_minutes(
        frame.rename(columns={"code": "ts_code", "time": "trade_time"})
    )


def normalize_rt_min_daily(frame: pd.DataFrame) -> pd.DataFrame:
    return _normalize_minutes(
        frame.rename(columns={"code": "ts_code", "time": "trade_time"})
    )


def expected_minute_rows(signal_slot: str) -> int:
    if signal_slot not in SIGNAL_SLOTS:
        raise ValueError(f"unsupported V34 signal slot: {signal_slot}")
    signal = pd.Timestamp(f"2000-01-01 {signal_slot}:00")
    afternoon_start = pd.Timestamp("2000-01-01 13:01:00")
    afternoon_rows = int((signal - afternoon_start).total_seconds() // 60) + 1
    return 121 + afternoon_rows


def build_intraday_path_features(
    candidates: pd.DataFrame,
    minutes: pd.DataFrame,
) -> pd.DataFrame:
    required = {*IDENTITY_COLUMNS, "fold", "signal_price"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"V34 candidates missing columns: {missing}")
    source = candidates.loc[
        :,
        [*IDENTITY_COLUMNS, "fold", "signal_price"],
    ].copy()
    for column in IDENTITY_COLUMNS:
        source[column] = source[column].astype(str)
    if source.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("V34 candidate identities are duplicated")
    normalized = normalize_historical_minutes(minutes)
    grouped = {
        (str(code), str(date)): group.reset_index(drop=True)
        for (code, date), group in normalized.groupby(
            ["ts_code", "trade_date"],
            sort=False,
        )
    }
    empty = normalized.head(0)
    rows: list[dict[str, Any]] = []
    for record in source.to_dict(orient="records"):
        trade_date = str(record["trade_date"])
        signal_slot = str(record["signal_slot"])
        signal_time = pd.Timestamp(
            f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} "
            f"{signal_slot}:00"
        )
        stock_day = grouped.get((str(record["ts_code"]), trade_date), empty)
        bars = stock_day.loc[stock_day["trade_time"].le(signal_time)].copy()
        bars = bars.loc[_continuous_session_mask(bars["trade_time"])]
        bars.sort_values("trade_time", kind="stable", inplace=True)
        expected = expected_minute_rows(signal_slot)
        observed = int(len(bars))
        latest = bars["trade_time"].max() if observed else pd.NaT
        coverage = min(observed / max(expected, 1), 1.0)
        causal = bool(pd.isna(latest) or latest <= signal_time)
        signal_price = _finite_float(record["signal_price"])
        final_close = (
            _finite_float(bars["close"].iloc[-1]) if observed else np.nan
        )
        price_error_bps = _ratio_distance_bps(final_close, signal_price)
        parity = bool(
            np.isfinite(price_error_bps)
            and price_error_bps <= MAX_SIGNAL_PRICE_ERROR_BPS
        )
        values = _path_feature_values(bars)
        finite = bool(
            all(np.isfinite(values[column]) for column in values)
        )
        complete = bool(
            coverage >= MINIMUM_ROW_COVERAGE
            and causal
            and parity
            and finite
            and observed > 0
        )
        rows.append(
            {
                **record,
                "v34_observed_rows": observed,
                "v34_expected_rows": expected,
                "v34_coverage_ratio": float(coverage),
                "v34_latest_time": (
                    latest.isoformat() if not pd.isna(latest) else None
                ),
                "v34_causal_ok": causal,
                "v34_signal_price_error_bps": (
                    float(price_error_bps)
                    if np.isfinite(price_error_bps)
                    else np.nan
                ),
                "v34_signal_price_parity_ok": parity,
                "v34_path_complete": complete,
                **values,
            }
        )
    result = pd.DataFrame(rows)
    result.sort_values(
        ["fold", *IDENTITY_COLUMNS],
        kind="stable",
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)
    if not result["v34_causal_ok"].all():
        raise RuntimeError("V34 path features crossed a signal timestamp")
    return result


def audit_intraday_path_coverage(
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    query_failures: int,
) -> dict[str, Any]:
    identity = list(IDENTITY_COLUMNS)
    left = (
        candidates[identity]
        .astype(str)
        .sort_values(identity)
        .reset_index(drop=True)
    )
    right = (
        features[identity]
        .astype(str)
        .sort_values(identity)
        .reset_index(drop=True)
    )
    identity_exact = bool(
        len(left) == len(right)
        and left.equals(right)
        and not features.duplicated(identity).any()
    )
    numeric = features[list(V34_INTRADAY_PATH_FEATURE_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    finite_rows = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    complete = features["v34_path_complete"].fillna(False).astype(bool)
    coverage = float(complete.mean()) if len(features) else 0.0
    causal = bool(features["v34_causal_ok"].fillna(False).all())
    parity_rate = float(
        features["v34_signal_price_parity_ok"]
        .fillna(False)
        .astype(bool)
        .mean()
    ) if len(features) else 0.0
    finite_rate = float(finite_rows.mean()) if len(features) else 0.0
    diverse_features = int(
        sum(numeric[column].nunique(dropna=True) >= 10 for column in numeric)
    )
    date_coverage = {
        str(date): float(group["v34_path_complete"].astype(bool).mean())
        for date, group in features.groupby("trade_date", sort=True)
    }
    forbidden = sorted(
        column
        for column in features.columns
        if any(token in column.lower() for token in FORBIDDEN_TOKENS)
    )
    passed = bool(
        query_failures == 0
        and identity_exact
        and len(features) == len(candidates)
        and coverage >= MINIMUM_DATASET_COVERAGE
        and parity_rate >= MINIMUM_DATASET_COVERAGE
        and finite_rate >= MINIMUM_DATASET_COVERAGE
        and causal
        and diverse_features >= int(
            np.ceil(len(V34_INTRADAY_PATH_FEATURE_COLUMNS) * 0.90)
        )
        and len(date_coverage) >= 8
        and all(value >= 0.95 for value in date_coverage.values())
        and not forbidden
    )
    return {
        "candidate_rows": int(len(candidates)),
        "feature_rows": int(len(features)),
        "identity_exact": identity_exact,
        "duplicate_identities": int(features.duplicated(identity).sum()),
        "query_failures": int(query_failures),
        "complete_row_coverage": coverage,
        "signal_price_parity_rate": parity_rate,
        "finite_feature_row_rate": finite_rate,
        "causal_timestamps": causal,
        "diverse_feature_count": diverse_features,
        "feature_count": len(V34_INTRADAY_PATH_FEATURE_COLUMNS),
        "date_complete_coverage": date_coverage,
        "forbidden_columns": forbidden,
        "coverage_passed": passed,
    }


def _normalize_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "ts_code",
        "trade_time",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
    )
    result = frame.reindex(columns=columns).copy()
    result["ts_code"] = result["ts_code"].astype(str)
    result["trade_time"] = pd.to_datetime(
        result["trade_time"],
        errors="coerce",
    )
    result.dropna(subset=["ts_code", "trade_time"], inplace=True)
    result["trade_date"] = result["trade_time"].dt.strftime("%Y%m%d")
    for column in ("open", "high", "low", "close", "vol", "amount"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.loc[_continuous_session_mask(result["trade_time"])].copy()
    result.sort_values(["ts_code", "trade_time"], kind="stable", inplace=True)
    result.drop_duplicates(["ts_code", "trade_time"], keep="last", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def _continuous_session_mask(values: pd.Series) -> pd.Series:
    clock = pd.to_datetime(values, errors="coerce").dt.strftime("%H:%M")
    morning = clock.between("09:30", "11:30", inclusive="both")
    afternoon = clock.between("13:00", "15:00", inclusive="both")
    return morning | afternoon


def _path_feature_values(bars: pd.DataFrame) -> dict[str, float]:
    if bars.empty:
        return {
            column: 0.0 for column in V34_INTRADAY_PATH_FEATURE_COLUMNS
        }
    close = _series(bars, "close")
    open_price = _series(bars, "open")
    high = _series(bars, "high")
    low = _series(bars, "low")
    amount = _series(bars, "amount").clip(lower=0.0)
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    clock = bars["trade_time"].dt.strftime("%H:%M")
    morning_mask = clock.le("11:30")
    afternoon_mask = clock.ge("13:00")
    opening_mask = clock.le("10:00")
    post_1400_mask = clock.ge("14:00")
    morning = bars.loc[morning_mask]
    afternoon = bars.loc[afternoon_mask]
    opening = bars.loc[opening_mask]
    post_1400 = bars.loc[post_1400_mask]
    first_open = _finite_float(open_price.iloc[0])
    final_close = _finite_float(close.iloc[-1])
    session_high = _finite_float(high.max())
    session_low = _finite_float(low.min())
    total_amount = float(amount.sum())
    weighted_price = (
        float((close * amount).sum() / total_amount)
        if total_amount > 0
        else float(close.mean())
    )
    signed_amount = np.sign(returns.to_numpy(dtype=float)) * amount.to_numpy(
        dtype=float
    )
    cumulative_peak = close.cummax().replace(0.0, np.nan)
    drawdown = close / cumulative_peak - 1.0
    denominator = float(returns.abs().sum())
    high_position = int(high.to_numpy(dtype=float).argmax())
    low_position = int(low.to_numpy(dtype=float).argmin())
    span = max(len(bars) - 1, 1)
    recent = returns.tail(30)
    prior = returns.iloc[-60:-30] if len(returns) >= 60 else returns.iloc[:-30]
    recent_vol = _safe_std(recent)
    prior_vol = _safe_std(prior)
    recent_ratio = recent_vol / prior_vol if prior_vol > 0 else 0.0
    values = {
        "v34_session_return_pct": _return_pct(final_close, first_open),
        "v34_opening_30m_return_pct": _frame_return_pct(opening),
        "v34_morning_return_pct": _frame_return_pct(morning),
        "v34_afternoon_return_pct": _frame_return_pct(afternoon),
        "v34_lunch_gap_pct": _lunch_gap_pct(morning, afternoon),
        "v34_post_1400_return_pct": _frame_return_pct(post_1400),
        "v34_morning_range_pct": _frame_range_pct(morning),
        "v34_afternoon_range_pct": _frame_range_pct(afternoon),
        "v34_session_realized_volatility_pct": _safe_std(returns) * 100.0,
        "v34_session_downside_volatility_pct": (
            _safe_std(returns.loc[returns.lt(0.0)]) * 100.0
        ),
        "v34_directional_efficiency": (
            abs(final_close / first_open - 1.0) / denominator
            if first_open > 0 and denominator > 0
            else 0.0
        ),
        "v34_max_drawdown_pct": float(drawdown.min() * 100.0),
        "v34_rebound_from_low_pct": _return_pct(final_close, session_low),
        "v34_reversal_from_high_pct": _return_pct(final_close, session_high),
        "v34_session_vwap_gap_pct": _return_pct(final_close, weighted_price),
        "v34_above_vwap_share": float(close.gt(weighted_price).mean()),
        "v34_session_close_position": _close_position(
            final_close,
            session_low,
            session_high,
        ),
        "v34_opening_30m_amount_share": _amount_share(
            opening,
            total_amount,
        ),
        "v34_morning_amount_share": _amount_share(morning, total_amount),
        "v34_last_30m_amount_share": (
            float(amount.tail(30).sum() / total_amount)
            if total_amount > 0
            else 0.0
        ),
        "v34_amount_acceleration_15m": _amount_acceleration(amount),
        "v34_signed_amount_imbalance": _imbalance(signed_amount, amount),
        "v34_morning_signed_amount_imbalance": _frame_imbalance(
            bars,
            morning_mask,
        ),
        "v34_afternoon_signed_amount_imbalance": _frame_imbalance(
            bars,
            afternoon_mask,
        ),
        "v34_post_1400_signed_amount_imbalance": _frame_imbalance(
            bars,
            post_1400_mask,
        ),
        "v34_flow_regime_agreement": _flow_agreement(
            _frame_imbalance(bars, morning_mask),
            _frame_imbalance(bars, afternoon_mask),
        ),
        "v34_price_amount_correlation": _safe_corr(returns, amount),
        "v34_amihud_proxy": float(
            (returns.abs() / amount.replace(0.0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .mean()
            * 1_000_000.0
        ),
        "v34_high_time_fraction": float(high_position / span),
        "v34_low_time_fraction": float(low_position / span),
        "v34_minutes_since_high_fraction": float(
            (len(bars) - 1 - high_position) / span
        ),
        "v34_minutes_since_low_fraction": float(
            (len(bars) - 1 - low_position) / span
        ),
        "v34_zero_amount_share": float(amount.le(0.0).mean()),
        "v34_return_autocorr1": _safe_autocorr(returns),
        "v34_recent_vs_prior_volatility_ratio": float(recent_ratio),
    }
    return {
        column: _finite_or_zero(values[column])
        for column in V34_INTRADAY_PATH_FEATURE_COLUMNS
    }


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _frame_return_pct(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return _return_pct(
        _finite_float(frame["close"].iloc[-1]),
        _finite_float(frame["open"].iloc[0]),
    )


def _frame_range_pct(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return _return_pct(
        _finite_float(pd.to_numeric(frame["high"], errors="coerce").max()),
        _finite_float(pd.to_numeric(frame["low"], errors="coerce").min()),
    )


def _lunch_gap_pct(morning: pd.DataFrame, afternoon: pd.DataFrame) -> float:
    if morning.empty or afternoon.empty:
        return 0.0
    return _return_pct(
        _finite_float(afternoon["open"].iloc[0]),
        _finite_float(morning["close"].iloc[-1]),
    )


def _return_pct(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return 0.0
    return float((numerator / denominator - 1.0) * 100.0)


def _close_position(close: float, low: float, high: float) -> float:
    spread = high - low
    if not np.isfinite(spread) or spread <= 0:
        return 0.5
    return float(np.clip((close - low) / spread, 0.0, 1.0))


def _amount_share(frame: pd.DataFrame, total: float) -> float:
    if frame.empty or total <= 0:
        return 0.0
    amount = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0)
    return float(amount.clip(lower=0.0).sum() / total)


def _amount_acceleration(amount: pd.Series) -> float:
    recent = float(amount.tail(15).sum())
    prior = float(amount.iloc[-30:-15].sum()) if len(amount) >= 30 else 0.0
    return float(recent / prior - 1.0) if prior > 0 else 0.0


def _frame_imbalance(frame: pd.DataFrame, mask: pd.Series) -> float:
    selected = frame.loc[mask].copy()
    if selected.empty:
        return 0.0
    close = pd.to_numeric(selected["close"], errors="coerce").astype(float)
    amount = (
        pd.to_numeric(selected["amount"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    returns = close.pct_change().fillna(0.0)
    signed = np.sign(returns.to_numpy(dtype=float)) * amount.to_numpy(
        dtype=float
    )
    return _imbalance(signed, amount)


def _imbalance(signed_amount: np.ndarray, amount: pd.Series) -> float:
    denominator = float(amount.sum())
    return float(np.sum(signed_amount) / denominator) if denominator > 0 else 0.0


def _flow_agreement(morning: float, afternoon: float) -> float:
    if morning == 0.0 or afternoon == 0.0:
        return 0.0
    return float(np.sign(morning) * np.sign(afternoon))


def _safe_std(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.std(ddof=0)) if len(clean) >= 2 else 0.0


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat(
        [
            pd.to_numeric(left, errors="coerce"),
            pd.to_numeric(right, errors="coerce"),
        ],
        axis=1,
    ).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return 0.0
    return _finite_or_zero(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def _safe_autocorr(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 3 or clean.nunique() < 2:
        return 0.0
    return _finite_or_zero(clean.autocorr(lag=1))


def _ratio_distance_bps(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right) or right <= 0:
        return np.nan
    return float(abs(left / right - 1.0) * 10_000.0)


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _finite_or_zero(value: Any) -> float:
    result = _finite_float(value)
    return float(result) if np.isfinite(result) else 0.0
