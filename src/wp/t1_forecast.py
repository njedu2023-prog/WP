from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


T1_FORECAST_MODEL_VERSION = "t1_ohlc_quantile_v1"
QUANTILES = (0.10, 0.50, 0.90)
TARGETS = ("open", "high", "low", "close")

FORECAST_COLUMNS = [
    "forecast_model_version",
    "forecast_mode",
    "forecast_warning",
    "forecast_sample_count",
    "forecast_live_sample_count",
    "forecast_proxy_sample_count",
    "forecast_effective_sample_count",
    "forecast_confidence",
    "forecast_open_q10_pct",
    "forecast_open_q50_pct",
    "forecast_open_q90_pct",
    "forecast_high_q10_pct",
    "forecast_high_q50_pct",
    "forecast_high_q90_pct",
    "forecast_low_q10_pct",
    "forecast_low_q50_pct",
    "forecast_low_q90_pct",
    "forecast_close_q10_pct",
    "forecast_close_q50_pct",
    "forecast_close_q90_pct",
    "forecast_open_q10_price",
    "forecast_open_q50_price",
    "forecast_open_q90_price",
    "forecast_high_q10_price",
    "forecast_high_q50_price",
    "forecast_high_q90_price",
    "forecast_low_q10_price",
    "forecast_low_q50_price",
    "forecast_low_q90_price",
    "forecast_close_q10_price",
    "forecast_close_q50_price",
    "forecast_close_q90_price",
    "forecast_profit_probability",
    "forecast_touch_plus3_probability",
    "forecast_touch_minus3_probability",
    "forecast_open_below_minus3_probability",
    "forecast_limit_touch_probability",
    "forecast_expected_net_return_pct",
    "forecast_downside_q10_pct",
    "forecast_risk_adjusted_utility",
    "forecast_path_status",
]

DEFAULT_FORECAST_CONFIG = {
    "forecast_min_total_samples": 24,
    "forecast_min_live_samples": 20,
    "forecast_proxy_weight": 0.30,
    "forecast_round_trip_cost_pct": 0.20,
    "forecast_downside_penalty": 0.20,
}

FEATURE_SCALES = {
    "entry_pct_chg": 1.5,
    "tail_profit_score": 18.0,
    "risk_penalty_score": 20.0,
    "amount_ratio_5d": 0.8,
}


@dataclass
class T1ForecastResult:
    table: pd.DataFrame
    summary: dict


def _num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _text(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="object")
    return frame[column].fillna(default).astype(str)


def _eligible_code(value: object) -> bool:
    code = str(value or "").strip()
    raw = code.split(".", 1)[0]
    if code.endswith(".BJ") or raw.startswith(("8", "9", "300", "301", "688")):
        return False
    return bool(raw)


def _weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
    value_num = pd.to_numeric(values, errors="coerce")
    weight_num = pd.to_numeric(weights, errors="coerce")
    valid = value_num.notna() & weight_num.notna() & weight_num.gt(0)
    if not valid.any():
        return float("nan")
    ordered = pd.DataFrame({"value": value_num[valid], "weight": weight_num[valid]}).sort_values("value")
    cumulative = ordered["weight"].cumsum()
    cutoff = float(ordered["weight"].sum()) * float(quantile)
    position = int(np.searchsorted(cumulative.to_numpy(), cutoff, side="left"))
    return float(ordered.iloc[min(position, len(ordered) - 1)]["value"])


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    value_num = pd.to_numeric(values, errors="coerce")
    weight_num = pd.to_numeric(weights, errors="coerce")
    valid = value_num.notna() & weight_num.notna() & weight_num.gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(value_num[valid], weights=weight_num[valid]))


def _weighted_probability(mask: pd.Series, weights: pd.Series) -> float:
    weight_num = pd.to_numeric(weights, errors="coerce")
    valid = mask.notna() & weight_num.notna() & weight_num.gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(mask[valid].astype(float), weights=weight_num[valid]) * 100)


def _sample_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    date_column: str,
    live: bool,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=frame.index)
    out["sample_trade_date"] = _text(frame, date_column).str.replace("-", "", regex=False).str.replace(r"\.0$", "", regex=True)
    out["ts_code"] = _text(frame, "ts_code").str.strip()
    out["sector_name"] = _text(frame, "sector_name")
    out["entry_pct_chg"] = _num(frame, "pct_chg_plan" if live else "pct_chg")
    out["tail_profit_score"] = _num(frame, "tail_profit_score")
    out["risk_penalty_score"] = _num(frame, "risk_penalty_score")
    out["amount_ratio_5d"] = _num(frame, "amount_ratio_5d")
    if live:
        out["open_return"] = _num(frame, "return_open_pct")
        out["high_return"] = _num(frame, "return_high_pct")
        out["low_return"] = _num(frame, "return_low_pct")
        out["close_return"] = _num(frame, "return_close_pct")
    else:
        out["open_return"] = _num(frame, "next_day_open_pct")
        out["high_return"] = _num(frame, "next_day_max_pct")
        out["low_return"] = _num(frame, "next_day_drawdown_pct")
        out["close_return"] = _num(frame, "next_day_close_pct")
    out["sample_source"] = source
    out = out[out["ts_code"].map(_eligible_code)].copy()
    out = out[out[["open_return", "high_return", "low_return", "close_return"]].notna().all(axis=1)]
    return out.reset_index(drop=True)


def _live_samples(validation: pd.DataFrame, as_of_trade_date: str) -> pd.DataFrame:
    if validation is None or validation.empty:
        return pd.DataFrame()
    verified = validation.copy()
    if "truth_status" in verified.columns:
        verified = verified[verified["truth_status"].fillna("").astype(str).eq("verified")]
    out = _sample_frame(
        verified,
        source="live_validation",
        date_column="plan_trade_date",
        live=True,
    )
    if as_of_trade_date:
        out = out[out["sample_trade_date"].lt(as_of_trade_date)]
    return out.drop_duplicates(["sample_trade_date", "ts_code"], keep="first")


def _read_proxy_samples(output_root: Path, as_of_trade_date: str) -> pd.DataFrame:
    samples: list[pd.DataFrame] = []
    for path in sorted((output_root / "backtests").glob("*/buy_trades.csv")):
        try:
            frame = pd.read_csv(path, keep_default_na=False, dtype={"ts_code": str, "backtest_trade_date": str})
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        sample = _sample_frame(
            frame,
            source="eod_proxy",
            date_column="backtest_trade_date",
            live=False,
        )
        if not sample.empty:
            samples.append(sample)
    if not samples:
        return pd.DataFrame()
    out = pd.concat(samples, ignore_index=True, sort=False)
    if as_of_trade_date:
        out = out[out["sample_trade_date"].lt(as_of_trade_date)]
    return out.drop_duplicates(["sample_trade_date", "ts_code"], keep="last")


def build_training_samples(
    validation: pd.DataFrame,
    output_root: str | Path,
    as_of_trade_date: str,
    *,
    proxy_samples: pd.DataFrame | None = None,
) -> pd.DataFrame:
    live = _live_samples(validation, as_of_trade_date)
    if proxy_samples is None:
        proxy = _read_proxy_samples(Path(output_root), as_of_trade_date)
    else:
        proxy = _sample_frame(
            proxy_samples,
            source="eod_proxy",
            date_column="backtest_trade_date",
            live=False,
        )
        if as_of_trade_date:
            proxy = proxy[proxy["sample_trade_date"].lt(as_of_trade_date)]
        proxy = proxy.drop_duplicates(["sample_trade_date", "ts_code"], keep="last")
    if live.empty:
        return proxy.reset_index(drop=True)
    if proxy.empty:
        return live.reset_index(drop=True)
    return pd.concat([live, proxy], ignore_index=True, sort=False)


def _config(config: dict | None) -> dict:
    values = DEFAULT_FORECAST_CONFIG.copy()
    values.update({key: value for key, value in (config or {}).items() if key in values})
    return values


def _similarity_weights(samples: pd.DataFrame, row: pd.Series, cfg: dict) -> pd.Series:
    base = pd.Series(1.0, index=samples.index, dtype="float64")
    proxy_weight = max(float(cfg["forecast_proxy_weight"]), 0.0)
    base.loc[samples["sample_source"].eq("eod_proxy")] = proxy_weight
    used = pd.Series(0, index=samples.index, dtype="int64")
    distance = pd.Series(0.0, index=samples.index, dtype="float64")
    row_sources = {
        "entry_pct_chg": row.get("pct_chg"),
        "tail_profit_score": row.get("tail_profit_score"),
        "risk_penalty_score": row.get("risk_penalty_score"),
        "amount_ratio_5d": row.get("amount_ratio_5d"),
    }
    for column, scale in FEATURE_SCALES.items():
        target = pd.to_numeric(pd.Series([row_sources[column]]), errors="coerce").iloc[0]
        values = pd.to_numeric(samples[column], errors="coerce")
        valid = values.notna() & pd.notna(target)
        distance.loc[valid] += ((values.loc[valid] - float(target)).abs() / scale).clip(upper=4.0)
        used.loc[valid] += 1
    normalized = distance / used.replace(0, 1)
    base *= np.exp(-normalized)
    sector = str(row.get("sector_name") or "").strip()
    if sector:
        base.loc[samples["sector_name"].eq(sector)] *= 1.15
    return base.fillna(0.0).clip(lower=0.0)


def _confidence(mode: str, effective_count: float, live_count: int) -> float:
    base = {"\u5b9e\u65f6\u6837\u672c\u6821\u51c6": 62.0, "\u6df7\u5408\u5148\u9a8c": 42.0, "\u65e5\u7ebf\u4ee3\u7406\u5148\u9a8c": 28.0}.get(mode, 0.0)
    return round(min(90.0, base + min(np.sqrt(max(effective_count, 0.0)) * 3.0, 18.0) + min(live_count, 30) * 0.3), 2)


def _blank_forecast() -> dict:
    row = {column: "" for column in FORECAST_COLUMNS}
    row.update(
        {
            "forecast_model_version": T1_FORECAST_MODEL_VERSION,
            "forecast_mode": "\u6837\u672c\u4e0d\u8db3",
            "forecast_warning": "\u53ef\u7528\u5386\u53f2\u6837\u672c\u4e0d\u8db3\uff0c\u4e0d\u8f93\u51fa\u6570\u503c\u5efa\u8bae",
            "forecast_path_status": "\u5f85\u79ef\u7d2fT+1\u5206\u65f6\u8def\u5f84",
        }
    )
    return row


def _forecast_row(row: pd.Series, samples: pd.DataFrame, cfg: dict) -> dict:
    result = _blank_forecast()
    weights = _similarity_weights(samples, row, cfg)
    if int(samples["sample_source"].eq("live_validation").sum()) >= int(cfg["forecast_min_live_samples"]):
        weights.loc[samples["sample_source"].eq("eod_proxy")] = 0.0
    usable = weights.gt(0)
    sample_count = int(usable.sum())
    live_count = int((samples.loc[usable, "sample_source"] == "live_validation").sum())
    proxy_count = int((samples.loc[usable, "sample_source"] == "eod_proxy").sum())
    sum_weights = float(weights[usable].sum())
    sum_square = float((weights[usable] ** 2).sum())
    effective_count = (sum_weights**2 / sum_square) if sum_square > 0 else 0.0
    result.update(
        {
            "forecast_sample_count": sample_count,
            "forecast_live_sample_count": live_count,
            "forecast_proxy_sample_count": proxy_count,
            "forecast_effective_sample_count": round(effective_count, 2),
        }
    )
    if sample_count < int(cfg["forecast_min_total_samples"]) or effective_count < 8:
        return result

    min_live = int(cfg["forecast_min_live_samples"])
    if live_count >= min_live:
        mode = "\u5b9e\u65f6\u6837\u672c\u6821\u51c6"
        warning = ""
    elif live_count:
        mode = "\u6df7\u5408\u5148\u9a8c"
        warning = "\u771f\u5b9e\u5c3e\u76d8\u6837\u672c\u5c1a\u5c11\uff0c\u4ecd\u542b\u6536\u76d8\u65e5\u7ebf\u4ee3\u7406\u5148\u9a8c"
    else:
        mode = "\u65e5\u7ebf\u4ee3\u7406\u5148\u9a8c"
        warning = "\u4ec5\u7528\u4e8e\u65b9\u5411\u53c2\u8003\uff0c\u4e0d\u662f14:20\u540e\u771f\u5b9e\u5feb\u7167\u6821\u51c6"
    result["forecast_mode"] = mode
    result["forecast_warning"] = warning
    result["forecast_confidence"] = _confidence(mode, effective_count, live_count)

    quantile_values: dict[tuple[str, int], float] = {}
    for target in TARGETS:
        values = samples[f"{target}_return"]
        for quantile in QUANTILES:
            suffix = int(quantile * 100)
            quantile_values[(target, suffix)] = _weighted_quantile(values, weights, quantile)
    for suffix in (10, 50, 90):
        open_value = quantile_values[("open", suffix)]
        close_value = quantile_values[("close", suffix)]
        quantile_values[("low", suffix)] = min(quantile_values[("low", suffix)], open_value, close_value)
        quantile_values[("high", suffix)] = max(quantile_values[("high", suffix)], open_value, close_value)

    entry_price = pd.to_numeric(pd.Series([row.get("price")]), errors="coerce").iloc[0]
    for target in TARGETS:
        for suffix in (10, 50, 90):
            value = round(float(quantile_values[(target, suffix)]), 4)
            result[f"forecast_{target}_q{suffix}_pct"] = value
            if pd.notna(entry_price) and float(entry_price) > 0:
                result[f"forecast_{target}_q{suffix}_price"] = round(float(entry_price) * (1 + value / 100), 4)

    cost = float(cfg["forecast_round_trip_cost_pct"])
    close_values = pd.to_numeric(samples["close_return"], errors="coerce")
    high_values = pd.to_numeric(samples["high_return"], errors="coerce")
    low_values = pd.to_numeric(samples["low_return"], errors="coerce")
    open_values = pd.to_numeric(samples["open_return"], errors="coerce")
    expected_net = _weighted_mean(close_values, weights) - cost
    downside = float(quantile_values[("low", 10)])
    result.update(
        {
            "forecast_profit_probability": round(_weighted_probability(close_values.gt(cost), weights), 2),
            "forecast_touch_plus3_probability": round(_weighted_probability(high_values.ge(3.0), weights), 2),
            "forecast_touch_minus3_probability": round(_weighted_probability(low_values.le(-3.0), weights), 2),
            "forecast_open_below_minus3_probability": round(_weighted_probability(open_values.le(-3.0), weights), 2),
            "forecast_limit_touch_probability": round(_weighted_probability(high_values.ge(9.5), weights), 2),
            "forecast_expected_net_return_pct": round(expected_net, 4),
            "forecast_downside_q10_pct": round(downside, 4),
            "forecast_risk_adjusted_utility": round(
                expected_net - float(cfg["forecast_downside_penalty"]) * max(0.0, -downside),
                4,
            ),
            "forecast_path_status": "\u5df2\u9884\u6d4b\u89e6\u53ca\u6982\u7387\uff1b\u5148\u6da8\u540e\u8dcc\u987a\u5e8f\u5f85T+1\u5206\u65f6\u6837\u672c",
        }
    )
    return result


def build_t1_forecasts(
    frame: pd.DataFrame,
    validation: pd.DataFrame,
    output_root: str | Path,
    trade_date: str,
    config: dict | None = None,
    *,
    proxy_samples: pd.DataFrame | None = None,
) -> T1ForecastResult:
    cfg = _config(config)
    out = frame.copy()
    for column in FORECAST_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    samples = build_training_samples(
        validation,
        output_root,
        str(trade_date or "").replace("-", ""),
        proxy_samples=proxy_samples,
    )
    if not out.empty:
        forecasts = pd.DataFrame(
            [_forecast_row(row, samples, cfg) for _, row in out.iterrows()],
            index=out.index,
        )
        for column in FORECAST_COLUMNS:
            out[column] = forecasts[column]
    modes = out.get("forecast_mode", pd.Series(dtype="object")).fillna("").astype(str)
    summary = {
        "model_version": T1_FORECAST_MODEL_VERSION,
        "forecast_count": int(len(out)),
        "numeric_forecast_count": int((~modes.eq("\u6837\u672c\u4e0d\u8db3")).sum()),
        "training_sample_count": int(len(samples)),
        "live_sample_count": int(samples.get("sample_source", pd.Series(dtype="object")).eq("live_validation").sum()),
        "proxy_sample_count": int(samples.get("sample_source", pd.Series(dtype="object")).eq("eod_proxy").sum()),
        "manual_decision_support_only": True,
    }
    return T1ForecastResult(out, summary)
