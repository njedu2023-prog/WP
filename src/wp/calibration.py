from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _load_history(
    root: Path,
    model_version: str | None = None,
    before_date: str | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((root / "outputs" / "backtests").glob("*/trades.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        probability_column = "p_limitup_t1_raw" if "p_limitup_t1_raw" in frame.columns else "p_limitup_t1"
        if probability_column not in frame.columns or "label_t1_limitup" not in frame.columns:
            continue
        keep = [
            column
            for column in [
                probability_column,
                "label_t1_limitup",
                "backtest_trade_date",
                "ts_code",
                "model_version",
                "backtest_data_mode",
                "calibration_eligible",
            ]
            if column in frame.columns
        ]
        part = frame[keep].copy().rename(columns={probability_column: "p_limitup_t1_raw"})
        frames.append(part)
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["p_limitup_t1_raw"] = pd.to_numeric(out["p_limitup_t1_raw"], errors="coerce")
    out["label_t1_limitup"] = pd.to_numeric(out["label_t1_limitup"], errors="coerce")
    out = out.dropna(subset=["p_limitup_t1_raw", "label_t1_limitup"])
    out = out[out["label_t1_limitup"].isin([0, 1])]

    if before_date and "backtest_trade_date" in out.columns:
        dates = out["backtest_trade_date"].astype(str).str.replace("-", "", regex=False)
        out = out[dates < str(before_date).replace("-", "")]
    if model_version:
        if "model_version" not in out.columns:
            return pd.DataFrame()
        out = out[out["model_version"].fillna("").astype(str).eq(model_version)]
        if "backtest_data_mode" in out.columns:
            out = out[out["backtest_data_mode"].fillna("").astype(str).eq("intraday_1420")]
        if "calibration_eligible" in out.columns:
            eligible = out["calibration_eligible"].fillna(False).astype(str).str.lower().isin({"1", "true", "yes"})
            out = out[eligible]

    dedupe_keys = [column for column in ["backtest_trade_date", "ts_code"] if column in out.columns]
    if len(dedupe_keys) == 2:
        out = out.drop_duplicates(dedupe_keys, keep="last")
    return out.reset_index(drop=True)


def _logit(probability: float) -> float:
    value = float(np.clip(probability, 1e-5, 1 - 1e-5))
    return float(np.log(value / (1 - value)))


def apply_statistical_calibration(
    df: pd.DataFrame,
    root: Path,
    min_samples: int = 80,
    model_version: str | None = None,
    before_date: str | None = None,
) -> pd.DataFrame:
    if df.empty or "p_limitup_t1" not in df.columns:
        return df.copy()
    out = df.copy()
    raw = pd.to_numeric(out["p_limitup_t1"], errors="coerce").fillna(0).clip(0.2, 99.8)
    out["p_limitup_t1_raw"] = raw
    out["calibration_sample_count"] = 0
    out["self_learning_adjustment"] = 0.0
    out["calibration_method"] = "none"
    history = _load_history(root, model_version=model_version, before_date=before_date)
    if len(history) < min_samples or history["label_t1_limitup"].nunique() < 2:
        return out

    predicted_rate = float(history["p_limitup_t1_raw"].clip(0.2, 99.8).mean()) / 100
    positives = float(history["label_t1_limitup"].sum())
    observed_rate = (positives + 1.0) / (len(history) + 2.0)
    reliability = min(0.85, len(history) / (len(history) + 150.0))
    intercept_shift = (_logit(observed_rate) - _logit(predicted_rate)) * reliability
    calibrated = 1 / (1 + np.exp(-((raw / 100).map(_logit) + intercept_shift))) * 100
    calibrated = calibrated.clip(0.2, 35.0)
    out["p_limitup_t1"] = calibrated
    out["calibration_sample_count"] = int(len(history))
    out["self_learning_adjustment"] = calibrated - raw
    out["calibration_method"] = "logit_intercept_v1"
    return out
