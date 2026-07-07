from __future__ import annotations

from pathlib import Path

import pandas as pd


BIN_EDGES = [0, 15, 25, 35, 45, 60, 101]


def _load_history(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((root / "outputs" / "backtests").glob("*/trades.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if {"p_limitup_t1", "label_t1_limitup"}.issubset(frame.columns):
            frames.append(frame[["p_limitup_t1", "label_t1_limitup"]].copy())
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["p_limitup_t1"] = pd.to_numeric(out["p_limitup_t1"], errors="coerce")
    out["label_t1_limitup"] = pd.to_numeric(out["label_t1_limitup"], errors="coerce")
    return out.dropna()


def _bin_index(value: float) -> int:
    for index in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[index] <= value < BIN_EDGES[index + 1]:
            return index
    return len(BIN_EDGES) - 2


def apply_statistical_calibration(df: pd.DataFrame, root: Path, min_samples: int = 30) -> pd.DataFrame:
    if df.empty or "p_limitup_t1" not in df.columns:
        return df.copy()
    out = df.copy()
    history = _load_history(root)
    out["p_limitup_t1_raw"] = out["p_limitup_t1"]
    out["calibration_sample_count"] = 0
    out["self_learning_adjustment"] = 0.0
    if len(history) < min_samples:
        return out

    history["bin"] = history["p_limitup_t1"].map(_bin_index)
    global_rate = float(history["label_t1_limitup"].mean()) * 100
    stats = history.groupby("bin")["label_t1_limitup"].agg(["count", "mean"])

    calibrated = []
    samples = []
    adjustments = []
    for value in pd.to_numeric(out["p_limitup_t1"], errors="coerce").fillna(0):
        bin_id = _bin_index(float(value))
        if bin_id in stats.index and int(stats.loc[bin_id, "count"]) >= 5:
            empirical = float(stats.loc[bin_id, "mean"]) * 100
            count = int(stats.loc[bin_id, "count"])
        else:
            empirical = global_rate
            count = int(len(history))
        reliability = min(0.90, count / (count + 25))
        score_prior = max(0.0, min(35.0, float(value) * 0.25))
        adjusted = max(0.0, min(100.0, score_prior * (1 - reliability) + empirical * reliability))
        calibrated.append(adjusted)
        samples.append(count)
        adjustments.append(adjusted - float(value))
    out["p_limitup_t1"] = calibrated
    out["calibration_sample_count"] = samples
    out["self_learning_adjustment"] = adjustments
    return out
