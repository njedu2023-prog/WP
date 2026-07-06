from __future__ import annotations

import numpy as np
import pandas as pd


def add_limitup_probability(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    z = (
        0.030 * out["sector_strength_score"]
        + 0.028 * out["stock_strength_score"]
        + 0.026 * out["acceptance_score"]
        + 0.014 * out["momentum_score"]
        + 0.012 * out["capital_score"]
        + 0.008 * out["pattern_score"]
        - 0.035 * out["risk_penalty_score"]
        - 4.0
    )
    out["p_limitup_t1"] = (1 / (1 + np.exp(-z)) * 100).clip(0, 100)
    return out
