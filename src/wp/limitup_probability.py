from __future__ import annotations

import numpy as np
import pandas as pd


def add_limitup_probability(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    coverage_source = out["feature_coverage"] if "feature_coverage" in out.columns else pd.Series(70.0, index=out.index)
    coverage = pd.to_numeric(coverage_source, errors="coerce").fillna(70).clip(0, 100)
    reliability = 0.55 + coverage / 100 * 0.45
    if "ranking_score" in out.columns:
        evidence_score = pd.to_numeric(out["ranking_score"], errors="coerce").fillna(50)
    elif "wp_score" in out.columns:
        evidence_score = pd.to_numeric(out["wp_score"], errors="coerce").fillna(50)
    else:
        components = [
            pd.to_numeric(out[column], errors="coerce").fillna(50)
            for column in [
                "sector_strength_score",
                "stock_strength_score",
                "acceptance_score",
                "momentum_score",
                "capital_score",
                "pattern_score",
            ]
            if column in out.columns
        ]
        evidence_score = pd.concat(components, axis=1).mean(axis=1) if components else pd.Series(50.0, index=out.index)
    evidence = 0.040 * (evidence_score - 50)
    base_rate = 0.01
    base_logit = np.log(base_rate / (1 - base_rate))
    z = base_logit + evidence * reliability
    out["p_limitup_t1"] = (1 / (1 + np.exp(-z)) * 100).clip(0.2, 35.0)
    return out
