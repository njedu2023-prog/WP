from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from diagnose_wp_v24_evidence import (
    benjamini_hochberg,
    binary_auc,
    causal_feature_diagnostics,
    diagnose,
)
from wp.v3.v24_cross_section import MODEL_FEATURES


def synthetic_scored() -> pd.DataFrame:
    rows = []
    feature = MODEL_FEATURES[0]
    for day in range(120):
        year = 2023 + day // 30
        trade_date = f"{year}{(day % 12) + 1:02d}{(day % 27) + 1:02d}"
        for rank in range(5):
            signal = float(rank)
            net = signal - 2.0
            rows.append(
                {
                    "trade_date": trade_date,
                    "signal_slot": "14:20",
                    "ts_code": f"{600000 + rank:06d}.SH",
                    "net_return_pct": net,
                    feature: signal,
                    "v23_p_positive": 0.20 + rank * 0.15,
                    "v23_p_margin": 0.15 + rank * 0.15,
                    "v23_p_severe_loss": 0.80 - rank * 0.15,
                    "v23_expected_net_return_pct": net,
                    "v23_expected_net_return_lower_pct": net - 0.1,
                    "v23_economic_score": net,
                    "v24_cross_section_score": net,
                    "v24_source_fold": 5 + day // 10,
                    "v20_stock_rank_in_slot": rank + 1,
                    "v20_leader_appearances_so_far": 1,
                }
            )
    return pd.DataFrame(rows)


def test_auc_and_bh_helpers_are_deterministic() -> None:
    target = pd.Series([0, 0, 1, 1])
    score = pd.Series([0.1, 0.2, 0.8, 0.9])
    assert binary_auc(target, score) == 1.0
    adjusted = benjamini_hochberg([0.01, 0.04, 0.20])
    assert np.allclose(adjusted, [0.03, 0.06, 0.20])


def test_stable_within_slot_feature_is_detected() -> None:
    diagnostics = causal_feature_diagnostics(synthetic_scored())
    row = diagnostics.loc[
        diagnostics["feature"].eq(MODEL_FEATURES[0])
    ].iloc[0]
    assert row["cross_sections"] == 120
    assert row["mean_daily_slot_ic"] > 0.99
    assert row["positive_years"] == 4
    assert row["stable_exploratory_signal"]


def test_diagnosis_rejects_selected_count_mismatch() -> None:
    scored = synthetic_scored()
    selected = scored.head(10).copy()
    summary = {
        "schema_version": "wp_v24_cross_section_microstructure_1",
        "nested_oos_metrics": {"events": 11},
        "folds": [],
    }
    with pytest.raises(RuntimeError, match="selected count mismatch"):
        diagnose(summary, scored, selected)
