from __future__ import annotations

import pandas as pd

from scripts.diagnose_wp_v16_evidence import (
    classify_bottleneck,
    policy_attrition,
)
from wp.v3.v16_policy import ExpertPolicy


def test_policy_attrition_reports_fixed_threshold_bottleneck() -> None:
    frame = pd.DataFrame(
        [
            {
                "trade_date": "20260720",
                "signal_slot": "14:20",
                "ts_code": "600001.SH",
                "expert_count": 2,
                "expert_p_positive_lower": 0.51,
                "expert_expected_return_lower_pct": 0.20,
                "expert_p_severe": 0.10,
                "p_round_trip_fill_lower": 0.99,
                "expert_probability_spread": 0.05,
                "expert_score": 0.40,
            }
        ]
    )
    policy = ExpertPolicy(0.52, 0.0, 0.35, 0.95, 1, 0.18, 0.90, 3, "all")

    attrition = policy_attrition(frame, policy)

    assert attrition[0]["rows"] == 1
    assert attrition[-1]["rows"] == 0
    assert (
        classify_bottleneck(
            attrition,
            positive_mean_with_sample=0,
            stress_nonnegative=0,
            q_significant=0,
        )
        == "fixed_thresholds_eliminate_all_candidates"
    )
