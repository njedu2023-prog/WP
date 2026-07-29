from __future__ import annotations

import numpy as np
import pandas as pd

from wp.v3.contracts import V3Config
from wp.v3.diagnostics import build_prediction_diagnostics, diagnostics_tables


def _predictions() -> pd.DataFrame:
    rows = []
    for day in range(20):
        for slot in ("14:20", "14:25"):
            for stock in range(20):
                score = stock / 20 + day / 2_000
                positive = stock >= 12
                rows.append(
                    {
                        "trade_date": f"202601{day + 1:02d}",
                        "signal_slot": slot,
                        "ts_code": f"600{stock:03d}.SH",
                        "target_net_positive": int(positive),
                        "net_return_pct": 1.0 if positive else -1.0,
                        "target_entry_fillable": 1,
                        "target_exit_fillable": 1,
                        "p_net_positive": 0.2 + 0.6 * score,
                        "p_net_positive_lower": 0.15 + 0.55 * score,
                        "p_entry_fill": 0.98 + 0.01 * score,
                        "p_exit_fill_given_entry": 0.985 + 0.01 * score,
                        "p_round_trip_fill": 0.97 + 0.01 * score,
                        "p_conditional_net_positive": 0.25 + 0.6 * score,
                        "expected_utility_pct": -0.8 + 2.0 * score,
                        "expected_utility_lower_pct": -0.9 + 2.0 * score,
                        "expected_return_model_spread": 0.05,
                        "conditional_expected_net_return_pct": -0.7 + 2.0 * score,
                        "downside_q10_pct": -5.0 + 3.0 * score,
                        "probability_model_spread": 0.04,
                        "execution_eligible": True,
                        "passes_probability": score >= 2 / 3,
                        "passes_probability_lower": score >= 2 / 3,
                        "passes_expected_utility": score >= 0.55,
                        "passes_expected_utility_lower": score >= 0.55,
                        "passes_downside": True,
                        "passes_prior_oos_evidence": score >= 2 / 3,
                        "passes_stability": True,
                        "passes_freshness": True,
                        "passes_policy": score >= 2 / 3,
                    }
                )
    return pd.DataFrame(rows)


def test_prediction_diagnostics_exposes_discrimination_and_policy_funnel():
    diagnostics = build_prediction_diagnostics(_predictions(), V3Config())

    assert (
        diagnostics["score_quality"]["executable_positive_probability"]["roc_auc"]
        > 0.99
    )
    assert diagnostics["score_deciles"]
    probability_top_one = next(
        row
        for row in diagnostics["top_n_per_slot"]
        if row["score"] == "executable_positive_probability"
        and row["top_n"] == 1
    )
    assert probability_top_one["win_rate"] == 1.0
    assert diagnostics["policy_funnel"][-1]["gate"] == "final_policy"
    assert (
        diagnostics["policy_funnel"][-1]["cumulative_pass_count"]
        == int(_predictions()["passes_policy"].sum())
    )


def test_diagnostic_tables_are_flat_and_exportable():
    tables = diagnostics_tables(
        build_prediction_diagnostics(_predictions(), V3Config())
    )

    assert set(tables) == {
        "policy_funnel",
        "score_deciles",
        "top_n_per_slot",
        "extreme_rank_cohorts",
        "joint_gate_cohorts",
        "slot_quality",
    }
    assert not tables["top_n_per_slot"].empty
    assert not tables["extreme_rank_cohorts"].empty
    assert not tables["joint_gate_cohorts"].empty
    assert np.isfinite(
        tables["top_n_per_slot"]["mean_net_return_pct"].dropna()
    ).all()
