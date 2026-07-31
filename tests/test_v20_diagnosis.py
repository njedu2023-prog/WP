from __future__ import annotations

import pandas as pd

from scripts.diagnose_wp_v20_evidence import (
    diagnose,
    quantile_groups,
)


def test_quantile_bins_do_not_depend_on_outcomes() -> None:
    frame = _frame()
    first = quantile_groups(frame, "v20_gate_score")
    changed = frame.copy()
    changed["net_return_pct"] *= -100.0
    second = quantile_groups(changed, "v20_gate_score")

    pd.testing.assert_frame_equal(
        first[
            ["dimension", "group", "group_order", "bin_lower", "bin_upper"]
        ],
        second[
            ["dimension", "group", "group_order", "bin_lower", "bin_upper"]
        ],
    )


def test_diagnosis_is_exploratory_and_count_bound() -> None:
    frame = _frame()
    summary = {
        "schema_version": "wp_v20_hierarchical_opportunity_1",
        "nested_oos_metrics": {"events": len(frame)},
    }

    result, groups, correlations = diagnose(summary, frame)

    assert not result["successor_policy_authorized"]
    assert not result["outcome_driven_threshold_selection_allowed"]
    assert result["overall"]["events"] == len(frame)
    assert not groups.empty
    assert not correlations.empty


def _frame() -> pd.DataFrame:
    rows = []
    for index in range(40):
        rows.append(
            {
                "trade_date": f"2026{index // 20 + 1:02d}{index % 20 + 1:02d}",
                "signal_slot": "14:20" if index % 2 == 0 else "14:35",
                "ts_code": f"{index:06d}.SZ",
                "net_return_pct": (index % 7 - 3) * 0.50,
                "v20_source_fold": 20 + index // 20,
                "v20_stock_rank_in_slot": index % 3 + 1,
                "v20_leader_appearances_so_far": index % 4 + 1,
                "v20_gate_score": index / 40.0,
                "p_severe_loss": 0.30 - index / 400.0,
            }
        )
    return pd.DataFrame(rows)
