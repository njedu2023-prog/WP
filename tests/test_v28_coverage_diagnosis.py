from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.diagnose_wp_v28_feature_coverage import coverage_diagnosis
from wp.v3.v28_industry_peer import V28_PEER_FEATURE_COLUMNS


def diagnostic_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "trade_date": ["20250102", "20250103", "20250106"],
            "signal_slot": ["14:20", "14:20", "14:20"],
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
        }
    )
    for column in V28_PEER_FEATURE_COLUMNS:
        frame[column] = 1.0
    frame["v28_l2_peer_count"] = [4, 4, 4]
    frame["v28_l3_peer_count"] = [2, 1, 2]
    frame.loc[2, "v28_l3_peer_own_log_amount_excess"] = np.nan
    return frame


def test_diagnosis_separates_shallow_peers_from_engineering_missingness() -> None:
    result = coverage_diagnosis(diagnostic_frame())

    assert result["rows"] == 3
    assert result["complete_rows"] == 1
    assert result["l3_peer_depth_below_2"] == 1
    assert result["incomplete_with_sufficient_peer_depth"] == 1
    assert result["amount_only_missing_with_sufficient_depth"] == 1


def test_diagnostic_source_does_not_read_outcomes() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "diagnose_wp_v28_feature_coverage.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "gross_return_pct",
        "net_return_pct",
        "t1_close",
        "target_net_return",
        "label_available",
    )

    assert not any(token in source for token in forbidden)
