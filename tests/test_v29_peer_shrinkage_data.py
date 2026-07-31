from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wp.v3.v28_industry_peer import PEER_FEATURE_SUFFIXES
from wp.v3.v29_peer_shrinkage import (
    V29_FEATURE_COLUMNS,
    audit_hierarchical_feature_coverage,
    build_hierarchical_peer_features,
)


def source_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "trade_date": ["20250102", "20250103", "20250106"],
            "signal_slot": ["14:20", "14:25", "14:30"],
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "fold": [1, 1, 1],
        }
    )
    for level in ("l2", "l3"):
        for suffix in PEER_FEATURE_SUFFIXES:
            frame[f"v28_{level}_peer_{suffix}"] = 1.0
    frame["v28_l2_peer_count"] = [10.0, 5.0, 0.0]
    frame["v28_l3_peer_count"] = [6.0, 0.0, 0.0]
    frame["v28_l2_peer_return_count"] = [10.0, 5.0, 0.0]
    frame["v28_l3_peer_return_count"] = [6.0, 0.0, 0.0]
    for suffix in PEER_FEATURE_SUFFIXES:
        if suffix in {"count", "return_count"}:
            continue
        frame[f"v28_l2_peer_{suffix}"] = [2.0, 4.0, np.nan]
        frame[f"v28_l3_peer_{suffix}"] = [6.0, np.nan, np.nan]
    return frame


def test_hierarchical_features_shrink_l3_to_l2_and_fallback() -> None:
    features = build_hierarchical_peer_features(source_frame())

    assert features.loc[0, "v29_peer_return_median_pct"] == 4.0
    assert features.loc[0, "v29_peer_l3_weight"] == 0.5
    assert features.loc[1, "v29_peer_return_median_pct"] == 4.0
    assert features.loc[1, "v29_peer_l3_weight"] == 0.0
    assert features.loc[2, "v29_peer_return_median_pct"] == 0.0
    assert features.loc[2, "v29_peer_no_peer_context"] == 1.0
    assert not features.loc[:, V29_FEATURE_COLUMNS].isna().any().any()


def test_hierarchical_feature_audit_requires_exact_identity() -> None:
    source = source_frame()
    features = build_hierarchical_peer_features(source)
    candidates = source[
        ["trade_date", "signal_slot", "ts_code", "fold"]
    ].copy()

    audit = audit_hierarchical_feature_coverage(features, candidates)

    assert audit["candidate_identity_match"]
    assert audit["finite_feature_coverage"] == 1.0
    assert audit["coverage_passed"]
    assert audit["no_peer_context_rows"] == 1


def test_v29_data_builder_does_not_read_profit_outcomes() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / "src" / "wp" / "v3" / "v29_peer_shrinkage.py",
        root / "scripts" / "build_wp_v29_peer_shrinkage_data.py",
    ]
    forbidden = (
        "gross_return_pct",
        "net_return_pct",
        "target_net_return",
        "t1_close",
        "label_available",
    )
    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden)
