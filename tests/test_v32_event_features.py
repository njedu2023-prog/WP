from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wp.v3.v31_public_event import SOURCE_SPECS
from wp.v3.v32_event_features import (
    ADMITTED_SOURCES,
    V32_EVENT_FEATURE_COLUMNS,
    audit_candidate_event_features,
    build_candidate_event_features,
)
from wp.v3.v32_public_event import normalize_a_share_event_frame


def event_frame(
    source: str,
    *,
    rows: list[dict[str, object]],
) -> pd.DataFrame:
    spec = SOURCE_SPECS[source]
    base_rows = []
    for row in rows:
        values = {
            column: None for column in spec["fields"].split(",")
        }
        values.update(row)
        base_rows.append(values)
    return normalize_a_share_event_frame(
        pd.DataFrame(base_rows),
        source=source,
    )


def empty_events() -> dict[str, pd.DataFrame]:
    return {
        source: pd.DataFrame(
            columns=[
                *SOURCE_SPECS[source]["fields"].split(","),
                "event_date",
                "event_source",
            ]
        )
        for source in ADMITTED_SOURCES
    }


def candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20260723", "20260723"],
            "signal_slot": ["14:20", "14:25"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "fold": [1, 1],
            "signal_price": [10.0, 10.5],
        }
    )


def test_event_features_are_causal_and_signal_price_relative() -> None:
    events = empty_events()
    events["forecast"] = event_frame(
        "forecast",
        rows=[
            {
                "ts_code": "000001.SZ",
                "ann_date": "20260722",
                "p_change_min": 10.0,
                "p_change_max": 20.0,
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": "20260718",
                "p_change_min": -5.0,
                "p_change_max": 5.0,
            },
        ],
    )
    events["block_trade"] = event_frame(
        "block_trade",
        rows=[
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260722",
                "price": 9.5,
                "amount": 100.0,
                "vol": 10.0,
            }
        ],
    )
    lookback = {
        "20260723": [
            "20260717",
            "20260718",
            "20260720",
            "20260721",
            "20260722",
        ]
    }

    result = build_candidate_event_features(
        candidates(),
        events,
        lookback,
    )

    assert len(result) == 2
    assert result["v32_forecast_event_count_5d"].eq(2).all()
    assert result["v32_forecast_latest_age_td"].eq(1).all()
    assert result["v32_forecast_p_change_mid_mean"].eq(7.5).all()
    assert result["v32_block_trade_event_count_5d"].eq(1).all()
    assert np.isclose(
        result.loc[
            result["signal_slot"].eq("14:20"),
            "v32_block_trade_weighted_price_to_signal_pct",
        ].iloc[0],
        -5.0,
    )
    assert result.loc[
        result["signal_slot"].eq("14:25"),
        "v32_block_trade_weighted_price_to_signal_pct",
    ].iloc[0] < -9.0


def test_no_event_rows_have_complete_counts_and_flags() -> None:
    result = build_candidate_event_features(
        candidates(),
        empty_events(),
        {
            "20260723": [
                "20260717",
                "20260718",
                "20260720",
                "20260721",
                "20260722",
            ]
        },
    )

    for source in ADMITTED_SOURCES:
        assert result[f"v32_{source}_event_count_5d"].eq(0).all()
        assert result[f"v32_{source}_active_5d"].eq(0).all()
    assert set(V32_EVENT_FEATURE_COLUMNS).issubset(result.columns)


def test_feature_audit_requires_detail_coverage_for_active_rows() -> None:
    events = empty_events()
    events["block_trade"] = event_frame(
        "block_trade",
        rows=[
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260722",
                "price": 9.5,
                "amount": 100.0,
                "vol": 10.0,
            }
        ],
    )
    result = build_candidate_event_features(
        candidates(),
        events,
        {
            "20260723": [
                "20260717",
                "20260718",
                "20260720",
                "20260721",
                "20260722",
            ]
        },
    )
    audit = audit_candidate_event_features(result, candidates())

    assert audit["identity_match"]
    assert audit["common_features_complete"]
    assert audit["active_flags_consistent"]
    assert audit["detail_coverage_by_source"]["block_trade"] == 1.0


def test_full_builder_source_does_not_read_profit_outcomes() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "scripts" / "build_wp_v32_public_event_data.py",
        root / "src" / "wp" / "v3" / "v32_event_features.py",
    ]
    forbidden = (
        "gross_return",
        "net_return",
        "target_return",
        "outcome_label",
        "t1_close",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert not any(token in text for token in forbidden)
