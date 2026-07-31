from __future__ import annotations

import json

import numpy as np
import pandas as pd

from wp.v3.io import file_sha256
from wp.v3.sharding import (
    SHARD_MANIFEST_NAME,
    SHARD_PREDICTIONS_NAME,
    SHARD_SCHEMA_VERSION,
)
from wp.v3.v23_data import (
    SOURCE_SELECTION_COLUMNS,
    V23_FEATURE_COLUMNS,
    assemble_v23_feature_frame,
    attach_previous_trade_dates,
    build_auction_features,
    build_minute_features,
    build_moneyflow_features,
    feature_coverage_audit,
    load_v23_source_leaders,
    required_stock_months,
)


def leaders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20260724", "20260724"],
            "signal_slot": ["14:20", "14:50"],
            "ts_code": ["600000.SH", "600000.SH"],
            "fold": [10, 10],
            "signal_price": [10.20, 10.50],
            "ret_from_prev_close_pct": [2.0, 5.0],
            "v23_prev_trade_date": ["20260723", "20260723"],
        }
    )


def minute_bars() -> pd.DataFrame:
    times = pd.date_range(
        "2026-07-24 13:55:00",
        "2026-07-24 14:50:00",
        freq="1min",
    )
    close = 10.0 + np.arange(len(times)) * 0.01
    return pd.DataFrame(
        {
            "ts_code": "600000.SH",
            "trade_time": times,
            "open": close - 0.005,
            "high": close + 0.02,
            "low": close - 0.02,
            "close": close,
            "vol": 10_000.0 + np.arange(len(times)) * 10.0,
            "amount": 100_000.0 + np.arange(len(times)) * 100.0,
        }
    )


def auctions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["600000.SH"],
            "trade_date": ["20260724"],
            "close": [10.0],
            "open": [9.95],
            "high": [10.05],
            "low": [9.90],
            "vol": [1_000_000.0],
            "amount": [10_000_000.0],
            "vwap": [9.98],
        }
    )


def moneyflow() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["600000.SH"],
            "trade_date": ["20260723"],
            "buy_sm_amount": [100.0],
            "sell_sm_amount": [120.0],
            "buy_md_amount": [200.0],
            "sell_md_amount": [180.0],
            "buy_lg_amount": [300.0],
            "sell_lg_amount": [220.0],
            "buy_elg_amount": [400.0],
            "sell_elg_amount": [260.0],
            "net_mf_amount": [220.0],
        }
    )


def test_source_projection_is_outcome_blind() -> None:
    forbidden = (
        "target",
        "label",
        "truth",
        "future",
        "gross_return",
        "net_return",
        "t1_",
        "exit_",
    )
    assert not [
        column
        for column in SOURCE_SELECTION_COLUMNS
        if any(token in column.lower() for token in forbidden)
    ]


def test_source_loader_reads_only_outcome_blind_projection(tmp_path) -> None:
    rows = []
    for index, code in enumerate(("600000.SH", "000001.SZ")):
        row = {column: np.nan for column in SOURCE_SELECTION_COLUMNS}
        row.update(
            {
                "trade_date": "20260724",
                "signal_slot": "14:20",
                "ts_code": code,
                "fold": 1,
                "signal_price": 10.0 + index,
                "ret_from_prev_close_pct": 2.0 + index,
                "execution_eligible": True,
                "data_age_seconds": 30.0,
                "p_net_positive": 0.60 - index * 0.01,
                "p_net_positive_lower": 0.55 - index * 0.01,
                "p_conditional_net_positive": 0.62,
                "p_cross_section_top": 0.58,
                "p_severe_loss": 0.20,
                "p_round_trip_fill_lower": 0.99,
                "probability_model_spread": 0.05,
                "expected_return_model_spread": 0.05,
                "expected_utility_pct": 0.30,
                "expected_utility_lower_pct": 0.20,
                "selection_score": 0.80 - index * 0.10,
                "model_version": "v9",
                "model_fingerprint": "model-1",
                "policy_fingerprint": "policy-1",
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    prediction_path = tmp_path / SHARD_PREDICTIONS_NAME
    frame.to_parquet(prediction_path, index=False)
    manifest = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "expected_folds": [1],
        "produced_folds": [1],
        "prediction_rows": len(frame),
        "prediction_sha256": file_sha256(prediction_path),
        "dataset_manifest_sha256": "dataset-1",
    }
    (tmp_path / SHARD_MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    loaded, source = load_v23_source_leaders(
        tmp_path,
        evaluation_end="20260724",
        top_per_source=2,
        exploration_per_slot=0,
    )

    assert len(loaded) == 1
    assert loaded.iloc[0]["ts_code"] == "600000.SH"
    assert source["profit_outcomes_read"] is False
    assert source["source_integrity"] is True


def test_source_loader_accepts_legacy_missing_data_age(tmp_path) -> None:
    rows = []
    for index, code in enumerate(("600000.SH", "000001.SZ")):
        row = {
            column: np.nan
            for column in SOURCE_SELECTION_COLUMNS
            if column != "data_age_seconds"
        }
        row.update(
            {
                "trade_date": "20260724",
                "signal_slot": "14:20",
                "ts_code": code,
                "fold": 1,
                "signal_price": 10.0 + index,
                "ret_from_prev_close_pct": 2.0 + index,
                "execution_eligible": True,
                "p_net_positive": 0.60 - index * 0.01,
                "p_net_positive_lower": 0.55 - index * 0.01,
                "p_conditional_net_positive": 0.62,
                "p_cross_section_top": 0.58,
                "p_severe_loss": 0.20,
                "p_round_trip_fill_lower": 0.99,
                "probability_model_spread": 0.05,
                "expected_return_model_spread": 0.05,
                "expected_utility_pct": 0.30,
                "expected_utility_lower_pct": 0.20,
                "selection_score": 0.80 - index * 0.10,
                "model_version": "v9",
                "model_fingerprint": "model-1",
                "policy_fingerprint": "policy-1",
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    prediction_path = tmp_path / SHARD_PREDICTIONS_NAME
    frame.to_parquet(prediction_path, index=False)
    manifest = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "expected_folds": [1],
        "produced_folds": [1],
        "prediction_rows": len(frame),
        "prediction_sha256": file_sha256(prediction_path),
        "dataset_manifest_sha256": "dataset-1",
    }
    (tmp_path / SHARD_MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    loaded, source = load_v23_source_leaders(
        tmp_path,
        evaluation_end="20260724",
        top_per_source=2,
        exploration_per_slot=0,
    )

    assert len(loaded) == 1
    assert source["source_integrity"] is True
    assert source["shards"][0]["missing_optional_columns"] == [
        "data_age_seconds"
    ]


def test_required_pairs_and_previous_trade_date_are_causal() -> None:
    source = leaders().drop(columns="v23_prev_trade_date")
    requirements = required_stock_months(source)
    assert requirements == {
        "202607": {"600000.SH": ("20260724",)}
    }
    attached = attach_previous_trade_dates(
        source,
        ["20260722", "20260723", "20260724"],
    )
    assert set(attached["v23_prev_trade_date"]) == {"20260723"}
    assert (
        attached["v23_prev_trade_date"] < attached["trade_date"]
    ).all()


def test_minute_features_never_read_after_signal() -> None:
    source = leaders()
    bars = minute_bars()
    original = build_minute_features(source, bars)
    modified = bars.copy()
    modified.loc[
        modified["trade_time"].dt.strftime("%H:%M").gt("14:20"),
        ["open", "high", "low", "close"],
    ] *= 5.0
    changed = build_minute_features(source, modified)

    first = original.loc[original["signal_slot"].eq("14:20")].iloc[0]
    first_changed = changed.loc[
        changed["signal_slot"].eq("14:20")
    ].iloc[0]
    assert first["v23_minute_latest_time"].endswith("14:20:00")
    assert first["v23_m1_coverage_ratio"] == 1.0
    assert first["v23_m1_ret_5m_pct"] == first_changed[
        "v23_m1_ret_5m_pct"
    ]
    assert first["v23_m1_vwap_gap_pct"] == first_changed[
        "v23_m1_vwap_gap_pct"
    ]
    later = original.loc[original["signal_slot"].eq("14:50")].iloc[0]
    later_changed = changed.loc[
        changed["signal_slot"].eq("14:50")
    ].iloc[0]
    assert later["v23_m1_ret_from_1400_pct"] != later_changed[
        "v23_m1_ret_from_1400_pct"
    ]


def test_complete_point_in_time_feature_frame_passes_audit() -> None:
    source = leaders()
    minute = build_minute_features(source, minute_bars())
    auction = build_auction_features(source, auctions())
    flow = build_moneyflow_features(source, moneyflow())
    features = assemble_v23_feature_frame(
        source,
        minute,
        auction,
        flow,
    )
    audit = feature_coverage_audit(features)

    assert len(features) == len(source)
    assert set(V23_FEATURE_COLUMNS).issubset(features.columns)
    assert features["v23_point_in_time_complete"].all()
    assert audit["coverage_passed"]
    assert audit["complete_coverage_rate"] == 1.0
