from __future__ import annotations

import numpy as np
import pandas as pd

from wp.v3.v25_positioning import (
    V25_FEATURE_COLUMNS,
    attach_candidate_signal_price,
    attach_positioning_features,
    positioning_coverage_audit,
)


def source() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": "20260724",
                "signal_slot": "14:20",
                "ts_code": "600000.SH",
                "signal_price": 11.0,
                "v23_prev_trade_date": "20260723",
            },
            {
                "trade_date": "20260724",
                "signal_slot": "14:25",
                "ts_code": "000001.SZ",
                "signal_price": 10.0,
                "v23_prev_trade_date": "20260723",
            },
        ]
    )


def cyq() -> pd.DataFrame:
    rows = []
    for code in ("600000.SH", "000001.SZ"):
        rows.append(
            {
                "trade_date": "20260723",
                "ts_code": code,
                "his_low": 5.0,
                "his_high": 15.0,
                "cost_5pct": 8.0,
                "cost_15pct": 8.5,
                "cost_50pct": 9.5,
                "cost_85pct": 10.5,
                "cost_95pct": 11.0,
                "weight_avg": 9.6,
                "winner_rate": 63.0,
            }
        )
    return pd.DataFrame(rows)


def margin() -> pd.DataFrame:
    rows = []
    for date, multiplier in (("20260722", 1.0), ("20260723", 1.1)):
        for code in ("600000.SH", "000001.SZ"):
            rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "name": code,
                    "rzye": 1_000_000 * multiplier,
                    "rqye": 10_000 * multiplier,
                    "rzmre": 100_000,
                    "rqyl": 1_000,
                    "rzche": 80_000,
                    "rqchl": 800,
                    "rqmcl": 1_200,
                    "rzrqye": 1_010_000 * multiplier,
                }
            )
    return pd.DataFrame(rows)


def top_list() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": "20260723",
                "ts_code": "600000.SH",
                "name": "sample",
                "close": 10.0,
                "pct_change": 9.9,
                "turnover_rate": 12.0,
                "amount": 1_000_000,
                "l_sell": 300_000,
                "l_buy": 500_000,
                "l_amount": 800_000,
                "net_amount": 200_000,
                "net_rate": 0.25,
                "amount_rate": 0.10,
                "float_values": 2_000_000_000,
                "reason": "reason-a",
            }
        ]
    )


def test_positioning_features_use_previous_dates_and_keep_sparse_zero() -> None:
    result = attach_positioning_features(
        source(),
        cyq(),
        margin(),
        top_list(),
        open_dates=("20260722", "20260723", "20260724"),
    )

    assert len(result) == 2
    assert result[list(V25_FEATURE_COLUMNS)].shape[1] == len(
        V25_FEATURE_COLUMNS
    )
    first = result.loc[result["ts_code"].eq("600000.SH")].iloc[0]
    second = result.loc[result["ts_code"].eq("000001.SZ")].iloc[0]
    assert np.isclose(first["v25_cyq_signal_vs_weight_avg_pct"], 14.583333)
    assert np.isclose(
        first["v25_margin_financing_balance_change_pct"],
        10.0,
    )
    assert first["v25_toplist_flag"]
    assert not second["v25_toplist_flag"]
    assert second["v25_toplist_records"] == 0.0


def test_positioning_coverage_requires_core_families() -> None:
    result = attach_positioning_features(
        source(),
        cyq(),
        margin(),
        top_list(),
        open_dates=("20260722", "20260723", "20260724"),
    )
    audit = positioning_coverage_audit(result)

    assert audit["cyq_coverage"] == 1.0
    assert audit["margin_coverage"] == 1.0
    assert audit["top_list_event_rate"] == 0.5
    assert audit["coverage_passed"]


def test_missing_margin_is_measured_not_imputed_as_position() -> None:
    result = attach_positioning_features(
        source(),
        cyq(),
        margin().loc[lambda frame: frame["ts_code"].eq("600000.SH")],
        top_list(),
        open_dates=("20260722", "20260723", "20260724"),
    )
    missing = result.loc[result["ts_code"].eq("000001.SZ")].iloc[0]

    assert not missing["v25_margin_available"]
    assert np.isnan(missing["v25_margin_financing_balance_log"])


def test_candidate_index_restores_signal_price_without_reordering() -> None:
    feature_source = source().drop(columns="signal_price").iloc[::-1]
    candidate_index = source().loc[
        :,
        ["trade_date", "signal_slot", "ts_code", "signal_price"],
    ]

    result = attach_candidate_signal_price(
        feature_source,
        candidate_index,
    )

    assert result["ts_code"].tolist() == feature_source["ts_code"].tolist()
    assert result["signal_price"].tolist() == [10.0, 11.0]


def test_candidate_index_rejects_identity_mismatch() -> None:
    feature_source = source().drop(columns="signal_price")
    candidate_index = source().loc[
        :,
        ["trade_date", "signal_slot", "ts_code", "signal_price"],
    ].iloc[:1]

    with np.testing.assert_raises_regex(
        RuntimeError,
        "identities differ",
    ):
        attach_candidate_signal_price(feature_source, candidate_index)
