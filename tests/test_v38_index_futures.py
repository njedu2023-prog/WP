from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from wp.v3.v38_index_futures import (
    PAIR_SPECS,
    SIGNAL_SLOTS,
    V38_FEATURE_COLUMNS,
    audit_probe_contract,
    build_regime_features,
    expected_minute_rows,
    normalize_historical_etf_minutes,
    normalize_historical_future_minutes,
    normalize_mapping,
    normalize_realtime_etf_minutes,
    normalize_realtime_future_minutes,
)


def _times(trade_date: str) -> pd.DatetimeIndex:
    iso = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    morning = pd.date_range(
        f"{iso} 09:30:00",
        f"{iso} 11:30:00",
        freq="1min",
    )
    afternoon = pd.date_range(
        f"{iso} 13:01:00",
        f"{iso} 14:50:00",
        freq="1min",
    )
    return morning.append(afternoon)


def _market_minutes(
    trade_date: str,
    ts_code: str,
    *,
    future: bool,
    offset: float,
) -> pd.DataFrame:
    times = _times(trade_date)
    sequence = np.arange(len(times), dtype=float)
    base = (3500.0 if future else 3.5) + offset
    scale = 0.08 if future else 0.00008
    close = (
        base
        + sequence * scale
        + np.sin(sequence / (7.0 + offset)) * scale * 6.0
    )
    result = pd.DataFrame(
        {
            "ts_code": ts_code,
            "trade_time": times,
            "open": close - scale,
            "close": close,
            "high": close + scale * 2.0,
            "low": close - scale * 2.0,
            "vol": 1000.0 + sequence * (3.0 + offset),
            "amount": (1000.0 + sequence * (3.0 + offset)) * close,
        }
    )
    if future:
        result["oi"] = 100000.0 + sequence * (2.0 + offset)
    return result


def _probe_frames(
    dates: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mappings: list[dict[str, str]] = []
    etfs: list[pd.DataFrame] = []
    futures: list[pd.DataFrame] = []
    for date_index, trade_date in enumerate(dates):
        for pair_index, spec in enumerate(PAIR_SPECS):
            contract = (
                f"{spec['continuous_code'][:2]}"
                f"{23 + date_index:02d}{pair_index + 1:02d}.CFX"
            )
            mappings.append(
                {
                    "ts_code": spec["continuous_code"],
                    "trade_date": trade_date,
                    "mapping_ts_code": contract,
                }
            )
            offset = float(date_index * 3 + pair_index + 1)
            etfs.append(
                _market_minutes(
                    trade_date,
                    spec["etf_code"],
                    future=False,
                    offset=offset / 100.0,
                )
            )
            futures.append(
                _market_minutes(
                    trade_date,
                    contract,
                    future=True,
                    offset=offset,
                )
            )
    return (
        pd.DataFrame(mappings),
        pd.concat(etfs, ignore_index=True),
        pd.concat(futures, ignore_index=True),
    )


def test_expected_rows_follow_a_share_sessions() -> None:
    assert expected_minute_rows("14:20") == 201
    assert expected_minute_rows("14:50") == 231


def test_historical_and_realtime_schemas_have_parity() -> None:
    etf = _market_minutes(
        "20260723",
        "510300.SH",
        future=False,
        offset=0.1,
    ).head(3)
    future = _market_minutes(
        "20260723",
        "IF2608.CFX",
        future=True,
        offset=1.0,
    ).head(3)
    realtime_etf = etf.rename(
        columns={"ts_code": "code", "trade_time": "time"}
    ).assign(freq="1MIN")
    realtime_future = future.rename(
        columns={"ts_code": "code", "trade_time": "time"}
    ).assign(freq="1MIN")
    pdt.assert_frame_equal(
        normalize_historical_etf_minutes(etf),
        normalize_realtime_etf_minutes(realtime_etf),
    )
    pdt.assert_frame_equal(
        normalize_historical_future_minutes(future),
        normalize_realtime_future_minutes(realtime_future),
    )


def test_mapping_normalization_rejects_non_index_future_contracts() -> None:
    frame = pd.DataFrame(
        {
            "ts_code": ["IF.CFX", "CU.SHF"],
            "trade_date": ["20260723", "20260723"],
            "mapping_ts_code": ["IF2608.CFX", "CU2608.SHF"],
        }
    )
    result = normalize_mapping(frame, required_dates=("20260723",))
    assert result.to_dict(orient="records") == [
        {
            "ts_code": "IF.CFX",
            "trade_date": "20260723",
            "mapping_ts_code": "IF2608.CFX",
        }
    ]


def test_post_signal_mutation_cannot_change_earlier_regime_features() -> None:
    dates = ("20260723",)
    mappings, etfs, futures = _probe_frames(dates)
    baseline = build_regime_features(dates, mappings, etfs, futures)
    cutoff = pd.Timestamp("2026-07-23 14:20:00")
    changed_etfs = etfs.copy()
    changed_futures = futures.copy()
    changed_etfs.loc[
        changed_etfs["trade_time"].gt(cutoff),
        ["open", "close", "high", "low", "amount"],
    ] *= 100.0
    changed_futures.loc[
        changed_futures["trade_time"].gt(cutoff),
        ["open", "close", "high", "low", "amount", "oi"],
    ] *= 100.0
    replay = build_regime_features(
        dates,
        mappings,
        changed_etfs,
        changed_futures,
    )
    left = baseline.loc[
        baseline["signal_slot"].eq("14:20"),
        list(V38_FEATURE_COLUMNS),
    ].reset_index(drop=True)
    right = replay.loc[
        replay["signal_slot"].eq("14:20"),
        list(V38_FEATURE_COLUMNS),
    ].reset_index(drop=True)
    pdt.assert_frame_equal(left, right)


def test_complete_probe_builds_all_date_slot_rows() -> None:
    dates = (
        "20230825",
        "20231229",
        "20240315",
        "20240927",
        "20250115",
        "20250723",
        "20260115",
        "20260723",
    )
    mappings, etfs, futures = _probe_frames(dates)
    features = build_regime_features(dates, mappings, etfs, futures)
    assert len(features) == len(dates) * len(SIGNAL_SLOTS)
    assert features["v38_all_pairs_complete"].all()
    assert features["v38_causal_ok"].all()
    assert features["v38_finite_features"].all()
    assert np.isfinite(
        features[list(V38_FEATURE_COLUMNS)].to_numpy(dtype=float)
    ).all()
    audit = audit_probe_contract(
        features,
        mappings,
        probe_dates=dates,
        query_failures=0,
    )
    assert audit["mapping_exact"]
    assert audit["identity_exact"]
    assert audit["coverage_passed"]


def test_audit_rejects_outcome_fields_and_query_failures() -> None:
    dates = (
        "20230825",
        "20231229",
        "20240315",
        "20240927",
        "20250115",
        "20250723",
        "20260115",
        "20260723",
    )
    mappings, etfs, futures = _probe_frames(dates)
    features = build_regime_features(dates, mappings, etfs, futures)
    contaminated = features.assign(target_net_positive=1)
    audit = audit_probe_contract(
        contaminated,
        mappings,
        probe_dates=dates,
        query_failures=1,
    )
    assert not audit["coverage_passed"]
    assert "target_net_positive" in audit["forbidden_columns"]


def test_ambiguous_mapping_is_rejected() -> None:
    dates = ("20260723",)
    mappings, etfs, futures = _probe_frames(dates)
    duplicate = mappings.iloc[[0]].copy()
    duplicate["mapping_ts_code"] = "IF2609.CFX"
    with pytest.raises(ValueError, match="ambiguous"):
        build_regime_features(
            dates,
            pd.concat([mappings, duplicate], ignore_index=True),
            etfs,
            futures,
        )
