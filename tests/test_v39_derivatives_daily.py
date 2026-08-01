from __future__ import annotations

import numpy as np
import pandas as pd

from wp.v3.v39_derivatives_daily import (
    FUTURE_SPECS,
    OPTION_SPECS,
    V39_FUTURE_FEATURE_COLUMNS,
    V39_OPTION_FEATURE_COLUMNS,
    audit_probe_contract,
    build_derivative_features,
    normalize_option_basic,
)


DATES = (
    "20230825",
    "20231229",
    "20240315",
    "20240927",
    "20250115",
    "20250723",
    "20260115",
    "20260723",
)
PREVIOUS = {
    date: (pd.Timestamp(date) - pd.offsets.BDay(1)).strftime("%Y%m%d")
    for date in DATES
}


def _frames() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    mappings: list[dict[str, object]] = []
    futures: list[dict[str, object]] = []
    indices: list[dict[str, object]] = []
    basics: list[dict[str, object]] = []
    options: list[dict[str, object]] = []
    funds: list[dict[str, object]] = []
    for date_index, target_date in enumerate(DATES):
        source_date = PREVIOUS[target_date]
        for spec_index, spec in enumerate(FUTURE_SPECS):
            month = 2309 + date_index * 5 + spec_index
            year = 23 + (month // 100 - 23)
            numeric_month = month % 100
            if numeric_month > 12:
                year += numeric_month // 12
                numeric_month = ((numeric_month - 1) % 12) + 1
            code = f"{spec['prefix']}{year:02d}{numeric_month:02d}.CFX"
            next_month = numeric_month + 1
            next_year = year
            if next_month > 12:
                next_month = 1
                next_year += 1
            next_code = (
                f"{spec['prefix']}{next_year:02d}{next_month:02d}.CFX"
            )
            mappings.append(
                {
                    "ts_code": spec["continuous_code"],
                    "trade_date": source_date,
                    "mapping_ts_code": code,
                }
            )
            index_close = 3000.0 + date_index * 40 + spec_index * 100
            indices.append(
                {
                    "ts_code": spec["index_code"],
                    "trade_date": source_date,
                    "pre_close": index_close - 8,
                    "open": index_close - 3,
                    "high": index_close + 10,
                    "low": index_close - 12,
                    "close": index_close,
                    "pct_chg": 0.2,
                    "vol": 1e8,
                    "amount": 2e11,
                }
            )
            for contract_index, contract in enumerate((code, next_code)):
                settle = index_close * (
                    1.0
                    + 0.002 * (contract_index + 1)
                    + date_index * 0.0001
                )
                oi = 100000 + date_index * 1000 + spec_index * 5000
                contract_oi = oi / (contract_index + 1)
                if contract_index:
                    contract_oi *= 1.0 + date_index * 0.015
                futures.append(
                    {
                        "ts_code": contract,
                        "trade_date": source_date,
                        "pre_close": settle - 5,
                        "pre_settle": settle - 4,
                        "open": settle - 2,
                        "high": settle + 8,
                        "low": settle - 10,
                        "close": settle + 1,
                        "settle": settle,
                        "vol": contract_oi
                        * (
                            0.6
                            + contract_index * 0.1
                            + date_index * 0.01
                            + spec_index * 0.005
                        ),
                        "amount": 2e9,
                        "oi": contract_oi,
                        "oi_chg": 100 + date_index * 7,
                    }
                )
        for option_index, spec in enumerate(OPTION_SPECS):
            spot = 3.0 + option_index + date_index * 0.02
            funds.append(
                {
                    "ts_code": spec["underlying_code"],
                    "trade_date": source_date,
                    "pre_close": spot - 0.01,
                    "open": spot,
                    "high": spot + 0.03,
                    "low": spot - 0.03,
                    "close": spot,
                    "pct_chg": 0.2,
                    "vol": 1e8,
                    "amount": 5e9,
                }
            )
            maturity = (
                pd.Timestamp(source_date) + pd.Timedelta(days=25)
            ).strftime("%Y%m%d")
            for strike_index in range(-5, 6):
                strike = spot + strike_index * 0.05
                for side in ("C", "P"):
                    code_number = (
                        10_000_000
                        + option_index * 1_000_000
                        + date_index * 10_000
                        + (strike_index + 5) * 2
                        + (0 if side == "C" else 1)
                    )
                    ts_code = f"{code_number:08d}.SH"
                    basics.append(
                        {
                            "ts_code": ts_code,
                            "symbol": str(code_number),
                            "exchange": "SSE",
                            "name": (
                                f"{spec['underlying_code']} "
                                f"{'购' if side == 'C' else '沽'}"
                            ),
                            "opt_code": f"OP{spec['underlying_code']}",
                            "opt_type": "ETF",
                            "call_put": side,
                            "exercise_price": strike,
                            "maturity_date": maturity,
                            "list_date": "20200101",
                            "delist_date": maturity,
                        }
                    )
                    intrinsic = (
                        max(spot - strike, 0)
                        if side == "C"
                        else max(strike - spot, 0)
                    )
                    premium = (
                        0.08
                        + intrinsic
                        + date_index * 0.002
                        + (0.01 if side == "P" else 0)
                    )
                    options.append(
                        {
                            "ts_code": ts_code,
                            "trade_date": source_date,
                            "exchange": "SSE",
                            "pre_settle": premium - 0.001,
                            "pre_close": premium - 0.001,
                            "open": premium,
                            "high": premium + 0.01,
                            "low": premium - 0.01,
                            "close": premium,
                            "settle": premium,
                            "vol": (
                                1000
                                + strike_index * 10
                                + date_index
                                * (3 if side == "P" else 1)
                            ),
                            "amount": (
                                100
                                + strike_index
                                + date_index
                                * (2 if side == "P" else 1)
                            ),
                            "oi": (
                                5000
                                + strike_index * 20
                                + date_index
                                * (4 if side == "P" else 1)
                            ),
                        }
                    )
    return (
        pd.DataFrame(mappings),
        pd.DataFrame(futures),
        pd.DataFrame(indices),
        pd.DataFrame(basics),
        pd.DataFrame(options),
        pd.DataFrame(funds),
    )


def test_option_basic_normalizes_underlying_and_side() -> None:
    *_, basics, _, _ = _frames()
    result = normalize_option_basic(basics)
    assert set(result["underlying_code"]) == {
        spec["underlying_code"] for spec in OPTION_SPECS
    }
    assert set(result["option_side"]) == {"call", "put"}


def test_probe_build_is_strictly_tminus1_and_finite() -> None:
    mappings, futures, indices, basics, options, funds = _frames()
    features = build_derivative_features(
        DATES,
        PREVIOUS,
        mappings,
        futures,
        indices,
        basics,
        options,
        funds,
    )
    assert len(features) == len(DATES)
    assert features["v39_tminus1_causal"].all()
    assert features["v39_futures_complete"].all()
    assert features["v39_options_complete"].all()
    assert np.isfinite(
        features[
            [
                *V39_FUTURE_FEATURE_COLUMNS,
                *V39_OPTION_FEATURE_COLUMNS,
            ]
        ].to_numpy(dtype=float)
    ).all()


def test_audit_accepts_both_complete_source_families() -> None:
    mappings, futures, indices, basics, options, funds = _frames()
    features = build_derivative_features(
        DATES,
        PREVIOUS,
        mappings,
        futures,
        indices,
        basics,
        options,
        funds,
    )
    audit = audit_probe_contract(
        features,
        mappings,
        target_dates=DATES,
        family_query_failures={"futures": 0, "options": 0},
    )
    assert audit["futures"]["passed"]
    assert audit["options"]["passed"]
    assert audit["full_backfill_authorized"]


def test_audit_can_admit_futures_when_options_permission_fails() -> None:
    mappings, futures, indices, basics, options, funds = _frames()
    features = build_derivative_features(
        DATES,
        PREVIOUS,
        mappings,
        futures,
        indices,
        basics,
        options,
        funds,
    )
    audit = audit_probe_contract(
        features,
        mappings,
        target_dates=DATES,
        family_query_failures={"futures": 0, "options": 1},
    )
    assert audit["futures"]["passed"]
    assert not audit["options"]["passed"]
    assert audit["selected_source_families"] == [
        "tminus1_index_futures_daily"
    ]


def test_audit_rejects_outcome_contamination() -> None:
    mappings, futures, indices, basics, options, funds = _frames()
    features = build_derivative_features(
        DATES,
        PREVIOUS,
        mappings,
        futures,
        indices,
        basics,
        options,
        funds,
    ).assign(target_net_return=1.0)
    audit = audit_probe_contract(
        features,
        mappings,
        target_dates=DATES,
        family_query_failures={"futures": 0, "options": 0},
    )
    assert not audit["full_backfill_authorized"]
    assert audit["forbidden_columns"] == ["target_net_return"]
