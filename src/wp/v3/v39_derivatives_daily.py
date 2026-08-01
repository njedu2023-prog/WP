from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCHEMA_VERSION = "wp_v39_tminus1_derivatives_probe_1"

FUTURE_SPECS = (
    {
        "family": "if300",
        "prefix": "IF",
        "continuous_code": "IF.CFX",
        "index_code": "000300.SH",
    },
    {
        "family": "ih50",
        "prefix": "IH",
        "continuous_code": "IH.CFX",
        "index_code": "000016.SH",
    },
    {
        "family": "ic500",
        "prefix": "IC",
        "continuous_code": "IC.CFX",
        "index_code": "000905.SH",
    },
    {
        "family": "im1000",
        "prefix": "IM",
        "continuous_code": "IM.CFX",
        "index_code": "000852.SH",
    },
)

OPTION_SPECS = (
    {"family": "etf50", "underlying_code": "510050.SH"},
    {"family": "etf300", "underlying_code": "510300.SH"},
)

FUTURE_FEATURE_NAMES = (
    "main_close_return_pct",
    "main_settle_return_pct",
    "main_oi_change_pct",
    "main_volume_to_oi",
    "main_basis_pct",
    "next_term_spread_pct",
    "main_oi_share",
)

OPTION_FEATURE_NAMES = (
    "put_call_volume_ratio",
    "put_call_oi_ratio",
    "put_call_amount_ratio",
    "atm_straddle_pct",
    "atm_put_call_premium_ratio",
    "option_volume_to_oi",
    "top5_oi_concentration",
)

V39_FUTURE_FEATURE_COLUMNS = tuple(
    f"v39_{spec['family']}_{name}"
    for spec in FUTURE_SPECS
    for name in FUTURE_FEATURE_NAMES
) + (
    "v39_future_small_minus_large_basis_pct",
    "v39_future_mid_minus_large_basis_pct",
    "v39_future_basis_mean_pct",
    "v39_future_basis_dispersion_pct",
    "v39_future_oi_change_mean_pct",
    "v39_future_oi_change_dispersion_pct",
)

V39_OPTION_FEATURE_COLUMNS = tuple(
    f"v39_{spec['family']}_{name}"
    for spec in OPTION_SPECS
    for name in OPTION_FEATURE_NAMES
) + (
    "v39_option_put_call_oi_mean",
    "v39_option_atm_straddle_mean_pct",
    "v39_option_atm_straddle_dispersion_pct",
)

FORBIDDEN_COLUMN_TOKENS = (
    "target",
    "truth",
    "t1",
    "t_1",
    "next_trade",
    "exit",
    "gross_return",
    "net_return",
    "outcome",
    "label",
)


def normalize_mapping(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reindex(
        columns=["ts_code", "trade_date", "mapping_ts_code"]
    ).copy()
    for column in result.columns:
        result[column] = result[column].astype(str).str.strip()
    result = result.loc[
        result["ts_code"].isin(
            spec["continuous_code"] for spec in FUTURE_SPECS
        )
        & result["trade_date"].str.fullmatch(r"\d{8}")
        & result["mapping_ts_code"].str.fullmatch(
            r"(?:IF|IH|IC|IM)\d{4}\.CFX",
            na=False,
        )
    ].copy()
    result.drop_duplicates(inplace=True)
    duplicate = result.duplicated(["trade_date", "ts_code"], keep=False)
    if duplicate.any():
        raise ValueError("ambiguous V39 futures mapping")
    result.sort_values(["trade_date", "ts_code"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def normalize_futures_daily(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ts_code",
        "trade_date",
        "pre_close",
        "pre_settle",
        "open",
        "high",
        "low",
        "close",
        "settle",
        "vol",
        "amount",
        "oi",
        "oi_chg",
    ]
    result = _normalize_daily(frame, columns)
    result = result.loc[
        result["ts_code"].str.fullmatch(
            r"(?:IF|IH|IC|IM)\d{4}\.CFX",
            na=False,
        )
    ].copy()
    return result


def normalize_index_daily(frame: pd.DataFrame) -> pd.DataFrame:
    return _normalize_daily(
        frame,
        [
            "ts_code",
            "trade_date",
            "pre_close",
            "open",
            "high",
            "low",
            "close",
            "pct_chg",
            "vol",
            "amount",
        ],
    )


def normalize_fund_daily(frame: pd.DataFrame) -> pd.DataFrame:
    return _normalize_daily(
        frame,
        [
            "ts_code",
            "trade_date",
            "pre_close",
            "open",
            "high",
            "low",
            "close",
            "pct_chg",
            "vol",
            "amount",
        ],
    )


def normalize_option_basic(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ts_code",
        "symbol",
        "exchange",
        "name",
        "opt_code",
        "opt_type",
        "call_put",
        "exercise_price",
        "maturity_date",
        "list_date",
        "delist_date",
    ]
    result = frame.reindex(columns=columns).copy()
    text_columns = [
        "ts_code",
        "symbol",
        "exchange",
        "name",
        "opt_code",
        "opt_type",
        "call_put",
        "maturity_date",
        "list_date",
        "delist_date",
    ]
    for column in text_columns:
        result[column] = result[column].astype(str).str.strip()
    result["exercise_price"] = pd.to_numeric(
        result["exercise_price"],
        errors="coerce",
    )
    result["option_side"] = result["call_put"].map(_option_side)
    result["underlying_code"] = result.apply(
        lambda row: _underlying_code(
            row["opt_code"],
            row["name"],
            row["symbol"],
        ),
        axis=1,
    )
    result = result.loc[
        result["exchange"].eq("SSE")
        & result["ts_code"].str.fullmatch(r"\d{8}\.SH", na=False)
        & result["option_side"].isin(["call", "put"])
        & result["underlying_code"].isin(
            spec["underlying_code"] for spec in OPTION_SPECS
        )
        & result["exercise_price"].gt(0)
        & result["maturity_date"].str.fullmatch(r"\d{8}", na=False)
    ].copy()
    result.drop_duplicates("ts_code", keep="last", inplace=True)
    result.sort_values("ts_code", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def normalize_options_daily(frame: pd.DataFrame) -> pd.DataFrame:
    return _normalize_daily(
        frame,
        [
            "ts_code",
            "trade_date",
            "exchange",
            "pre_settle",
            "pre_close",
            "open",
            "high",
            "low",
            "close",
            "settle",
            "vol",
            "amount",
            "oi",
        ],
    )


def build_derivative_features(
    target_dates: Iterable[str],
    previous_dates: dict[str, str],
    mappings: pd.DataFrame,
    futures_daily: pd.DataFrame,
    index_daily: pd.DataFrame,
    option_basic: pd.DataFrame,
    options_daily: pd.DataFrame,
    fund_daily: pd.DataFrame,
) -> pd.DataFrame:
    normalized_mapping = normalize_mapping(mappings)
    futures = normalize_futures_daily(futures_daily)
    indices = normalize_index_daily(index_daily)
    basics = normalize_option_basic(option_basic)
    options = normalize_options_daily(options_daily)
    funds = normalize_fund_daily(fund_daily)
    rows: list[dict[str, Any]] = []
    for target_date in tuple(str(value) for value in target_dates):
        previous = str(previous_dates.get(target_date, ""))
        row: dict[str, Any] = {
            "trade_date": target_date,
            "source_trade_date": previous,
            "v39_tminus1_causal": bool(
                previous.isdigit()
                and len(previous) == 8
                and previous < target_date
            ),
        }
        future_values, future_complete = _future_features_for_date(
            previous,
            normalized_mapping,
            futures,
            indices,
        )
        option_values, option_complete = _option_features_for_date(
            previous,
            basics,
            options,
            funds,
        )
        row.update(future_values)
        row.update(option_values)
        row["v39_futures_complete"] = future_complete
        row["v39_options_complete"] = option_complete
        row["v39_futures_finite"] = _all_finite(
            row,
            V39_FUTURE_FEATURE_COLUMNS,
        )
        row["v39_options_finite"] = _all_finite(
            row,
            V39_OPTION_FEATURE_COLUMNS,
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    for column in (
        *V39_FUTURE_FEATURE_COLUMNS,
        *V39_OPTION_FEATURE_COLUMNS,
    ):
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result.sort_values("trade_date", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def audit_probe_contract(
    features: pd.DataFrame,
    mappings: pd.DataFrame,
    *,
    target_dates: Iterable[str],
    family_query_failures: dict[str, int],
) -> dict[str, Any]:
    dates = tuple(str(value) for value in target_dates)
    normalized_mapping = normalize_mapping(mappings)
    expected_mapping = len(dates) * len(FUTURE_SPECS)
    identity_exact = (
        len(features) == len(dates)
        and features["trade_date"].astype(str).tolist() == sorted(dates)
        and not features["trade_date"].duplicated().any()
    )
    causal = bool(
        len(features)
        and features["v39_tminus1_causal"].fillna(False).astype(bool).all()
    )
    forbidden = sorted(
        column
        for column in features.columns
        if any(token in column.lower() for token in FORBIDDEN_COLUMN_TOKENS)
    )
    future_complete_rate = _boolean_rate(
        features,
        "v39_futures_complete",
    )
    option_complete_rate = _boolean_rate(
        features,
        "v39_options_complete",
    )
    future_finite_rate = _boolean_rate(features, "v39_futures_finite")
    option_finite_rate = _boolean_rate(features, "v39_options_finite")
    future_diverse = _diverse_count(
        features,
        V39_FUTURE_FEATURE_COLUMNS,
    )
    option_diverse = _diverse_count(
        features,
        V39_OPTION_FEATURE_COLUMNS,
    )
    futures_passed = bool(
        identity_exact
        and causal
        and not forbidden
        and int(family_query_failures.get("futures", 0)) == 0
        and len(normalized_mapping) == expected_mapping
        and future_complete_rate == 1.0
        and future_finite_rate == 1.0
        and future_diverse
        >= int(np.ceil(len(V39_FUTURE_FEATURE_COLUMNS) * 0.80))
    )
    options_passed = bool(
        identity_exact
        and causal
        and not forbidden
        and int(family_query_failures.get("options", 0)) == 0
        and option_complete_rate == 1.0
        and option_finite_rate == 1.0
        and option_diverse
        >= int(np.ceil(len(V39_OPTION_FEATURE_COLUMNS) * 0.80))
    )
    selected = [
        family
        for family, passed in (
            ("tminus1_index_futures_daily", futures_passed),
            ("tminus1_etf_options_daily", options_passed),
        )
        if passed
    ]
    return {
        "target_dates": len(dates),
        "feature_rows": int(len(features)),
        "identity_exact": identity_exact,
        "causal_tminus1_dates": causal,
        "forbidden_columns": forbidden,
        "mapping_rows": int(len(normalized_mapping)),
        "expected_mapping_rows": expected_mapping,
        "family_query_failures": {
            key: int(value) for key, value in family_query_failures.items()
        },
        "futures": {
            "complete_rate": future_complete_rate,
            "finite_rate": future_finite_rate,
            "diverse_features": future_diverse,
            "feature_count": len(V39_FUTURE_FEATURE_COLUMNS),
            "passed": futures_passed,
        },
        "options": {
            "complete_rate": option_complete_rate,
            "finite_rate": option_finite_rate,
            "diverse_features": option_diverse,
            "feature_count": len(V39_OPTION_FEATURE_COLUMNS),
            "passed": options_passed,
        },
        "selected_source_families": selected,
        "full_backfill_authorized": bool(selected),
    }


def _future_features_for_date(
    trade_date: str,
    mappings: pd.DataFrame,
    futures: pd.DataFrame,
    indices: pd.DataFrame,
) -> tuple[dict[str, float], bool]:
    values: dict[str, float] = {}
    family_values: dict[str, dict[str, float]] = {}
    complete = True
    for spec in FUTURE_SPECS:
        mapping = mappings.loc[
            mappings["trade_date"].eq(trade_date)
            & mappings["ts_code"].eq(spec["continuous_code"])
        ]
        main_code = (
            str(mapping["mapping_ts_code"].iloc[0])
            if len(mapping) == 1
            else ""
        )
        family = futures.loc[
            futures["trade_date"].eq(trade_date)
            & futures["ts_code"].str.startswith(spec["prefix"])
        ].copy()
        main = family.loc[family["ts_code"].eq(main_code)]
        index = indices.loc[
            indices["trade_date"].eq(trade_date)
            & indices["ts_code"].eq(spec["index_code"])
        ]
        current: dict[str, float]
        if len(main) != 1 or len(index) != 1:
            current = {name: np.nan for name in FUTURE_FEATURE_NAMES}
            complete = False
        else:
            main_row = main.iloc[0]
            next_row = _next_contract(family, main_code)
            index_close = _finite(index["close"].iloc[0])
            main_settle = _finite(main_row["settle"])
            oi = _finite(main_row["oi"])
            oi_change = _finite(main_row["oi_chg"])
            family_oi = pd.to_numeric(
                family["oi"],
                errors="coerce",
            ).clip(lower=0).sum()
            current = {
                "main_close_return_pct": _return_pct(
                    main_row["pre_settle"],
                    main_row["close"],
                ),
                "main_settle_return_pct": _return_pct(
                    main_row["pre_settle"],
                    main_row["settle"],
                ),
                "main_oi_change_pct": _safe_ratio(
                    oi_change,
                    oi - oi_change,
                )
                * 100.0,
                "main_volume_to_oi": _safe_ratio(main_row["vol"], oi),
                "main_basis_pct": _return_pct(index_close, main_settle),
                "next_term_spread_pct": (
                    _return_pct(main_settle, next_row["settle"])
                    if next_row is not None
                    else np.nan
                ),
                "main_oi_share": _safe_ratio(oi, family_oi),
            }
            if not all(np.isfinite(list(current.values()))):
                complete = False
        family_values[spec["family"]] = current
        for name, value in current.items():
            values[f"v39_{spec['family']}_{name}"] = value
    if all(
        family in family_values for family in ("if300", "ic500", "im1000")
    ):
        basis = np.asarray(
            [
                item["main_basis_pct"]
                for item in family_values.values()
            ],
            dtype=float,
        )
        oi_change = np.asarray(
            [
                item["main_oi_change_pct"]
                for item in family_values.values()
            ],
            dtype=float,
        )
        values.update(
            {
                "v39_future_small_minus_large_basis_pct": (
                    family_values["im1000"]["main_basis_pct"]
                    - family_values["if300"]["main_basis_pct"]
                ),
                "v39_future_mid_minus_large_basis_pct": (
                    family_values["ic500"]["main_basis_pct"]
                    - family_values["if300"]["main_basis_pct"]
                ),
                "v39_future_basis_mean_pct": float(np.mean(basis)),
                "v39_future_basis_dispersion_pct": float(np.std(basis)),
                "v39_future_oi_change_mean_pct": float(
                    np.mean(oi_change)
                ),
                "v39_future_oi_change_dispersion_pct": float(
                    np.std(oi_change)
                ),
            }
        )
    return values, bool(
        complete and _all_finite(values, V39_FUTURE_FEATURE_COLUMNS)
    )


def _option_features_for_date(
    trade_date: str,
    basics: pd.DataFrame,
    options: pd.DataFrame,
    funds: pd.DataFrame,
) -> tuple[dict[str, float], bool]:
    values: dict[str, float] = {}
    family_values: dict[str, dict[str, float]] = {}
    complete = True
    daily = options.loc[options["trade_date"].eq(trade_date)].merge(
        basics,
        on="ts_code",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_basic"),
    )
    for spec in OPTION_SPECS:
        underlying = spec["underlying_code"]
        active = daily.loc[
            daily["underlying_code"].eq(underlying)
            & daily["list_date"].le(trade_date)
            & daily["delist_date"].ge(trade_date)
        ].copy()
        spot_rows = funds.loc[
            funds["trade_date"].eq(trade_date)
            & funds["ts_code"].eq(underlying)
        ]
        current = _option_family_features(
            active,
            _finite(spot_rows["close"].iloc[0])
            if len(spot_rows) == 1
            else np.nan,
            trade_date,
        )
        family_values[spec["family"]] = current
        if not all(np.isfinite(list(current.values()))):
            complete = False
        for name, value in current.items():
            values[f"v39_{spec['family']}_{name}"] = value
    if family_values:
        oi_ratios = np.asarray(
            [
                item["put_call_oi_ratio"]
                for item in family_values.values()
            ],
            dtype=float,
        )
        straddles = np.asarray(
            [
                item["atm_straddle_pct"]
                for item in family_values.values()
            ],
            dtype=float,
        )
        values.update(
            {
                "v39_option_put_call_oi_mean": float(
                    np.mean(oi_ratios)
                ),
                "v39_option_atm_straddle_mean_pct": float(
                    np.mean(straddles)
                ),
                "v39_option_atm_straddle_dispersion_pct": float(
                    np.std(straddles)
                ),
            }
        )
    return values, bool(
        complete and _all_finite(values, V39_OPTION_FEATURE_COLUMNS)
    )


def _option_family_features(
    frame: pd.DataFrame,
    spot: float,
    trade_date: str,
) -> dict[str, float]:
    empty = {name: np.nan for name in OPTION_FEATURE_NAMES}
    if frame.empty or not np.isfinite(spot) or spot <= 0:
        return empty
    calls = frame.loc[frame["option_side"].eq("call")]
    puts = frame.loc[frame["option_side"].eq("put")]
    if len(calls) < 5 or len(puts) < 5:
        return empty
    trade_timestamp = pd.Timestamp(
        f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    )
    maturity = pd.to_datetime(
        frame["maturity_date"],
        format="%Y%m%d",
        errors="coerce",
    )
    dte = (maturity - trade_timestamp).dt.days
    eligible = frame.loc[dte.between(7, 45, inclusive="both")].copy()
    if eligible.empty:
        return empty
    nearest = eligible["maturity_date"].min()
    near = eligible.loc[eligible["maturity_date"].eq(nearest)].copy()
    paired = (
        near.pivot_table(
            index="exercise_price",
            columns="option_side",
            values="close",
            aggfunc="last",
        )
        .dropna(subset=["call", "put"])
        .reset_index()
    )
    if paired.empty:
        return empty
    paired["_distance"] = (
        pd.to_numeric(paired["exercise_price"], errors="coerce") - spot
    ).abs()
    atm = paired.sort_values("_distance", kind="stable").iloc[0]
    vol = pd.to_numeric(frame["vol"], errors="coerce").clip(lower=0)
    oi = pd.to_numeric(frame["oi"], errors="coerce").clip(lower=0)
    amount = pd.to_numeric(
        frame["amount"],
        errors="coerce",
    ).clip(lower=0)
    call_mask = frame["option_side"].eq("call")
    put_mask = frame["option_side"].eq("put")
    total_oi = float(oi.sum())
    return {
        "put_call_volume_ratio": _safe_ratio(
            vol.loc[put_mask].sum() + 1.0,
            vol.loc[call_mask].sum() + 1.0,
        ),
        "put_call_oi_ratio": _safe_ratio(
            oi.loc[put_mask].sum() + 1.0,
            oi.loc[call_mask].sum() + 1.0,
        ),
        "put_call_amount_ratio": _safe_ratio(
            amount.loc[put_mask].sum() + 1.0,
            amount.loc[call_mask].sum() + 1.0,
        ),
        "atm_straddle_pct": _safe_ratio(
            _finite(atm["call"]) + _finite(atm["put"]),
            spot,
        )
        * 100.0,
        "atm_put_call_premium_ratio": _safe_ratio(
            atm["put"],
            atm["call"],
        ),
        "option_volume_to_oi": _safe_ratio(vol.sum(), total_oi),
        "top5_oi_concentration": _safe_ratio(
            oi.nlargest(5).sum(),
            total_oi,
        ),
    }


def _next_contract(
    family: pd.DataFrame,
    main_code: str,
) -> pd.Series | None:
    main_month = _contract_month(main_code)
    candidates = family.assign(
        _contract_month=family["ts_code"].map(_contract_month)
    )
    candidates = candidates.loc[
        candidates["_contract_month"].gt(main_month)
    ].sort_values("_contract_month", kind="stable")
    if candidates.empty:
        return None
    return candidates.iloc[0]


def _contract_month(value: Any) -> int:
    match = re.search(r"(?:IF|IH|IC|IM)(\d{4})", str(value))
    if not match:
        return -1
    code = match.group(1)
    return 200000 + int(code[:2]) * 100 + int(code[2:])


def _underlying_code(
    opt_code: Any,
    name: Any,
    symbol: Any,
) -> str:
    text = " ".join(str(value) for value in (opt_code, name, symbol))
    for spec in OPTION_SPECS:
        digits = spec["underlying_code"].split(".")[0]
        if digits in text:
            return spec["underlying_code"]
    return ""


def _option_side(value: Any) -> str:
    normalized = str(value).strip().upper()
    if normalized in {"C", "CALL", "认购", "购"}:
        return "call"
    if normalized in {"P", "PUT", "认沽", "沽"}:
        return "put"
    return ""


def _normalize_daily(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    result = frame.reindex(columns=columns).copy()
    for column in ("ts_code", "trade_date", "exchange"):
        if column in result:
            result[column] = result[column].astype(str).str.strip()
    numeric = [
        column
        for column in columns
        if column not in {"ts_code", "trade_date", "exchange"}
    ]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.loc[
        result["ts_code"].ne("")
        & result["trade_date"].str.fullmatch(r"\d{8}", na=False)
    ].copy()
    result.drop_duplicates(
        ["ts_code", "trade_date"],
        keep="last",
        inplace=True,
    )
    result.sort_values(["trade_date", "ts_code"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def _return_pct(start: Any, end: Any) -> float:
    start_value = _finite(start)
    end_value = _finite(end)
    if (
        not np.isfinite(start_value)
        or start_value <= 0
        or not np.isfinite(end_value)
    ):
        return np.nan
    return float((end_value / start_value - 1.0) * 100.0)


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    numerator_value = _finite(numerator)
    denominator_value = _finite(denominator)
    if (
        not np.isfinite(numerator_value)
        or not np.isfinite(denominator_value)
        or abs(denominator_value) <= 1e-12
    ):
        return np.nan
    return float(numerator_value / denominator_value)


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _all_finite(
    values: dict[str, Any] | pd.Series,
    columns: Iterable[str],
) -> bool:
    return bool(
        np.isfinite(
            np.asarray([values.get(column, np.nan) for column in columns])
        ).all()
    )


def _boolean_rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return float(frame[column].fillna(False).astype(bool).mean())


def _diverse_count(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> int:
    return int(
        sum(
            pd.to_numeric(frame[column], errors="coerce").nunique(
                dropna=True
            )
            >= 4
            for column in columns
            if column in frame
        )
    )
