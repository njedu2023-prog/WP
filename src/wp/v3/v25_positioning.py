from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = "wp_v25_positioning_features_1"
MINIMUM_CYQ_COVERAGE = 0.95
MINIMUM_MARGIN_COVERAGE = 0.65

CYQ_RAW_COLUMNS = (
    "ts_code",
    "trade_date",
    "his_low",
    "his_high",
    "cost_5pct",
    "cost_15pct",
    "cost_50pct",
    "cost_85pct",
    "cost_95pct",
    "weight_avg",
    "winner_rate",
)
MARGIN_RAW_COLUMNS = (
    "trade_date",
    "ts_code",
    "name",
    "rzye",
    "rqye",
    "rzmre",
    "rqyl",
    "rzche",
    "rqchl",
    "rqmcl",
    "rzrqye",
)
TOP_LIST_RAW_COLUMNS = (
    "trade_date",
    "ts_code",
    "name",
    "close",
    "pct_change",
    "turnover_rate",
    "amount",
    "l_sell",
    "l_buy",
    "l_amount",
    "net_amount",
    "net_rate",
    "amount_rate",
    "float_values",
    "reason",
)

CYQ_FEATURE_COLUMNS = (
    "v25_cyq_winner_rate_pct",
    "v25_cyq_signal_vs_weight_avg_pct",
    "v25_cyq_signal_vs_cost_50_pct",
    "v25_cyq_signal_vs_cost_85_pct",
    "v25_cyq_signal_vs_cost_95_pct",
    "v25_cyq_cost_90_width_pct",
    "v25_cyq_cost_70_width_pct",
    "v25_cyq_signal_position_90",
    "v25_cyq_upper_overhang_85_pct",
    "v25_cyq_profit_cushion_pct",
    "v25_cyq_available",
)
MARGIN_FEATURE_COLUMNS = (
    "v25_margin_financing_balance_log",
    "v25_margin_short_balance_log",
    "v25_margin_total_balance_log",
    "v25_margin_financing_flow_ratio",
    "v25_margin_short_flow_ratio",
    "v25_margin_financing_balance_change_pct",
    "v25_margin_short_balance_change_pct",
    "v25_margin_short_to_financing_ratio",
    "v25_margin_available",
)
TOP_LIST_FEATURE_COLUMNS = (
    "v25_toplist_flag",
    "v25_toplist_records",
    "v25_toplist_reason_count",
    "v25_toplist_net_amount_signed_log",
    "v25_toplist_net_buy_ratio",
    "v25_toplist_buy_sell_ratio",
    "v25_toplist_amount_rate_max",
)
V25_FEATURE_COLUMNS = (
    *CYQ_FEATURE_COLUMNS,
    *MARGIN_FEATURE_COLUMNS,
    *TOP_LIST_FEATURE_COLUMNS,
)
SOURCE_IDENTITY_COLUMNS = ("trade_date", "signal_slot", "ts_code")


def previous_date_map(open_dates: Iterable[str]) -> dict[str, str]:
    ordered = sorted({str(value) for value in open_dates})
    return {
        current: previous
        for previous, current in zip(ordered, ordered[1:], strict=False)
    }


def attach_candidate_signal_price(
    source: pd.DataFrame,
    candidate_index: pd.DataFrame,
) -> pd.DataFrame:
    """Restore the immutable V24 signal price without changing identities."""
    source_required = {*SOURCE_IDENTITY_COLUMNS}
    candidate_required = {*SOURCE_IDENTITY_COLUMNS, "signal_price"}
    source_missing = sorted(source_required - set(source.columns))
    candidate_missing = sorted(candidate_required - set(candidate_index.columns))
    if source_missing:
        raise ValueError(
            f"V25 feature source missing identities: {source_missing}"
        )
    if candidate_missing:
        raise ValueError(
            f"V25 candidate index missing columns: {candidate_missing}"
        )

    left = source.copy()
    right = candidate_index.loc[
        :,
        [*SOURCE_IDENTITY_COLUMNS, "signal_price"],
    ].copy()
    for column in SOURCE_IDENTITY_COLUMNS:
        left[column] = left[column].astype(str)
        right[column] = right[column].astype(str)
    if left.duplicated(list(SOURCE_IDENTITY_COLUMNS)).any():
        raise RuntimeError("V25 feature source contains duplicate identities")
    if right.duplicated(list(SOURCE_IDENTITY_COLUMNS)).any():
        raise RuntimeError("V25 candidate index contains duplicate identities")

    left_keys = set(
        left.loc[:, SOURCE_IDENTITY_COLUMNS].itertuples(
            index=False,
            name=None,
        )
    )
    right_keys = set(
        right.loc[:, SOURCE_IDENTITY_COLUMNS].itertuples(
            index=False,
            name=None,
        )
    )
    if left_keys != right_keys:
        raise RuntimeError(
            "V25 feature source and candidate index identities differ"
        )

    existing = (
        pd.to_numeric(left["signal_price"], errors="coerce")
        if "signal_price" in left
        else None
    )
    left.drop(columns="signal_price", errors="ignore", inplace=True)
    left["_v25_source_order"] = np.arange(len(left))
    result = left.merge(
        right,
        on=list(SOURCE_IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    result.sort_values("_v25_source_order", kind="stable", inplace=True)
    result.drop(columns="_v25_source_order", inplace=True)
    result.reset_index(drop=True, inplace=True)
    signal_price = pd.to_numeric(result["signal_price"], errors="coerce")
    if (
        signal_price.isna().any()
        or not np.isfinite(signal_price).all()
        or not signal_price.gt(0.0).all()
    ):
        raise RuntimeError("V25 candidate index has invalid signal prices")
    if existing is not None and not np.allclose(
        existing.to_numpy(dtype=float),
        signal_price.to_numpy(dtype=float),
        equal_nan=False,
    ):
        raise RuntimeError(
            "V25 feature source signal prices differ from candidate index"
        )
    return result


def attach_positioning_features(
    source: pd.DataFrame,
    cyq: pd.DataFrame,
    margin: pd.DataFrame,
    top_list: pd.DataFrame,
    *,
    open_dates: Iterable[str],
) -> pd.DataFrame:
    required = {
        "trade_date",
        "signal_slot",
        "ts_code",
        "signal_price",
        "v23_prev_trade_date",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"V25 source missing columns: {missing}")
    result = source.copy()
    for column in ("trade_date", "signal_slot", "ts_code"):
        result[column] = result[column].astype(str)
    result["v23_prev_trade_date"] = result[
        "v23_prev_trade_date"
    ].astype(str)
    result["v25_margin_prev2_trade_date"] = result[
        "v23_prev_trade_date"
    ].map(previous_date_map(open_dates))

    cyq_prepared = prepare_cyq(cyq)
    result = result.merge(
        cyq_prepared,
        on=["v23_prev_trade_date", "ts_code"],
        how="left",
        validate="many_to_one",
    )
    result = add_cyq_features(result)

    margin_prepared = prepare_margin(margin)
    current_margin = margin_prepared.rename(
        columns={
            "trade_date": "v23_prev_trade_date",
            **{
                column: f"_v25_margin_current_{column}"
                for column in margin_numeric_columns()
            },
        }
    )
    lag_margin = margin_prepared.rename(
        columns={
            "trade_date": "v25_margin_prev2_trade_date",
            **{
                column: f"_v25_margin_lag_{column}"
                for column in margin_numeric_columns()
            },
        }
    )
    result = result.merge(
        current_margin,
        on=["v23_prev_trade_date", "ts_code"],
        how="left",
        validate="many_to_one",
    )
    result = result.merge(
        lag_margin,
        on=["v25_margin_prev2_trade_date", "ts_code"],
        how="left",
        validate="many_to_one",
    )
    result = add_margin_features(result)

    top_prepared = prepare_top_list(top_list).rename(
        columns={"trade_date": "v23_prev_trade_date"}
    )
    result = result.merge(
        top_prepared,
        on=["v23_prev_trade_date", "ts_code"],
        how="left",
        validate="many_to_one",
    )
    result = add_top_list_features(result)
    result["v25_positioning_core_complete"] = result[
        "v25_cyq_available"
    ].fillna(False).astype(bool)
    return result


def prepare_cyq(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reindex(columns=CYQ_RAW_COLUMNS).copy()
    result["trade_date"] = result["trade_date"].astype(str)
    result["ts_code"] = result["ts_code"].astype(str)
    if result.duplicated(["trade_date", "ts_code"]).any():
        raise RuntimeError("V25 CYQ source contains duplicate stock dates")
    numeric_columns = [
        column
        for column in CYQ_RAW_COLUMNS
        if column not in {"trade_date", "ts_code"}
    ]
    for column in numeric_columns:
        result[column] = _numeric(result, column)
    ordered = result[
        [
            "his_low",
            "cost_5pct",
            "cost_15pct",
            "cost_50pct",
            "cost_85pct",
            "cost_95pct",
            "his_high",
        ]
    ].diff(axis=1).iloc[:, 1:].ge(0.0).all(axis=1)
    valid = (
        result[numeric_columns].notna().all(axis=1)
        & ordered
        & result["weight_avg"].gt(0.0)
        & result["winner_rate"].between(0.0, 100.0)
    )
    result["_v25_cyq_valid"] = valid
    return result.rename(columns={"trade_date": "v23_prev_trade_date"})


def add_cyq_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    signal = _numeric(result, "signal_price")
    weighted = _numeric(result, "weight_avg")
    cost_5 = _numeric(result, "cost_5pct")
    cost_15 = _numeric(result, "cost_15pct")
    cost_50 = _numeric(result, "cost_50pct")
    cost_85 = _numeric(result, "cost_85pct")
    cost_95 = _numeric(result, "cost_95pct")
    available = result.get(
        "_v25_cyq_valid",
        pd.Series(False, index=result.index),
    ).fillna(False).astype(bool)
    result["v25_cyq_winner_rate_pct"] = _numeric(
        result,
        "winner_rate",
    ).where(available)
    result["v25_cyq_signal_vs_weight_avg_pct"] = _relative_pct(
        signal,
        weighted,
    ).where(available)
    result["v25_cyq_signal_vs_cost_50_pct"] = _relative_pct(
        signal,
        cost_50,
    ).where(available)
    result["v25_cyq_signal_vs_cost_85_pct"] = _relative_pct(
        signal,
        cost_85,
    ).where(available)
    result["v25_cyq_signal_vs_cost_95_pct"] = _relative_pct(
        signal,
        cost_95,
    ).where(available)
    result["v25_cyq_cost_90_width_pct"] = (
        100.0 * (cost_95 - cost_5) / weighted.replace(0.0, np.nan)
    ).where(available)
    result["v25_cyq_cost_70_width_pct"] = (
        100.0 * (cost_85 - cost_15) / weighted.replace(0.0, np.nan)
    ).where(available)
    result["v25_cyq_signal_position_90"] = (
        (signal - cost_5) / (cost_95 - cost_5).replace(0.0, np.nan)
    ).clip(-2.0, 3.0).where(available)
    result["v25_cyq_upper_overhang_85_pct"] = (
        100.0
        * (cost_85 - signal).clip(lower=0.0)
        / signal.replace(0.0, np.nan)
    ).where(available)
    result["v25_cyq_profit_cushion_pct"] = (
        100.0
        * (signal - weighted).clip(lower=0.0)
        / signal.replace(0.0, np.nan)
    ).where(available)
    result["v25_cyq_available"] = available
    return result.drop(
        columns=[*CYQ_RAW_COLUMNS[2:], "_v25_cyq_valid"]
    )


def prepare_margin(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reindex(columns=MARGIN_RAW_COLUMNS).copy()
    result["trade_date"] = result["trade_date"].astype(str)
    result["ts_code"] = result["ts_code"].astype(str)
    if result.duplicated(["trade_date", "ts_code"]).any():
        raise RuntimeError("V25 margin source contains duplicate stock dates")
    for column in margin_numeric_columns():
        result[column] = _numeric(result, column)
    return result.drop(columns="name")


def margin_numeric_columns() -> tuple[str, ...]:
    return tuple(
        column
        for column in MARGIN_RAW_COLUMNS
        if column not in {"trade_date", "ts_code", "name"}
    )


def add_margin_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    current = {
        column: _numeric(result, f"_v25_margin_current_{column}")
        for column in margin_numeric_columns()
    }
    lag = {
        column: _numeric(result, f"_v25_margin_lag_{column}")
        for column in margin_numeric_columns()
    }
    available = current["rzye"].notna() & current["rzrqye"].notna()
    result["v25_margin_financing_balance_log"] = np.log1p(
        current["rzye"].clip(lower=0.0)
    ).where(available)
    result["v25_margin_short_balance_log"] = np.log1p(
        current["rqye"].clip(lower=0.0)
    ).where(available)
    result["v25_margin_total_balance_log"] = np.log1p(
        current["rzrqye"].clip(lower=0.0)
    ).where(available)
    result["v25_margin_financing_flow_ratio"] = _signed_flow_ratio(
        current["rzmre"],
        current["rzche"],
    ).where(available)
    result["v25_margin_short_flow_ratio"] = _signed_flow_ratio(
        current["rqmcl"],
        current["rqchl"],
    ).where(available)
    result["v25_margin_financing_balance_change_pct"] = _relative_pct(
        current["rzye"],
        lag["rzye"],
    ).where(available & lag["rzye"].notna())
    result["v25_margin_short_balance_change_pct"] = _relative_pct(
        current["rqye"],
        lag["rqye"],
    ).where(available & lag["rqye"].notna())
    result["v25_margin_short_to_financing_ratio"] = (
        current["rqye"] / current["rzye"].replace(0.0, np.nan)
    ).clip(0.0, 10.0).where(available)
    result["v25_margin_available"] = available
    drop_columns = [
        column
        for column in result.columns
        if column.startswith("_v25_margin_current_")
        or column.startswith("_v25_margin_lag_")
    ]
    return result.drop(columns=drop_columns)


def prepare_top_list(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reindex(columns=TOP_LIST_RAW_COLUMNS).copy()
    if result.empty:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "ts_code",
                "_v25_toplist_records",
                "_v25_toplist_reason_count",
                "_v25_toplist_net_amount",
                "_v25_toplist_l_buy",
                "_v25_toplist_l_sell",
                "_v25_toplist_l_amount",
                "_v25_toplist_amount_rate_max",
            ]
        )
    result["trade_date"] = result["trade_date"].astype(str)
    result["ts_code"] = result["ts_code"].astype(str)
    for column in (
        "net_amount",
        "l_buy",
        "l_sell",
        "l_amount",
        "amount_rate",
    ):
        result[column] = _numeric(result, column).fillna(0.0)
    result["reason"] = result["reason"].fillna("").astype(str)
    return (
        result.groupby(["trade_date", "ts_code"], as_index=False)
        .agg(
            _v25_toplist_records=("ts_code", "size"),
            _v25_toplist_reason_count=("reason", "nunique"),
            _v25_toplist_net_amount=("net_amount", "sum"),
            _v25_toplist_l_buy=("l_buy", "sum"),
            _v25_toplist_l_sell=("l_sell", "sum"),
            _v25_toplist_l_amount=("l_amount", "sum"),
            _v25_toplist_amount_rate_max=("amount_rate", "max"),
        )
    )


def add_top_list_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    records = _numeric(result, "_v25_toplist_records").fillna(0.0)
    net = _numeric(result, "_v25_toplist_net_amount").fillna(0.0)
    buy = _numeric(result, "_v25_toplist_l_buy").fillna(0.0)
    sell = _numeric(result, "_v25_toplist_l_sell").fillna(0.0)
    amount = _numeric(result, "_v25_toplist_l_amount").fillna(0.0)
    result["v25_toplist_flag"] = records.gt(0.0)
    result["v25_toplist_records"] = records
    result["v25_toplist_reason_count"] = _numeric(
        result,
        "_v25_toplist_reason_count",
    ).fillna(0.0)
    result["v25_toplist_net_amount_signed_log"] = np.sign(net) * np.log1p(
        net.abs()
    )
    result["v25_toplist_net_buy_ratio"] = (
        net / amount.replace(0.0, np.nan)
    ).fillna(0.0).clip(-5.0, 5.0)
    result["v25_toplist_buy_sell_ratio"] = (
        buy / sell.replace(0.0, np.nan)
    ).fillna(0.0).clip(0.0, 20.0)
    result["v25_toplist_amount_rate_max"] = _numeric(
        result,
        "_v25_toplist_amount_rate_max",
    ).fillna(0.0)
    drop_columns = [
        column
        for column in result.columns
        if column.startswith("_v25_toplist_")
    ]
    return result.drop(columns=drop_columns)


def positioning_coverage_audit(frame: pd.DataFrame) -> dict[str, Any]:
    cyq_coverage = float(
        frame["v25_cyq_available"].fillna(False).astype(bool).mean()
    )
    margin_coverage = float(
        frame["v25_margin_available"].fillna(False).astype(bool).mean()
    )
    top_list_rate = float(
        frame["v25_toplist_flag"].fillna(False).astype(bool).mean()
    )
    feature_coverage = {
        column: float(
            pd.to_numeric(frame[column], errors="coerce").notna().mean()
        )
        for column in V25_FEATURE_COLUMNS
        if column not in {
            "v25_cyq_available",
            "v25_margin_available",
            "v25_toplist_flag",
        }
    }
    return {
        "rows": int(len(frame)),
        "trade_dates": int(frame["trade_date"].astype(str).nunique()),
        "cyq_coverage": cyq_coverage,
        "margin_coverage": margin_coverage,
        "top_list_event_rate": top_list_rate,
        "feature_non_null_coverage": feature_coverage,
        "coverage_passed": bool(
            cyq_coverage >= MINIMUM_CYQ_COVERAGE
            and margin_coverage >= MINIMUM_MARGIN_COVERAGE
        ),
    }


def _relative_pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return 100.0 * (
        numerator / denominator.replace(0.0, np.nan) - 1.0
    )


def _signed_flow_ratio(inflow: pd.Series, outflow: pd.Series) -> pd.Series:
    total = inflow.abs() + outflow.abs()
    return (inflow - outflow) / total.replace(0.0, np.nan)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")
