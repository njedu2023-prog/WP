from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import V3Config
from .features import enrich_feature_frame


IDENTITY_COLUMNS = ("trade_date", "signal_slot", "ts_code")


@dataclass(frozen=True)
class DatasetAudit:
    rows: int
    trade_days: int
    symbols: int
    eligible_rows: int
    labelled_rows: int
    positive_rows: int


def execution_eligibility(frame: pd.DataFrame, config: V3Config) -> pd.Series:
    execution = config.execution
    price = _numeric(frame, "signal_price")
    listing_days = _numeric(frame, "listing_days")
    prev_amount = _numeric(frame, "prev_20d_amount")
    slot_amount = _numeric(frame, "slot_amount")
    up_distance = _numeric(frame, "distance_to_up_limit_pct")
    down_distance = _numeric(frame, "distance_to_down_limit_pct")

    eligible = (
        price.between(execution.min_price, execution.max_price, inclusive="both")
        & listing_days.ge(execution.min_listing_days)
        & prev_amount.ge(execution.min_prev_20d_amount)
        & slot_amount.ge(execution.min_slot_amount)
        & slot_amount.mul(execution.max_entry_pct_of_slot_amount).ge(
            execution.reference_order_notional
        )
        & up_distance.ge(execution.min_distance_to_up_limit_pct)
        & down_distance.ge(execution.min_distance_to_down_limit_pct)
    )
    for flag in ("is_st", "is_suspended", "one_word_up_limit"):
        if flag in frame:
            eligible &= ~_boolean(frame[flag])
    if "board" in frame and config.strategy.board_scope == "main_board":
        eligible &= frame["board"].astype(str).eq("main_board")
    if "adj_factor" in frame:
        eligible &= _numeric(frame, "adj_factor").gt(0)
    if "slot_bar_lag_minutes" in frame:
        eligible &= _numeric(frame, "slot_bar_lag_minutes").between(
            0,
            5,
            inclusive="both",
        )
    if "intraday_snapshot_count" in frame:
        eligible &= _numeric(frame, "intraday_snapshot_count").ge(
            execution.min_intraday_snapshot_count
        )
    return eligible.fillna(False)


def build_supervised_panel(frame: pd.DataFrame, config: V3Config) -> pd.DataFrame:
    missing = sorted(
        set(
            IDENTITY_COLUMNS
            + (
                "signal_price",
                "entry_benchmark_price",
                "entry_fillable",
                "exit_fillable",
                "adj_factor",
                "t1_close",
                "t1_total_return_close",
            )
        )
        - set(frame.columns)
    )
    if missing:
        raise ValueError(f"cannot build labels; missing columns: {missing}")

    panel = enrich_feature_frame(frame)
    panel["execution_eligible"] = execution_eligibility(panel, config)
    panel["entry_fillable"] = _boolean(panel["entry_fillable"])
    panel["exit_fillable"] = _boolean(panel["exit_fillable"])

    signal_price = _numeric(panel, "signal_price")
    entry_reference = _numeric(panel, "entry_benchmark_price")
    exit_price = _numeric(panel, "t1_total_return_close")
    slippage = config.execution.entry_slippage_bps / 10_000.0
    cost_pct = config.execution.round_trip_cost_bps / 100.0
    panel["entry_price"] = entry_reference * (1.0 + slippage)
    panel["conditional_gross_return_pct"] = (
        exit_price / panel["entry_price"] - 1.0
    ) * 100.0
    panel["conditional_net_return_pct"] = (
        panel["conditional_gross_return_pct"] - cost_pct
    )

    entry_truth_known = entry_reference.gt(0) | ~panel["entry_fillable"]
    exit_truth_known = exit_price.gt(0) | ~panel["exit_fillable"]
    observable = (
        signal_price.gt(0)
        & entry_truth_known
        & (~panel["entry_fillable"] | exit_truth_known)
    )
    round_trip_fill = (
        panel["execution_eligible"]
        & panel["entry_fillable"]
        & panel["exit_fillable"]
        & observable
    )
    panel["label_available"] = observable
    panel["execution_success"] = round_trip_fill
    panel["target_entry_fillable"] = np.where(
        observable,
        panel["entry_fillable"].astype("int8"),
        np.nan,
    )
    panel["target_exit_fillable"] = np.where(
        observable & panel["entry_fillable"],
        panel["exit_fillable"].astype("int8"),
        np.nan,
    )
    panel["target_conditional_net_positive"] = np.where(
        round_trip_fill,
        panel["conditional_net_return_pct"].gt(0).astype("int8"),
        np.nan,
    )
    panel["target_conditional_severe_loss"] = np.where(
        round_trip_fill,
        panel["conditional_net_return_pct"]
        .le(config.model.severe_loss_threshold_pct)
        .astype("int8"),
        np.nan,
    )
    # A failed entry leaves cash uninvested; it is an execution miss, not a
    # fabricated trading loss. A failed T+1 exit leaves an open position and
    # therefore receives the explicit conservative contract penalty.
    panel["gross_return_pct"] = np.where(
        round_trip_fill,
        panel["conditional_gross_return_pct"],
        np.where(observable & ~panel["entry_fillable"], 0.0, np.nan),
    )
    panel["net_return_pct"] = np.where(
        round_trip_fill,
        panel["conditional_net_return_pct"],
        np.where(
            observable & ~panel["entry_fillable"],
            0.0,
            np.where(
                observable
                & panel["entry_fillable"]
                & ~panel["exit_fillable"],
                config.execution.non_fill_penalty_pct,
                np.nan,
            ),
        ),
    )
    panel["target_net_positive"] = np.where(
        observable,
        (
            round_trip_fill
            & panel["conditional_net_return_pct"].gt(0)
        ).astype("int8"),
        np.nan,
    )
    panel["target_severe_loss"] = np.where(
        observable,
        (
            (
                panel["entry_fillable"]
                & ~panel["exit_fillable"]
            )
            | (
                round_trip_fill
                & panel["conditional_net_return_pct"].le(
                    config.model.severe_loss_threshold_pct
                )
            )
        ).astype("int8"),
        np.nan,
    )

    group_keys = [panel["trade_date"], panel["signal_slot"]]
    # V8 learns the full executable outcome, not only the easy subset that
    # completed both legs. Entry misses remain zero-return cash outcomes and
    # failed exits retain the explicit conservative penalty in the rank target.
    eligible_return = panel["net_return_pct"].where(
        panel["execution_eligible"] & observable
    )
    return_rank = eligible_return.groupby(group_keys, sort=False).rank(
        method="average",
        pct=True,
    )
    panel["_target_net_return_rank"] = return_rank
    panel["target_cross_section_top"] = np.where(
        panel["execution_eligible"] & observable,
        return_rank.ge(1.0 - config.model.cross_section_top_fraction).astype("int8"),
        np.nan,
    )
    return panel


def first_crossing_candidates(
    predictions: pd.DataFrame,
    config: V3Config,
    *,
    status_column: str = "passes_policy",
) -> pd.DataFrame:
    required = {"trade_date", "signal_slot", "ts_code", status_column}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"cannot select first crossing; missing columns: {missing}")
    allowed = predictions["signal_slot"].astype(str).isin(config.strategy.signal_slots)
    passed = _boolean(predictions[status_column])
    selected = predictions.loc[allowed & passed].copy()
    if selected.empty:
        return selected
    selected["_slot_order"] = pd.Categorical(
        selected["signal_slot"],
        categories=config.strategy.signal_slots,
        ordered=True,
    )
    selected = selected.sort_values(
        ["trade_date", "ts_code", "_slot_order"],
        kind="stable",
    )
    return (
        selected.drop_duplicates(["trade_date", "ts_code"], keep="first")
        .drop(columns="_slot_order")
        .reset_index(drop=True)
    )


def audit_panel(panel: pd.DataFrame) -> DatasetAudit:
    labels = pd.to_numeric(panel.get("target_net_positive"), errors="coerce")
    return DatasetAudit(
        rows=int(len(panel)),
        trade_days=int(panel.get("trade_date", pd.Series(dtype=str)).nunique()),
        symbols=int(panel.get("ts_code", pd.Series(dtype=str)).nunique()),
        eligible_rows=int(_boolean(panel.get("execution_eligible", False)).sum()),
        labelled_rows=int(labels.notna().sum()),
        positive_rows=int(labels.eq(1).sum()),
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean(values: pd.Series | bool) -> pd.Series:
    if isinstance(values, bool):
        return pd.Series(dtype=bool)
    if values.dtype == bool:
        return values.fillna(False)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y", "qualified", "pass"})


def _optional_bool(frame: pd.DataFrame, column: str, default: bool) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    return _boolean(frame[column]).reindex(frame.index, fill_value=default)
