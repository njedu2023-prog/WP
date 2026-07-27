from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .contracts import V3Config
from .dataset import execution_eligibility
from .features import enrich_feature_frame
from .history import (
    ADJ_FIELDS,
    DAILY_BASIC_FIELDS,
    DAILY_FIELDS,
    LIMIT_FIELDS,
    MINUTE_FIELDS,
    TushareHistoryClient,
    _add_industry_context,
    _add_market_context,
    _board,
    _build_prior_day_features,
    _industry_at,
    _load_industry_intervals,
    _load_stock_basic,
    _minute_universe_quality,
    _slot_features,
)


def build_live_feature_frame(
    client: TushareHistoryClient,
    *,
    trade_date: str,
    signal_slot: str,
    config: V3Config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if signal_slot not in config.strategy.signal_slots:
        raise ValueError(f"{signal_slot} is not a V3 signal slot")
    calendar_start = (
        datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=90)
    ).strftime("%Y%m%d")
    calendar_end = (
        datetime.strptime(trade_date, "%Y%m%d") + timedelta(days=10)
    ).strftime("%Y%m%d")
    calendar = client.query(
        "trade_cal",
        cache_key=f"{calendar_start}_{calendar_end}_sse",
        exchange="SSE",
        start_date=calendar_start,
        end_date=calendar_end,
        is_open="1",
        fields="exchange,cal_date,is_open,pretrade_date",
    )
    open_dates = sorted(
        calendar.loc[calendar["is_open"].astype(str).eq("1"), "cal_date"].astype(str)
    )
    prior_dates = [date for date in open_dates if date < trade_date][-35:]
    future_dates = [date for date in open_dates if date > trade_date]
    if len(prior_dates) < 22 or not future_dates:
        raise RuntimeError("insufficient trading calendar depth for live feature construction")
    target_trade_date = future_dates[0]

    daily_frames: list[pd.DataFrame] = []
    basic_frames: list[pd.DataFrame] = []
    adjustment_frames: list[pd.DataFrame] = []
    for date in prior_dates:
        daily_frames.append(
            client.query(
                "daily",
                cache_key=date,
                trade_date=date,
                fields=DAILY_FIELDS,
            )
        )
        basic_frames.append(
            client.query(
                "daily_basic",
                cache_key=date,
                trade_date=date,
                fields=DAILY_BASIC_FIELDS,
            )
        )
        adjustment_frames.append(
            client.query(
                "adj_factor",
                cache_key=date,
                trade_date=date,
                fields=ADJ_FIELDS,
            )
        )
    daily = pd.concat(daily_frames, ignore_index=True).merge(
        pd.concat(adjustment_frames, ignore_index=True),
        on=["ts_code", "trade_date"],
        how="left",
    )
    daily_basic = pd.concat(basic_frames, ignore_index=True)
    symbols = daily.loc[daily["trade_date"].astype(str).eq(prior_dates[-1]), "ts_code"].dropna().unique()
    daily_stub = pd.DataFrame({"ts_code": symbols, "trade_date": trade_date})
    for column in DAILY_FIELDS.split(","):
        if column not in daily_stub:
            daily_stub[column] = np.nan
    basic_stub = pd.DataFrame({"ts_code": symbols, "trade_date": trade_date})
    for column in DAILY_BASIC_FIELDS.split(","):
        if column not in basic_stub:
            basic_stub[column] = np.nan
    feature_history = _build_prior_day_features(
        pd.concat([daily, daily_stub], ignore_index=True),
        pd.concat([daily_basic, basic_stub], ignore_index=True),
    )
    current_features = feature_history.loc[
        feature_history["trade_date"].astype(str).eq(trade_date)
    ].copy()
    previous_close = (
        daily.loc[
            daily["trade_date"].astype(str).eq(prior_dates[-1]),
            ["ts_code", "close"],
        ]
        .rename(columns={"close": "pre_close"})
        .drop_duplicates("ts_code")
    )
    current_adjustment = client.query(
        "adj_factor",
        cache_key=trade_date,
        trade_date=trade_date,
        fields=ADJ_FIELDS,
    )
    if not current_adjustment.empty:
        current_adjustment = (
            current_adjustment[["ts_code", "adj_factor"]]
            .drop_duplicates("ts_code")
            .rename(columns={"adj_factor": "current_adj_factor"})
        )
        previous_close = previous_close.merge(
            current_adjustment,
            on="ts_code",
            how="left",
        )
        previous_close["adj_factor"] = pd.to_numeric(
            previous_close["current_adj_factor"],
            errors="coerce",
        )
        previous_close = previous_close.drop(columns="current_adj_factor")
    else:
        previous_close["adj_factor"] = np.nan
    limits = client.query(
        "stk_limit",
        cache_key=trade_date,
        trade_date=trade_date,
        fields=LIMIT_FIELDS,
    )
    stock_basic = _load_stock_basic(client)
    industry_intervals = _load_industry_intervals(client, include_history=False)
    base = (
        previous_close.merge(
            limits[["ts_code", "up_limit", "down_limit"]],
            on="ts_code",
            how="left",
        )
        .merge(current_features.drop(columns=["trade_date"]), on="ts_code", how="left")
        .merge(stock_basic, on="ts_code", how="left")
    )
    for column in ("pre_close", "up_limit", "down_limit"):
        base[column] = pd.to_numeric(base[column], errors="coerce")
    base["trade_date"] = trade_date
    base["target_trade_date"] = target_trade_date
    base["listing_days"] = (
        pd.Timestamp(datetime.strptime(trade_date, "%Y%m%d"))
        - pd.to_datetime(base["list_date"], format="%Y%m%d", errors="coerce")
    ).dt.days
    base["is_st"] = base["name"].fillna("").astype(str).str.upper().str.contains("ST")
    base["industry"] = [
        _industry_at(code, trade_date, industry_intervals)
        for code in base["ts_code"]
    ]

    day_dash = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    open_minutes = client.query(
        "stk_mins",
        cache_key=f"{trade_date}_open_5m",
        paged=True,
        start_date=f"{day_dash} 09:30:00",
        end_date=f"{day_dash} 09:35:00",
        freq="5min",
        fields=MINUTE_FIELDS,
    )
    tail_minutes = client.query(
        "stk_mins",
        cache_key=f"{trade_date}_tail_{signal_slot.replace(':', '')}_5m",
        paged=True,
        start_date=f"{day_dash} 14:00:00",
        end_date=f"{day_dash} {signal_slot}:59",
        freq="5min",
        fields=MINUTE_FIELDS,
    )
    if open_minutes.empty or tail_minutes.empty:
        raise RuntimeError("live all-market five-minute bars are missing")
    expected_symbols = set(
        base.loc[
            base["board"].astype(str).eq(config.strategy.board_scope)
            & ~base["is_st"].fillna(True).astype(bool)
            & pd.to_numeric(base["adj_factor"], errors="coerce").gt(0),
            "ts_code",
        ].astype(str)
    )
    quality = _minute_universe_quality(
        expected_symbols,
        set(open_minutes["ts_code"].dropna().astype(str)),
        set(tail_minutes["ts_code"].dropna().astype(str)),
        config,
        trade_date=trade_date,
    )
    for frame in (open_minutes, tail_minutes):
        frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
        for column in ("open", "high", "low", "close", "amount"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    open_price = (
        open_minutes.sort_values(["ts_code", "trade_time"])
        .drop_duplicates("ts_code", keep="first")[["ts_code", "open"]]
        .rename(columns={"open": "day_open"})
    )
    tail = tail_minutes.dropna(subset=["ts_code", "trade_time"]).sort_values(
        ["ts_code", "trade_time"]
    )
    tail = tail.groupby("ts_code", group_keys=False).tail(5)
    snapshots = _slot_features(tail, signal_slot)
    snapshots = snapshots.merge(open_price, on="ts_code", how="left").merge(
        base, on="ts_code", how="inner"
    )
    snapshots["signal_slot"] = signal_slot
    snapshots["signal_price"] = snapshots["slot_close"]
    snapshots["ret_from_prev_close_pct"] = (
        snapshots["signal_price"] / snapshots["pre_close"] - 1.0
    ) * 100.0
    snapshots["ret_from_open_pct"] = (
        snapshots["signal_price"] / snapshots["day_open"] - 1.0
    ) * 100.0
    snapshots["distance_to_up_limit_pct"] = (
        snapshots["up_limit"] / snapshots["signal_price"] - 1.0
    ) * 100.0
    snapshots["distance_to_down_limit_pct"] = (
        snapshots["signal_price"] / snapshots["down_limit"] - 1.0
    ) * 100.0
    snapshots["entry_fillable"] = (
        snapshots["slot_amount"].ge(config.execution.min_slot_amount)
        & snapshots["distance_to_up_limit_pct"].ge(
            config.execution.max_distance_to_up_limit_pct
        )
        & snapshots["slot_bar_lag_minutes"].between(0, 5, inclusive="both")
    )
    snapshots = _add_market_context(snapshots)
    snapshots = _add_industry_context(snapshots)
    snapshots = enrich_feature_frame(snapshots)
    snapshots["execution_eligible"] = execution_eligibility(snapshots, config)
    snapshots["market_data_time"] = tail["trade_time"].max().strftime("%Y-%m-%d %H:%M:%S")
    if len(snapshots) < 1_000:
        raise RuntimeError(f"live feature universe has only {len(snapshots)} rows")
    actual_slot = pd.Timestamp(tail["trade_time"].max()).strftime("%H:%M")
    lag_minutes = (
        _minute_value(signal_slot) - _minute_value(actual_slot)
    )
    if lag_minutes > 5:
        raise RuntimeError(
            f"live minute data is stale: requested {signal_slot}, latest {actual_slot}"
        )
    manifest = {
        "schema_version": "wp_v3_live_features_1",
        "trade_date": trade_date,
        "target_trade_date": target_trade_date,
        "signal_slot": signal_slot,
        "market_data_time": snapshots["market_data_time"].iloc[0],
        "latest_bar_slot": actual_slot,
        "row_count": int(len(snapshots)),
        "eligible_count": int(snapshots["execution_eligible"].sum()),
        "feature_version": config.model.feature_version,
        **quality,
    }
    return snapshots, manifest


def _minute_value(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)
