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

RTK_FIELDS = (
    "ts_code,name,pre_close,high,open,low,close,vol,amount,trade_time"
)
RTK_WILDCARDS = "6*.SH,0*.SZ"


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

    current_rt = client.query(
        "rt_k",
        cache_key=f"{trade_date}_{signal_slot.replace(':', '')}_main",
        ts_code=RTK_WILDCARDS,
        fields=RTK_FIELDS,
    )
    current_rt = _normalize_rt_k(current_rt)
    if current_rt.empty:
        raise RuntimeError("live all-market rt_k snapshot is missing")
    actual_market_time = current_rt["source_trade_time"].max()
    if pd.isna(actual_market_time):
        raise RuntimeError("live rt_k snapshot has no valid trade_time")
    actual_slot = pd.Timestamp(actual_market_time).strftime("%H:%M")
    source_lag_minutes = _minute_value(signal_slot) - _minute_value(actual_slot)
    if source_lag_minutes < -1 or source_lag_minutes > 5:
        raise RuntimeError(
            f"live rt_k data is stale or future-dated: requested {signal_slot}, "
            f"latest {actual_slot}"
        )
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
        set(current_rt["ts_code"].dropna().astype(str)),
        set(current_rt["ts_code"].dropna().astype(str)),
        config,
        trade_date=trade_date,
    )
    open_price = (
        current_rt.drop_duplicates("ts_code", keep="last")[["ts_code", "open"]]
        .rename(columns={"open": "day_open"})
    )
    tail = _load_rt_k_session_snapshots(
        client,
        trade_date=trade_date,
        signal_slot=signal_slot,
        current=current_rt,
        signal_slots=config.strategy.signal_slots,
    )
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
    snapshots["slot_bar_lag_minutes"] = max(0, source_lag_minutes)
    snapshots["market_data_time"] = pd.Timestamp(actual_market_time).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    snapshots["entry_fillable"] &= snapshots["slot_bar_lag_minutes"].between(
        0,
        5,
        inclusive="both",
    )
    if len(snapshots) < 1_000:
        raise RuntimeError(f"live feature universe has only {len(snapshots)} rows")
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


def _normalize_rt_k(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "ts_code",
                "open",
                "high",
                "low",
                "close",
                "vol",
                "amount",
                "source_trade_time",
            ]
        )
    result = frame.rename(columns={"code": "ts_code", "time": "trade_time"}).copy()
    required = {"ts_code", "open", "high", "low", "close", "vol", "amount"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise RuntimeError(f"rt_k response is missing columns: {missing}")
    for column in ("open", "high", "low", "close", "vol", "amount", "pre_close"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
    result["ts_code"] = result["ts_code"].astype(str)
    result["source_trade_time"] = pd.to_datetime(
        result.get("trade_time"),
        errors="coerce",
    )
    return result.drop_duplicates("ts_code", keep="last")


def _load_rt_k_session_snapshots(
    client: TushareHistoryClient,
    *,
    trade_date: str,
    signal_slot: str,
    current: pd.DataFrame,
    signal_slots: tuple[str, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for slot in signal_slots:
        if slot > signal_slot:
            continue
        cache_path = (
            client.cache_dir
            / "rt_k"
            / f"{trade_date}_{slot.replace(':', '')}_main.parquet"
        )
        raw = current if slot == signal_slot else (
            _normalize_rt_k(pd.read_parquet(cache_path))
            if cache_path.exists()
            else pd.DataFrame()
        )
        if raw.empty:
            continue
        normalized = raw.copy()
        normalized["trade_time"] = pd.Timestamp(
            f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} {slot}:00"
        )
        normalized["amount"] = (
            pd.to_numeric(normalized["amount"], errors="coerce")
            / _completed_five_minute_bars(slot)
        )
        frames.append(
            normalized[
                ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"]
            ]
        )
    if not frames:
        raise RuntimeError("no live rt_k session snapshots are available")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["ts_code", "trade_time"],
        kind="stable",
    )


def _completed_five_minute_bars(slot: str) -> int:
    absolute = _minute_value(slot)
    if absolute <= 11 * 60 + 30:
        return max(1, (absolute - (9 * 60 + 30)) // 5)
    return 24 + max(1, (absolute - 13 * 60) // 5)


def _minute_value(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)
