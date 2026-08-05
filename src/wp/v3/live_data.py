from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
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
    _observation_slots,
    _ordered_bounded_map,
    _slot_features,
)

RTMIN_FIELDS = "code,time,open,close,high,low,vol,amount"
RTMIN_BATCH_SIZE = 800
RTK_FIELDS = "ts_code,pre_close,open,trade_time"
RTK_WILDCARDS = "6*.SH,0*.SZ"


def warm_live_reference_cache(
    client: TushareHistoryClient,
    *,
    trade_date: str,
) -> dict[str, Any]:
    """Populate only the rolling reference queries needed by the live build."""
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
        calendar.loc[
            calendar["is_open"].astype(str).eq("1"),
            "cal_date",
        ].astype(str)
    )
    prior_dates = [date for date in open_dates if date < trade_date][-35:]
    future_dates = [date for date in open_dates if date > trade_date]
    if len(prior_dates) < 22 or not future_dates:
        raise RuntimeError(
            "insufficient trading calendar depth for live reference warmup"
        )

    for date in prior_dates:
        client.query(
            "daily",
            cache_key=date,
            trade_date=date,
            fields=DAILY_FIELDS,
        )
        client.query(
            "daily_basic",
            cache_key=date,
            trade_date=date,
            fields=DAILY_BASIC_FIELDS,
        )
        client.query(
            "adj_factor",
            cache_key=date,
            trade_date=date,
            fields=ADJ_FIELDS,
        )

    client.query(
        "adj_factor",
        cache_key=trade_date,
        trade_date=trade_date,
        fields=ADJ_FIELDS,
    )
    client.query(
        "stk_limit",
        cache_key=trade_date,
        trade_date=trade_date,
        fields=LIMIT_FIELDS,
    )
    stock_basic = _load_stock_basic(client, cache_suffix=trade_date)
    industry_intervals = _load_industry_intervals(
        client,
        include_history=False,
        cache_suffix=trade_date,
    )
    return {
        "schema_version": "wp_v41_runtime_reference_1",
        "trade_date": trade_date,
        "target_trade_date": future_dates[0],
        "prior_trade_date_count": len(prior_dates),
        "prior_trade_date_start": prior_dates[0],
        "prior_trade_date_end": prior_dates[-1],
        "stock_basic_rows": int(len(stock_basic)),
        "industry_symbol_count": int(len(industry_intervals)),
    }


def build_live_feature_frame(
    client: TushareHistoryClient,
    *,
    trade_date: str,
    signal_slot: str,
    config: V3Config,
    late_recovery: bool = False,
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
            ["ts_code", "close", "adj_factor"],
        ]
        .rename(
            columns={
                "close": "pre_close",
                "adj_factor": "prior_adj_factor",
            }
        )
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
        ).fillna(
            pd.to_numeric(previous_close["prior_adj_factor"], errors="coerce")
        )
        previous_close = previous_close.drop(
            columns=["current_adj_factor", "prior_adj_factor"]
        )
    else:
        previous_close["adj_factor"] = pd.to_numeric(
            previous_close.pop("prior_adj_factor"),
            errors="coerce",
        )
    limits = client.query(
        "stk_limit",
        cache_key=trade_date,
        trade_date=trade_date,
        fields=LIMIT_FIELDS,
    )
    stock_basic = _load_stock_basic(client, cache_suffix=trade_date)
    industry_intervals = _load_industry_intervals(
        client,
        include_history=False,
        cache_suffix=trade_date,
    )
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

    expected_symbols = set(
        base.loc[
            base["board"].astype(str).eq(config.strategy.board_scope)
            & ~base["is_st"].fillna(True).astype(bool)
            & pd.to_numeric(base["adj_factor"], errors="coerce").gt(0),
            "ts_code",
        ].astype(str)
    )
    replay = pd.DataFrame()
    direct_replay = pd.DataFrame()
    direct_replay_codes: list[str] = []
    if late_recovery:
        replay = _fetch_rt_min_daily_replay(
            client,
            trade_date=trade_date,
            through_slot=signal_slot,
            ts_codes=sorted(expected_symbols),
            workers=config.history.minute_fetch_workers,
        )
        current_minute = replay.loc[
            pd.to_datetime(replay["trade_time"], errors="coerce")
            .dt.strftime("%H:%M")
            .eq(signal_slot)
        ].copy()
    else:
        current_minute = _fetch_rt_min_snapshot(
            client,
            trade_date=trade_date,
            observation_slot=signal_slot,
            ts_codes=sorted(expected_symbols),
        )
    if current_minute.empty:
        raise RuntimeError("live all-market rt_min 5MIN snapshot is missing")
    current_minute["bar_slot"] = pd.to_datetime(
        current_minute["trade_time"],
        errors="coerce",
    ).dt.strftime("%H:%M")
    current_minute = current_minute.loc[
        current_minute["bar_slot"].eq(signal_slot)
    ].drop(columns="bar_slot")
    if current_minute.empty:
        raise RuntimeError(
            f"live all-market rt_min has no completed {signal_slot} bar"
        )
    if not late_recovery:
        direct_replay_codes = _direct_replay_candidate_codes(
            current_minute,
            base=base,
            config=config,
        )
        if direct_replay_codes:
            direct_replay = _fetch_rt_min_daily_replay(
                client,
                trade_date=trade_date,
                through_slot=signal_slot,
                ts_codes=direct_replay_codes,
                workers=config.history.minute_fetch_workers,
            )
    requested_time = pd.Timestamp(
        f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} {signal_slot}:00"
    )
    current_age = (
        requested_time - pd.to_datetime(current_minute["trade_time"], errors="coerce")
    ).dt.total_seconds()
    fresh_current = current_minute.loc[
        current_age.between(
            -60,
            config.execution.max_market_data_age_seconds,
            inclusive="both",
        )
    ].copy()
    actual_market_time = current_minute["trade_time"].max()
    if pd.isna(actual_market_time):
        raise RuntimeError("live rt_min snapshot has no valid time")
    actual_slot = pd.Timestamp(actual_market_time).strftime("%H:%M")

    if late_recovery:
        day_snapshot = (
            replay.sort_values(["ts_code", "trade_time"], kind="stable")
            .drop_duplicates("ts_code", keep="first")
            [["ts_code", "open"]]
            .rename(columns={"open": "day_open"})
            .merge(
                previous_close[["ts_code", "pre_close"]],
                on="ts_code",
                how="inner",
            )
            .rename(columns={"pre_close": "rt_pre_close"})
        )
    else:
        open_snapshot = client.query(
            "rt_k",
            cache_key=f"{trade_date}_{signal_slot.replace(':', '')}_open",
            refresh=True,
            ts_code=RTK_WILDCARDS,
            fields=RTK_FIELDS,
        )
        day_snapshot = _normalize_rt_k_day(open_snapshot)
    if day_snapshot.empty:
        raise RuntimeError("live all-market rt_k day-open/previous-close snapshot is missing")
    quality = _minute_universe_quality(
        expected_symbols,
        set(day_snapshot["ts_code"].dropna().astype(str)),
        set(fresh_current["ts_code"].dropna().astype(str)),
        config,
        trade_date=trade_date,
    )
    observation_slots = _observation_slots(config.strategy.signal_slots)
    if late_recovery:
        tail = replay.loc[
            pd.to_datetime(replay["trade_time"], errors="coerce")
            .dt.strftime("%H:%M")
            .isin(observation_slots)
        ].copy()
    else:
        tail = _merge_direct_replay_tail(
            current_minute,
            direct_replay=direct_replay,
            observation_slots=observation_slots,
        )
    snapshots = _slot_features(tail, signal_slot)
    snapshots = snapshots.merge(day_snapshot, on="ts_code", how="left").merge(
        base, on="ts_code", how="inner"
    )
    snapshots["pre_close"] = pd.to_numeric(
        snapshots.pop("rt_pre_close"),
        errors="coerce",
    ).fillna(pd.to_numeric(snapshots["pre_close"], errors="coerce"))
    snapshots["signal_slot"] = signal_slot
    snapshots["signal_price"] = snapshots["slot_close"]
    snapshots["ret_from_prev_close_pct"] = (
        snapshots["signal_price"] / snapshots["pre_close"] - 1.0
    ) * 100.0
    snapshots["ret_from_open_pct"] = (
        snapshots["signal_price"] / snapshots["day_open"] - 1.0
    ) * 100.0
    snapshots["gap_open_pct"] = (
        snapshots["day_open"] / snapshots["pre_close"] - 1.0
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
            config.execution.min_distance_to_up_limit_pct
        )
        & snapshots["slot_bar_lag_minutes"].between(0, 5, inclusive="both")
        & snapshots["intraday_snapshot_count"].ge(
            config.execution.min_intraday_snapshot_count
        )
    )
    snapshots = _add_market_context(snapshots)
    snapshots = _add_industry_context(snapshots)
    snapshots = enrich_feature_frame(snapshots)
    capture_time = (
        requested_time + pd.Timedelta(2, unit="min")
        if late_recovery
        else pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
    )
    snapshots["data_age_seconds"] = (
        capture_time
        - pd.to_datetime(snapshots["slot_bar_time"], errors="coerce")
    ).dt.total_seconds()
    snapshots["execution_eligible"] = execution_eligibility(snapshots, config)
    snapshots["market_data_time"] = pd.Timestamp(actual_market_time).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    snapshots["entry_fillable"] &= snapshots["data_age_seconds"].between(
        -60,
        config.execution.max_market_data_age_seconds,
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
        "fresh_row_count": int(
            snapshots["data_age_seconds"]
            .between(
                -60,
                config.execution.max_market_data_age_seconds,
                inclusive="both",
            )
            .sum()
        ),
        "eligible_count": int(snapshots["execution_eligible"].sum()),
        "feature_version": config.model.feature_version,
        "market_data_source": (
            "rt_min_daily"
            if late_recovery
            else "rt_min+selective_rt_min_daily"
        ),
        "direct_replay_candidate_count": len(direct_replay_codes),
        "direct_replay_row_count": int(len(direct_replay)),
        **quality,
    }
    return snapshots, manifest


def _direct_replay_candidate_codes(
    current_minute: pd.DataFrame,
    *,
    base: pd.DataFrame,
    config: V3Config,
) -> list[str]:
    if current_minute.empty or base.empty:
        return []
    probe = (
        current_minute[
            ["ts_code", "close", "slot_amount"]
        ]
        .rename(columns={"close": "signal_price"})
        .merge(
            base[
                [
                    "ts_code",
                    "adj_factor",
                    "board",
                    "is_st",
                    "listing_days",
                    "prev_20d_amount",
                    "up_limit",
                    "down_limit",
                ]
            ],
            on="ts_code",
            how="inner",
        )
    )
    probe["distance_to_up_limit_pct"] = (
        pd.to_numeric(probe["up_limit"], errors="coerce")
        / pd.to_numeric(probe["signal_price"], errors="coerce")
        - 1.0
    ) * 100.0
    probe["distance_to_down_limit_pct"] = (
        pd.to_numeric(probe["signal_price"], errors="coerce")
        / pd.to_numeric(probe["down_limit"], errors="coerce")
        - 1.0
    ) * 100.0
    eligible = execution_eligibility(probe, config)
    return sorted(probe.loc[eligible, "ts_code"].dropna().astype(str).unique())


def _merge_direct_replay_tail(
    current_minute: pd.DataFrame,
    *,
    direct_replay: pd.DataFrame,
    observation_slots: tuple[str, ...],
) -> pd.DataFrame:
    tail = pd.concat(
        [current_minute, direct_replay],
        ignore_index=True,
    )
    tail_slots = pd.to_datetime(
        tail["trade_time"],
        errors="coerce",
    ).dt.strftime("%H:%M")
    return (
        tail.loc[tail_slots.isin(observation_slots)]
        .sort_values(["ts_code", "trade_time"], kind="stable")
        .drop_duplicates(["ts_code", "trade_time"], keep="last")
    )


def capture_live_minute_snapshot(
    client: TushareHistoryClient,
    *,
    trade_date: str,
    observation_slot: str,
    config: V3Config,
) -> dict[str, Any]:
    stock_basic = _load_stock_basic(client, cache_suffix=trade_date)
    active = stock_basic.loc[
        stock_basic["board"].astype(str).eq(config.strategy.board_scope)
        & stock_basic["list_status"].astype(str).eq("L")
        & ~stock_basic["name"].fillna("").astype(str).str.upper().str.contains("ST"),
        "ts_code",
    ].dropna()
    expected = set(active.astype(str))
    snapshot = _fetch_rt_min_snapshot(
        client,
        trade_date=trade_date,
        observation_slot=observation_slot,
        ts_codes=sorted(expected),
    )
    requested = pd.Timestamp(
        f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} {observation_slot}:00"
    )
    snapshot["bar_slot"] = pd.to_datetime(
        snapshot["trade_time"],
        errors="coerce",
    ).dt.strftime("%H:%M")
    snapshot = snapshot.loc[snapshot["bar_slot"].eq(observation_slot)].drop(
        columns="bar_slot"
    )
    age = (
        requested - pd.to_datetime(snapshot["trade_time"], errors="coerce")
    ).dt.total_seconds()
    fresh = set(
        snapshot.loc[
            age.between(
                -60,
                config.execution.max_market_data_age_seconds,
                inclusive="both",
            ),
            "ts_code",
        ].astype(str)
    )
    quality = _minute_universe_quality(
        expected,
        fresh,
        fresh,
        config,
        trade_date=trade_date,
    )
    return {
        "trade_date": trade_date,
        "observation_slot": observation_slot,
        "row_count": int(len(snapshot)),
        "latest_bar_time": (
            pd.Timestamp(snapshot["trade_time"].max()).strftime("%Y-%m-%d %H:%M:%S")
            if not snapshot.empty
            else None
        ),
        **quality,
    }


def capture_entry_settlement_frame(
    client: TushareHistoryClient,
    *,
    trade_date: str,
    settlement_slot: str,
    ts_codes: list[str],
    config: V3Config,
    late_recovery: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [
        "ts_code",
        "entry_benchmark_slot",
        "entry_benchmark_price",
        "entry_benchmark_amount",
        "entry_benchmark_bar_time",
        "data_age_seconds",
        "up_limit",
        "entry_benchmark_distance_to_up_limit_pct",
    ]
    requested = pd.Timestamp(
        f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} "
        f"{settlement_slot}:00"
    )
    codes = sorted({str(code) for code in ts_codes if str(code)})
    if not codes:
        return pd.DataFrame(columns=columns), {
            "schema_version": "wp_v9_entry_settlement_1",
            "trade_date": trade_date,
            "settlement_slot": settlement_slot,
            "requested_symbols": 0,
            "observed_symbols": 0,
            "fresh_symbols": 0,
            "capture_contract": (
                "retrospective_same_day_rt_min_daily"
                if late_recovery
                else "exact_next_5m_bar"
            ),
            "market_data_source": (
                "rt_min_daily" if late_recovery else "rt_min"
            ),
        }

    if late_recovery:
        snapshot = _fetch_rt_min_daily_replay(
            client,
            trade_date=trade_date,
            through_slot=settlement_slot,
            ts_codes=codes,
            workers=config.history.minute_fetch_workers,
        )
    else:
        snapshot = _fetch_rt_min_snapshot(
            client,
            trade_date=trade_date,
            observation_slot=settlement_slot,
            ts_codes=codes,
        ).copy()
    snapshot["bar_slot"] = pd.to_datetime(
        snapshot["trade_time"],
        errors="coerce",
    ).dt.strftime("%H:%M")
    snapshot = snapshot.loc[snapshot["bar_slot"].eq(settlement_slot)].copy()
    limits = client.query(
        "stk_limit",
        cache_key=trade_date,
        trade_date=trade_date,
        fields=LIMIT_FIELDS,
    )
    limits = limits[["ts_code", "up_limit"]].copy()
    limits["ts_code"] = limits["ts_code"].astype(str)
    limits["up_limit"] = pd.to_numeric(limits["up_limit"], errors="coerce")
    result = snapshot.merge(limits, on="ts_code", how="left")
    result["entry_benchmark_slot"] = settlement_slot
    result["entry_benchmark_price"] = pd.to_numeric(
        result["close"],
        errors="coerce",
    )
    result["entry_benchmark_amount"] = pd.to_numeric(
        result["slot_amount"],
        errors="coerce",
    )
    result["entry_benchmark_bar_time"] = pd.to_datetime(
        result["trade_time"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    capture_time = (
        requested + pd.Timedelta(2, unit="min")
        if late_recovery
        else pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
    )
    result["data_age_seconds"] = (
        capture_time - pd.to_datetime(result["trade_time"], errors="coerce")
    ).dt.total_seconds()
    result["entry_benchmark_distance_to_up_limit_pct"] = (
        result["up_limit"]
        / result["entry_benchmark_price"].replace(0, np.nan)
        - 1.0
    ) * 100.0
    result = result.reindex(columns=columns).drop_duplicates(
        "ts_code",
        keep="last",
    )
    fresh = result["data_age_seconds"].between(
        -60,
        config.execution.max_market_data_age_seconds,
        inclusive="both",
    )
    return result.reset_index(drop=True), {
        "schema_version": "wp_v9_entry_settlement_1",
        "trade_date": trade_date,
        "settlement_slot": settlement_slot,
        "requested_symbols": len(codes),
        "observed_symbols": int(len(result)),
        "fresh_symbols": int(fresh.sum()),
        "capture_contract": (
            "retrospective_same_day_rt_min_daily"
            if late_recovery
            else "exact_next_5m_bar"
        ),
        "market_data_source": (
            "rt_min_daily" if late_recovery else "rt_min"
        ),
        "latest_bar_time": (
            result["entry_benchmark_bar_time"].max()
            if not result.empty
            else None
        ),
    }


def _fetch_rt_min_daily_replay(
    client: TushareHistoryClient,
    *,
    trade_date: str,
    through_slot: str,
    ts_codes: list[str],
    workers: int,
) -> pd.DataFrame:
    requested_codes = sorted({str(code) for code in ts_codes if str(code)})

    def fetch(code: str) -> pd.DataFrame:
        raw = client.query(
            "rt_min_daily",
            cache_key=(
                f"{trade_date}_{through_slot.replace(':', '')}_{code}"
            ),
            refresh=True,
            ts_code=code,
            freq="5MIN",
        )
        normalized = _normalize_rt_min(raw, trade_date=trade_date)
        if normalized.empty:
            return normalized
        trade_time = pd.to_datetime(
            normalized["trade_time"],
            errors="coerce",
        )
        slots = trade_time.dt.strftime("%H:%M")
        dates = trade_time.dt.strftime("%Y%m%d")
        return normalized.loc[
            dates.eq(trade_date) & slots.le(through_slot)
        ].copy()

    frames: list[pd.DataFrame] = []
    for index, frame in enumerate(
        _ordered_bounded_map(
            fetch,
            requested_codes,
            workers=max(1, workers),
        ),
        start=1,
    ):
        if not frame.empty:
            frames.append(frame)
        if index % 250 == 0 or index == len(requested_codes):
            print(
                "WP same-day minute replay progress: "
                f"{index}/{len(requested_codes)}"
            )
    if not frames:
        raise RuntimeError("rt_min_daily returned no same-day minute rows")
    replay = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ts_code", "trade_time"], kind="stable")
        .drop_duplicates(["ts_code", "trade_time"], keep="last")
        .reset_index(drop=True)
    )
    exact_codes = set(
        replay.loc[
            pd.to_datetime(replay["trade_time"], errors="coerce")
            .dt.strftime("%H:%M")
            .eq(through_slot),
            "ts_code",
        ].astype(str)
    )
    if not exact_codes:
        raise RuntimeError(
            f"rt_min_daily has no completed {through_slot} bars"
        )
    return replay


def _fetch_rt_min_snapshot(
    client: TushareHistoryClient,
    *,
    trade_date: str,
    observation_slot: str,
    ts_codes: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for index in range(0, len(ts_codes), RTMIN_BATCH_SIZE):
        batch = ts_codes[index : index + RTMIN_BATCH_SIZE]
        digest = hashlib.sha256(",".join(batch).encode("ascii")).hexdigest()[:10]
        raw = client.query(
            "rt_min",
            cache_key=(
                f"{trade_date}_{observation_slot.replace(':', '')}_"
                f"{index // RTMIN_BATCH_SIZE:02d}_{digest}"
            ),
            refresh=True,
            ts_code=",".join(batch),
            freq="5MIN",
            fields=RTMIN_FIELDS,
        )
        normalized = _normalize_rt_min(raw, trade_date=trade_date)
        if normalized.empty:
            continue
        frames.append(normalized)
    if not frames:
        raise RuntimeError("no live rt_min rows returned for the main-board universe")
    snapshot = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ts_code", "trade_time"], kind="stable")
        .drop_duplicates("ts_code", keep="last")
        .reset_index(drop=True)
    )
    session_path = _rt_min_session_path(
        client,
        trade_date=trade_date,
        observation_slot=observation_slot,
    )
    session_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = session_path.with_suffix(".parquet.tmp")
    snapshot.to_parquet(temporary, index=False)
    temporary.replace(session_path)
    return snapshot


def _normalize_rt_min(frame: pd.DataFrame, *, trade_date: str) -> pd.DataFrame:
    columns = [
        "ts_code",
        "trade_time",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "slot_amount",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = frame.rename(columns={"code": "ts_code", "time": "trade_time"}).copy()
    required = {"ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise RuntimeError(f"rt_min response is missing columns: {missing}")
    raw_time = result["trade_time"].astype(str).str.strip()
    date_prefix = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    full_time = raw_time.where(
        raw_time.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", regex=True),
        date_prefix + " " + raw_time,
    )
    result["trade_time"] = pd.to_datetime(full_time, errors="coerce")
    for column in ("open", "high", "low", "close", "vol", "amount"):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
    result["ts_code"] = result["ts_code"].astype(str)
    result["slot_amount"] = result["amount"]
    return result.dropna(
        subset=["ts_code", "trade_time", "open", "high", "low", "close", "amount"]
    )[columns]


def _normalize_rt_k_day(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["ts_code", "rt_pre_close", "day_open"])
    result = frame.rename(columns={"code": "ts_code"}).copy()
    required = {"ts_code", "pre_close", "open"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise RuntimeError(f"rt_k response is missing day-contract columns: {missing}")
    result["ts_code"] = result["ts_code"].astype(str)
    result["rt_pre_close"] = pd.to_numeric(result["pre_close"], errors="coerce")
    result["day_open"] = pd.to_numeric(result["open"], errors="coerce")
    return result.dropna(subset=["ts_code", "rt_pre_close", "day_open"])[
        ["ts_code", "rt_pre_close", "day_open"]
    ].drop_duplicates("ts_code", keep="last")


def _load_rt_min_session_snapshots(
    client: TushareHistoryClient,
    *,
    trade_date: str,
    observation_slot: str,
    current: pd.DataFrame,
    observation_slots: tuple[str, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for slot in observation_slots:
        if slot > observation_slot:
            continue
        cache_path = _rt_min_session_path(
            client,
            trade_date=trade_date,
            observation_slot=slot,
        )
        frame = current if slot == observation_slot else (
            pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
        )
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        raise RuntimeError("no live rt_min session snapshots are available")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["ts_code", "trade_time"],
        kind="stable",
    )


def _rt_min_session_path(
    client: TushareHistoryClient,
    *,
    trade_date: str,
    observation_slot: str,
) -> Path:
    return (
        client.cache_dir
        / "rt_min_session"
        / f"{trade_date}_{observation_slot.replace(':', '')}_main.parquet"
    )
