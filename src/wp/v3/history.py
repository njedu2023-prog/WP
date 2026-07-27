from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import V3Config
from .dataset import audit_panel, build_supervised_panel


DAILY_FIELDS = (
    "ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount"
)
DAILY_BASIC_FIELDS = (
    "ts_code,trade_date,turnover_rate,volume_ratio,pe_ttm,pb,total_mv,circ_mv"
)
LIMIT_FIELDS = "trade_date,ts_code,up_limit,down_limit"
ADJ_FIELDS = "ts_code,trade_date,adj_factor"
MINUTE_FIELDS = "ts_code,trade_time,open,high,low,close,vol,amount"
MINUTE_STORE_COLUMNS = (
    "ts_code",
    "trade_date",
    "trade_time",
    "day_open",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
    "slot_amount",
)
WARMUP_SLOTS = ("14:00", "14:05", "14:10", "14:15")
INDUSTRY_FIELDS = (
    "l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,"
    "ts_code,name,in_date,out_date,is_new"
)
T = TypeVar("T")
R = TypeVar("R")


class _RequestStartLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self._minimum_interval = 60.0 / float(requests_per_minute)
        self._next_start = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            current = time.monotonic()
            scheduled = max(current, self._next_start)
            self._next_start = scheduled + self._minimum_interval
        delay = scheduled - current
        if delay > 0:
            time.sleep(delay)


class TushareHistoryClient:
    def __init__(
        self,
        pro: Any,
        cache_dir: str | Path,
        *,
        page_size: int = 8_000,
        attempts: int = 6,
        requests_per_minute: int = 180,
    ) -> None:
        self.pro = pro
        self.cache_dir = Path(cache_dir)
        self.page_size = page_size
        self.attempts = attempts
        self._request_limiter = _RequestStartLimiter(requests_per_minute)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def query(
        self,
        api_name: str,
        *,
        cache_key: str,
        paged: bool = False,
        refresh: bool = False,
        fields: str = "",
        **params: Any,
    ) -> pd.DataFrame:
        cache_path = self.cache_dir / api_name / f"{cache_key}.parquet"
        if cache_path.exists() and not refresh:
            return pd.read_parquet(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if paged:
            frame = self._paged_query(api_name, fields=fields, **params)
        else:
            frame = self._request(api_name, fields=fields, **params)
        temporary = cache_path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(cache_path)
        return frame

    def _paged_query(self, api_name: str, *, fields: str, **params: Any) -> pd.DataFrame:
        pages: list[pd.DataFrame] = []
        page_digests: set[str] = set()
        offset = 0
        for _ in range(10_000):
            page = self._request(
                api_name,
                fields=fields,
                limit=self.page_size,
                offset=offset,
                **params,
            )
            if page.empty:
                break
            page = page.drop_duplicates()
            digest = hashlib.sha256(
                pd.util.hash_pandas_object(
                    page,
                    index=False,
                    categorize=True,
                ).to_numpy(dtype=np.uint64).tobytes()
            ).hexdigest()
            if digest in page_digests:
                break
            page_digests.add(digest)
            pages.append(page)
            offset += len(page)
        else:
            raise RuntimeError(f"Tushare {api_name} pagination exceeded 10,000 pages")
        if not pages:
            return pd.DataFrame(columns=[item for item in fields.split(",") if item])
        return pd.concat(pages, ignore_index=True).drop_duplicates()

    def _request(self, api_name: str, *, fields: str, **params: Any) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(self.attempts):
            try:
                self._request_limiter.wait()
                frame = self.pro.query(api_name, fields=fields, **params)
                return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
            except Exception as error:  # Tushare raises several transport-specific types.
                last_error = error
                if attempt >= self.attempts - 1:
                    break
                time.sleep(min(2**attempt, 20))
        raise RuntimeError(f"Tushare {api_name} failed after retries: {last_error}") from last_error


def build_three_year_panel(
    client: TushareHistoryClient,
    config: V3Config,
    output_dir: str | Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    partition_dir = output / "panel"
    partition_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest = _reusable_panel_manifest(
        output / "wp_v3_dataset_manifest.json",
        partition_dir=partition_dir,
        config=config,
    )
    if existing_manifest is not None:
        print(
            "reusing verified WP V4 causal panel "
            f"{config.history.start_date}-{config.history.end_date}",
            flush=True,
        )
        return existing_manifest

    start = _date(config.history.start_date)
    warmup_start = (start - timedelta(days=120)).strftime("%Y%m%d")
    end = config.history.end_date

    calendar = client.query(
        "trade_cal",
        cache_key=f"{warmup_start}_{end}_sse",
        exchange="SSE",
        start_date=warmup_start,
        end_date=(datetime.strptime(end, "%Y%m%d") + timedelta(days=10)).strftime("%Y%m%d"),
        is_open="1",
        fields="exchange,cal_date,is_open,pretrade_date",
    )
    trade_dates = sorted(calendar.loc[calendar["is_open"].astype(str).eq("1"), "cal_date"].astype(str))
    target_dates = [date for date in trade_dates if config.history.start_date <= date <= end]
    if not target_dates:
        raise RuntimeError("Tushare trade calendar returned no target dates")
    target_months = {date[:6] for date in target_dates}
    for stale_partition in partition_dir.glob("wp_v3_panel_*.parquet"):
        if stale_partition.stem.rsplit("_", 1)[-1] not in target_months:
            stale_partition.unlink()
    next_date = {
        date: trade_dates[index + 1]
        for index, date in enumerate(trade_dates[:-1])
        if date in target_dates
    }
    required_daily_dates = [
        date
        for date in trade_dates
        if warmup_start <= date <= max(next_date.values(), default=end)
    ]

    basic = _load_stock_basic(client)
    industry_intervals = _load_industry_intervals(client, include_history=True)
    st_intervals = _load_st_intervals(client, warmup_start, max(required_daily_dates))
    daily, daily_basic, limits, adjustments = _load_daily_history(
        client,
        required_daily_dates,
        workers=config.history.minute_fetch_workers,
    )
    daily = daily.merge(
        adjustments.drop_duplicates(["ts_code", "trade_date"], keep="last"),
        on=["ts_code", "trade_date"],
        how="left",
    )
    adjustment_coverage = float(
        pd.to_numeric(daily["adj_factor"], errors="coerce").gt(0).mean()
    )
    if adjustment_coverage < 0.98:
        raise RuntimeError(
            f"adjustment-factor coverage {adjustment_coverage:.2%} is below 98%"
        )
    daily_features = _build_prior_day_features(daily, daily_basic)
    daily = _index_by_trade_date(daily)
    daily_features = _index_by_trade_date(daily_features)
    limits = _index_by_trade_date(limits)
    minute_partitions = _build_historical_minute_partitions(
        client,
        stock_basic=basic,
        start_date=config.history.start_date,
        end_date=end,
        output_dir=output / "minute",
        config=config,
    )

    failures: list[dict[str, str]] = []
    partitions: list[dict[str, Any]] = []
    daily_quality: list[dict[str, Any]] = []
    month_rows: list[pd.DataFrame] = []
    current_month: str | None = None
    current_minutes_by_date: dict[str, pd.DataFrame] = {}
    covered_dates: list[str] = []

    for index, trade_date in enumerate(target_dates, start=1):
        month = trade_date[:6]
        if current_month is not None and month != current_month and month_rows:
            partitions.append(_write_partition(month_rows, current_month, partition_dir))
            month_rows = []
        if month != current_month:
            minute_path = minute_partitions.get(month)
            if minute_path is None or not minute_path.exists():
                raise RuntimeError(f"historical minute partition is missing for {month}")
            current_minutes = pd.read_parquet(minute_path)
            current_minutes_by_date = {
                str(date): group.copy()
                for date, group in current_minutes.groupby("trade_date", sort=False)
            }
        current_month = month
        try:
            frame = _build_day_panel(
                trade_date=trade_date,
                target_trade_date=next_date[trade_date],
                daily=daily,
                daily_features=daily_features,
                limits=limits,
                stock_basic=basic,
                industry_intervals=industry_intervals,
                st_intervals=st_intervals,
                minute_bars=current_minutes_by_date.get(
                    trade_date,
                    pd.DataFrame(columns=MINUTE_STORE_COLUMNS),
                ),
                config=config,
            )
            if frame.empty:
                raise RuntimeError("no causal slot rows produced")
            daily_quality.append(dict(frame.attrs.get("data_quality", {})))
            month_rows.append(frame)
            covered_dates.append(trade_date)
        except Exception as error:
            failures.append({"trade_date": trade_date, "error": str(error)[:500]})
        if index % 20 == 0:
            print(
                f"history progress {index}/{len(target_dates)}; "
                f"covered={len(covered_dates)} failed={len(failures)}",
                flush=True,
            )

    if current_month and month_rows:
        partitions.append(_write_partition(month_rows, current_month, partition_dir))

    coverage = len(covered_dates) / len(target_dates)
    if not allow_partial and coverage < 0.98:
        raise RuntimeError(
            f"history coverage {coverage:.2%} is below the mandatory 98%; "
            f"failed dates: {failures[:8]}"
        )
    manifest = {
        "schema_version": "wp_point_in_time_panel_2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy_id": config.strategy.strategy_id,
        "feature_version": config.model.feature_version,
        "requested_start": config.history.start_date,
        "requested_end": config.history.end_date,
        "requested_trade_days": len(target_dates),
        "covered_trade_days": len(covered_dates),
        "coverage": coverage,
        "adjustment_factor_coverage": adjustment_coverage,
        "minute_universe_coverage": {
            "minimum_required": config.history.minimum_minute_universe_coverage,
            "minimum_open": min(
                (
                    float(item["open_universe_coverage"])
                    for item in daily_quality
                    if item.get("open_universe_coverage") is not None
                ),
                default=None,
            ),
            "minimum_tail": min(
                (
                    float(item["tail_universe_coverage"])
                    for item in daily_quality
                    if item.get("tail_universe_coverage") is not None
                ),
                default=None,
            ),
            "median_expected_symbols": float(
                np.median(
                    [
                        int(item["expected_symbols"])
                        for item in daily_quality
                        if item.get("expected_symbols") is not None
                    ]
                )
            )
            if daily_quality
            else None,
        },
        "failed_trade_days": failures,
        "partitions": partitions,
        "minute_partitions": {
            month: str(path.as_posix())
            for month, path in minute_partitions.items()
        },
        "execution_contract": asdict(config.execution),
        "signal_slots": list(config.strategy.signal_slots),
        "exit_contract": config.strategy.exit_contract,
    }
    manifest_path = output / "wp_v3_dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_panel_partitions(path: str | Path) -> pd.DataFrame:
    files = sorted(Path(path).glob("wp_v3_panel_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no WP V4 panel partitions under {path}")
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


def _load_stock_basic(
    client: TushareHistoryClient,
    *,
    cache_suffix: str = "",
) -> pd.DataFrame:
    frames = []
    suffix = f"_{cache_suffix}" if cache_suffix else ""
    for status in ("L", "D", "P"):
        frames.append(
            client.query(
                "stock_basic",
                cache_key=f"all_{status}{suffix}",
                list_status=status,
                fields="ts_code,name,industry,market,list_date,delist_date,list_status",
            )
        )
    basic = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="first")
    basic["board"] = basic["ts_code"].map(_board)
    return basic


def _load_industry_intervals(
    client: TushareHistoryClient,
    *,
    include_history: bool,
    cache_suffix: str = "",
) -> dict[str, list[tuple[str, str, str]]]:
    statuses = ("Y", "N") if include_history else ("Y",)
    suffix = f"_{cache_suffix}" if cache_suffix else ""
    frames = [
        client.query(
            "index_member_all",
            cache_key=f"sw_l1_{status}{suffix}",
            paged=True,
            is_new=status,
            fields=INDUSTRY_FIELDS,
        )
        for status in statuses
    ]
    memberships = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["ts_code", "l1_code", "in_date", "out_date"],
        keep="last",
    )
    intervals: dict[str, list[tuple[str, str, str]]] = {}
    for row in memberships.to_dict(orient="records"):
        code = str(row.get("ts_code") or "")
        industry = str(row.get("l1_name") or row.get("l1_code") or "")
        if not code or not industry:
            continue
        start = _date_field(row.get("in_date"), "19000101")
        end = _date_field(row.get("out_date"), "29991231")
        intervals.setdefault(code, []).append((start, end, industry))
    for code in intervals:
        intervals[code].sort(key=lambda item: item[0])
    return intervals


def _load_st_intervals(
    client: TushareHistoryClient,
    start_date: str,
    end_date: str,
) -> dict[str, list[tuple[str, str]]]:
    changes = client.query(
        "namechange",
        cache_key="all_history",
        paged=True,
        fields="ts_code,name,start_date,end_date,change_reason",
    )
    intervals: dict[str, list[tuple[str, str]]] = {}
    for row in changes.to_dict(orient="records"):
        name = str(row.get("name") or "").upper()
        if "ST" not in name:
            continue
        start = _date_field(row.get("start_date"), "19000101")
        end = _date_field(row.get("end_date"), "29991231")
        intervals.setdefault(str(row.get("ts_code")), []).append((start, end))
    return intervals


def _load_daily_history(
    client: TushareHistoryClient,
    trade_dates: list[str],
    *,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_frames: list[pd.DataFrame] = []
    basic_frames: list[pd.DataFrame] = []
    limit_frames: list[pd.DataFrame] = []
    adjustment_frames: list[pd.DataFrame] = []

    def fetch_trade_date(
        item: tuple[int, str],
    ) -> tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        index, trade_date = item
        daily = client.query(
            "daily",
            cache_key=trade_date,
            trade_date=trade_date,
            fields=DAILY_FIELDS,
        )
        basic = client.query(
            "daily_basic",
            cache_key=trade_date,
            trade_date=trade_date,
            fields=DAILY_BASIC_FIELDS,
        )
        limits = client.query(
            "stk_limit",
            cache_key=trade_date,
            trade_date=trade_date,
            fields=LIMIT_FIELDS,
        )
        adjustments = client.query(
            "adj_factor",
            cache_key=trade_date,
            trade_date=trade_date,
            fields=ADJ_FIELDS,
        )
        return index, daily, basic, limits, adjustments

    indexed_dates = list(enumerate(trade_dates, start=1))
    for index, daily, basic, limits, adjustments in _ordered_bounded_map(
        fetch_trade_date,
        indexed_dates,
        workers=workers,
    ):
        daily_frames.append(daily)
        basic_frames.append(basic)
        limit_frames.append(limits)
        adjustment_frames.append(adjustments)
        if index % 50 == 0:
            print(f"daily history {index}/{len(trade_dates)}", flush=True)
    return (
        pd.concat(daily_frames, ignore_index=True),
        pd.concat(basic_frames, ignore_index=True),
        pd.concat(limit_frames, ignore_index=True),
        pd.concat(adjustment_frames, ignore_index=True),
    )


def _build_prior_day_features(daily: pd.DataFrame, basic: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    if "adj_factor" not in frame:
        frame["adj_factor"] = 1.0
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame = frame.sort_values(["ts_code", "trade_date"], kind="stable")
    for column in (
        "open",
        "close",
        "high",
        "low",
        "pre_close",
        "pct_chg",
        "amount",
        "adj_factor",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # Tushare daily.amount is reported in thousand RMB; minute amount is RMB.
    frame["amount"] = frame["amount"] * 1_000.0
    grouped = frame.groupby("ts_code", sort=False, group_keys=False)
    adjusted_close = frame["close"] * frame["adj_factor"]
    adjusted_grouped = adjusted_close.groupby(frame["ts_code"], sort=False)
    total_return = (adjusted_close / adjusted_grouped.shift(1) - 1.0) * 100.0
    frame["prev_1d_return_pct"] = total_return.groupby(frame["ts_code"]).shift(1)
    for days in (2, 3, 5, 10, 20):
        previous_close = adjusted_grouped.shift(1)
        base_close = adjusted_grouped.shift(days + 1)
        frame[f"prev_{days}d_return_pct"] = (previous_close / base_close - 1.0) * 100.0
    shifted_return = total_return.groupby(frame["ts_code"]).shift(1)
    for days, minimum in ((5, 3), (10, 7), (20, 15)):
        frame[f"prev_{days}d_positive_share"] = (
            shifted_return.gt(0)
            .astype(float)
            .groupby(frame["ts_code"])
            .rolling(days, min_periods=minimum)
            .mean()
            .reset_index(level=0, drop=True)
        )
    frame["prev_20d_volatility_pct"] = (
        shifted_return.groupby(frame["ts_code"]).rolling(20, min_periods=15).std().reset_index(level=0, drop=True)
    )
    downside = shifted_return.where(shifted_return.lt(0), 0.0)
    frame["prev_20d_downside_volatility_pct"] = (
        downside.groupby(frame["ts_code"])
        .rolling(20, min_periods=15)
        .std()
        .reset_index(level=0, drop=True)
    )
    prior_adjusted_close = adjusted_grouped.shift(1)
    prior_20d_high = (
        prior_adjusted_close.groupby(frame["ts_code"])
        .rolling(20, min_periods=15)
        .max()
        .reset_index(level=0, drop=True)
    )
    frame["prev_20d_drawdown_pct"] = (
        prior_adjusted_close / prior_20d_high.replace(0, np.nan) - 1.0
    ) * 100.0
    day_gap = (frame["open"] / frame["pre_close"].replace(0, np.nan) - 1.0) * 100.0
    day_intraday = (frame["close"] / frame["open"].replace(0, np.nan) - 1.0) * 100.0
    day_range = frame["high"] - frame["low"]
    day_close_position = np.where(
        day_range.gt(0),
        (frame["close"] - frame["low"]) / day_range,
        0.5,
    )
    day_upper_wick = (
        (frame["high"] - frame[["open", "close"]].max(axis=1))
        / frame["pre_close"].replace(0, np.nan)
        * 100.0
    )
    day_lower_wick = (
        (frame[["open", "close"]].min(axis=1) - frame["low"])
        / frame["pre_close"].replace(0, np.nan)
        * 100.0
    )
    for name, values in (
        ("prev_day_gap_pct", day_gap),
        ("prev_day_intraday_return_pct", day_intraday),
        ("prev_day_close_position", pd.Series(day_close_position, index=frame.index)),
        ("prev_day_upper_wick_pct", day_upper_wick),
        ("prev_day_lower_wick_pct", day_lower_wick),
    ):
        frame[name] = values.groupby(frame["ts_code"]).shift(1)
    amplitude = (frame["high"] - frame["low"]) / frame["pre_close"].replace(0, np.nan) * 100.0
    shifted_amplitude = amplitude.groupby(frame["ts_code"]).shift(1)
    frame["prev_5d_amplitude_pct"] = (
        shifted_amplitude.groupby(frame["ts_code"]).rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    )
    shifted_amount = grouped["amount"].shift(1)
    frame["prev_20d_amount"] = (
        shifted_amount.groupby(frame["ts_code"]).rolling(20, min_periods=15).mean().reset_index(level=0, drop=True)
    )
    previous_5d_amount = (
        shifted_amount.groupby(frame["ts_code"])
        .rolling(5, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    frame["prev_amount_ratio_20d"] = shifted_amount / frame[
        "prev_20d_amount"
    ].replace(0, np.nan)
    frame["prev_5d_amount_ratio_20d"] = previous_5d_amount / frame[
        "prev_20d_amount"
    ].replace(0, np.nan)

    basic_columns = [
        "ts_code",
        "trade_date",
        "turnover_rate",
        "volume_ratio",
        "pe_ttm",
        "pb",
        "total_mv",
        "circ_mv",
    ]
    basics = basic.reindex(columns=basic_columns).copy()
    basics["trade_date"] = basics["trade_date"].astype(str)
    basics = basics.sort_values(["ts_code", "trade_date"], kind="stable")
    for column in basic_columns[2:]:
        basics[column] = pd.to_numeric(basics[column], errors="coerce")
        if column in {"total_mv", "circ_mv"}:
            basics[column] = basics[column] * 10_000.0
        basics[column] = basics.groupby("ts_code", sort=False)[column].shift(1)
    turnover_20d = (
        basics["turnover_rate"]
        .groupby(basics["ts_code"])
        .rolling(20, min_periods=15)
        .mean()
        .reset_index(level=0, drop=True)
    )
    basics["prev_turnover_ratio_20d"] = (
        basics["turnover_rate"] / turnover_20d.replace(0, np.nan)
    )
    basics = basics.rename(
        columns={
            "turnover_rate": "prev_turnover_rate",
            "volume_ratio": "prev_volume_ratio",
            "pe_ttm": "prev_pe_ttm",
            "pb": "prev_pb",
            "circ_mv": "float_mv",
        }
    )
    keep = [
        "ts_code",
        "trade_date",
        "prev_day_gap_pct",
        "prev_day_intraday_return_pct",
        "prev_day_close_position",
        "prev_day_upper_wick_pct",
        "prev_day_lower_wick_pct",
        "prev_1d_return_pct",
        "prev_2d_return_pct",
        "prev_3d_return_pct",
        "prev_5d_return_pct",
        "prev_10d_return_pct",
        "prev_20d_return_pct",
        "prev_5d_positive_share",
        "prev_10d_positive_share",
        "prev_20d_positive_share",
        "prev_20d_volatility_pct",
        "prev_20d_downside_volatility_pct",
        "prev_20d_drawdown_pct",
        "prev_5d_amplitude_pct",
        "prev_amount_ratio_20d",
        "prev_5d_amount_ratio_20d",
        "prev_20d_amount",
    ]
    return frame[keep].merge(basics, on=["ts_code", "trade_date"], how="left")


def _build_day_panel(
    *,
    trade_date: str,
    target_trade_date: str,
    daily: pd.DataFrame,
    daily_features: pd.DataFrame,
    limits: pd.DataFrame,
    stock_basic: pd.DataFrame,
    industry_intervals: dict[str, list[tuple[str, str, str]]],
    st_intervals: dict[str, list[tuple[str, str]]],
    minute_bars: pd.DataFrame,
    config: V3Config,
) -> pd.DataFrame:
    if minute_bars.empty:
        raise RuntimeError("missing historical five-minute decision snapshots")

    current_daily = _day(daily, trade_date)
    current_limit = _day(limits, trade_date)
    target_daily = _day(daily, target_trade_date).rename(
        columns={
            "open": "t1_open",
            "high": "t1_high",
            "low": "t1_low",
            "close": "t1_close",
            "vol": "t1_vol",
            "adj_factor": "t1_adj_factor",
        }
    )
    target_limit = _day(limits, target_trade_date).rename(
        columns={"up_limit": "t1_up_limit", "down_limit": "t1_down_limit"}
    )
    base = (
        current_daily[["ts_code", "pre_close", "vol", "adj_factor"]]
        .rename(columns={"vol": "current_day_vol"})
        .merge(current_limit[["ts_code", "up_limit", "down_limit"]], on="ts_code", how="left")
        .merge(_day(daily_features, trade_date), on=["ts_code"], how="left")
        .merge(stock_basic, on="ts_code", how="left")
        .merge(
            target_daily[
                [
                    "ts_code",
                    "t1_open",
                    "t1_high",
                    "t1_low",
                    "t1_close",
                    "t1_vol",
                    "t1_adj_factor",
                ]
            ],
            on="ts_code",
            how="left",
        )
        .merge(
            target_limit[["ts_code", "t1_down_limit", "t1_up_limit"]],
            on="ts_code",
            how="left",
        )
    )
    base["trade_date"] = trade_date
    base["target_trade_date"] = target_trade_date
    base["listing_days"] = (
        pd.Timestamp(_date(trade_date))
        - pd.to_datetime(base["list_date"], format="%Y%m%d", errors="coerce")
    ).dt.days
    base["is_st"] = [
        _is_st(code, trade_date, st_intervals, name)
        for code, name in zip(base["ts_code"], base["name"], strict=False)
    ]
    base["industry"] = [
        _industry_at(code, trade_date, industry_intervals)
        for code in base["ts_code"]
    ]
    numeric = (
        "pre_close",
        "up_limit",
        "down_limit",
        "t1_open",
        "t1_high",
        "t1_low",
        "t1_close",
        "t1_vol",
        "t1_down_limit",
        "current_day_vol",
        "adj_factor",
        "t1_adj_factor",
    )
    for column in numeric:
        base[column] = pd.to_numeric(base[column], errors="coerce")

    expected_symbols = set(
        base.loc[
            base["board"].astype(str).eq(config.strategy.board_scope)
            & ~base["is_st"].fillna(True).astype(bool)
            & base["current_day_vol"].gt(0)
            & base["adj_factor"].gt(0),
            "ts_code",
        ].astype(str)
    )
    minute_symbols = set(minute_bars["ts_code"].dropna().astype(str))
    quality = _minute_universe_quality(
        expected_symbols,
        minute_symbols,
        minute_symbols,
        config,
        trade_date=trade_date,
    )
    # A closing price pinned at the down limit is treated as non-executable.
    # This intentionally penalizes the strategy rather than assuming queue priority.
    base["exit_fillable"] = (
        base["t1_close"].notna()
        & base["t1_vol"].gt(0)
        & (
            base["t1_down_limit"].isna()
            | base["t1_close"].gt(base["t1_down_limit"] * 1.0001)
        )
    )
    base["t1_total_return_close"] = np.where(
        base["adj_factor"].gt(0) & base["t1_adj_factor"].gt(0),
        base["t1_close"] * base["t1_adj_factor"] / base["adj_factor"],
        np.nan,
    )

    bars = minute_bars.copy()
    bars["trade_time"] = pd.to_datetime(bars["trade_time"], errors="coerce")
    bars = bars.dropna(subset=["trade_time", "ts_code"]).sort_values(["ts_code", "trade_time"])
    for column in ("open", "high", "low", "close", "amount"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    open_price = (
        bars[["ts_code", "day_open"]]
        .drop_duplicates("ts_code", keep="last")
    )
    observation_slots = _observation_slots(config.strategy.signal_slots)
    tail = bars[
        bars["trade_time"].dt.strftime("%H:%M").isin(observation_slots)
    ].copy()
    all_snapshots = _slot_features_for_slots(
        tail,
        config.strategy.signal_slots,
    )
    rows: list[pd.DataFrame] = []
    for slot in config.strategy.signal_slots:
        snapshots = all_snapshots.loc[
            all_snapshots["signal_slot"].eq(slot)
        ].copy()
        snapshots = snapshots.merge(open_price, on="ts_code", how="left").merge(
            base, on="ts_code", how="inner"
        )
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
        )
        rows.append(snapshots)
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.loc[
        panel["board"].astype(str).eq(config.strategy.board_scope)
        & ~panel["is_st"].fillna(True).astype(bool)
        & pd.to_numeric(panel["adj_factor"], errors="coerce").gt(0)
    ].copy()
    panel = _add_market_context(panel)
    panel = _add_industry_context(panel)
    panel = build_supervised_panel(panel, config)
    panel = panel.loc[
        panel["execution_eligible"].fillna(False)
        & panel["label_available"].fillna(False)
    ].reset_index(drop=True)
    panel.attrs["data_quality"] = quality
    return panel


def _build_historical_minute_partitions(
    client: TushareHistoryClient,
    *,
    stock_basic: pd.DataFrame,
    start_date: str,
    end_date: str,
    output_dir: Path,
    config: V3Config,
) -> dict[str, Path]:
    months = sorted(
        pd.period_range(
            pd.Timestamp(_date(start_date)),
            pd.Timestamp(_date(end_date)),
            freq="M",
        ).strftime("%Y%m")
    )
    manifest_path = output_dir / "manifest.json"
    existing = _read_json(manifest_path)
    if (
        existing.get("start_date") == start_date
        and existing.get("end_date") == end_date
        and existing.get("signal_slots") == list(config.strategy.signal_slots)
        and existing.get("observation_slots")
        == list(_observation_slots(config.strategy.signal_slots))
    ):
        paths = {
            month: output_dir / f"wp_v3_minutes_{month}.parquet"
            for month in months
        }
        if all(path.exists() for path in paths.values()):
            return paths

    building = output_dir / "_building"
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, pq.ParquetWriter] = {}
    paths = {
        month: output_dir / f"wp_v3_minutes_{month}.parquet"
        for month in months
    }
    temporary_paths = {
        month: building / f"wp_v3_minutes_{month}.parquet"
        for month in months
    }
    query_failures: list[dict[str, str]] = []
    codes = _historical_minute_codes(stock_basic, start_date, end_date)
    if len(codes) < 1_000:
        raise RuntimeError(
            f"historical main-board minute universe has only {len(codes)} symbols"
        )
    records = list(enumerate(codes.to_dict(orient="records"), start=1))

    def fetch_symbol(
        item: tuple[int, dict[str, Any]],
    ) -> tuple[int, str, pd.DataFrame, Exception | None]:
        index, row = item
        code = str(row["ts_code"])
        symbol_start = max(start_date, _date_field(row.get("list_date"), start_date))
        symbol_end = min(end_date, _date_field(row.get("delist_date"), end_date))
        try:
            raw = client.query(
                "stk_mins",
                cache_key=(
                    f"{code.replace('.', '_')}_{symbol_start}_{symbol_end}_5min"
                ),
                paged=True,
                ts_code=code,
                start_date=f"{_dash(symbol_start)} 09:30:00",
                end_date=f"{_dash(symbol_end)} 15:00:00",
                freq="5min",
                fields=MINUTE_FIELDS,
            )
            selected = _normalize_historical_minutes(
                raw,
                signal_slots=config.strategy.signal_slots,
            )
        except Exception as error:
            return index, code, pd.DataFrame(), error
        return index, code, selected, None

    try:
        for index, code, selected, error in _ordered_bounded_map(
            fetch_symbol,
            records,
            workers=config.history.minute_fetch_workers,
        ):
            if error is not None:
                query_failures.append({"ts_code": code, "error": str(error)[:300]})
                if index == 1:
                    raise RuntimeError(
                        "historical minute probe failed; verify Tushare stk_mins "
                        f"permission and parameters: {error}"
                    ) from error
                continue
            for month, group in selected.groupby(
                selected["trade_date"].astype(str).str[:6],
                sort=False,
            ):
                if month not in temporary_paths or group.empty:
                    continue
                table = pa.Table.from_pandas(
                    group.reindex(columns=MINUTE_STORE_COLUMNS),
                    preserve_index=False,
                )
                writer = writers.get(month)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary_paths[month],
                        table.schema,
                        compression="zstd",
                    )
                    writers[month] = writer
                writer.write_table(table)
            if index % 50 == 0:
                print(
                    f"minute history {index}/{len(codes)}; "
                    f"failed_symbols={len(query_failures)}",
                    flush=True,
                )
    finally:
        for writer in writers.values():
            writer.close()

    failure_rate = len(query_failures) / len(codes)
    if failure_rate > 0.02:
        raise RuntimeError(
            f"historical minute symbol failure rate {failure_rate:.2%} exceeds 2%; "
            f"examples={query_failures[:5]}"
        )
    missing_months = [
        month for month, path in temporary_paths.items() if not path.exists()
    ]
    if missing_months:
        raise RuntimeError(f"historical minute months are missing: {missing_months}")
    for month, temporary in temporary_paths.items():
        temporary.replace(paths[month])
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "wp_v3_historical_minutes_2",
                "start_date": start_date,
                "end_date": end_date,
                "signal_slots": list(config.strategy.signal_slots),
                "observation_slots": list(
                    _observation_slots(config.strategy.signal_slots)
                ),
                "symbol_count": len(codes),
                "failed_symbols": query_failures,
                "partitions": {
                    month: str(path.as_posix())
                    for month, path in paths.items()
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(building)
    return paths


def _ordered_bounded_map(
    function: Callable[[T], R],
    items: list[T],
    *,
    workers: int,
) -> Iterator[R]:
    if workers <= 1:
        for item in items:
            yield function(item)
        return

    maximum_pending = workers * 2
    next_submit = 0
    pending: dict[int, Future[R]] = {}
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="wp-v3-minute",
    ) as executor:
        while next_submit < min(maximum_pending, len(items)):
            pending[next_submit] = executor.submit(function, items[next_submit])
            next_submit += 1
        for index in range(len(items)):
            future = pending.pop(index)
            yield future.result()
            if next_submit < len(items):
                pending[next_submit] = executor.submit(
                    function,
                    items[next_submit],
                )
                next_submit += 1


def _historical_minute_codes(
    stock_basic: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    frame = stock_basic.copy()
    listed = frame["list_date"].map(lambda value: _date_field(value, "29991231"))
    delisted = frame["delist_date"].map(lambda value: _date_field(value, "29991231"))
    return frame.loc[
        frame["board"].astype(str).eq("main_board")
        & listed.le(end_date)
        & delisted.ge(start_date),
        ["ts_code", "list_date", "delist_date"],
    ].drop_duplicates("ts_code")


def _normalize_historical_minutes(
    frame: pd.DataFrame,
    *,
    signal_slots: tuple[str, ...],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=MINUTE_STORE_COLUMNS)
    result = frame.reindex(columns=MINUTE_FIELDS.split(",")).copy()
    result["trade_time"] = pd.to_datetime(result["trade_time"], errors="coerce")
    result = result.dropna(subset=["ts_code", "trade_time"]).sort_values(
        ["ts_code", "trade_time"],
        kind="stable",
    )
    for column in ("open", "high", "low", "close", "vol", "amount"):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
    result["ts_code"] = result["ts_code"].astype(str)
    result["trade_date"] = result["trade_time"].dt.strftime("%Y%m%d")
    grouped = result.groupby(["ts_code", "trade_date"], sort=False)
    result["day_open"] = grouped["open"].transform("first")
    result["slot_amount"] = result["amount"]
    selected = result["trade_time"].dt.strftime("%H:%M").isin(
        _observation_slots(signal_slots)
    )
    return result.loc[selected, MINUTE_STORE_COLUMNS].reset_index(drop=True)


def _slot_features(bars: pd.DataFrame, slot: str) -> pd.DataFrame:
    result = _slot_features_for_slots(bars, (slot,))
    return result.drop(columns=["signal_slot"], errors="ignore")


def _slot_features_for_slots(
    bars: pd.DataFrame,
    slots: tuple[str, ...],
) -> pd.DataFrame:
    output_columns = [
        "ts_code",
        "signal_slot",
        "slot_close",
        "slot_amount",
        "slot_bar_time",
        "slot_bar_lag_minutes",
        "intraday_snapshot_count",
        "ret_5m_pct",
        "ret_10m_pct",
        "ret_20m_pct",
        "bar_body_pct",
        "bar_range_pct",
        "bar_upper_wick_pct",
        "bar_lower_wick_pct",
        "tail_range_10m_pct",
        "tail_close_position_10m",
        "tail_return_from_1400_pct",
        "tail_range_since_1400_pct",
        "tail_close_position_since_1400",
        "tail_amount_weighted_price_gap_pct",
        "tail_realized_volatility_pct",
        "tail_mean_abs_return_pct",
        "tail_up_bar_share",
        "tail_down_bar_share",
        "tail_directional_efficiency",
        "tail_trend_slope_pct",
        "tail_trend_r2",
        "tail_max_drawdown_pct",
        "tail_rebound_from_low_pct",
        "tail_cumulative_amount",
        "tail_latest_amount_share",
        "tail_amount_concentration",
        "tail_amount_acceleration",
    ]
    if bars.empty:
        return pd.DataFrame(columns=output_columns)

    frame = bars.sort_values(
        ["ts_code", "trade_time"],
        kind="stable",
    ).reset_index(drop=True)
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    frame = frame.dropna(subset=["ts_code", "trade_time"])
    for column in ("open", "high", "low", "close", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grouped = frame.groupby("ts_code", sort=False)
    frame["_snapshot_count"] = grouped.cumcount() + 1
    for periods, column in (
        (1, "_ret_5m_pct"),
        (2, "_ret_10m_pct"),
        (4, "_ret_20m_pct"),
    ):
        base = grouped["close"].shift(periods)
        frame[column] = np.where(
            base.gt(0),
            (frame["close"] / base - 1.0) * 100.0,
            np.nan,
        )
    frame["_range_high"] = (
        grouped["high"]
        .rolling(2, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )
    frame["_range_low"] = (
        grouped["low"]
        .rolling(2, min_periods=1)
        .min()
        .reset_index(level=0, drop=True)
    )
    previous_amount = grouped["amount"].shift(1)
    previous_three_mean = (
        previous_amount.groupby(frame["ts_code"], sort=False)
        .rolling(3, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    frame["_amount_acceleration"] = np.where(
        previous_three_mean.gt(0),
        frame["amount"] / previous_three_mean,
        np.nan,
    )
    previous_close = grouped["close"].shift(1)
    frame["_bar_log_return"] = np.log(
        frame["close"]
        / previous_close.fillna(frame["open"]).replace(0, np.nan)
    )
    frame["_bar_abs_return"] = frame["_bar_log_return"].abs()
    frame["_bar_up"] = frame["_bar_log_return"].gt(0).astype(float)
    frame["_bar_down"] = frame["_bar_log_return"].lt(0).astype(float)
    frame["_return_sum"] = frame.groupby("ts_code", sort=False)[
        "_bar_log_return"
    ].cumsum()
    frame["_return_sq_sum"] = (
        frame["_bar_log_return"].pow(2).groupby(frame["ts_code"], sort=False).cumsum()
    )
    frame["_abs_return_sum"] = frame.groupby("ts_code", sort=False)[
        "_bar_abs_return"
    ].cumsum()
    frame["_up_count"] = frame.groupby("ts_code", sort=False)["_bar_up"].cumsum()
    frame["_down_count"] = frame.groupby("ts_code", sort=False)["_bar_down"].cumsum()
    count = frame["_snapshot_count"].astype(float)
    variance = (
        frame["_return_sq_sum"] - frame["_return_sum"].pow(2) / count
    ) / (count - 1.0).replace(0, np.nan)
    frame["tail_realized_volatility_pct"] = np.sqrt(
        variance.clip(lower=0)
    ) * 100.0
    frame["tail_mean_abs_return_pct"] = (
        frame["_abs_return_sum"] / count * 100.0
    )
    frame["tail_up_bar_share"] = frame["_up_count"] / count
    frame["tail_down_bar_share"] = frame["_down_count"] / count

    first_open = grouped["open"].transform("first")
    cumulative_high = grouped["high"].cummax()
    cumulative_low = grouped["low"].cummin()
    cumulative_amount = grouped["amount"].cumsum()
    cumulative_max_amount = grouped["amount"].cummax()
    typical_price = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    cumulative_weighted_price = (
        (typical_price * frame["amount"])
        .groupby(frame["ts_code"], sort=False)
        .cumsum()
        / cumulative_amount.replace(0, np.nan)
    )
    frame["tail_return_from_1400_pct"] = (
        frame["close"] / first_open.replace(0, np.nan) - 1.0
    ) * 100.0
    frame["tail_range_since_1400_pct"] = (
        cumulative_high / cumulative_low.replace(0, np.nan) - 1.0
    ) * 100.0
    cumulative_width = cumulative_high - cumulative_low
    frame["tail_close_position_since_1400"] = np.where(
        cumulative_width.gt(0),
        (frame["close"] - cumulative_low) / cumulative_width,
        0.5,
    )
    frame["tail_amount_weighted_price_gap_pct"] = (
        frame["close"] / cumulative_weighted_price.replace(0, np.nan) - 1.0
    ) * 100.0
    frame["tail_directional_efficiency"] = np.where(
        frame["_abs_return_sum"].gt(0),
        np.log(frame["close"] / first_open.replace(0, np.nan))
        / frame["_abs_return_sum"],
        0.0,
    )
    running_high_close = grouped["close"].cummax()
    running_low_close = grouped["close"].cummin()
    current_drawdown = (
        frame["close"] / running_high_close.replace(0, np.nan) - 1.0
    ) * 100.0
    frame["tail_max_drawdown_pct"] = current_drawdown.groupby(
        frame["ts_code"],
        sort=False,
    ).cummin()
    frame["tail_rebound_from_low_pct"] = (
        frame["close"] / running_low_close.replace(0, np.nan) - 1.0
    ) * 100.0
    frame["tail_cumulative_amount"] = cumulative_amount
    frame["tail_latest_amount_share"] = (
        frame["amount"] / cumulative_amount.replace(0, np.nan)
    )
    frame["tail_amount_concentration"] = (
        cumulative_max_amount / cumulative_amount.replace(0, np.nan)
    )

    x = grouped.cumcount().astype(float)
    sum_x = count * (count - 1.0) / 2.0
    sum_x2 = count * (count - 1.0) * (2.0 * count - 1.0) / 6.0
    log_close = np.log(frame["close"].replace(0, np.nan))
    sum_y = log_close.groupby(frame["ts_code"], sort=False).cumsum()
    sum_y2 = log_close.pow(2).groupby(frame["ts_code"], sort=False).cumsum()
    sum_xy = (x * log_close).groupby(frame["ts_code"], sort=False).cumsum()
    covariance_numerator = count * sum_xy - sum_x * sum_y
    x_variance_numerator = count * sum_x2 - sum_x.pow(2)
    y_variance_numerator = count * sum_y2 - sum_y.pow(2)
    frame["tail_trend_slope_pct"] = (
        covariance_numerator / x_variance_numerator.replace(0, np.nan) * 100.0
    )
    frame["tail_trend_r2"] = (
        covariance_numerator.pow(2)
        / (
            x_variance_numerator.replace(0, np.nan)
            * y_variance_numerator.replace(0, np.nan)
        )
    ).clip(lower=0.0, upper=1.0)

    high_body = frame[["open", "close"]].max(axis=1)
    low_body = frame[["open", "close"]].min(axis=1)
    frame["bar_body_pct"] = (
        frame["close"] / frame["open"].replace(0, np.nan) - 1.0
    ) * 100.0
    frame["bar_range_pct"] = (
        frame["high"] / frame["low"].replace(0, np.nan) - 1.0
    ) * 100.0
    frame["bar_upper_wick_pct"] = (
        frame["high"] / high_body.replace(0, np.nan) - 1.0
    ) * 100.0
    frame["bar_lower_wick_pct"] = (
        low_body / frame["low"].replace(0, np.nan) - 1.0
    ) * 100.0

    frame["slot_amount"] = pd.to_numeric(
        frame["slot_amount"] if "slot_amount" in frame else frame["amount"],
        errors="coerce",
    )
    range_width = frame["_range_high"] - frame["_range_low"]
    frame["tail_range_10m_pct"] = np.where(
        frame["_range_low"].gt(0),
        (frame["_range_high"] / frame["_range_low"] - 1.0) * 100.0,
        np.nan,
    )
    frame["tail_close_position_10m"] = np.where(
        range_width.gt(0),
        (frame["close"] - frame["_range_low"]) / range_width,
        0.5,
    )
    frame = frame.rename(
        columns={
            "close": "slot_close",
            "_snapshot_count": "intraday_snapshot_count",
            "_ret_5m_pct": "ret_5m_pct",
            "_ret_10m_pct": "ret_10m_pct",
            "_ret_20m_pct": "ret_20m_pct",
            "_amount_acceleration": "tail_amount_acceleration",
        }
    )
    day = frame["trade_time"].dt.normalize().min()
    targets = pd.MultiIndex.from_product(
        [
            frame["ts_code"].drop_duplicates().astype(str),
            slots,
        ],
        names=["ts_code", "signal_slot"],
    ).to_frame(index=False)
    targets["target_time"] = day + pd.to_timedelta(
        targets["signal_slot"].map(_minute_value).astype(int),
        unit="m",
    )
    source = frame[
        [
            "ts_code",
            "trade_time",
            "slot_close",
            "slot_amount",
            "intraday_snapshot_count",
            "ret_5m_pct",
            "ret_10m_pct",
            "ret_20m_pct",
            "tail_range_10m_pct",
            "tail_close_position_10m",
            "bar_body_pct",
            "bar_range_pct",
            "bar_upper_wick_pct",
            "bar_lower_wick_pct",
            "tail_return_from_1400_pct",
            "tail_range_since_1400_pct",
            "tail_close_position_since_1400",
            "tail_amount_weighted_price_gap_pct",
            "tail_realized_volatility_pct",
            "tail_mean_abs_return_pct",
            "tail_up_bar_share",
            "tail_down_bar_share",
            "tail_directional_efficiency",
            "tail_trend_slope_pct",
            "tail_trend_r2",
            "tail_max_drawdown_pct",
            "tail_rebound_from_low_pct",
            "tail_cumulative_amount",
            "tail_latest_amount_share",
            "tail_amount_concentration",
            "tail_amount_acceleration",
        ]
    ].copy()
    targets["ts_code"] = targets["ts_code"].astype(str)
    source["ts_code"] = source["ts_code"].astype(str)
    matched = pd.merge_asof(
        targets.sort_values(["target_time", "ts_code"], kind="stable"),
        source.sort_values(["trade_time", "ts_code"], kind="stable"),
        left_on="target_time",
        right_on="trade_time",
        by="ts_code",
        direction="backward",
        allow_exact_matches=True,
    ).dropna(subset=["trade_time", "slot_close"])
    matched["slot_bar_lag_minutes"] = (
        matched["target_time"] - matched["trade_time"]
    ).dt.total_seconds() / 60.0
    matched["slot_bar_time"] = matched["trade_time"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return matched[output_columns].reset_index(drop=True)


def _add_market_context(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    grouped = result.groupby(["trade_date", "signal_slot"], sort=False)
    result["market_return_pct"] = grouped["ret_from_prev_close_pct"].transform("median")
    result["market_breadth"] = grouped["ret_from_prev_close_pct"].transform(
        lambda values: values.gt(0).mean()
    )
    result["market_breadth_above_2pct"] = grouped[
        "ret_from_prev_close_pct"
    ].transform(lambda values: values.gt(2.0).mean())
    result["market_breadth_above_5pct"] = grouped[
        "ret_from_prev_close_pct"
    ].transform(lambda values: values.gt(5.0).mean())
    result["market_return_dispersion_pct"] = grouped[
        "ret_from_prev_close_pct"
    ].transform("std")
    result["market_gap_pct"] = grouped["gap_open_pct"].transform("median")
    result["market_tail_return_pct"] = grouped["ret_20m_pct"].transform("median")
    result["market_tail_breadth"] = grouped["ret_20m_pct"].transform(
        lambda values: values.gt(0).mean()
    )
    result["market_tail_dispersion_pct"] = grouped["ret_20m_pct"].transform("std")
    result["market_prev_5d_return_pct"] = grouped["prev_5d_return_pct"].transform(
        "median"
    )
    result["market_prev_20d_volatility_pct"] = grouped[
        "prev_20d_volatility_pct"
    ].transform("median")
    result["up_limit_count"] = grouped["distance_to_up_limit_pct"].transform(
        lambda values: values.le(0.10).sum()
    )
    result["down_limit_count"] = grouped["distance_to_down_limit_pct"].transform(
        lambda values: values.le(0.10).sum()
    )
    return result


def _add_industry_context(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    valid = result["industry"].fillna("").astype(str).ne("")
    result["industry_return_pct"] = np.nan
    result["industry_breadth"] = np.nan
    result["industry_tail_return_pct"] = np.nan
    result["industry_tail_breadth"] = np.nan
    if valid.any():
        grouped = result.loc[valid].groupby(
            ["trade_date", "signal_slot", "industry"], sort=False
        )
        result.loc[valid, "industry_return_pct"] = grouped[
            "ret_from_prev_close_pct"
        ].transform("median")
        result.loc[valid, "industry_breadth"] = grouped[
            "ret_from_prev_close_pct"
        ].transform(lambda values: values.gt(0).mean())
        result.loc[valid, "industry_tail_return_pct"] = grouped[
            "ret_20m_pct"
        ].transform("median")
        result.loc[valid, "industry_tail_breadth"] = grouped[
            "ret_20m_pct"
        ].transform(lambda values: values.gt(0).mean())
    return result


def _write_partition(
    frames: list[pd.DataFrame],
    month: str,
    partition_dir: Path,
) -> dict[str, Any]:
    frame = pd.concat(frames, ignore_index=True)
    path = partition_dir / f"wp_v3_panel_{month}.parquet"
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = audit_panel(frame)
    return {
        "path": str(path.as_posix()),
        "sha256": digest,
        **asdict(audit),
        "start_trade_date": str(frame["trade_date"].min()),
        "end_trade_date": str(frame["trade_date"].max()),
    }


def _day(frame: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if frame.index.name == "_trade_date_index":
        try:
            return frame.loc[[str(trade_date)]].copy()
        except KeyError:
            return frame.iloc[0:0].copy()
    return frame.loc[frame["trade_date"].astype(str).eq(str(trade_date))].copy()


def _index_by_trade_date(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = pd.Index(
        result["trade_date"].astype(str),
        name="_trade_date_index",
    )
    return result.sort_index(kind="stable")


def _return(closes: np.ndarray, periods: int) -> float:
    if len(closes) <= periods or closes[-periods - 1] <= 0:
        return np.nan
    return float((closes[-1] / closes[-periods - 1] - 1.0) * 100.0)


def _board(code: str) -> str:
    value = str(code)
    number = value.split(".")[0]
    if number.startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
        return "main_board"
    if number.startswith("300"):
        return "chinext"
    if number.startswith("688"):
        return "star"
    if number.startswith(("4", "8", "9")):
        return "bse"
    return "other"


def _is_st(
    code: str,
    trade_date: str,
    intervals: dict[str, list[tuple[str, str]]],
    fallback_name: Any,
) -> bool:
    entries = intervals.get(str(code), [])
    if entries:
        return any(start <= trade_date <= (end or "29991231") for start, end in entries)
    return "ST" in str(fallback_name or "").upper()


def _industry_at(
    code: str,
    trade_date: str,
    intervals: dict[str, list[tuple[str, str, str]]],
) -> str:
    active = [
        (start, industry)
        for start, end, industry in intervals.get(str(code), [])
        if start <= trade_date <= (end or "29991231")
    ]
    return max(active, key=lambda item: item[0])[1] if active else ""


def _minute_universe_quality(
    expected_symbols: set[str],
    open_symbols: set[str],
    tail_symbols: set[str],
    config: V3Config,
    *,
    trade_date: str,
) -> dict[str, Any]:
    if len(expected_symbols) < 1_000:
        raise RuntimeError(
            f"{trade_date} expected executable market has only "
            f"{len(expected_symbols)} symbols"
        )
    open_coverage = len(expected_symbols & open_symbols) / len(expected_symbols)
    tail_coverage = len(expected_symbols & tail_symbols) / len(expected_symbols)
    minimum = config.history.minimum_minute_universe_coverage
    if open_coverage < minimum or tail_coverage < minimum:
        raise RuntimeError(
            f"{trade_date} minute-universe coverage is incomplete: "
            f"open={open_coverage:.2%}, tail={tail_coverage:.2%}, "
            f"required={minimum:.2%}"
        )
    return {
        "trade_date": trade_date,
        "expected_symbols": len(expected_symbols),
        "open_symbols": len(expected_symbols & open_symbols),
        "tail_symbols": len(expected_symbols & tail_symbols),
        "open_universe_coverage": open_coverage,
        "tail_universe_coverage": tail_coverage,
    }


def _minute_value(value: str) -> int:
    hour, minute = str(value).split(":")
    return int(hour) * 60 + int(minute)


def _observation_slots(signal_slots: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*WARMUP_SLOTS, *signal_slots)))


def _date_field(value: Any, default: str) -> str:
    if value is None or pd.isna(value):
        return default
    normalized = str(value).strip().replace("-", "").replace(".0", "")
    return normalized if len(normalized) == 8 and normalized.isdigit() else default


def _date(value: str) -> datetime:
    return datetime.strptime(str(value), "%Y%m%d")


def _dash(value: str) -> str:
    parsed = _date(value)
    return parsed.strftime("%Y-%m-%d")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _reusable_panel_manifest(
    manifest_path: Path,
    *,
    partition_dir: Path,
    config: V3Config,
) -> dict[str, Any] | None:
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") not in {
        "wp_v3_causal_panel_1",
        "wp_point_in_time_panel_2",
    }:
        return None
    expected = {
        "requested_start": config.history.start_date,
        "requested_end": config.history.end_date,
        "execution_contract": json.loads(json.dumps(asdict(config.execution))),
        "signal_slots": list(config.strategy.signal_slots),
        "exit_contract": config.strategy.exit_contract,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    if float(manifest.get("coverage", 0.0) or 0.0) < 0.98:
        return None
    if manifest.get("failed_trade_days"):
        return None

    partitions = manifest.get("partitions")
    if not isinstance(partitions, list) or not partitions:
        return None
    for item in partitions:
        if not isinstance(item, dict):
            return None
        path_value = item.get("path")
        expected_digest = str(item.get("sha256") or "")
        if not path_value or not expected_digest:
            return None
        partition_path = partition_dir / Path(str(path_value)).name
        if (
            not partition_path.is_file()
            or partition_path.stat().st_size <= 0
            or _file_sha256(partition_path) != expected_digest
        ):
            return None
    return manifest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
