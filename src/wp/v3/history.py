from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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
INDUSTRY_FIELDS = (
    "l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,"
    "ts_code,name,in_date,out_date,is_new"
)


class TushareHistoryClient:
    def __init__(
        self,
        pro: Any,
        cache_dir: str | Path,
        *,
        page_size: int = 8_000,
        attempts: int = 6,
    ) -> None:
        self.pro = pro
        self.cache_dir = Path(cache_dir)
        self.page_size = page_size
        self.attempts = attempts
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def query(
        self,
        api_name: str,
        *,
        cache_key: str,
        paged: bool = False,
        fields: str = "",
        **params: Any,
    ) -> pd.DataFrame:
        cache_path = self.cache_dir / api_name / f"{cache_key}.parquet"
        if cache_path.exists():
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
    current_minutes = pd.DataFrame(columns=MINUTE_STORE_COLUMNS)
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
                minute_bars=current_minutes.loc[
                    current_minutes["trade_date"].astype(str).eq(trade_date)
                ].copy(),
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
        "schema_version": "wp_v3_causal_panel_1",
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
        raise FileNotFoundError(f"no WP V3 panel partitions under {path}")
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


def _load_stock_basic(client: TushareHistoryClient) -> pd.DataFrame:
    frames = []
    for status in ("L", "D", "P"):
        frames.append(
            client.query(
                "stock_basic",
                cache_key=f"all_{status}",
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
) -> dict[str, list[tuple[str, str, str]]]:
    statuses = ("Y", "N") if include_history else ("Y",)
    frames = [
        client.query(
            "index_member_all",
            cache_key=f"sw_l1_{status}",
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
    try:
        changes = client.query(
            "namechange",
            cache_key="all_history",
            paged=True,
            fields="ts_code,name,start_date,end_date,change_reason",
        )
    except RuntimeError:
        return {}
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_frames: list[pd.DataFrame] = []
    basic_frames: list[pd.DataFrame] = []
    limit_frames: list[pd.DataFrame] = []
    adjustment_frames: list[pd.DataFrame] = []
    for index, trade_date in enumerate(trade_dates, start=1):
        daily_frames.append(
            client.query(
                "daily",
                cache_key=trade_date,
                trade_date=trade_date,
                fields=DAILY_FIELDS,
            )
        )
        basic_frames.append(
            client.query(
                "daily_basic",
                cache_key=trade_date,
                trade_date=trade_date,
                fields=DAILY_BASIC_FIELDS,
            )
        )
        limit_frames.append(
            client.query(
                "stk_limit",
                cache_key=trade_date,
                trade_date=trade_date,
                fields=LIMIT_FIELDS,
            )
        )
        adjustment_frames.append(
            client.query(
                "adj_factor",
                cache_key=trade_date,
                trade_date=trade_date,
                fields=ADJ_FIELDS,
            )
        )
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
    for days in (2, 5, 10, 20):
        previous_close = adjusted_grouped.shift(1)
        base_close = adjusted_grouped.shift(days + 1)
        frame[f"prev_{days}d_return_pct"] = (previous_close / base_close - 1.0) * 100.0
    shifted_return = total_return.groupby(frame["ts_code"]).shift(1)
    frame["prev_20d_volatility_pct"] = (
        shifted_return.groupby(frame["ts_code"]).rolling(20, min_periods=15).std().reset_index(level=0, drop=True)
    )
    amplitude = (frame["high"] - frame["low"]) / frame["pre_close"].replace(0, np.nan) * 100.0
    shifted_amplitude = amplitude.groupby(frame["ts_code"]).shift(1)
    frame["prev_5d_amplitude_pct"] = (
        shifted_amplitude.groupby(frame["ts_code"]).rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    )
    shifted_amount = grouped["amount"].shift(1)
    frame["prev_20d_amount"] = (
        shifted_amount.groupby(frame["ts_code"]).rolling(20, min_periods=15).mean().reset_index(level=0, drop=True)
    )

    basic_columns = ["ts_code", "trade_date", "turnover_rate", "pe_ttm", "pb", "total_mv", "circ_mv"]
    basics = basic.reindex(columns=basic_columns).copy()
    basics["trade_date"] = basics["trade_date"].astype(str)
    basics = basics.sort_values(["ts_code", "trade_date"], kind="stable")
    for column in basic_columns[2:]:
        basics[column] = pd.to_numeric(basics[column], errors="coerce")
        if column in {"total_mv", "circ_mv"}:
            basics[column] = basics[column] * 10_000.0
        basics[column] = basics.groupby("ts_code", sort=False)[column].shift(1)
    basics = basics.rename(
        columns={
            "turnover_rate": "prev_turnover_rate",
            "circ_mv": "float_mv",
        }
    )
    keep = [
        "ts_code",
        "trade_date",
        "prev_1d_return_pct",
        "prev_2d_return_pct",
        "prev_5d_return_pct",
        "prev_10d_return_pct",
        "prev_20d_return_pct",
        "prev_20d_volatility_pct",
        "prev_5d_amplitude_pct",
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
    tail = bars[bars["trade_time"].dt.strftime("%H:%M").between("14:20", "14:50")].copy()
    rows: list[pd.DataFrame] = []
    for slot in config.strategy.signal_slots:
        slot_bars = tail[tail["trade_time"].dt.strftime("%H:%M").le(slot)].copy()
        slot_bars = slot_bars.groupby("ts_code", group_keys=False).tail(5)
        snapshots = _slot_features(slot_bars, slot)
        snapshots = snapshots.merge(open_price, on="ts_code", how="left").merge(
            base, on="ts_code", how="inner"
        )
        snapshots["signal_slot"] = slot
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
        rows.append(snapshots)
    panel = pd.concat(rows, ignore_index=True)
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
    try:
        for index, row in enumerate(codes.to_dict(orient="records"), start=1):
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
                "schema_version": "wp_v3_historical_minutes_1",
                "start_date": start_date,
                "end_date": end_date,
                "signal_slots": list(config.strategy.signal_slots),
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
    result["high"] = grouped["high"].cummax()
    result["low"] = grouped["low"].cummin()
    cumulative_amount = grouped["amount"].cumsum()
    bar_count = grouped.cumcount() + 1
    result["slot_amount"] = cumulative_amount / bar_count
    selected = result["trade_time"].dt.strftime("%H:%M").isin(signal_slots)
    return result.loc[selected, MINUTE_STORE_COLUMNS].reset_index(drop=True)


def _slot_features(bars: pd.DataFrame, slot: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for code, group in bars.groupby("ts_code", sort=False):
        group = group.sort_values("trade_time")
        last = group.iloc[-1]
        closes = group["close"].to_numpy(dtype=float)
        amounts = group["amount"].to_numpy(dtype=float)
        highs = group["high"].tail(2)
        lows = group["low"].tail(2)
        range_high = float(highs.max())
        range_low = float(lows.min())
        records.append(
            {
                "ts_code": code,
                "slot_close": float(last["close"]),
                "slot_amount": float(last["amount"]),
                "slot_bar_time": pd.Timestamp(last["trade_time"]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "slot_bar_lag_minutes": (
                    _minute_value(slot)
                    - (
                        int(pd.Timestamp(last["trade_time"]).hour) * 60
                        + int(pd.Timestamp(last["trade_time"]).minute)
                    )
                ),
                "ret_5m_pct": _return(closes, 1),
                "ret_10m_pct": _return(closes, 2),
                "ret_20m_pct": _return(closes, 4),
                "tail_range_10m_pct": (range_high / range_low - 1.0) * 100.0
                if range_low > 0
                else np.nan,
                "tail_close_position_10m": (float(last["close"]) - range_low)
                / (range_high - range_low)
                if range_high > range_low
                else 0.5,
                "tail_amount_acceleration": float(amounts[-1] / np.mean(amounts[-4:-1]))
                if len(amounts) >= 4 and np.mean(amounts[-4:-1]) > 0
                else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def _add_market_context(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    grouped = result.groupby(["trade_date", "signal_slot"], sort=False)
    result["market_return_pct"] = grouped["ret_from_prev_close_pct"].transform("median")
    result["market_breadth"] = grouped["ret_from_prev_close_pct"].transform(
        lambda values: values.gt(0).mean()
    )
    result["market_tail_return_pct"] = grouped["ret_20m_pct"].transform("median")
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
    return frame.loc[frame["trade_date"].astype(str).eq(str(trade_date))].copy()


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
