from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


MEMBER_FIELDS = (
    "l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,"
    "ts_code,name,in_date,out_date,is_new"
)
SIGNAL_SLOTS = (
    "14:20",
    "14:25",
    "14:30",
    "14:35",
    "14:40",
    "14:45",
    "14:50",
)
PEER_LEVELS = ("l2_code", "l3_code")


def normalize_membership(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.reindex(columns=MEMBER_FIELDS.split(",")).copy()
    for column in (
        "l1_code",
        "l1_name",
        "l2_code",
        "l2_name",
        "l3_code",
        "l3_name",
        "ts_code",
        "is_new",
    ):
        data[column] = data[column].fillna("").astype(str).str.strip()
    data["in_timestamp"] = pd.to_datetime(
        data["in_date"].fillna("").astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    out_raw = data["out_date"].fillna("").astype(str).str.strip()
    data["out_timestamp"] = pd.to_datetime(
        out_raw.where(out_raw.ne("")),
        format="%Y%m%d",
        errors="coerce",
    )
    return data.drop_duplicates(
        [
            "ts_code",
            "l1_code",
            "l2_code",
            "l3_code",
            "in_date",
            "out_date",
        ],
        keep="last",
    ).reset_index(drop=True)


def active_membership(
    membership: pd.DataFrame,
    *,
    trade_date: str,
) -> pd.DataFrame:
    timestamp = pd.Timestamp(datetime.strptime(str(trade_date), "%Y%m%d"))
    valid_interval = membership["in_timestamp"].notna() & (
        membership["out_timestamp"].isna()
        | membership["out_timestamp"].ge(membership["in_timestamp"])
    )
    active = membership.loc[
        valid_interval
        & membership["in_timestamp"].le(timestamp)
        & (
            membership["out_timestamp"].isna()
            | membership["out_timestamp"].gt(timestamp)
        )
    ].copy()
    active.sort_values(
        ["ts_code", "in_timestamp"],
        kind="stable",
        inplace=True,
    )
    return active.drop_duplicates("ts_code", keep="last").reset_index(
        drop=True
    )


def build_stock_slot_frame(
    minute_bars: pd.DataFrame,
    daily_preclose: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    trade_date: str,
    signal_slots: Iterable[str] = SIGNAL_SLOTS,
) -> pd.DataFrame:
    slots = tuple(signal_slots)
    required_minute = {
        "ts_code",
        "trade_date",
        "trade_time",
        "close",
        "amount",
    }
    missing_minute = sorted(required_minute - set(minute_bars.columns))
    if missing_minute:
        raise ValueError(f"minute frame missing columns: {missing_minute}")
    required_daily = {"ts_code", "trade_date", "pre_close"}
    missing_daily = sorted(required_daily - set(daily_preclose.columns))
    if missing_daily:
        raise ValueError(f"daily frame missing columns: {missing_daily}")

    bars = minute_bars.loc[:, sorted(required_minute)].copy()
    bars["ts_code"] = bars["ts_code"].fillna("").astype(str).str.strip()
    bars["trade_date"] = bars["trade_date"].map(normalize_trade_date)
    bars["trade_timestamp"] = pd.to_datetime(
        bars["trade_time"],
        errors="coerce",
    ).astype("datetime64[ns]")
    bars["signal_slot"] = bars["trade_timestamp"].dt.strftime("%H:%M")
    bars = bars.loc[
        bars["trade_date"].eq(str(trade_date))
        & bars["signal_slot"].isin(slots)
    ].copy()
    if bars.duplicated(["ts_code", "trade_timestamp"]).any():
        raise ValueError("minute frame contains duplicate stock timestamps")
    for column in ("close", "amount"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")

    daily = daily_preclose.loc[:, sorted(required_daily)].copy()
    daily["ts_code"] = daily["ts_code"].fillna("").astype(str).str.strip()
    daily["trade_date"] = daily["trade_date"].map(normalize_trade_date)
    daily["pre_close"] = pd.to_numeric(daily["pre_close"], errors="coerce")
    daily = daily.loc[daily["trade_date"].eq(str(trade_date))].drop_duplicates(
        "ts_code",
        keep="last",
    )
    active = active_membership(membership, trade_date=str(trade_date))
    active = active.reindex(
        columns=["ts_code", "l1_code", "l2_code", "l3_code"]
    )

    bars["previous_20m_timestamp"] = (
        bars["trade_timestamp"] - np.timedelta64(20, "m")
    )
    history = minute_bars.reindex(
        columns=["ts_code", "trade_date", "trade_time", "close"]
    ).copy()
    history["ts_code"] = history["ts_code"].fillna("").astype(str).str.strip()
    history["trade_date"] = history["trade_date"].map(normalize_trade_date)
    history["previous_20m_timestamp"] = pd.to_datetime(
        history["trade_time"],
        errors="coerce",
    ).astype("datetime64[ns]")
    history["close_20m_ago"] = pd.to_numeric(
        history["close"],
        errors="coerce",
    )
    history = history.loc[history["trade_date"].eq(str(trade_date))].drop_duplicates(
        ["ts_code", "previous_20m_timestamp"],
        keep="last",
    )

    result = (
        bars.merge(
            history[
                ["ts_code", "previous_20m_timestamp", "close_20m_ago"]
            ],
            on=["ts_code", "previous_20m_timestamp"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            daily[["ts_code", "pre_close"]],
            on="ts_code",
            how="left",
            validate="many_to_one",
        )
        .merge(
            active,
            on="ts_code",
            how="left",
            validate="many_to_one",
        )
    )
    result["ret_from_prev_close_pct"] = (
        result["close"] / result["pre_close"] - 1.0
    ) * 100.0
    result["ret_20m_pct"] = (
        result["close"] / result["close_20m_ago"] - 1.0
    ) * 100.0
    result.sort_values(
        ["trade_date", "signal_slot", "ts_code"],
        kind="stable",
        inplace=True,
    )
    return result.reset_index(drop=True)


def peer_group_count(
    stock_slots: pd.DataFrame,
    *,
    level: str,
    minimum_members: int,
) -> int:
    if level not in PEER_LEVELS:
        raise ValueError(f"unsupported peer level: {level}")
    valid = stock_slots[level].fillna("").astype(str).str.strip().ne("")
    sizes = (
        stock_slots.loc[valid]
        .groupby(["trade_date", "signal_slot", level], sort=False)[
            "ts_code"
        ]
        .nunique()
    )
    return int(sizes.ge(int(minimum_members)).sum())


def leave_one_out_peer_features(
    stock_slots: pd.DataFrame,
    candidate_index: pd.DataFrame,
    *,
    level: str,
) -> pd.DataFrame:
    if level not in PEER_LEVELS:
        raise ValueError(f"unsupported peer level: {level}")
    identity = ["trade_date", "signal_slot", "ts_code"]
    candidates = candidate_index.reindex(columns=identity).drop_duplicates()
    universe = stock_slots.copy()
    if universe.duplicated(identity).any():
        raise ValueError("peer universe contains duplicate identities")
    selected = candidates.merge(
        universe,
        on=identity,
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    group_keys = ["trade_date", "signal_slot", level]
    universe_groups = {
        key: group.copy()
        for key, group in universe.groupby(group_keys, sort=False)
    }
    prefix = f"v28_{level[:2]}_peer"
    for row in selected.to_dict(orient="records"):
        key = (
            str(row.get("trade_date") or ""),
            str(row.get("signal_slot") or ""),
            str(row.get(level) or ""),
        )
        group = universe_groups.get(key, universe.iloc[0:0])
        peers = group.loc[
            group["ts_code"].astype(str).ne(str(row.get("ts_code") or ""))
        ].copy()
        returns = pd.to_numeric(
            peers.reindex(columns=["ret_from_prev_close_pct"])[
                "ret_from_prev_close_pct"
            ],
            errors="coerce",
        ).dropna()
        tail_returns = pd.to_numeric(
            peers.reindex(columns=["ret_20m_pct"])["ret_20m_pct"],
            errors="coerce",
        ).dropna()
        own_return = _finite_or_nan(row.get("ret_from_prev_close_pct"))
        peer_median = float(returns.median()) if len(returns) else np.nan
        rows.append(
            {
                **{column: row.get(column) for column in identity},
                f"{prefix}_count": int(peers["ts_code"].nunique()),
                f"{prefix}_return_median_pct": peer_median,
                f"{prefix}_return_dispersion_pct": (
                    float(returns.std(ddof=0)) if len(returns) else np.nan
                ),
                f"{prefix}_positive_share": (
                    float(returns.gt(0).mean()) if len(returns) else np.nan
                ),
                f"{prefix}_above_2pct_share": (
                    float(returns.gt(2.0).mean()) if len(returns) else np.nan
                ),
                f"{prefix}_tail_median_pct": (
                    float(tail_returns.median())
                    if len(tail_returns)
                    else np.nan
                ),
                f"{prefix}_tail_positive_share": (
                    float(tail_returns.gt(0).mean())
                    if len(tail_returns)
                    else np.nan
                ),
                f"{prefix}_own_excess_pct": (
                    own_return - peer_median
                    if np.isfinite(own_return) and np.isfinite(peer_median)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def normalize_trade_date(value: Any) -> str:
    raw = str(value).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    digits = "".join(character for character in raw if character.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _finite_or_nan(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan
