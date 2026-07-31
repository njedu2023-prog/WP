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
PEER_FEATURE_SUFFIXES = (
    "count",
    "return_count",
    "return_median_pct",
    "return_mean_pct",
    "return_q75_pct",
    "return_iqr_pct",
    "return_dispersion_pct",
    "return_max_pct",
    "positive_share",
    "above_2pct_share",
    "above_5pct_share",
    "above_7pct_share",
    "tail_median_pct",
    "tail_q75_pct",
    "tail_positive_share",
    "own_excess_pct",
    "own_gap_to_q75_pct",
    "own_gap_to_max_pct",
    "own_percentile",
    "own_log_amount_excess",
)
V28_PEER_FEATURE_COLUMNS = tuple(
    f"v28_{level[:2]}_peer_{suffix}"
    for level in PEER_LEVELS
    for suffix in PEER_FEATURE_SUFFIXES
)
MINIMUM_L2_PEERS = 4
MINIMUM_L3_PEERS = 2
MINIMUM_STOCKS_PER_SLOT = 1_000
MINIMUM_FIELD_COVERAGE = 0.95
MINIMUM_PRECLOSE_COVERAGE = 0.98
MINIMUM_L2_GROUPS = 80
MINIMUM_L3_GROUPS = 120


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


def audit_stock_slot_frame(
    stock_slots: pd.DataFrame,
    *,
    trade_date: str,
) -> dict[str, Any]:
    timestamps = pd.to_datetime(
        stock_slots.reindex(columns=["trade_timestamp"])["trade_timestamp"],
        errors="coerce",
    )
    date_consistent = bool(
        len(stock_slots)
        and timestamps.notna().all()
        and timestamps.dt.strftime("%Y%m%d").eq(str(trade_date)).all()
    )
    duplicate_identity = bool(
        stock_slots.duplicated(
            ["trade_date", "signal_slot", "ts_code"]
        ).any()
    )
    slot_records: list[dict[str, Any]] = []
    for slot in SIGNAL_SLOTS:
        frame = stock_slots.loc[
            stock_slots["signal_slot"].astype(str).eq(slot)
        ].copy()
        rows = int(len(frame))
        preclose_coverage = _positive_coverage(frame, "pre_close")
        l2_coverage = _text_coverage(frame, "l2_code")
        l3_coverage = _text_coverage(frame, "l3_code")
        return_coverage = _numeric_coverage(
            frame,
            "ret_from_prev_close_pct",
        )
        tail_coverage = _numeric_coverage(frame, "ret_20m_pct")
        l2_groups = peer_group_count(
            frame,
            level="l2_code",
            minimum_members=5,
        )
        l3_groups = peer_group_count(
            frame,
            level="l3_code",
            minimum_members=3,
        )
        passed = bool(
            rows >= MINIMUM_STOCKS_PER_SLOT
            and preclose_coverage >= MINIMUM_PRECLOSE_COVERAGE
            and l2_coverage >= MINIMUM_FIELD_COVERAGE
            and l3_coverage >= MINIMUM_FIELD_COVERAGE
            and return_coverage >= MINIMUM_FIELD_COVERAGE
            and tail_coverage >= MINIMUM_FIELD_COVERAGE
            and l2_groups >= MINIMUM_L2_GROUPS
            and l3_groups >= MINIMUM_L3_GROUPS
        )
        slot_records.append(
            {
                "signal_slot": slot,
                "rows": rows,
                "preclose_coverage": preclose_coverage,
                "l2_coverage": l2_coverage,
                "l3_coverage": l3_coverage,
                "return_coverage": return_coverage,
                "tail_20m_coverage": tail_coverage,
                "eligible_l2_groups": l2_groups,
                "eligible_l3_groups": l3_groups,
                "coverage_pass": passed,
            }
        )
    return {
        "trade_date": str(trade_date),
        "rows": int(len(stock_slots)),
        "date_consistent": date_consistent,
        "duplicate_identity": duplicate_identity,
        "slot_count": len(slot_records),
        "slot_records": slot_records,
        "coverage_pass": bool(
            date_consistent
            and not duplicate_identity
            and len(slot_records) == len(SIGNAL_SLOTS)
            and all(record["coverage_pass"] for record in slot_records)
        ),
    }


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
            _clean_key(row.get("trade_date")),
            _clean_key(row.get("signal_slot")),
            _clean_key(row.get(level)),
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
        amounts = pd.to_numeric(
            peers.reindex(columns=["amount"])["amount"],
            errors="coerce",
        )
        amounts = amounts.loc[amounts.ge(0)].dropna()
        own_return = _finite_or_nan(row.get("ret_from_prev_close_pct"))
        own_amount = _finite_or_nan(row.get("amount"))
        peer_median = float(returns.median()) if len(returns) else np.nan
        peer_q75 = (
            float(returns.quantile(0.75)) if len(returns) else np.nan
        )
        peer_q25 = (
            float(returns.quantile(0.25)) if len(returns) else np.nan
        )
        peer_max = float(returns.max()) if len(returns) else np.nan
        log_amount_median = (
            float(np.log1p(amounts).median()) if len(amounts) else np.nan
        )
        rows.append(
            {
                **{column: row.get(column) for column in identity},
                f"{prefix}_count": int(peers["ts_code"].nunique()),
                f"{prefix}_return_count": int(len(returns)),
                f"{prefix}_return_median_pct": peer_median,
                f"{prefix}_return_mean_pct": (
                    float(returns.mean()) if len(returns) else np.nan
                ),
                f"{prefix}_return_q75_pct": peer_q75,
                f"{prefix}_return_iqr_pct": (
                    peer_q75 - peer_q25
                    if np.isfinite(peer_q75) and np.isfinite(peer_q25)
                    else np.nan
                ),
                f"{prefix}_return_dispersion_pct": (
                    float(returns.std(ddof=0)) if len(returns) else np.nan
                ),
                f"{prefix}_return_max_pct": peer_max,
                f"{prefix}_positive_share": (
                    float(returns.gt(0).mean()) if len(returns) else np.nan
                ),
                f"{prefix}_above_2pct_share": (
                    float(returns.gt(2.0).mean()) if len(returns) else np.nan
                ),
                f"{prefix}_above_5pct_share": (
                    float(returns.gt(5.0).mean()) if len(returns) else np.nan
                ),
                f"{prefix}_above_7pct_share": (
                    float(returns.gt(7.0).mean()) if len(returns) else np.nan
                ),
                f"{prefix}_tail_median_pct": (
                    float(tail_returns.median())
                    if len(tail_returns)
                    else np.nan
                ),
                f"{prefix}_tail_q75_pct": (
                    float(tail_returns.quantile(0.75))
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
                f"{prefix}_own_gap_to_q75_pct": (
                    own_return - peer_q75
                    if np.isfinite(own_return) and np.isfinite(peer_q75)
                    else np.nan
                ),
                f"{prefix}_own_gap_to_max_pct": (
                    own_return - peer_max
                    if np.isfinite(own_return) and np.isfinite(peer_max)
                    else np.nan
                ),
                f"{prefix}_own_percentile": (
                    float(returns.le(own_return).mean())
                    if len(returns) and np.isfinite(own_return)
                    else np.nan
                ),
                f"{prefix}_own_log_amount_excess": (
                    float(np.log1p(own_amount) - log_amount_median)
                    if own_amount >= 0 and np.isfinite(log_amount_median)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).reindex(
        columns=[*identity, *peer_feature_columns(level)]
    )


def peer_feature_columns(level: str) -> tuple[str, ...]:
    if level not in PEER_LEVELS:
        raise ValueError(f"unsupported peer level: {level}")
    prefix = f"v28_{level[:2]}_peer"
    return tuple(f"{prefix}_{suffix}" for suffix in PEER_FEATURE_SUFFIXES)


def audit_peer_feature_coverage(
    features: pd.DataFrame,
    candidate_index: pd.DataFrame,
) -> dict[str, Any]:
    identity = ["trade_date", "signal_slot", "ts_code"]
    missing = sorted(
        set(identity).union(V28_PEER_FEATURE_COLUMNS) - set(features.columns)
    )
    expected = candidate_index.reindex(columns=identity).drop_duplicates()
    duplicate_identity = bool(features.duplicated(identity).any())
    actual = features.reindex(columns=identity).drop_duplicates()
    candidate_match = bool(
        len(actual) == len(expected)
        and actual.merge(
            expected.assign(_expected=1),
            on=identity,
            how="outer",
            indicator=True,
        )["_merge"].eq("both").all()
    )
    if missing:
        complete = pd.Series(False, index=features.index, dtype=bool)
    else:
        numeric = features.loc[:, V28_PEER_FEATURE_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )
        complete = (
            numeric.notna().all(axis=1)
            & numeric["v28_l2_peer_count"].ge(MINIMUM_L2_PEERS)
            & numeric["v28_l3_peer_count"].ge(MINIMUM_L3_PEERS)
        )
    complete_rate = float(complete.mean()) if len(features) else 0.0
    return {
        "expected_candidate_rows": int(len(expected)),
        "feature_rows": int(len(features)),
        "missing_columns": missing,
        "duplicate_identity": duplicate_identity,
        "candidate_identity_match": candidate_match,
        "complete_feature_rows": int(complete.sum()),
        "complete_feature_coverage": complete_rate,
        "minimum_complete_feature_coverage": 0.98,
        "coverage_passed": bool(
            not missing
            and not duplicate_identity
            and candidate_match
            and complete_rate >= 0.98
        ),
    }


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


def _clean_key(value: Any) -> str:
    if value is None or bool(pd.isna(value)):
        return ""
    return str(value).strip()


def _numeric_coverage(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return float(
        pd.to_numeric(frame[column], errors="coerce").notna().mean()
    )


def _positive_coverage(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").gt(0).mean())


def _text_coverage(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return float(
        frame[column].fillna("").astype(str).str.strip().ne("").mean()
    )
