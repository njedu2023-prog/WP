from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .meta_alpha import IDENTITY_COLUMNS
from .v28_industry_peer import active_membership
from .v30_limit_event import (
    CURRENT_DAY_FORBIDDEN_COLUMNS,
    KPL_FIELDS,
    build_causal_event_projection,
    normalize_kpl_frame,
)


SCHEMA_VERSION = "wp_v33_limit_industry_ecology_probe_1"
LEVELS = ("l2", "l3")
_A_SHARE_CODE = re.compile(r"^\d{6}\.(?:SH|SZ|BJ)$")

MARKET_CURRENT_COLUMNS = (
    "v33_market_limit_hit_count",
    "v33_market_limit_open_count",
    "v33_market_limit_down_count",
    "v33_market_limit_hit_last_10m",
    "v33_market_limit_open_last_10m",
    "v33_market_net_sealed_count",
    "v33_market_open_to_hit_ratio",
)
LEVEL_CURRENT_SUFFIXES = (
    "member_count",
    "limit_hit_count",
    "limit_open_count",
    "limit_down_count",
    "limit_hit_last_10m",
    "limit_open_last_10m",
    "net_sealed_count",
    "limit_hit_rate",
    "limit_down_rate",
    "open_to_hit_ratio",
    "market_hit_share",
    "active_before_signal",
)
MARKET_PREVIOUS_COLUMNS = (
    "v33_prev_market_limit_hit_count",
    "v33_prev_market_limit_open_count",
    "v33_prev_market_limit_down_count",
    "v33_prev_market_net_sealed_count",
    "v33_prev_market_open_to_hit_ratio",
)
LEVEL_PREVIOUS_SUFFIXES = (
    "member_count",
    "limit_hit_count",
    "limit_open_count",
    "limit_down_count",
    "net_sealed_count",
    "limit_hit_rate",
    "open_to_hit_ratio",
    "active",
)
V33_LIMIT_ECOLOGY_FEATURE_COLUMNS = (
    *MARKET_CURRENT_COLUMNS,
    *(
        f"v33_{level}_{suffix}"
        for level in LEVELS
        for suffix in LEVEL_CURRENT_SUFFIXES
    ),
    *MARKET_PREVIOUS_COLUMNS,
    *(
        f"v33_prev_{level}_{suffix}"
        for level in LEVELS
        for suffix in LEVEL_PREVIOUS_SUFFIXES
    ),
)


def audit_decision_tape_frame(
    frame: pd.DataFrame,
    *,
    trade_date: str,
    requested_tag: str,
) -> dict[str, Any]:
    required = set(KPL_FIELDS.split(","))
    if not required.issubset(frame.columns):
        return {
            "trade_date": trade_date,
            "requested_tag": requested_tag,
            "rows": int(len(frame)),
            "schema_ok": False,
            "coverage_pass": False,
        }
    normalized = normalize_kpl_frame(
        frame,
        trade_date=trade_date,
        requested_tag=requested_tag,
    )
    relevant_raw = (
        normalized["ld_time"]
        if requested_tag == "跌停"
        else normalized["lu_time"]
    )
    relevant_timestamp = (
        normalized["_ld_timestamp"]
        if requested_tag == "跌停"
        else normalized["_lu_timestamp"]
    )
    supplied_relevant = _supplied_time(relevant_raw)
    relevant_parse = relevant_timestamp.notna()
    relevant_coverage = (
        float(relevant_parse.mean()) if len(normalized) else 1.0
    )
    supplied_parse_rate = (
        float(relevant_parse.loc[supplied_relevant].mean())
        if supplied_relevant.any()
        else 1.0
    )
    supplied_open = _supplied_time(normalized["open_time"])
    open_parse_rate = (
        float(normalized.loc[supplied_open, "_open_timestamp"].notna().mean())
        if supplied_open.any()
        else 1.0
    )
    open_order = normalized.loc[
        normalized["_lu_timestamp"].notna()
        & normalized["_open_timestamp"].notna()
    ]
    open_after_touch = bool(
        open_order.empty
        or open_order["_open_timestamp"]
        .ge(open_order["_lu_timestamp"])
        .all()
    )
    date_ok = bool(
        normalized.empty
        or normalized["trade_date"].astype(str).eq(str(trade_date)).all()
    )
    code_ok = bool(
        normalized["ts_code"].astype(str).str.match(_A_SHARE_CODE).all()
        if len(normalized)
        else True
    )
    duplicates = bool(normalized["ts_code"].duplicated().any())
    coverage_pass = bool(
        date_ok
        and code_ok
        and not duplicates
        and relevant_coverage >= 0.90
        and supplied_parse_rate >= 1.0
        and open_parse_rate >= 1.0
        and open_after_touch
    )
    return {
        "trade_date": trade_date,
        "requested_tag": requested_tag,
        "rows": int(len(normalized)),
        "schema_ok": True,
        "date_ok": date_ok,
        "a_share_code_ok": code_ok,
        "duplicate_codes": duplicates,
        "relevant_time_coverage": relevant_coverage,
        "supplied_relevant_time_parse_rate": supplied_parse_rate,
        "supplied_open_time_parse_rate": open_parse_rate,
        "open_not_before_first_touch": open_after_touch,
        "post_1450_events_used_for_current_features": False,
        "coverage_pass": coverage_pass,
    }


def build_date_candidate_ecology(
    candidates: pd.DataFrame,
    *,
    trade_date: str,
    previous_trade_date: str,
    current_stock_events: pd.DataFrame,
    previous_stock_events: pd.DataFrame,
    market_projection: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    required = {*IDENTITY_COLUMNS, "fold"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"V33 candidates missing columns: {missing}")
    date_candidates = candidates.loc[
        candidates["trade_date"].astype(str).eq(str(trade_date)),
        [*IDENTITY_COLUMNS, "fold"],
    ].copy()
    if date_candidates.empty:
        raise ValueError(f"V33 has no candidates for {trade_date}")
    if date_candidates.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("V33 candidates contain duplicate identities")

    current_members = _active_member_frame(membership, trade_date)
    previous_members = _active_member_frame(
        membership,
        previous_trade_date,
    )
    result = date_candidates.merge(
        current_members,
        on="ts_code",
        how="left",
        validate="many_to_one",
    )
    result["v33_membership_available"] = (
        result["l2_code"].fillna("").astype(str).ne("")
        & result["l3_code"].fillna("").astype(str).ne("")
    ).astype(float)

    projection = market_projection.loc[
        market_projection["trade_date"].astype(str).eq(str(trade_date))
    ].copy()
    projection.rename(
        columns={
            "market_limit_hit_count": "v33_market_limit_hit_count",
            "market_limit_open_count": "v33_market_limit_open_count",
            "market_limit_down_count": "v33_market_limit_down_count",
            "market_limit_hit_last_10m": (
                "v33_market_limit_hit_last_10m"
            ),
            "market_limit_open_last_10m": (
                "v33_market_limit_open_last_10m"
            ),
            "market_net_sealed_count": "v33_market_net_sealed_count",
            "market_open_to_hit_ratio": (
                "v33_market_open_to_hit_ratio"
            ),
        },
        inplace=True,
    )
    result = result.merge(
        projection[["trade_date", "signal_slot", *MARKET_CURRENT_COLUMNS]],
        on=["trade_date", "signal_slot"],
        how="left",
        validate="many_to_one",
    )

    current_events = _attach_membership(
        current_stock_events,
        current_members,
    )
    previous_events = _attach_membership(
        previous_stock_events,
        previous_members,
    )
    slot_parts: list[pd.DataFrame] = []
    for slot, slot_candidates in result.groupby("signal_slot", sort=True):
        cutoff = pd.Timestamp(f"{trade_date} {slot}:00")
        lower = cutoff - pd.Timedelta(minutes=10)
        part = slot_candidates.copy()
        own = _event_state(current_events, cutoff=cutoff, lower=lower)
        own = own[
            [
                "ts_code",
                "event_limit_hit",
                "event_limit_open",
                "event_limit_down",
                "event_limit_hit_recent",
                "event_limit_open_recent",
            ]
        ]
        part = part.merge(
            own,
            on="ts_code",
            how="left",
            validate="many_to_one",
        )
        for column in (
            "event_limit_hit",
            "event_limit_open",
            "event_limit_down",
            "event_limit_hit_recent",
            "event_limit_open_recent",
        ):
            part[column] = part[column].fillna(False).astype(bool)
        state = _event_state(current_events, cutoff=cutoff, lower=lower)
        for level in LEVELS:
            part = _merge_current_level(
                part,
                state,
                current_members,
                level=level,
            )
        part.drop(
            columns=[
                "event_limit_hit",
                "event_limit_open",
                "event_limit_down",
                "event_limit_hit_recent",
                "event_limit_open_recent",
            ],
            inplace=True,
        )
        slot_parts.append(part)
    result = pd.concat(slot_parts, ignore_index=True)

    previous_state = _event_state(previous_events)
    result = _merge_previous_market(result, previous_state)
    for level in LEVELS:
        result = _merge_previous_level(
            result,
            previous_state,
            previous_members,
            level=level,
        )

    result["v33_ecology_active_before_signal"] = (
        result["v33_l2_limit_hit_count"].gt(0)
        | result["v33_l3_limit_hit_count"].gt(0)
        | result["v33_l2_limit_down_count"].gt(0)
        | result["v33_l3_limit_down_count"].gt(0)
    ).astype(float)
    for column in V33_LIMIT_ECOLOGY_FEATURE_COLUMNS:
        if column not in result:
            result[column] = 0.0
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).fillna(0.0)
    result.replace([np.inf, -np.inf], 0.0, inplace=True)
    result.sort_values(
        ["fold", *IDENTITY_COLUMNS],
        kind="stable",
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)
    return result[
        [
            *IDENTITY_COLUMNS,
            "fold",
            "l2_code",
            "l3_code",
            "v33_membership_available",
            "v33_ecology_active_before_signal",
            *V33_LIMIT_ECOLOGY_FEATURE_COLUMNS,
        ]
    ]


def audit_ecology_feature_coverage(
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    current_event_membership_coverage: float,
    previous_event_membership_coverage: float,
) -> dict[str, Any]:
    identity = list(IDENTITY_COLUMNS)
    left = candidates[identity].astype(str).sort_values(identity)
    right = features[identity].astype(str).sort_values(identity)
    identity_match = bool(
        len(left) == len(right)
        and left.reset_index(drop=True).equals(
            right.reset_index(drop=True)
        )
        and not features.duplicated(identity).any()
    )
    numeric = features[list(V33_LIMIT_ECOLOGY_FEATURE_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    numeric_complete = bool(np.isfinite(numeric.to_numpy()).all())
    membership_rate = float(features["v33_membership_available"].mean())
    l2_active = features["v33_l2_limit_hit_count"].gt(0)
    l3_active = features["v33_l3_limit_hit_count"].gt(0)
    target_dates = int(features["trade_date"].astype(str).nunique())
    l2_dates = int(features.loc[l2_active, "trade_date"].astype(str).nunique())
    l3_dates = int(features.loc[l3_active, "trade_date"].astype(str).nunique())
    forbidden_output = sorted(
        set(CURRENT_DAY_FORBIDDEN_COLUMNS).intersection(features.columns)
    )
    coverage_passed = bool(
        identity_match
        and numeric_complete
        and membership_rate >= 0.98
        and current_event_membership_coverage >= 0.90
        and previous_event_membership_coverage >= 0.90
        and float(l2_active.mean()) >= 0.20
        and float(l3_active.mean()) >= 0.10
        and l2_dates >= 6
        and l3_dates >= 4
        and not forbidden_output
    )
    return {
        "candidate_rows": int(len(candidates)),
        "feature_rows": int(len(features)),
        "identity_match": identity_match,
        "numeric_features_complete": numeric_complete,
        "candidate_membership_rate": membership_rate,
        "minimum_candidate_membership_rate": 0.98,
        "current_event_membership_coverage": (
            current_event_membership_coverage
        ),
        "previous_event_membership_coverage": (
            previous_event_membership_coverage
        ),
        "minimum_event_membership_coverage": 0.90,
        "l2_active_rows": int(l2_active.sum()),
        "l2_active_row_rate": float(l2_active.mean()),
        "minimum_l2_active_row_rate": 0.20,
        "l3_active_rows": int(l3_active.sum()),
        "l3_active_row_rate": float(l3_active.mean()),
        "minimum_l3_active_row_rate": 0.10,
        "target_dates": target_dates,
        "l2_active_dates": l2_dates,
        "minimum_l2_active_dates": 6,
        "l3_active_dates": l3_dates,
        "minimum_l3_active_dates": 4,
        "forbidden_output_columns": forbidden_output,
        "coverage_passed": coverage_passed,
    }


def event_membership_coverage(
    stock_events: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    trade_date: str,
) -> float:
    members = _active_member_frame(membership, trade_date)
    event_codes = stock_events.loc[
        stock_events[
            [
                "first_limit_touch",
                "first_limit_open",
                "first_limit_down",
            ]
        ]
        .notna()
        .any(axis=1),
        ["ts_code"],
    ].drop_duplicates()
    if event_codes.empty:
        return 1.0
    joined = event_codes.merge(
        members,
        on="ts_code",
        how="left",
        validate="one_to_one",
    )
    available = (
        joined["l2_code"].fillna("").astype(str).ne("")
        & joined["l3_code"].fillna("").astype(str).ne("")
    )
    return float(available.mean())


def build_projection(
    frames: list[pd.DataFrame],
    *,
    trade_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_causal_event_projection(frames, trade_date=trade_date)


def _active_member_frame(
    membership: pd.DataFrame,
    trade_date: str,
) -> pd.DataFrame:
    frame = active_membership(membership, trade_date=trade_date)[
        ["ts_code", "l2_code", "l3_code"]
    ].copy()
    frame["ts_code"] = frame["ts_code"].astype(str)
    return frame


def _attach_membership(
    stock_events: pd.DataFrame,
    members: pd.DataFrame,
) -> pd.DataFrame:
    event_columns = [
        "ts_code",
        "first_limit_touch",
        "first_limit_open",
        "first_limit_down",
    ]
    frame = stock_events.reindex(columns=event_columns).copy()
    frame["ts_code"] = frame["ts_code"].astype(str)
    return frame.merge(
        members,
        on="ts_code",
        how="left",
        validate="one_to_one",
    )


def _event_state(
    events: pd.DataFrame,
    *,
    cutoff: pd.Timestamp | None = None,
    lower: pd.Timestamp | None = None,
) -> pd.DataFrame:
    state = events.copy()
    if cutoff is None:
        state["event_limit_hit"] = state["first_limit_touch"].notna()
        state["event_limit_open"] = state["first_limit_open"].notna()
        state["event_limit_down"] = state["first_limit_down"].notna()
        state["event_limit_hit_recent"] = False
        state["event_limit_open_recent"] = False
    else:
        state["event_limit_hit"] = state["first_limit_touch"].le(cutoff)
        state["event_limit_open"] = state["first_limit_open"].le(cutoff)
        state["event_limit_down"] = state["first_limit_down"].le(cutoff)
        state["event_limit_hit_recent"] = state[
            "first_limit_touch"
        ].between(lower, cutoff, inclusive="right")
        state["event_limit_open_recent"] = state[
            "first_limit_open"
        ].between(lower, cutoff, inclusive="right")
    for column in (
        "event_limit_hit",
        "event_limit_open",
        "event_limit_down",
        "event_limit_hit_recent",
        "event_limit_open_recent",
    ):
        state[column] = state[column].fillna(False).astype(bool)
    return state


def _group_counts(
    state: pd.DataFrame,
    level_column: str,
) -> pd.DataFrame:
    valid = state.loc[
        state[level_column].fillna("").astype(str).ne("")
    ].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                level_column,
                "limit_hit_count",
                "limit_open_count",
                "limit_down_count",
                "limit_hit_last_10m",
                "limit_open_last_10m",
            ]
        )
    return (
        valid.groupby(level_column, sort=True)
        .agg(
            limit_hit_count=("event_limit_hit", "sum"),
            limit_open_count=("event_limit_open", "sum"),
            limit_down_count=("event_limit_down", "sum"),
            limit_hit_last_10m=("event_limit_hit_recent", "sum"),
            limit_open_last_10m=("event_limit_open_recent", "sum"),
        )
        .reset_index()
    )


def _member_counts(
    members: pd.DataFrame,
    level_column: str,
) -> pd.DataFrame:
    valid = members.loc[
        members[level_column].fillna("").astype(str).ne("")
    ]
    return (
        valid.groupby(level_column, sort=True)["ts_code"]
        .nunique()
        .rename("member_count")
        .reset_index()
    )


def _merge_current_level(
    candidates: pd.DataFrame,
    state: pd.DataFrame,
    members: pd.DataFrame,
    *,
    level: str,
) -> pd.DataFrame:
    code = f"{level}_code"
    counts = _group_counts(state, code)
    counts = counts.merge(
        _member_counts(members, code),
        on=code,
        how="outer",
        validate="one_to_one",
    )
    renamed = {
        column: f"v33_{level}_{column}"
        for column in counts.columns
        if column != code
    }
    counts.rename(columns=renamed, inplace=True)
    result = candidates.merge(
        counts,
        on=code,
        how="left",
        validate="many_to_one",
    )
    for suffix, own_column in (
        ("limit_hit_count", "event_limit_hit"),
        ("limit_open_count", "event_limit_open"),
        ("limit_down_count", "event_limit_down"),
        ("limit_hit_last_10m", "event_limit_hit_recent"),
        ("limit_open_last_10m", "event_limit_open_recent"),
    ):
        column = f"v33_{level}_{suffix}"
        result[column] = (
            pd.to_numeric(result[column], errors="coerce")
            .fillna(0.0)
            .sub(result[own_column].astype(float))
            .clip(lower=0.0)
        )
    member_column = f"v33_{level}_member_count"
    candidate_has_level = result[code].fillna("").astype(str).ne("")
    result[member_column] = (
        pd.to_numeric(result[member_column], errors="coerce")
        .fillna(0.0)
        .sub(candidate_has_level.astype(float))
        .clip(lower=0.0)
    )
    hit = result[f"v33_{level}_limit_hit_count"]
    opened = result[f"v33_{level}_limit_open_count"]
    down = result[f"v33_{level}_limit_down_count"]
    result[f"v33_{level}_net_sealed_count"] = (
        hit - opened
    ).clip(lower=0.0)
    result[f"v33_{level}_limit_hit_rate"] = _safe_ratio(
        hit,
        result[member_column],
    )
    result[f"v33_{level}_limit_down_rate"] = _safe_ratio(
        down,
        result[member_column],
    )
    result[f"v33_{level}_open_to_hit_ratio"] = _safe_ratio(
        opened,
        hit,
    )
    result[f"v33_{level}_market_hit_share"] = _safe_ratio(
        hit,
        result["v33_market_limit_hit_count"],
    )
    result[f"v33_{level}_active_before_signal"] = hit.gt(0).astype(float)
    return result


def _merge_previous_market(
    candidates: pd.DataFrame,
    previous_state: pd.DataFrame,
) -> pd.DataFrame:
    result = candidates.copy()
    hit = float(previous_state["event_limit_hit"].sum())
    opened = float(previous_state["event_limit_open"].sum())
    down = float(previous_state["event_limit_down"].sum())
    result["v33_prev_market_limit_hit_count"] = hit
    result["v33_prev_market_limit_open_count"] = opened
    result["v33_prev_market_limit_down_count"] = down
    result["v33_prev_market_net_sealed_count"] = max(hit - opened, 0.0)
    result["v33_prev_market_open_to_hit_ratio"] = (
        opened / hit if hit else 0.0
    )
    return result


def _merge_previous_level(
    candidates: pd.DataFrame,
    previous_state: pd.DataFrame,
    previous_members: pd.DataFrame,
    *,
    level: str,
) -> pd.DataFrame:
    code = f"{level}_code"
    counts = _group_counts(previous_state, code)
    counts = counts.merge(
        _member_counts(previous_members, code),
        on=code,
        how="outer",
        validate="one_to_one",
    )
    keep = [
        code,
        "member_count",
        "limit_hit_count",
        "limit_open_count",
        "limit_down_count",
    ]
    counts = counts.reindex(columns=keep)
    counts.rename(
        columns={
            column: f"v33_prev_{level}_{column}"
            for column in counts.columns
            if column != code
        },
        inplace=True,
    )
    result = candidates.merge(
        counts,
        on=code,
        how="left",
        validate="many_to_one",
    )
    previous_candidate_group = previous_members[
        ["ts_code", code]
    ].rename(columns={code: f"_v33_prev_member_{code}"})
    result = result.merge(
        previous_candidate_group,
        on="ts_code",
        how="left",
        validate="many_to_one",
    )
    own = previous_state[
        [
            "ts_code",
            code,
            "event_limit_hit",
            "event_limit_open",
            "event_limit_down",
        ]
    ].rename(
        columns={
            code: f"_v33_prev_{code}",
            "event_limit_hit": "_v33_prev_own_hit",
            "event_limit_open": "_v33_prev_own_open",
            "event_limit_down": "_v33_prev_own_down",
        }
    )
    result = result.merge(
        own,
        on="ts_code",
        how="left",
        validate="many_to_one",
    )
    same_group = result[f"_v33_prev_{code}"].eq(result[code])
    for suffix, own_column in (
        ("limit_hit_count", "_v33_prev_own_hit"),
        ("limit_open_count", "_v33_prev_own_open"),
        ("limit_down_count", "_v33_prev_own_down"),
    ):
        column = f"v33_prev_{level}_{suffix}"
        own_value = (
            result[own_column].fillna(False).astype(bool) & same_group
        ).astype(float)
        result[column] = (
            pd.to_numeric(result[column], errors="coerce")
            .fillna(0.0)
            .sub(own_value)
            .clip(lower=0.0)
        )
    member_column = f"v33_prev_{level}_member_count"
    result[member_column] = pd.to_numeric(
        result[member_column],
        errors="coerce",
    ).fillna(0.0).sub(
        result[f"_v33_prev_member_{code}"]
        .eq(result[code])
        .astype(float)
    ).clip(lower=0.0)
    hit = result[f"v33_prev_{level}_limit_hit_count"]
    opened = result[f"v33_prev_{level}_limit_open_count"]
    result[f"v33_prev_{level}_net_sealed_count"] = (
        hit - opened
    ).clip(lower=0.0)
    result[f"v33_prev_{level}_limit_hit_rate"] = _safe_ratio(
        hit,
        result[member_column],
    )
    result[f"v33_prev_{level}_open_to_hit_ratio"] = _safe_ratio(
        opened,
        hit,
    )
    result[f"v33_prev_{level}_active"] = hit.gt(0).astype(float)
    result.drop(
        columns=[
            f"_v33_prev_{code}",
            f"_v33_prev_member_{code}",
            "_v33_prev_own_hit",
            "_v33_prev_own_open",
            "_v33_prev_own_down",
        ],
        inplace=True,
    )
    return result


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    left = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    right = pd.to_numeric(denominator, errors="coerce").fillna(0.0)
    return pd.Series(
        np.where(right.gt(0), left / right, 0.0),
        index=left.index,
        dtype=float,
    )


def _supplied_time(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return ~normalized.isin({"", "0", "none", "nan", "nat", "--"})
