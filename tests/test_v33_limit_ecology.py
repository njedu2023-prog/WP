from __future__ import annotations

import pandas as pd

from wp.v3.v28_industry_peer import normalize_membership
from wp.v3.v30_limit_event import KPL_FIELDS, normalize_kpl_frame
from wp.v3.v33_limit_ecology import (
    V33_LIMIT_ECOLOGY_FEATURE_COLUMNS,
    audit_decision_tape_frame,
    audit_ecology_feature_coverage,
    build_date_candidate_ecology,
    build_projection,
)


def _raw_event(
    *,
    code: str,
    tag: str,
    lu_time: str = "",
    open_time: str = "",
    ld_time: str = "",
) -> pd.DataFrame:
    row = {column: "" for column in KPL_FIELDS.split(",")}
    row.update(
        {
            "ts_code": code,
            "name": "sample",
            "trade_date": "20260723",
            "tag": tag,
            "lu_time": lu_time,
            "open_time": open_time,
            "ld_time": ld_time,
            "last_time": "150004",
        }
    )
    return pd.DataFrame([row], columns=KPL_FIELDS.split(","))


def _normalized(
    *,
    code: str,
    tag: str,
    lu_time: str = "",
    open_time: str = "",
    ld_time: str = "",
    trade_date: str = "20260723",
) -> pd.DataFrame:
    raw = _raw_event(
        code=code,
        tag=tag,
        lu_time=lu_time,
        open_time=open_time,
        ld_time=ld_time,
    )
    raw["trade_date"] = trade_date
    return normalize_kpl_frame(
        raw,
        trade_date=trade_date,
        requested_tag=tag,
    )


def _membership() -> pd.DataFrame:
    rows = [
        {
            "l1_code": "L1",
            "l1_name": "l1",
            "l2_code": "L2A",
            "l2_name": "l2a",
            "l3_code": "L3A",
            "l3_name": "l3a",
            "ts_code": "000001.SZ",
            "name": "candidate",
            "in_date": "20200101",
            "out_date": "",
            "is_new": "N",
        },
        {
            "l1_code": "L1",
            "l1_name": "l1",
            "l2_code": "L2A",
            "l2_name": "l2a",
            "l3_code": "L3A",
            "l3_name": "l3a",
            "ts_code": "000002.SZ",
            "name": "peer",
            "in_date": "20200101",
            "out_date": "",
            "is_new": "N",
        },
        {
            "l1_code": "L1",
            "l1_name": "l1",
            "l2_code": "L2A",
            "l2_name": "l2a",
            "l3_code": "L3B",
            "l3_name": "l3b",
            "ts_code": "000003.SZ",
            "name": "peer2",
            "in_date": "20200101",
            "out_date": "",
            "is_new": "N",
        },
    ]
    return normalize_membership(pd.DataFrame(rows))


def test_v33_audit_ignores_unused_last_time_after_close() -> None:
    record = audit_decision_tape_frame(
        _raw_event(
            code="000002.SZ",
            tag="涨停",
            lu_time="142000",
        ),
        trade_date="20260723",
        requested_tag="涨停",
    )

    assert record["coverage_pass"]
    assert not record["post_1450_events_used_for_current_features"]


def test_v33_audit_rejects_unparseable_required_time() -> None:
    record = audit_decision_tape_frame(
        _raw_event(
            code="000002.SZ",
            tag="涨停",
            lu_time="bad-time",
        ),
        trade_date="20260723",
        requested_tag="涨停",
    )

    assert not record["coverage_pass"]


def test_v33_ecology_excludes_candidate_and_respects_signal_time() -> None:
    current_frames = [
        _normalized(
            code="000001.SZ",
            tag="涨停",
            lu_time="141000",
        ),
        _normalized(
            code="000002.SZ",
            tag="炸板",
            lu_time="142500",
            open_time="144500",
        ),
        _normalized(
            code="000003.SZ",
            tag="跌停",
            ld_time="144000",
        ),
    ]
    projection, current_stocks = build_projection(
        current_frames,
        trade_date="20260723",
    )
    previous_frames = [
        _normalized(
            code="000002.SZ",
            tag="涨停",
            lu_time="143000",
            trade_date="20260722",
        )
    ]
    _, previous_stocks = build_projection(
        previous_frames,
        trade_date="20260722",
    )
    candidates = pd.DataFrame(
        {
            "trade_date": ["20260723", "20260723"],
            "signal_slot": ["14:20", "14:50"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "fold": [1, 1],
        }
    )

    result = build_date_candidate_ecology(
        candidates,
        trade_date="20260723",
        previous_trade_date="20260722",
        current_stock_events=current_stocks,
        previous_stock_events=previous_stocks,
        market_projection=projection,
        membership=_membership(),
    ).sort_values("signal_slot")

    early = result.iloc[0]
    late = result.iloc[1]
    assert early["v33_l2_limit_hit_count"] == 0
    assert late["v33_l2_limit_hit_count"] == 1
    assert late["v33_l2_limit_open_count"] == 1
    assert late["v33_l2_limit_down_count"] == 1
    assert late["v33_l3_limit_down_count"] == 0
    assert late["v33_prev_l2_limit_hit_count"] == 1
    assert late["v33_prev_l3_limit_hit_count"] == 1
    assert set(V33_LIMIT_ECOLOGY_FEATURE_COLUMNS).issubset(result.columns)


def test_v33_feature_audit_passes_covered_synthetic_probe() -> None:
    candidates = pd.DataFrame(
        {
            "trade_date": ["20260723"] * 10,
            "signal_slot": ["14:20"] * 10,
            "ts_code": [f"{index:06d}.SZ" for index in range(10)],
        }
    )
    features = candidates.copy()
    features["v33_membership_available"] = 1.0
    for column in V33_LIMIT_ECOLOGY_FEATURE_COLUMNS:
        features[column] = 0.0
    features.loc[:5, "v33_l2_limit_hit_count"] = 1.0
    features.loc[:3, "v33_l3_limit_hit_count"] = 1.0

    audit = audit_ecology_feature_coverage(
        features,
        candidates,
        current_event_membership_coverage=1.0,
        previous_event_membership_coverage=1.0,
    )

    assert not audit["coverage_passed"]
    assert audit["l2_active_row_rate"] == 0.6
    assert audit["l3_active_row_rate"] == 0.4
