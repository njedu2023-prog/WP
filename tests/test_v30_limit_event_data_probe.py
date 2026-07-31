from __future__ import annotations

from pathlib import Path

import pandas as pd

from wp.v3.v30_limit_event import (
    CURRENT_DAY_FORBIDDEN_COLUMNS,
    KPL_FIELDS,
    SIGNAL_SLOTS,
    attach_candidate_event_state,
    audit_kpl_frame,
    build_causal_event_projection,
    normalize_kpl_frame,
    parse_event_time,
)


def event_frame(
    *,
    tag: str,
    code: str = "000001.SZ",
    lu_time: str = "142000",
    open_time: str = "",
    ld_time: str = "",
) -> pd.DataFrame:
    row = {
        "ts_code": code,
        "name": "Sample",
        "trade_date": "20260723",
        "lu_time": lu_time,
        "ld_time": ld_time,
        "open_time": open_time,
        "last_time": "145000" if lu_time else "",
        "lu_desc": "reason",
        "tag": tag,
        "theme": "theme",
        "net_change": 1.0,
        "bid_amount": 1.0,
        "status": "first",
        "bid_change": 1.0,
        "bid_turnover": 1.0,
        "lu_bid_vol": 1.0,
        "pct_chg": 10.0,
        "bid_pct_chg": 1.0,
        "rt_pct_chg": 10.0,
        "limit_order": 1.0,
        "amount": 1.0,
        "turnover_rate": 1.0,
        "free_float": 1.0,
        "lu_limit_order": 1.0,
    }
    return pd.DataFrame([row], columns=KPL_FIELDS.split(","))


def test_parse_event_time_accepts_documented_formats() -> None:
    expected = pd.Timestamp("2026-07-23 14:20:00")
    assert parse_event_time("142000", "20260723") == expected
    assert parse_event_time("14:20:00", "20260723") == expected
    assert parse_event_time("20260723142000", "20260723") == expected
    assert pd.isna(parse_event_time("", "20260723"))


def test_probe_audit_accepts_valid_touch_and_rejects_future_date() -> None:
    valid = audit_kpl_frame(
        event_frame(tag="涨停"),
        trade_date="20260723",
        requested_tag="涨停",
    )
    wrong_date = event_frame(tag="涨停")
    wrong_date["trade_date"] = "20260724"
    invalid = audit_kpl_frame(
        wrong_date,
        trade_date="20260723",
        requested_tag="涨停",
    )

    assert valid["coverage_pass"]
    assert not invalid["coverage_pass"]


def test_probe_audit_rejects_open_before_first_touch() -> None:
    record = audit_kpl_frame(
        event_frame(
            tag="炸板",
            lu_time="143000",
            open_time="142500",
        ),
        trade_date="20260723",
        requested_tag="炸板",
    )

    assert not record["coverage_pass"]
    assert not record["open_not_before_first_touch"]


def test_probe_audit_accepts_valid_zero_event_category() -> None:
    empty = pd.DataFrame(columns=KPL_FIELDS.split(","))
    record = audit_kpl_frame(
        empty,
        trade_date="20260723",
        requested_tag="跌停",
    )

    assert record["coverage_pass"]
    assert record["rows"] == 0


def test_causal_projection_does_not_reveal_future_open() -> None:
    failed = normalize_kpl_frame(
        event_frame(
            tag="炸板",
            lu_time="142000",
            open_time="144500",
        ),
        trade_date="20260723",
        requested_tag="炸板",
    )
    down = normalize_kpl_frame(
        event_frame(
            tag="跌停",
            code="000002.SZ",
            lu_time="",
            ld_time="144000",
        ),
        trade_date="20260723",
        requested_tag="跌停",
    )
    projection, _ = build_causal_event_projection(
        [failed, down],
        trade_date="20260723",
    )
    at_1430 = projection.loc[projection["signal_slot"].eq("14:30")].iloc[0]
    at_1450 = projection.loc[projection["signal_slot"].eq("14:50")].iloc[0]

    assert at_1430["market_limit_hit_count"] == 1
    assert at_1430["market_limit_open_count"] == 0
    assert at_1430["market_limit_down_count"] == 0
    assert at_1450["market_limit_open_count"] == 1
    assert at_1450["market_limit_down_count"] == 1


def test_causal_projection_unions_final_categories_without_exposing_them() -> None:
    up = normalize_kpl_frame(
        event_frame(tag="涨停", lu_time="142000"),
        trade_date="20260723",
        requested_tag="涨停",
    )
    failed = normalize_kpl_frame(
        event_frame(
            tag="炸板",
            lu_time="142000",
            open_time="144500",
        ),
        trade_date="20260723",
        requested_tag="炸板",
    )
    projection, stocks = build_causal_event_projection(
        [up, failed],
        trade_date="20260723",
    )

    assert len(stocks) == 1
    assert projection["market_limit_hit_count"].max() == 1
    assert not any(
        column in projection.columns
        for column in CURRENT_DAY_FORBIDDEN_COLUMNS
    )


def test_candidate_projection_uses_only_events_before_each_signal() -> None:
    failed = normalize_kpl_frame(
        event_frame(
            tag="炸板",
            lu_time="142500",
            open_time="144500",
        ),
        trade_date="20260723",
        requested_tag="炸板",
    )
    projection, stocks = build_causal_event_projection(
        [failed],
        trade_date="20260723",
    )
    candidates = pd.DataFrame(
        {
            "trade_date": ["20260723", "20260723"],
            "signal_slot": ["14:20", "14:50"],
            "ts_code": ["000001.SZ", "000001.SZ"],
        }
    )
    result = attach_candidate_event_state(
        candidates,
        stocks,
        projection,
    ).sort_values("signal_slot")

    assert not result.iloc[0]["candidate_limit_hit_before_signal"]
    assert result.iloc[1]["candidate_limit_hit_before_signal"]
    assert result.iloc[1]["candidate_limit_open_before_signal"]
    assert list(projection["signal_slot"]) == list(SIGNAL_SLOTS)


def test_probe_source_does_not_read_return_outcomes() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / "scripts" / "probe_wp_v30_limit_event_data.py",
        root / "src" / "wp" / "v3" / "v30_limit_event.py",
    ]
    forbidden = (
        "gross_return",
        "net_return",
        "target_return",
        "outcome_label",
        "t1_close",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)

    assert not any(token in text for token in forbidden)
