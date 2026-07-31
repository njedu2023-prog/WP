from __future__ import annotations

from pathlib import Path

import pandas as pd

from wp.v3.v31_public_event import (
    SOURCE_SPECS,
    audit_event_frame,
    build_candidate_event_presence,
    build_lookback_map,
    causal_dates_valid,
    normalize_event_frame,
)


def source_frame(
    source: str,
    *,
    ts_code: str = "000001.SZ",
    event_date: str = "20260722",
) -> pd.DataFrame:
    spec = SOURCE_SPECS[source]
    values = {column: None for column in spec["fields"].split(",")}
    values["ts_code"] = ts_code
    values[spec["date_column"]] = event_date
    return pd.DataFrame([values])


def test_lookback_map_excludes_target_date() -> None:
    dates = [
        "20260716",
        "20260717",
        "20260720",
        "20260721",
        "20260722",
        "20260723",
    ]
    mapping = build_lookback_map(
        dates,
        ("20260723",),
        lookback=5,
    )

    assert mapping["20260723"] == dates[:5]
    assert "20260723" not in mapping["20260723"]


def test_event_audit_accepts_valid_and_empty_responses() -> None:
    source = "repurchase"
    valid = audit_event_frame(
        source_frame(source),
        source=source,
        requested_date="20260722",
    )
    empty = audit_event_frame(
        pd.DataFrame(columns=SOURCE_SPECS[source]["fields"].split(",")),
        source=source,
        requested_date="20260722",
    )

    assert valid["coverage_pass"]
    assert empty["coverage_pass"]


def test_event_audit_rejects_wrong_date_and_code() -> None:
    source = "block_trade"
    wrong_date = audit_event_frame(
        source_frame(source, event_date="20260723"),
        source=source,
        requested_date="20260722",
    )
    wrong_code = audit_event_frame(
        source_frame(source, ts_code="BAD"),
        source=source,
        requested_date="20260722",
    )

    assert not wrong_date["coverage_pass"]
    assert not wrong_code["coverage_pass"]


def test_candidate_presence_uses_only_frozen_lookback() -> None:
    candidates = pd.DataFrame(
        {
            "trade_date": ["20260723", "20260723"],
            "ts_code": ["000001.SZ", "000002.SZ"],
        }
    )
    prior = normalize_event_frame(
        source_frame("forecast", ts_code="000001.SZ"),
        source="forecast",
    )
    same_day = normalize_event_frame(
        source_frame(
            "forecast",
            ts_code="000002.SZ",
            event_date="20260723",
        ),
        source="forecast",
    )
    events = {
        source: pd.DataFrame(
            columns=[
                *spec["fields"].split(","),
                "event_date",
                "event_source",
            ]
        )
        for source, spec in SOURCE_SPECS.items()
    }
    events["forecast"] = pd.concat(
        [prior, same_day],
        ignore_index=True,
    )
    lookback = {"20260723": ["20260722"]}
    result = build_candidate_event_presence(
        candidates,
        events,
        lookback,
    ).set_index("ts_code")

    assert result.loc["000001.SZ", "event_forecast"]
    assert not result.loc["000002.SZ", "event_forecast"]
    assert not causal_dates_valid(events, lookback)


def test_probe_source_does_not_read_profit_outcomes() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / "scripts" / "probe_wp_v31_public_event_data.py",
        root / "src" / "wp" / "v3" / "v31_public_event.py",
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

