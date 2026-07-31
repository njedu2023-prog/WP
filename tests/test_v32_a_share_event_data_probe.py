from __future__ import annotations

from pathlib import Path

import pandas as pd

from wp.v3.v32_public_event import (
    SOURCE_SPECS,
    audit_a_share_event_frame,
    normalize_a_share_event_frame,
)


def source_frame(
    source: str,
    *,
    codes: tuple[str, ...],
    event_date: str = "20260722",
) -> pd.DataFrame:
    spec = SOURCE_SPECS[source]
    rows = []
    for code in codes:
        values = {
            column: None for column in spec["fields"].split(",")
        }
        values["ts_code"] = code
        values[spec["date_column"]] = event_date
        rows.append(values)
    return pd.DataFrame(rows)


def test_mixed_security_universe_is_filtered_not_rejected() -> None:
    source = "block_trade"
    frame = source_frame(
        source,
        codes=("000001.SZ", "900901.SH", "BAD"),
    )

    audit = audit_a_share_event_frame(
        frame,
        source=source,
        requested_date="20260722",
    )
    normalized = normalize_a_share_event_frame(frame, source=source)

    assert audit["coverage_pass"]
    assert audit["raw_rows"] == 3
    assert audit["rows"] == 1
    assert audit["excluded_non_a_share_rows"] == 2
    assert set(normalized["ts_code"]) == {"000001.SZ"}


def test_all_non_a_share_rows_are_valid_empty_observation() -> None:
    source = "repurchase"
    frame = source_frame(source, codes=("HK0001", "BAD"))

    audit = audit_a_share_event_frame(
        frame,
        source=source,
        requested_date="20260722",
    )
    normalized = normalize_a_share_event_frame(frame, source=source)

    assert audit["coverage_pass"]
    assert audit["rows"] == 0
    assert audit["excluded_non_a_share_rows"] == 2
    assert normalized.empty


def test_wrong_raw_date_still_fails_after_universe_filter() -> None:
    source = "share_float"
    frame = source_frame(
        source,
        codes=("000001.SZ", "BAD"),
        event_date="20260723",
    )

    audit = audit_a_share_event_frame(
        frame,
        source=source,
        requested_date="20260722",
    )

    assert not audit["coverage_pass"]
    assert not audit["date_ok"]


def test_retained_a_share_duplicates_fail() -> None:
    source = "forecast"
    frame = source_frame(
        source,
        codes=("000001.SZ", "000001.SZ"),
    )

    audit = audit_a_share_event_frame(
        frame,
        source=source,
        requested_date="20260722",
    )

    assert not audit["coverage_pass"]
    assert audit["exact_duplicates"]


def test_probe_source_does_not_read_profit_outcomes() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / "scripts" / "probe_wp_v32_a_share_event_data.py",
        root / "src" / "wp" / "v3" / "v32_public_event.py",
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
