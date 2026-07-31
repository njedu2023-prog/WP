from __future__ import annotations

from pathlib import Path

import pandas as pd

from probe_wp_v33_limit_industry_ecology import (
    audit_v30_projection_parity,
    membership_coverage,
    previous_trade_dates,
)
from wp.v3.v28_industry_peer import normalize_membership


def test_previous_trade_dates_uses_a_share_calendar() -> None:
    mapping = previous_trade_dates(
        ["20260720", "20260721", "20260722", "20260723"],
        ("20260721", "20260723"),
    )

    assert mapping == {
        "20260721": "20260720",
        "20260723": "20260722",
    }


def test_membership_coverage_is_event_code_weighted() -> None:
    membership = normalize_membership(
        pd.DataFrame(
            [
                {
                    "l1_code": "L1",
                    "l2_code": "L2",
                    "l3_code": "L3",
                    "ts_code": "000001.SZ",
                    "in_date": "20200101",
                    "out_date": "",
                }
            ]
        )
    )
    events = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "first_limit_touch": [
                pd.Timestamp("2026-07-23 14:20"),
                pd.Timestamp("2026-07-23 14:25"),
            ],
            "first_limit_open": [pd.NaT, pd.NaT],
            "first_limit_down": [pd.NaT, pd.NaT],
        }
    )

    result = membership_coverage({"20260723": events}, membership)

    assert result["event_codes"] == 2
    assert result["covered_codes"] == 1
    assert result["coverage"] == 0.5


def test_v30_projection_parity_detects_value_drift(monkeypatch) -> None:
    monkeypatch.setattr(
        "probe_wp_v33_limit_industry_ecology.PROBE_DATES",
        ("20260723",),
    )
    expected = pd.DataFrame(
        {
            "trade_date": ["20260723"],
            "signal_slot": ["14:20"],
            "market_limit_hit_count": [10],
        }
    )
    actual = expected.copy()
    passed = audit_v30_projection_parity(
        {"20260723": actual},
        expected,
    )
    drifted = actual.copy()
    drifted["market_limit_hit_count"] = 11
    failed = audit_v30_projection_parity(
        {"20260723": drifted},
        expected,
    )

    assert passed["passed"]
    assert not failed["passed"]
    assert failed["mismatch_rows"] == 1


def test_probe_source_does_not_read_profit_outcomes() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "scripts" / "probe_wp_v33_limit_industry_ecology.py",
        root / "src" / "wp" / "v3" / "v33_limit_ecology.py",
    ]
    forbidden = (
        "gross_return",
        "net_return",
        "target_return",
        "outcome_label",
        "t1_close",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert not any(token in text for token in forbidden)
