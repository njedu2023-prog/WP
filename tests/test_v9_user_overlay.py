from __future__ import annotations

import pandas as pd

from wp.v3.overlay import (
    attach_previous_limit_flags,
    build_limit_up_flags,
    overlay_mask,
    top_n_per_day,
)


def test_overlay_excludes_exact_previous_close_limit_and_requires_gt_seven():
    daily = pd.DataFrame(
        [
            {
                "trade_date": "20260720",
                "ts_code": "600001.SH",
                "close": 11.0,
                "high": 11.0,
            },
            {
                "trade_date": "20260720",
                "ts_code": "600002.SH",
                "close": 10.9,
                "high": 11.0,
            },
            {
                "trade_date": "20260720",
                "ts_code": "600003.SH",
                "close": 10.8,
                "high": 10.9,
            },
            {
                "trade_date": "20260720",
                "ts_code": "600004.SH",
                "close": 10.9,
                "high": 11.0,
            },
        ]
    )
    limits = pd.DataFrame(
        [
            {
                "trade_date": "20260720",
                "ts_code": code,
                "up_limit": 11.0,
            }
            for code in (
                "600001.SH",
                "600002.SH",
                "600003.SH",
                "600004.SH",
            )
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "trade_date": "20260721",
                "signal_slot": "14:20",
                "ts_code": "600001.SH",
                "ret_from_prev_close_pct": 8.0,
                "execution_eligible": True,
            },
            {
                "trade_date": "20260721",
                "signal_slot": "14:20",
                "ts_code": "600002.SH",
                "ret_from_prev_close_pct": 7.0,
                "execution_eligible": True,
            },
            {
                "trade_date": "20260721",
                "signal_slot": "14:20",
                "ts_code": "600003.SH",
                "ret_from_prev_close_pct": 7.01,
                "execution_eligible": True,
            },
            {
                "trade_date": "20260721",
                "signal_slot": "14:20",
                "ts_code": "600004.SH",
                "ret_from_prev_close_pct": 8.0,
                "execution_eligible": True,
            },
        ]
    )
    attached = attach_previous_limit_flags(
        predictions,
        build_limit_up_flags(daily, limits),
        trade_dates=["20260720", "20260721"],
    )
    primary = overlay_mask(attached, previous_limit_mode="closed")
    strict = overlay_mask(attached, previous_limit_mode="touched")
    assert attached.loc[0, "previous_day_closed_up_limit"]
    assert not attached.loc[1, "previous_day_closed_up_limit"]
    assert attached.loc[1, "previous_day_touched_up_limit"]
    assert primary.tolist() == [False, False, True, True]
    assert strict.tolist() == [False, False, True, False]


def test_top_n_per_day_is_deterministic():
    frame = pd.DataFrame(
        [
            {"trade_date": "20260721", "ts_code": "600002.SH", "score": 0.8},
            {"trade_date": "20260721", "ts_code": "600001.SH", "score": 0.8},
            {"trade_date": "20260721", "ts_code": "600003.SH", "score": 0.7},
            {"trade_date": "20260722", "ts_code": "600004.SH", "score": 0.6},
        ]
    )
    selected = top_n_per_day(frame, score_column="score", count=1)
    assert selected["ts_code"].tolist() == ["600001.SH", "600004.SH"]
