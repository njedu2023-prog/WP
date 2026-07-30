from __future__ import annotations

import pandas as pd

from wp.v3.v16_funnel import build_funnel, rejection_summary


def row(
    code: str,
    *,
    slot: str = "14:20",
    probability: float = 0.60,
    expected: float = 0.20,
    severe: float = 0.10,
    fill: float = 0.99,
    rank: float = 0.99,
    risk: float = 0.20,
    score: float = 1.0,
) -> dict[str, object]:
    return {
        "trade_date": "20260720",
        "signal_slot": slot,
        "ts_code": code,
        "meta_p_positive": probability,
        "meta_expected_net_return_pct": expected,
        "meta_p_severe_loss": severe,
        "p_round_trip_fill_lower": fill,
        "meta_rank_pct": rank,
        "risk_failure_rank_pct": risk,
        "meta_score": score,
    }


def test_funnel_attributes_first_failed_gate() -> None:
    frame = pd.DataFrame(
        [
            row("PASS", score=4.0),
            row("LATE", slot="14:40", score=3.0),
            row("PROB", probability=0.53, score=2.0),
            row("RISK", risk=0.60, score=1.0),
        ]
    )

    summary, attributed = build_funnel(
        frame,
        max_candidates_per_day=3,
    )
    stages = attributed.set_index("ts_code")["first_rejection_stage"].to_dict()

    assert stages == {
        "PASS": "passed_thresholds",
        "LATE": "early_slot",
        "PROB": "positive_probability",
        "RISK": "exit_failure_safe_half",
    }
    assert int(summary.iloc[-1]["rows_remaining"]) == 1


def test_funnel_deduplicates_symbol_before_daily_top_k() -> None:
    frame = pd.DataFrame(
        [
            row("A", slot="14:20", score=5.0),
            row("A", slot="14:25", score=4.0),
            row("B", slot="14:20", score=3.0),
            row("C", slot="14:20", score=2.0),
            row("D", slot="14:20", score=1.0),
        ]
    )

    _, attributed = build_funnel(frame, max_candidates_per_day=3)
    selected = attributed.loc[attributed["funnel_selected"], "ts_code"].tolist()

    assert selected == ["A", "B", "C"]
    assert (
        attributed.loc[
            (attributed["ts_code"].eq("A"))
            & (attributed["signal_slot"].eq("14:25")),
            "first_rejection_stage",
        ].item()
        == "daily_dedup_top_k"
    )


def test_rejection_summary_counts_rows_and_days() -> None:
    frame = pd.DataFrame(
        [
            row("PASS"),
            row("LATE", slot="14:40"),
        ]
    )
    _, attributed = build_funnel(frame)

    summary = rejection_summary(attributed).set_index(
        "first_rejection_stage"
    )

    assert int(summary.loc["passed_thresholds", "rows"]) == 1
    assert int(summary.loc["early_slot", "rows"]) == 1
