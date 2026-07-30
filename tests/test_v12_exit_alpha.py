from __future__ import annotations

from scripts.research_wp_v12_exit_alpha import (
    META_CALIBRATION_DAYS,
    META_TRAIN_DAYS,
    POLICY_CONFIRMATION_DAYS,
    POLICY_DESIGN_DAYS,
    PURGE_DAYS,
    research_readiness,
    rolling_segments,
)


def test_v12_rolling_segments_have_four_explicit_purges() -> None:
    needed = (
        META_TRAIN_DAYS
        + META_CALIBRATION_DAYS
        + POLICY_DESIGN_DAYS
        + POLICY_CONFIRMATION_DAYS
        + 4 * PURGE_DAYS
    )
    dates = [f"{index:08d}" for index in range(1, needed + 1)]
    train, calibration, design, confirmation = rolling_segments(dates)  # type: ignore[misc]
    assert len(train) == META_TRAIN_DAYS
    assert len(calibration) == META_CALIBRATION_DAYS
    assert len(design) == POLICY_DESIGN_DAYS
    assert len(confirmation) == POLICY_CONFIRMATION_DAYS
    assert int(calibration[0]) - int(train[-1]) == PURGE_DAYS + 1
    assert int(design[0]) - int(calibration[-1]) == PURGE_DAYS + 1
    assert int(confirmation[0]) - int(design[-1]) == PURGE_DAYS + 1
    assert int(dates[-1]) - int(confirmation[-1]) == PURGE_DAYS


def test_v12_readiness_requires_tail_and_50bp_stress() -> None:
    metrics = {
        "events": 300,
        "trade_days": 80,
        "win_rate": 0.60,
        "win_rate_day_clustered_lower": 0.55,
        "mean_net_return_pct": 0.40,
        "mean_net_return_day_clustered_lower_pct": 0.10,
        "profit_factor": 1.4,
        "entry_fill_rate": 0.99,
        "exit_fill_rate_given_entry": 0.99,
        "net_return_q10_pct": -3.5,
        "stress": {"50bps": {"positive_total_return": True}},
    }
    readiness = research_readiness(metrics)
    assert readiness["all_oos_gates_passed"] is False
    assert readiness["failed_gates"] == ["tail_q10_above_minus_3pct"]
