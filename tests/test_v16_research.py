from __future__ import annotations

import pandas as pd

from wp.v3.v16_research import (
    materialize_close_labels,
    pareto_mask,
    research_readiness,
    rolling_model_segments,
    rolling_policy_segments,
)


def dates(count: int) -> list[str]:
    return [f"{index:08d}" for index in range(count)]


def test_model_segments_keep_purges_out_of_train_and_calibration() -> None:
    values = dates(600)

    train, calibration = rolling_model_segments(values) or ([], [])

    assert len(train) == 504
    assert len(calibration) == 42
    assert int(calibration[0]) - int(train[-1]) == 3
    assert int(values[-1]) - int(calibration[-1]) == 2


def test_policy_segments_are_design_then_purged_confirmation() -> None:
    values = dates(130)

    design, confirmation = rolling_policy_segments(values) or ([], [])

    assert len(design) == 84
    assert len(confirmation) == 42
    assert int(confirmation[0]) - int(design[-1]) == 3
    assert int(values[-1]) - int(confirmation[-1]) == 2


def test_close_labels_preserve_missing_truth() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20260720", "20260720", "20260720"],
            "signal_slot": ["14:20", "14:25", "14:30"],
            "ts_code": ["A", "B", "C"],
            "entry_fillable": [True, True, True],
            "net_t1_close_auction_pct": [1.0, -2.5, None],
            "exit_t1_close_auction_fillable": [True, True, False],
        }
    )

    result = materialize_close_labels(
        frame,
        severe_loss_threshold_pct=-2.0,
    )

    assert result["label_available"].tolist() == [True, True, False]
    assert result["target_net_positive"].iloc[:2].tolist() == [True, False]
    assert result["target_severe_loss"].iloc[:2].tolist() == [False, True]
    assert pd.isna(result["target_net_positive"].iloc[2])


def test_pareto_mask_keeps_frequency_return_tradeoffs() -> None:
    frame = pd.DataFrame(
        {
            "candidate_day_rate": [0.10, 0.20, 0.15, 0.05],
            "mean_net_return_pct": [0.50, 0.30, 0.40, 0.10],
            "win_rate": [0.60, 0.55, 0.58, 0.50],
        }
    )

    assert pareto_mask(frame).tolist() == [True, True, True, False]


def test_readiness_requires_clustered_win_rate_lower_bound() -> None:
    metrics = {
        "events": 300,
        "candidate_days": 60,
        "win_rate": 0.60,
        "win_rate_wilson_lower": 0.55,
        "clustered_win_rate_lower": 0.49,
        "mean_net_return_pct": 0.80,
        "clustered_mean_lower_pct": 0.20,
        "profit_factor": 1.50,
        "stress_50bps_mean_net_return_pct": 0.30,
        "return_p10_pct": -2.0,
    }

    readiness = research_readiness(metrics, temporal_integrity=True)

    assert not readiness["all_historical_gates_passed"]
    assert "minimum_clustered_win_rate_lower" in readiness["failed_gates"]
