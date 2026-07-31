from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wp.v3.v19_recall import build_recall_frontier
from wp.v3.v21_margin import (
    FrozenMarginPolicy,
    MarginPolicySpec,
    apply_margin_policy,
    build_margin_leaders,
    calibrate_margin_policy,
    fit_margin_gate,
    rolling_margin_model_segments,
    v21_research_readiness,
)


def test_margin_leaders_do_not_use_future_truth() -> None:
    source = _source_frame(days=3, stocks=8, seed=21)
    first = build_margin_leaders(source, leaders_per_slot=3)
    changed = source.copy()
    changed["target_net_positive"] = 1 - changed["target_net_positive"]
    changed["net_return_pct"] = -changed["net_return_pct"] + 100.0
    second = build_margin_leaders(changed, leaders_per_slot=3)

    identity = ["trade_date", "signal_slot", "ts_code"]
    pd.testing.assert_frame_equal(first[identity], second[identity])
    assert first.groupby(["trade_date", "signal_slot"]).size().eq(3).all()
    assert "v21_stock_score" in first
    assert not any(column.startswith("v20_") for column in first.columns)


def test_margin_model_predicts_without_truth_columns() -> None:
    source = build_recall_frontier(
        _source_frame(days=70, stocks=12, seed=2121),
        top_per_source=8,
        exploration_per_slot=4,
    )
    leaders = build_margin_leaders(source, leaders_per_slot=3)
    dates = sorted(leaders["trade_date"].unique())
    train = leaders.loc[leaders["trade_date"].isin(dates[:50])]
    calibration = leaders.loc[leaders["trade_date"].isin(dates[50:60])]
    test = leaders.loc[leaders["trade_date"].isin(dates[60:])]

    bundle = fit_margin_gate(
        train,
        calibration,
        random_seed=21,
        minimum_train_rows=500,
        minimum_calibration_rows=100,
    )
    prediction_input = test.drop(
        columns=["net_return_pct", "target_net_positive"],
    )
    scored = bundle.predict(prediction_input)

    assert any(
        column.startswith("v21_slot_") for column in bundle.feature_columns
    )
    assert scored["v21_margin_probability_lower"].between(0.001, 0.999).all()
    assert scored["v21_tail_probability_upper"].between(0.001, 0.999).all()
    assert scored["v21_return_q20_pct"].notna().all()


def test_policy_locks_first_signal_and_daily_maximum() -> None:
    policy = FrozenMarginPolicy(
        spec=MarginPolicySpec(
            target_candidate_day_rate=0.12,
            max_candidates_per_day=2,
        ),
        margin_probability_lower_threshold=0.60,
        threshold_calibration_start="20260701",
        threshold_calibration_end="20260720",
        threshold_calibration_days=20,
        threshold_eligible_days=10,
    )
    scored = pd.DataFrame(
        [
            _scored_row("20260727", "14:20", "000001.SZ", 0.65, 0.70),
            _scored_row("20260727", "14:20", "000002.SZ", 0.70, 0.80),
            _scored_row("20260727", "14:25", "000001.SZ", 0.99, 0.90),
            _scored_row("20260727", "14:45", "000003.SZ", 0.95, 0.95),
        ]
    )

    selected = apply_margin_policy(scored, policy)

    assert len(selected) == 2
    assert set(selected["ts_code"]) == {"000001.SZ", "000002.SZ"}
    first = selected.set_index("ts_code").loc["000001.SZ"]
    assert first["signal_slot"] == "14:20"
    assert first["v21_margin_probability_lower"] == 0.65


def test_policy_threshold_uses_fixed_unlabelled_day_rate() -> None:
    dates = [f"202607{day:02d}" for day in range(1, 11)]
    scored = pd.DataFrame(
        [
            _scored_row(
                date,
                "14:20",
                f"{index:06d}.SZ",
                index / 10.0,
                index / 10.0,
            )
            for index, date in enumerate(dates, start=1)
        ]
    )
    first = calibrate_margin_policy(
        scored,
        calibration_dates=dates,
        spec=MarginPolicySpec(
            target_candidate_day_rate=0.20,
            max_candidates_per_day=2,
        ),
    )
    changed = scored.copy()
    changed["net_return_pct"] = np.linspace(-100.0, 100.0, len(changed))
    second = calibrate_margin_policy(
        changed,
        calibration_dates=dates,
        spec=MarginPolicySpec(
            target_candidate_day_rate=0.20,
            max_candidates_per_day=2,
        ),
    )

    assert first.margin_probability_lower_threshold == 0.9
    assert second.margin_probability_lower_threshold == 0.9
    assert first.threshold_calibration_days == 10


def test_margin_segments_are_ordered_and_purged() -> None:
    dates = [
        date.strftime("%Y%m%d")
        for date in pd.bdate_range("2025-01-02", periods=180)
    ]

    segments = rolling_margin_model_segments(dates)

    assert segments is not None
    train, calibration = segments
    assert len(train) == 126
    assert len(calibration) == 42
    assert train[-1] < calibration[0]
    assert dates.index(calibration[0]) - dates.index(train[-1]) == 3


def test_policy_rejects_incomplete_tail_contract() -> None:
    row = _scored_row("20260727", "14:20", "000001.SZ", 0.80, 0.90)
    row.pop("v21_tail_probability_upper")
    policy = FrozenMarginPolicy(
        spec=MarginPolicySpec(),
        margin_probability_lower_threshold=0.60,
        threshold_calibration_start="20260701",
        threshold_calibration_end="20260720",
        threshold_calibration_days=20,
        threshold_eligible_days=10,
    )

    with pytest.raises(ValueError, match="v21_tail_probability_upper"):
        apply_margin_policy(pd.DataFrame([row]), policy)


def test_readiness_requires_frequency_margin_tail_and_stability() -> None:
    metrics = {
        "events": 100,
        "candidate_days": 70,
        "candidate_day_rate": 0.12,
        "win_rate": 0.58,
        "win_rate_wilson_lower": 0.50,
        "clustered_win_rate_lower": 0.50,
        "margin_hit_rate": 0.48,
        "tail_loss_rate": 0.12,
        "mean_net_return_pct": 0.40,
        "clustered_mean_lower_pct": 0.05,
        "profit_factor": 1.35,
        "stress_50bps_mean_net_return_pct": 0.01,
        "return_p10_pct": -2.5,
    }
    yearly = [
        {"year": "2024", "events": 30, "mean_net_return_pct": 0.30},
        {"year": "2025", "events": 30, "mean_net_return_pct": 0.20},
        {"year": "2026", "events": 40, "mean_net_return_pct": 0.50},
    ]

    ready = v21_research_readiness(
        metrics,
        yearly=yearly,
        temporal_integrity=True,
        source_integrity=True,
    )
    assert ready["all_historical_gates_passed"]

    failed = dict(metrics)
    failed["stress_50bps_mean_net_return_pct"] = -0.01
    assert not v21_research_readiness(
        failed,
        yearly=yearly,
        temporal_integrity=True,
        source_integrity=True,
    )["all_historical_gates_passed"]


def _source_frame(
    *,
    days: int,
    stocks: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-06", periods=days)
    slots = (
        "14:20",
        "14:25",
        "14:30",
        "14:35",
        "14:40",
        "14:45",
        "14:50",
    )
    rows: list[dict[str, object]] = []
    for date_index, date in enumerate(dates):
        market = rng.normal(0.0, 0.5)
        quality = rng.normal(0.0, 1.0, stocks)
        returns = 0.55 * quality + rng.normal(0.0, 1.15, stocks)
        for slot_index, slot in enumerate(slots):
            for stock in range(stocks):
                probability = np.clip(
                    0.50
                    + 0.08 * quality[stock]
                    + 0.003 * slot_index
                    + rng.normal(0.0, 0.03),
                    0.05,
                    0.95,
                )
                intraday = (
                    1.5
                    + quality[stock]
                    + 0.05 * slot_index
                    + market
                    + rng.normal(0.0, 0.15)
                )
                rows.append(
                    {
                        "trade_date": date.strftime("%Y%m%d"),
                        "signal_slot": slot,
                        "ts_code": f"{stock:06d}.SZ",
                        "fold": date_index // 42 + 1,
                        "execution_eligible": True,
                        "label_available": True,
                        "entry_fillable": True,
                        "exit_fillable": True,
                        "net_return_pct": returns[stock],
                        "target_net_positive": float(returns[stock] > 0),
                        "target_severe_loss": float(returns[stock] <= -3.0),
                        "ret_from_prev_close_pct": intraday,
                        "p_entry_fill": 0.99,
                        "p_exit_fill_given_entry": 0.995,
                        "p_round_trip_fill_lower": 0.98,
                        "p_net_positive": probability,
                        "p_net_positive_lower": probability - 0.03,
                        "p_conditional_net_positive": np.clip(
                            probability + 0.01,
                            0.0,
                            1.0,
                        ),
                        "p_cross_section_top": np.clip(
                            probability + 0.02,
                            0.0,
                            1.0,
                        ),
                        "p_severe_loss": np.clip(
                            0.40 - probability,
                            0.02,
                            0.35,
                        ),
                        "selection_score": probability + intraday * 0.01,
                        "selection_rank_pct": (stock + 1) / stocks,
                        "expected_utility_pct": probability - 0.45,
                        "expected_utility_lower_pct": probability - 0.50,
                        "downside_q10_pct": -2.0,
                        "probability_model_spread": 0.05,
                        "fill_probability_model_spread": 0.01,
                        "selection_rank_spread": 0.03,
                        "expected_return_model_spread": 0.10,
                        "data_age_seconds": 30.0,
                    }
                )
    return pd.DataFrame(rows)


def _scored_row(
    trade_date: str,
    slot: str,
    ts_code: str,
    margin_probability_lower: float,
    stock_score: float,
) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "signal_slot": slot,
        "ts_code": ts_code,
        "net_return_pct": 1.0,
        "v21_margin_probability_lower": float(margin_probability_lower),
        "v21_margin_model_spread": 0.05,
        "v21_tail_probability_upper": 0.15,
        "v21_tail_model_spread": 0.04,
        "v21_return_q20_pct": -1.0,
        "v21_stock_score": float(stock_score),
        "p_severe_loss": 0.10,
        "p_round_trip_fill_lower": 0.98,
        "data_age_seconds": 30.0,
    }
