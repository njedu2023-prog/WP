from __future__ import annotations

import numpy as np
import pandas as pd

from wp.v3.v18_ranked import (
    FrozenRankedPolicy,
    RankedPolicySpec,
    apply_ranked_policy,
    calibrate_ranked_policy,
    fit_ranked_selector,
    prepare_ranked_frame,
    ranked_policy_grid,
    rolling_ranked_policy_segments,
    v18_research_readiness,
)


def test_policy_family_is_small_and_predeclared() -> None:
    policies = ranked_policy_grid()

    assert len(policies) == 16
    assert len({policy.spec_id for policy in policies}) == 16
    assert {
        policy.target_candidate_day_rate for policy in policies
    } == {0.08, 0.12, 0.16, 0.20}


def test_persistence_features_do_not_look_into_future_slots() -> None:
    source = _frame(days=2, stocks=3, seed=18)
    baseline = prepare_ranked_frame(source)
    changed = source.copy()
    future = changed["signal_slot"].eq("14:50")
    changed.loc[future, "p_net_positive_lower"] = 0.999
    changed.loc[future, "selection_score"] = 99.0
    replayed = prepare_ranked_frame(changed)

    keys = ["trade_date", "signal_slot", "ts_code"]
    early = baseline["signal_slot"].le("14:45")
    columns = [
        *keys,
        "v18_p_positive_mean_so_far",
        "v18_selection_score_mean_so_far",
        "v18_quality_hits_so_far",
    ]
    pd.testing.assert_frame_equal(
        baseline.loc[early, columns].reset_index(drop=True),
        replayed.loc[early, columns].reset_index(drop=True),
    )


def test_event_stream_never_replaces_an_earlier_locked_candidate() -> None:
    spec = RankedPolicySpec(
        target_candidate_day_rate=0.12,
        max_candidates_per_day=1,
        minimum_quality_hits=1,
    )
    policy = FrozenRankedPolicy(
        spec=spec,
        score_threshold=0.50,
        threshold_calibration_start="20260701",
        threshold_calibration_end="20260720",
        threshold_calibration_days=20,
        threshold_eligible_days=10,
    )
    scored = pd.DataFrame(
        [
            _scored_row("20260727", "14:20", "000001.SZ", 0.70),
            _scored_row("20260727", "14:45", "000002.SZ", 9.00),
            _scored_row("20260727", "14:50", "000001.SZ", 10.00),
        ]
    )

    selected = apply_ranked_policy(scored, policy)

    assert len(selected) == 1
    assert selected.iloc[0]["ts_code"] == "000001.SZ"
    assert selected.iloc[0]["signal_slot"] == "14:20"


def test_threshold_is_calibrated_to_prior_daily_maxima() -> None:
    spec = RankedPolicySpec(
        target_candidate_day_rate=0.20,
        max_candidates_per_day=1,
        minimum_quality_hits=1,
    )
    dates = [f"202607{day:02d}" for day in range(1, 11)]
    scored = pd.DataFrame(
        [
            _scored_row(date, "14:20", f"{index:06d}.SZ", float(index))
            for index, date in enumerate(dates, start=1)
        ]
    )

    policy = calibrate_ranked_policy(
        scored,
        spec,
        calibration_dates=dates,
    )

    assert policy.score_threshold == 9.0
    assert policy.threshold_calibration_days == 10
    assert policy.threshold_eligible_days == 10


def test_ranked_selector_scores_live_rows_with_causal_features() -> None:
    source = _frame(days=70, stocks=8, seed=1818)
    dates = sorted(source["trade_date"].unique())
    train = source.loc[source["trade_date"].isin(dates[:50])]
    calibration = source.loc[source["trade_date"].isin(dates[50:60])]
    test = source.loc[source["trade_date"].isin(dates[60:])]

    bundle = fit_ranked_selector(
        train,
        calibration,
        random_seed=1818,
        minimum_train_rows=300,
        minimum_calibration_rows=100,
    )
    live = bundle.predict(
        test.drop(columns=["net_return_pct", "target_net_positive"])
    )

    assert any(
        column.startswith("v18_") for column in bundle.feature_columns
    )
    assert live["selector_p_positive"].between(0.001, 0.999).all()
    assert live["v18_slots_seen"].between(1, 7).all()


def test_policy_segments_are_ordered_and_purged() -> None:
    dates = [
        date.strftime("%Y%m%d")
        for date in pd.bdate_range("2025-01-02", periods=150)
    ]

    segments = rolling_ranked_policy_segments(dates)

    assert segments is not None
    threshold, design, confirmation = segments
    assert len(threshold) == 42
    assert len(design) == 42
    assert len(confirmation) == 42
    assert threshold[-1] < design[0] < confirmation[0]
    assert dates.index(design[0]) - dates.index(threshold[-1]) == 3
    assert dates.index(confirmation[0]) - dates.index(design[-1]) == 3


def test_readiness_requires_frequency_profit_and_year_stability() -> None:
    metrics = {
        "events": 120,
        "candidate_days": 60,
        "candidate_day_rate": 0.15,
        "win_rate": 0.58,
        "win_rate_wilson_lower": 0.51,
        "clustered_win_rate_lower": 0.51,
        "mean_net_return_pct": 0.35,
        "clustered_mean_lower_pct": 0.05,
        "profit_factor": 1.30,
        "stress_50bps_mean_net_return_pct": 0.02,
        "return_p10_pct": -2.5,
    }
    yearly = [
        {"year": "2025", "events": 60, "mean_net_return_pct": 0.30},
        {"year": "2026", "events": 60, "mean_net_return_pct": 0.40},
    ]

    assert v18_research_readiness(
        metrics,
        yearly=yearly,
        temporal_integrity=True,
    )["all_historical_gates_passed"]
    assert not v18_research_readiness(
        {**metrics, "candidate_day_rate": 0.03},
        yearly=yearly,
        temporal_integrity=True,
    )["all_historical_gates_passed"]


def _frame(*, days: int, stocks: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-05", periods=days)
    slots = (
        "14:20",
        "14:25",
        "14:30",
        "14:35",
        "14:40",
        "14:45",
        "14:50",
    )
    rows = []
    for date in dates:
        market = rng.normal(0.0, 0.5)
        stock_return = rng.normal(0.4, 1.0, size=stocks)
        net_returns = 0.25 * stock_return + rng.normal(0.0, 0.9, stocks)
        for slot_index, slot in enumerate(slots):
            for stock in range(stocks):
                probability = np.clip(
                    0.48
                    + 0.04 * stock_return[stock]
                    + 0.005 * slot_index
                    + rng.normal(0.0, 0.03),
                    0.05,
                    0.95,
                )
                intraday_return = (
                    stock_return[stock]
                    + 0.08 * slot_index
                    + market
                    + rng.normal(0.0, 0.15)
                )
                rows.append(
                    {
                        "trade_date": date.strftime("%Y%m%d"),
                        "signal_slot": slot,
                        "ts_code": f"{stock:06d}.SZ",
                        "net_return_pct": net_returns[stock],
                        "target_net_positive": float(
                            net_returns[stock] > 0
                        ),
                        "ret_from_prev_close_pct": intraday_return,
                        "p_entry_fill": 0.99,
                        "p_exit_fill_given_entry": 0.995,
                        "p_round_trip_fill_lower": 0.97,
                        "p_net_positive": probability,
                        "p_net_positive_lower": probability - 0.03,
                        "p_conditional_net_positive": probability + 0.01,
                        "p_severe_loss": np.clip(
                            0.45 - probability,
                            0.02,
                            0.40,
                        ),
                        "selection_score": (
                            probability + intraday_return * 0.01
                        ),
                        "selection_rank_pct": (stock + 1) / stocks,
                        "expected_utility_pct": probability - 0.50,
                        "expected_utility_lower_pct": probability - 0.55,
                        "downside_q10_pct": -3.0 + probability,
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
    score: float,
) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "signal_slot": slot,
        "ts_code": ts_code,
        "net_return_pct": 1.0,
        "selector_p_positive_lower": 0.60,
        "selector_expected_net_return_pct": 0.40,
        "selector_return_q25_pct": -0.20,
        "selector_probability_spread": 0.05,
        "selector_score": score,
        "selector_score_rank_pct": 0.97,
        "p_severe_loss": 0.10,
        "p_round_trip_fill_lower": 0.98,
        "v18_quality_hits_so_far": 2,
    }
