from __future__ import annotations

import numpy as np
import pandas as pd

from wp.v3.v18_ranked import prepare_ranked_frame
from wp.v3.v19_recall import (
    FrozenRecallPolicy,
    RecallPolicySpec,
    apply_recall_policy,
    build_recall_frontier,
    calibrate_recall_policy,
    deterministic_stock_day_sample,
    fit_recall_selector,
    recall_policy_grid,
    rolling_recall_policy_segments,
    v19_research_readiness,
)


def test_recall_frontier_is_label_independent_and_keeps_exploration() -> None:
    source = _source_frame(days=2, stocks=40, seed=19)
    first = build_recall_frontier(
        source,
        top_per_source=3,
        exploration_per_slot=4,
    )
    changed = source.copy()
    changed["target_net_positive"] = 1 - changed["target_net_positive"]
    changed["net_return_pct"] = -changed["net_return_pct"] + 100.0
    second = build_recall_frontier(
        changed,
        top_per_source=3,
        exploration_per_slot=4,
    )

    keys = ["trade_date", "signal_slot", "ts_code"]
    pd.testing.assert_frame_equal(
        first.loc[:, keys],
        second.loc[:, keys],
    )
    assert first["v19_retrieval_exploration"].sum() > 0
    assert first["v19_full_context_universe_size"].eq(40).all()


def test_training_sample_is_deterministic_and_ignores_outcome() -> None:
    source = build_recall_frontier(
        _source_frame(days=5, stocks=30, seed=1901),
        top_per_source=8,
        exploration_per_slot=4,
    )
    first = deterministic_stock_day_sample(
        source,
        max_stocks_per_day=12,
    )
    changed = source.copy()
    changed["target_net_positive"] = 1
    changed["net_return_pct"] = 999.0
    second = deterministic_stock_day_sample(
        changed.sample(frac=1.0, random_state=7),
        max_stocks_per_day=12,
    )
    keys = ["trade_date", "signal_slot", "ts_code"]
    pd.testing.assert_frame_equal(
        first.loc[:, keys].sort_values(keys).reset_index(drop=True),
        second.loc[:, keys].sort_values(keys).reset_index(drop=True),
    )


def test_v19_persistence_does_not_look_into_later_slots() -> None:
    frontier = build_recall_frontier(
        _source_frame(days=2, stocks=12, seed=1919),
        top_per_source=12,
        exploration_per_slot=2,
    )
    baseline = prepare_ranked_frame(frontier)
    changed = frontier.copy()
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


def test_v19_selector_uses_full_context_and_retrieval_features() -> None:
    source = build_recall_frontier(
        _source_frame(days=70, stocks=12, seed=19190),
        top_per_source=8,
        exploration_per_slot=4,
    )
    dates = sorted(source["trade_date"].unique())
    train = source.loc[source["trade_date"].isin(dates[:50])]
    calibration = source.loc[source["trade_date"].isin(dates[50:60])]
    test = source.loc[source["trade_date"].isin(dates[60:])]

    bundle = fit_recall_selector(
        train,
        calibration,
        random_seed=19,
        max_stocks_per_day=12,
        minimum_train_rows=500,
        minimum_calibration_rows=100,
    )
    scored = bundle.predict(
        test.drop(columns=["net_return_pct", "target_net_positive"])
    )

    assert any(
        column.startswith("v19_full_")
        for column in bundle.feature_columns
    )
    assert any(
        column.startswith("v19_retrieval_")
        for column in bundle.feature_columns
    )
    assert scored["selector_p_positive"].between(0.001, 0.999).all()


def test_policy_family_is_predeclared_and_frequency_calibrated() -> None:
    policies = recall_policy_grid()

    assert len(policies) == 16
    assert {
        policy.target_candidate_day_rate for policy in policies
    } == {0.12, 0.18, 0.24, 0.30}

    spec = RecallPolicySpec(
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
    policy = calibrate_recall_policy(
        scored,
        spec,
        calibration_dates=dates,
    )

    assert policy.score_threshold == 9.0
    assert policy.threshold_calibration_days == 10


def test_event_stream_locks_first_candidate_before_later_scores() -> None:
    policy = FrozenRecallPolicy(
        spec=RecallPolicySpec(
            target_candidate_day_rate=0.18,
            max_candidates_per_day=1,
            minimum_quality_hits=1,
        ),
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

    selected = apply_recall_policy(scored, policy)

    assert len(selected) == 1
    assert selected.iloc[0]["ts_code"] == "000001.SZ"
    assert selected.iloc[0]["signal_slot"] == "14:20"


def test_policy_segments_are_long_ordered_and_purged() -> None:
    dates = [
        date.strftime("%Y%m%d")
        for date in pd.bdate_range("2025-01-02", periods=220)
    ]

    segments = rolling_recall_policy_segments(dates)

    assert segments is not None
    threshold, design, confirmation = segments
    assert len(threshold) == 42
    assert len(design) == 84
    assert len(confirmation) == 84
    assert threshold[-1] < design[0] < confirmation[0]
    assert dates.index(design[0]) - dates.index(threshold[-1]) == 3
    assert dates.index(confirmation[0]) - dates.index(design[-1]) == 3


def test_readiness_requires_frequency_profit_and_source_integrity() -> None:
    metrics = {
        "events": 100,
        "candidate_days": 60,
        "candidate_day_rate": 0.18,
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
        {"year": "2025", "events": 50, "mean_net_return_pct": 0.30},
        {"year": "2026", "events": 50, "mean_net_return_pct": 0.40},
    ]

    assert v19_research_readiness(
        metrics,
        yearly=yearly,
        temporal_integrity=True,
        source_integrity=True,
    )["all_historical_gates_passed"]
    assert not v19_research_readiness(
        metrics,
        yearly=yearly,
        temporal_integrity=True,
        source_integrity=False,
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
    rows = []
    for date_index, date in enumerate(dates):
        market = rng.normal(0.0, 0.5)
        stock_quality = rng.normal(0.0, 1.0, stocks)
        net_returns = (
            0.35 * stock_quality + rng.normal(0.0, 0.9, stocks)
        )
        for slot_index, slot in enumerate(slots):
            for stock in range(stocks):
                probability = np.clip(
                    0.50
                    + 0.08 * stock_quality[stock]
                    + 0.003 * slot_index
                    + rng.normal(0.0, 0.03),
                    0.05,
                    0.95,
                )
                intraday = (
                    1.5
                    + stock_quality[stock]
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
                        "net_return_pct": net_returns[stock],
                        "target_net_positive": float(
                            net_returns[stock] > 0
                        ),
                        "ret_from_prev_close_pct": intraday,
                        "p_entry_fill": 0.99,
                        "p_exit_fill_given_entry": 0.995,
                        "p_round_trip_fill_lower": 0.98,
                        "p_net_positive": probability,
                        "p_net_positive_lower": probability - 0.03,
                        "p_conditional_net_positive": probability + 0.01,
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
                        "selection_score": (
                            probability + intraday * 0.01
                        ),
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
        "data_age_seconds": 30.0,
    }
