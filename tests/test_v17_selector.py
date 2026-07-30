from __future__ import annotations

import numpy as np
import pandas as pd

from wp.v3.v17_selector import (
    SelectorPolicy,
    apply_selector_policy,
    fit_selector,
    selector_confirmation_gate,
    selector_design_gate,
    selector_policy_grid,
)


def test_policy_family_is_small_and_predeclared() -> None:
    policies = selector_policy_grid()

    assert len(policies) == 24
    assert len({policy.policy_id for policy in policies}) == 24
    assert {policy.max_candidates_per_day for policy in policies} == {2, 3}


def test_selector_fits_and_scores_causal_oos_rows() -> None:
    source = _frame(days=90, stocks=12, seed=17)
    train_dates = sorted(source["trade_date"].unique())[:60]
    calibration_dates = sorted(source["trade_date"].unique())[60:75]
    test_dates = sorted(source["trade_date"].unique())[75:]
    train = source.loc[source["trade_date"].isin(train_dates)]
    calibration = source.loc[
        source["trade_date"].isin(calibration_dates)
    ]
    test = source.loc[source["trade_date"].isin(test_dates)]

    bundle = fit_selector(
        train,
        calibration,
        random_seed=1701,
        minimum_train_rows=300,
        minimum_calibration_rows=100,
    )
    scored = bundle.predict(test)

    assert bundle.train_rows == len(train)
    assert bundle.calibration_rows == len(calibration)
    assert len(bundle.feature_columns) >= 12
    assert scored["selector_p_positive"].between(0.001, 0.999).all()
    assert (
        scored["selector_p_positive_lower"]
        <= scored["selector_p_positive"]
    ).all()
    assert scored["selector_score_rank_pct"].between(0.0, 1.0).all()
    assert scored["selector_return_q25_pct"].notna().all()

    live = bundle.predict(
        test.drop(columns=["net_return_pct", "target_net_positive"])
    )
    assert len(live) == len(test)
    assert live["selector_p_positive"].notna().all()


def test_policy_locks_first_signal_and_caps_daily_candidates() -> None:
    frame = pd.DataFrame(
        [
            _scored_row("20260727", "14:20", "000001.SZ", 2.1),
            _scored_row("20260727", "14:25", "000001.SZ", 2.0),
            _scored_row("20260727", "14:20", "000002.SZ", 1.9),
            _scored_row("20260727", "14:30", "000003.SZ", 1.8),
        ]
    )
    policy = SelectorPolicy(
        probability_lower_min=0.52,
        expected_return_min_pct=0.0,
        score_rank_min=0.90,
        max_candidates_per_day=2,
    )

    selected = apply_selector_policy(frame, policy)

    assert len(selected) == 2
    first = selected.loc[selected["ts_code"].eq("000001.SZ")].iloc[0]
    assert first["signal_slot"] == "14:20"
    assert selected["ts_code"].nunique() == len(selected)
    assert selected["selector_policy_id"].eq(policy.policy_id).all()


def test_frequency_and_profit_gates_are_explicit() -> None:
    passing = {
        "events": 50,
        "candidate_days": 24,
        "candidate_day_rate": 0.30,
        "win_rate": 0.56,
        "mean_net_return_pct": 0.40,
        "profit_factor": 1.30,
        "stress_50bps_mean_net_return_pct": 0.05,
        "mean_return_q_value": 0.08,
    }

    assert selector_design_gate(passing)
    assert selector_confirmation_gate(
        {
            **passing,
            "events": 24,
            "candidate_days": 12,
            "candidate_day_rate": 0.24,
        }
    )
    assert not selector_design_gate(
        {**passing, "candidate_day_rate": 0.05}
    )
    assert not selector_design_gate(
        {**passing, "stress_50bps_mean_net_return_pct": -0.01}
    )


def _frame(*, days: int, stocks: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-05", periods=days)
    rows = []
    for day_index, date in enumerate(dates):
        market = rng.normal(0.0, 0.5)
        for stock in range(stocks):
            probability = np.clip(
                0.42 + 0.018 * stock + rng.normal(0.0, 0.04),
                0.05,
                0.95,
            )
            intraday_return = (
                0.25 * stock + market + rng.normal(0.0, 0.7)
            )
            net_return = (
                -0.70
                + 2.10 * probability
                + 0.08 * intraday_return
                + rng.normal(0.0, 0.9)
            )
            rows.append(
                {
                    "trade_date": date.strftime("%Y%m%d"),
                    "signal_slot": (
                        "14:20" if (day_index + stock) % 2 == 0 else "14:40"
                    ),
                    "ts_code": f"{stock:06d}.SZ",
                    "net_return_pct": net_return,
                    "target_net_positive": float(net_return > 0),
                    "ret_from_prev_close_pct": intraday_return,
                    "p_entry_fill": 0.99,
                    "p_exit_fill_given_entry": 0.995,
                    "p_round_trip_fill_lower": 0.97,
                    "p_net_positive": probability,
                    "p_net_positive_lower": probability - 0.03,
                    "p_conditional_net_positive": probability + 0.01,
                    "p_severe_loss": np.clip(0.45 - probability, 0.02, 0.40),
                    "selection_score": probability + intraday_return * 0.01,
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
    }
