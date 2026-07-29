from __future__ import annotations

from dataclasses import replace

import pandas as pd

from wp.v3.contracts import V3Config
from wp.v3.policy import (
    CandidatePolicy,
    apply_candidate_policy,
    apply_nested_oos_policies,
    candidate_policy_diagnostics,
    select_candidate_policy,
)


def _profitable_predictions(days: int, *, start: str = "2025-01-02") -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range(start, periods=days)
    for day_index, date in enumerate(dates):
        for slot in ("14:20", "14:25"):
            for stock in range(4):
                rows.append(
                    {
                        "trade_date": date.strftime("%Y%m%d"),
                        "target_trade_date": (
                            date + pd.offsets.BDay(1)
                        ).strftime("%Y%m%d"),
                        "signal_slot": slot,
                        "ts_code": f"600{stock:03d}.SH",
                        "execution_eligible": True,
                        "data_age_seconds": 30,
                        "p_entry_fill": 0.995,
                        "p_exit_fill_given_entry": 0.998,
                        "p_round_trip_fill_lower": 0.993,
                        "p_net_positive": 0.66,
                        "p_net_positive_lower": 0.61,
                        "p_conditional_net_positive": 0.67,
                        "p_cross_section_top": 0.60,
                        "p_severe_loss": 0.12,
                        "selection_rank_pct": 0.999,
                        "expected_utility_pct": 0.55,
                        "downside_q10_pct": -1.0,
                        "probability_model_spread": 0.02,
                        "fill_probability_model_spread": 0.01,
                        "selection_rank_spread": 0.03,
                        "net_return_pct": 0.45 + 0.01 * ((day_index + stock) % 3),
                        "target_net_positive": 1,
                        "target_entry_fillable": 1,
                        "target_exit_fillable": 1,
                    }
                )
    return pd.DataFrame(rows)


def _test_config() -> V3Config:
    base = V3Config()
    return replace(
        base,
        model=replace(
            base.model,
            policy_design_days=20,
            policy_confirmation_days=10,
            policy_min_design_events=20,
            policy_min_design_days=5,
            policy_min_confirmation_events=10,
            policy_min_confirmation_days=3,
            policy_min_win_rate=0.50,
            policy_min_wilson_lower=0.40,
            policy_min_clustered_lower=0.40,
            policy_min_mean_net_return_pct=0.10,
            policy_min_profit_factor=1.0,
        ),
    )


def test_policy_is_selected_on_design_and_independently_confirmed():
    frame = _profitable_predictions(30)
    config = _test_config()
    dates = sorted(frame["trade_date"].unique())
    design = frame.loc[frame["trade_date"].isin(dates[:20])]
    confirmation = frame.loc[frame["trade_date"].isin(dates[20:])]

    selection = select_candidate_policy(design, confirmation, config)

    assert selection.policy.authorized is True
    assert selection.policy.reason == "design_champion_confirmed_once"
    assert selection.design["events"] == 80
    assert selection.confirmation["events"] == 40
    assert selection.search["confirmation_policies_evaluated"] == 1
    assert apply_candidate_policy(frame, selection.policy, config).all()


def test_nested_policy_cannot_use_current_or_future_fold_truth():
    frame = _profitable_predictions(60)
    dates = sorted(frame["trade_date"].unique())
    frame["fold"] = 0
    for fold, start in enumerate(range(0, 60, 10), start=1):
        frame.loc[frame["trade_date"].isin(dates[start : start + 10]), "fold"] = fold

    result, audit, final = apply_nested_oos_policies(frame, _test_config())

    early = result["fold"].le(3)
    later = result["fold"].ge(4)
    assert not result.loc[early, "passes_policy"].any()
    assert result.loc[later, "passes_policy"].all()
    assert audit[2]["policy"]["reason"] == "insufficient_prior_oos_policy_days"
    assert audit[3]["policy"]["authorized"] is True
    assert final.policy.authorized is True


def test_fast_policy_mask_matches_full_diagnostics():
    frame = _profitable_predictions(2)
    config = _test_config()
    selection = select_candidate_policy(
        _profitable_predictions(20),
        _profitable_predictions(10, start="2025-03-03"),
        config,
    )
    frame.loc[frame.index[0], "p_net_positive"] = 0.10
    frame.loc[frame.index[-1], "execution_eligible"] = False

    fast = apply_candidate_policy(frame, selection.policy, config)
    detailed = candidate_policy_diagnostics(
        frame,
        selection.policy,
        config,
    )["passes_policy"]

    pd.testing.assert_series_equal(fast, detailed)


def test_policy_rejects_each_execution_probability_below_its_gate():
    frame = _profitable_predictions(1).iloc[:3].copy()
    policy = CandidatePolicy(
        policy_id="execution-gate-test",
        authorized=True,
        reason="test",
        entry_fill_probability_min=0.985,
        exit_fill_probability_min=0.995,
        round_trip_fill_probability_min=0.980,
        probability_min=0.46,
        probability_lower_min=0.40,
        conditional_probability_min=0.52,
        severe_loss_probability_max=0.25,
        selection_rank_min=0.998,
        expected_utility_min_pct=0.10,
        downside_min_pct=-2.0,
    )
    frame.loc[frame.index[0], "p_entry_fill"] = 0.984
    frame.loc[frame.index[1], "p_exit_fill_given_entry"] = 0.994
    frame.loc[frame.index[2], "p_round_trip_fill_lower"] = 0.979

    diagnostics = candidate_policy_diagnostics(frame, policy, _test_config())

    assert not diagnostics["passes_policy"].any()
    assert (
        diagnostics.loc[frame.index[0], "passes_entry_fill_probability"]
        == False
    )
    assert (
        diagnostics.loc[frame.index[1], "passes_exit_fill_probability"]
        == False
    )
    assert (
        diagnostics.loc[frame.index[2], "passes_round_trip_fill_probability"]
        == False
    )
