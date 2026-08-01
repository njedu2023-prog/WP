from __future__ import annotations

import numpy as np
import pandas as pd

from wp.v3.v37_fast_entry import (
    BASE_ALERT_SLOTS,
    FastEntryPolicySpec,
    FrozenFastEntryPolicy,
    apply_fast_entry_policy,
    audit_fast_entry_outcomes,
    build_fast_entry_outcomes,
    calibrate_fast_entry_policy,
    fast_entry_timing,
    join_fast_entry_outcomes,
    selected_execution_audit,
    validate_selected_contract,
)


def test_fast_entry_timing_is_executable_before_1450() -> None:
    assert fast_entry_timing("14:20") == ("14:22", "14:23")
    assert fast_entry_timing("14:45") == ("14:47", "14:48")


def test_fast_entry_outcome_uses_exact_t_plus_three_bar() -> None:
    candidates = pd.DataFrame([_candidate("14:20")])
    outcomes = _outcomes(candidates, _minutes("14:23", close=10.0))
    row = outcomes.iloc[0]

    expected_entry = 10.0 * 1.001
    expected_gross = (10.5 / expected_entry - 1.0) * 100.0
    assert row["v37_publication_time"] == "14:22"
    assert row["v37_entry_benchmark_time"] == "14:23"
    assert bool(row["entry_fillable"])
    assert np.isclose(row["entry_price"], expected_entry)
    assert np.isclose(row["gross_return_pct"], expected_gross)
    assert np.isclose(row["net_return_pct"], expected_gross - 0.25)
    assert int(row["target_net_positive"]) == 1


def test_fast_entry_miss_is_zero_return_cash() -> None:
    candidates = pd.DataFrame([_candidate("14:20")])
    minutes = _minutes("14:23", close=10.0, amount=2_000_000.0)
    row = _outcomes(candidates, minutes).iloc[0]

    assert not bool(row["entry_fillable"])
    assert bool(row["label_available"])
    assert row["gross_return_pct"] == 0.0
    assert row["net_return_pct"] == 0.0
    assert int(row["target_net_positive"]) == 0


def test_failed_t1_exit_receives_conservative_penalty() -> None:
    candidate = _candidate("14:20")
    candidate["source_exit_fillable"] = False
    candidates = pd.DataFrame([candidate])
    row = _outcomes(candidates, _minutes("14:23", close=10.0)).iloc[0]

    assert bool(row["entry_fillable"])
    assert not bool(row["exit_fillable"])
    assert not bool(row["execution_success"])
    assert row["net_return_pct"] == -10.0
    assert int(row["target_net_positive"]) == 0


def test_fast_entry_join_replaces_old_contract_outcomes() -> None:
    candidate = _candidate("14:20")
    candidate.update(
        {
            "entry_price": 99.0,
            "net_return_pct": -99.0,
            "target_net_positive": 0,
            "label_available": True,
        }
    )
    candidates = pd.DataFrame([candidate])
    outcomes = _outcomes(candidates, _minutes("14:23", close=10.0))
    joined = join_fast_entry_outcomes(candidates, outcomes)

    assert len(joined) == 1
    assert joined.iloc[0]["entry_price"] != 99.0
    assert joined.iloc[0]["net_return_pct"] > 0.0


def test_fast_entry_outcome_audit_requires_exact_identity_and_labels() -> None:
    candidates = pd.DataFrame(
        [_candidate("14:20"), _candidate("14:50")]
    )
    outcomes = _outcomes(candidates, _minutes("14:23", close=10.0))
    audit = audit_fast_entry_outcomes(outcomes, candidates)

    assert audit["expected_rows"] == 1
    assert audit["outcome_rows"] == 1
    assert audit["coverage_passed"]


def test_policy_threshold_cannot_read_fast_entry_outcomes() -> None:
    dates = [
        value.strftime("%Y%m%d")
        for value in pd.bdate_range("2026-06-01", periods=10)
    ]
    scored = pd.DataFrame(
        [
            _scored(date, f"{index:06d}.SZ", index / 10.0)
            for index, date in enumerate(dates, start=1)
        ]
    )
    spec = FastEntryPolicySpec(target_candidate_day_rate=0.30)
    first = calibrate_fast_entry_policy(
        scored,
        calibration_dates=dates,
        spec=spec,
    )
    changed = scored.copy()
    changed["net_return_pct"] = np.linspace(-100.0, 100.0, len(changed))
    changed["target_net_positive"] = 1
    second = calibrate_fast_entry_policy(
        changed,
        calibration_dates=dates,
        spec=spec,
    )

    assert first.score_threshold == 0.8
    assert second.score_threshold == 0.8


def test_policy_keeps_first_signal_and_daily_capacity() -> None:
    policy = FrozenFastEntryPolicy(
        spec=FastEntryPolicySpec(max_candidates_per_day=3),
        score_threshold=0.60,
        calibration_start="20260501",
        calibration_end="20260630",
        calibration_days=42,
        eligible_days=20,
    )
    scored = pd.DataFrame(
        [
            _scored("20260727", "000001.SZ", 0.70, slot="14:20"),
            _scored("20260727", "000002.SZ", 0.80, slot="14:20"),
            _scored("20260727", "000003.SZ", 0.65, slot="14:20"),
            _scored("20260727", "000004.SZ", 0.95, slot="14:20"),
            _scored("20260727", "000001.SZ", 0.99, slot="14:25"),
        ]
    )
    selected = apply_fast_entry_policy(scored, policy)

    validate_selected_contract(selected, policy)
    assert selected["ts_code"].tolist() == [
        "000004.SZ",
        "000002.SZ",
        "000001.SZ",
    ]


def test_selected_execution_audit_is_explicit() -> None:
    selected = pd.DataFrame(
        {
            "entry_fillable": [True, True, False],
            "exit_fillable": [True, False, True],
            "execution_success": [True, False, False],
        }
    )
    audit = selected_execution_audit(selected)

    assert np.isclose(audit["entry_fill_rate"], 2 / 3)
    assert np.isclose(audit["round_trip_fill_rate"], 1 / 3)
    assert np.isclose(audit["exit_fill_rate_given_entry"], 0.5)


def _candidate(slot: str) -> dict[str, object]:
    return {
        "trade_date": "20260727",
        "signal_slot": slot,
        "ts_code": "000001.SZ",
        "fold": 1,
        "signal_price": 9.90,
        "execution_eligible": True,
        "source_label_available": True,
        "target_trade_date": "20260728",
        "t1_total_return_close": 10.50,
        "source_exit_fillable": True,
        "up_limit": 10.90,
        "down_limit": 8.91,
    }


def _minutes(
    hhmm: str,
    *,
    close: float,
    amount: float = 20_000_000.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_time": pd.Timestamp(f"2026-07-27 {hhmm}:00"),
                "open": close - 0.01,
                "high": close + 0.02,
                "low": close - 0.02,
                "close": close,
                "vol": 100_000.0,
                "amount": amount,
            }
        ]
    )


def _outcomes(
    candidates: pd.DataFrame,
    minutes: pd.DataFrame,
) -> pd.DataFrame:
    return build_fast_entry_outcomes(
        candidates,
        minutes,
        entry_slippage_bps=10.0,
        round_trip_cost_bps=25.0,
        min_slot_amount=3_000_000.0,
        reference_order_notional=100_000.0,
        max_entry_pct_of_slot_amount=0.01,
        min_distance_to_up_limit_pct=0.50,
        min_distance_to_down_limit_pct=1.00,
        non_fill_penalty_pct=-10.0,
    )


def _scored(
    trade_date: str,
    ts_code: str,
    score: float,
    *,
    slot: str = "14:20",
) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "signal_slot": slot,
        "ts_code": ts_code,
        "v23_point_in_time_complete": True,
        "v34_path_complete": True,
        "p_round_trip_fill_lower": 0.99,
        "p_severe_loss": 0.10,
        "v34_p_positive_lower": 0.70,
        "v34_p_margin_lower": 0.50,
        "v34_p_severe_loss_upper": 0.10,
        "v34_expected_net_return_lower_pct": 0.30,
        "v34_positive_model_spread": 0.05,
        "v34_margin_model_spread": 0.05,
        "v34_severe_model_spread": 0.05,
        "v34_expected_return_model_spread_pct": 0.10,
        "v34_path_score": score,
        "data_age_seconds": 0.0,
    }
