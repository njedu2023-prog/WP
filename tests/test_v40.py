from __future__ import annotations

import pandas as pd

from wp.v3.cohorts import select_live_cohorts
from wp.v3.contracts import V3Config
from wp.v3.v40_model import (
    META_CALIBRATION_DAYS,
    META_TRAIN_DAYS,
    PURGE_DAYS,
    RISK_CALIBRATION_DAYS,
    RISK_TRAIN_DAYS,
    V40_MODEL_SCHEMA_VERSION,
)
from wp.v3.v40 import (
    V40Policy,
    attach_v40_policy_gates,
    evaluate_v40_fixed_1430,
)


def row(
    date: str,
    code: str,
    *,
    probability: float = 0.60,
    expected: float = 0.20,
    severe: float = 0.10,
    fill: float = 0.99,
    rank: float = 0.99,
    risk_rank: float = 0.20,
    net_return: float = 1.0,
    slot: str = "14:30",
) -> dict[str, object]:
    return {
        "trade_date": date,
        "signal_slot": slot,
        "ts_code": code,
        "signal_price": 10.0,
        "execution_eligible": True,
        "meta_p_positive": probability,
        "meta_expected_net_return_pct": expected,
        "meta_p_severe_loss": severe,
        "p_round_trip_fill_lower": fill,
        "meta_rank_pct": rank,
        "risk_failure_rank_pct": risk_rank,
        "meta_score": probability + expected - severe,
        "entry_fillable": True,
        "exit_fillable": True,
        "label_available": True,
        "net_return_pct": net_return,
    }


def two_day_frame() -> pd.DataFrame:
    rows = []
    for day_index, date in enumerate(("20260504", "20260505")):
        rows.extend(
            [
                row(date, f"Q{day_index}A", net_return=1.0),
                row(
                    date,
                    f"Q{day_index}B",
                    probability=0.58,
                    net_return=-0.5,
                ),
            ]
        )
        for index in range(6):
            rows.append(
                row(
                    date,
                    f"O{day_index}{index}",
                    probability=0.53 - index * 0.01,
                    net_return=float(index - 2),
                )
            )
        rows.append(row(date, f"LATE{day_index}", slot="14:35"))
    return pd.DataFrame(rows)


def test_v40_uses_a_full_year_for_rare_exit_risk_only() -> None:
    assert V40_MODEL_SCHEMA_VERSION == "wp_v40_fixed_1430_bundle_2"
    assert META_TRAIN_DAYS == 126
    assert META_CALIBRATION_DAYS == 21
    assert RISK_TRAIN_DAYS == 252
    assert RISK_CALIBRATION_DAYS == 42
    assert PURGE_DAYS == 2


def test_v40_has_all_passers_and_exactly_five_separate_observations() -> None:
    result = evaluate_v40_fixed_1430(
        two_day_frame(),
        V3Config(),
        start_date="20260501",
        end_date="20260531",
    )

    assert len(result.qualified) == 4
    assert result.qualified.groupby("trade_date").size().eq(2).all()
    assert len(result.observations) == 10
    assert result.observations.groupby("trade_date").size().eq(5).all()
    assert set(result.qualified["ts_code"]).isdisjoint(
        result.observations["ts_code"]
    )
    assert result.qualified["candidate_cohort"].eq("QUALIFIED").all()
    assert result.observations["candidate_cohort"].eq("OBSERVATION").all()


def test_v40_selection_does_not_use_future_truth() -> None:
    source = two_day_frame()
    first = evaluate_v40_fixed_1430(
        source,
        V3Config(),
        start_date="20260501",
        end_date="20260531",
    )
    changed = source.copy()
    changed["net_return_pct"] = -changed["net_return_pct"] * 100.0
    changed["entry_fillable"] = ~changed["entry_fillable"]
    changed["exit_fillable"] = ~changed["exit_fillable"]
    second = evaluate_v40_fixed_1430(
        changed,
        V3Config(),
        start_date="20260501",
        end_date="20260531",
    )

    identity = ["trade_date", "signal_slot", "ts_code"]
    assert first.qualified[identity].equals(second.qualified[identity])
    assert first.observations[identity].equals(second.observations[identity])


def test_v40_zero_qualified_is_valid_and_does_not_lower_gate() -> None:
    source = pd.DataFrame(
        [
            row(
                "20260504",
                f"O{index}",
                probability=0.50 - index * 0.01,
            )
            for index in range(6)
        ]
    )
    result = evaluate_v40_fixed_1430(
        source,
        V3Config(),
        start_date="20260504",
        end_date="20260504",
    )

    assert result.qualified.empty
    assert len(result.observations) == V40Policy().observation_count
    assert (
        result.summary["interpretation"]["conclusion"]
        == "NO_QUALIFIED_EVENTS"
    )


def test_v40_reports_incomplete_source_and_truth_honestly() -> None:
    result = evaluate_v40_fixed_1430(
        two_day_frame(),
        V3Config(),
        start_date="20260501",
        end_date="20260731",
    )

    assert result.summary["status"] == "INCOMPLETE"
    assert not result.summary["source"]["source_range_complete"]
    assert not result.summary["source"]["truth_range_complete"]
    assert (
        result.summary["interpretation"]["conclusion"]
        == "INCOMPLETE_EVIDENCE"
    )


def test_v40_observation_shortfall_is_integrity_failure() -> None:
    source = pd.DataFrame(
        [
            row("20260504", "Q"),
            row("20260504", "O1", probability=0.50),
            row("20260504", "O2", probability=0.49),
        ]
    )
    result = evaluate_v40_fixed_1430(
        source,
        V3Config(),
        start_date="20260504",
        end_date="20260504",
    )

    assert result.summary["status"] == "INCOMPLETE"
    assert result.summary["integrity"]["observation_incomplete_days"] == [
        "20260504"
    ]


def test_live_and_retrospective_use_the_same_observation_ranking() -> None:
    config = V3Config()
    source = two_day_frame().loc[
        lambda frame: frame["trade_date"].eq("20260504")
        & frame["signal_slot"].eq("14:30")
    ].copy()
    historical = evaluate_v40_fixed_1430(
        source,
        config,
        start_date="20260504",
        end_date="20260504",
    )
    live_scored = attach_v40_policy_gates(source, V40Policy())
    live_scored["passes_freshness"] = True
    live_scored["model_fingerprint"] = "v40-test"
    live = select_live_cohorts(live_scored, config)

    assert live.observations["ts_code"].tolist() == (
        historical.observations["ts_code"].tolist()
    )


def test_live_observations_never_include_stale_market_data() -> None:
    config = V3Config()
    source = pd.DataFrame(
        [
            row("20260504", f"O{index}", probability=0.53 - index * 0.01)
            for index in range(6)
        ]
    )
    live_scored = attach_v40_policy_gates(source, V40Policy())
    live_scored["passes_freshness"] = True
    live_scored.loc[live_scored["ts_code"].eq("O0"), "passes_freshness"] = False
    live_scored["model_fingerprint"] = "v40-test"
    live = select_live_cohorts(live_scored, config)

    assert len(live.observations) == config.strategy.observation_count
    assert "O0" not in set(live.observations["ts_code"])
