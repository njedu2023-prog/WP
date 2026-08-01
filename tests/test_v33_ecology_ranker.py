from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wp.v3.v33_ecology_ranker import (
    EcologyPolicySpec,
    FrozenEcologyPolicy,
    active_ecology_features,
    apply_ecology_policy,
    calibrate_ecology_policy,
    fit_ecology_ranker,
    policy_eligible_rows,
    validate_feature_contract,
    validate_selected_contract,
)
from wp.v3.v33_limit_ecology import V33_LIMIT_ECOLOGY_FEATURE_COLUMNS


def scored_rows() -> pd.DataFrame:
    rows = [
        {
            "trade_date": "20260723",
            "signal_slot": "14:20",
            "ts_code": "000001.SZ",
            "v33_ecology_score": 0.80,
        },
        {
            "trade_date": "20260723",
            "signal_slot": "14:20",
            "ts_code": "000002.SZ",
            "v33_ecology_score": 0.70,
        },
        {
            "trade_date": "20260723",
            "signal_slot": "14:25",
            "ts_code": "000001.SZ",
            "v33_ecology_score": 0.90,
        },
        {
            "trade_date": "20260723",
            "signal_slot": "14:30",
            "ts_code": "000003.SZ",
            "v33_ecology_score": 0.60,
        },
        {
            "trade_date": "20260723",
            "signal_slot": "14:35",
            "ts_code": "000004.SZ",
            "v33_ecology_score": 0.55,
        },
    ]
    frame = pd.DataFrame(rows)
    frame["v23_point_in_time_complete"] = True
    frame["v33_ecology_features_complete"] = True
    frame["v33_ecology_active_before_signal"] = 1.0
    frame["p_round_trip_fill_lower"] = 0.99
    frame["p_severe_loss"] = 0.10
    frame["v33_p_positive"] = 0.60
    frame["v33_expected_net_return_pct"] = 0.20
    frame["v33_p_severe_loss"] = 0.10
    frame["v33_within_slot_rank_score"] = 0.60
    frame["data_age_seconds"] = 60.0
    return frame


def policy(threshold: float = 0.0) -> FrozenEcologyPolicy:
    return FrozenEcologyPolicy(
        spec=EcologyPolicySpec(
            target_candidate_day_rate=0.25,
            max_candidates_per_day=None,
        ),
        score_threshold=threshold,
        calibration_start="20260601",
        calibration_end="20260722",
        calibration_days=42,
        eligible_days=20,
    )


def test_policy_allows_multiple_candidates_in_same_slot() -> None:
    selected = apply_ecology_policy(scored_rows(), policy())

    assert len(selected) == 4
    assert set(selected["ts_code"]) == {
        "000001.SZ",
        "000002.SZ",
        "000003.SZ",
        "000004.SZ",
    }


def test_policy_preserves_first_qualifying_signal() -> None:
    selected = apply_ecology_policy(scored_rows(), policy())
    row = selected.loc[selected["ts_code"].eq("000001.SZ")].iloc[0]

    assert row["signal_slot"] == "14:20"
    assert row["v33_ecology_score"] == 0.80


def test_policy_eligibility_requires_absolute_edge() -> None:
    frame = scored_rows()
    frame.loc[0, "v33_expected_net_return_pct"] = -0.01
    frame.loc[1, "v33_p_positive"] = 0.49
    frame.loc[2, "v33_p_severe_loss"] = 0.36
    frame.loc[3, "v33_ecology_active_before_signal"] = 0.0
    frame.loc[4, "v33_within_slot_rank_score"] = 0.49

    eligible = policy_eligible_rows(frame)

    assert eligible.empty


def test_policy_rejects_bad_freshness_and_source_risk() -> None:
    frame = scored_rows()
    frame.loc[0, "data_age_seconds"] = np.nan
    frame.loc[1, "data_age_seconds"] = -1.0
    frame.loc[2, "p_round_trip_fill_lower"] = 0.94
    frame.loc[3, "p_severe_loss"] = 0.46
    frame.loc[4, "data_age_seconds"] = 421.0

    eligible = policy_eligible_rows(frame)

    assert eligible.empty


def test_policy_calibration_uses_prior_day_frequency_only() -> None:
    frame = pd.concat(
        [
            scored_rows().assign(
                trade_date=f"202607{day:02d}",
                v33_ecology_score=float(day),
            )
            for day in range(1, 9)
        ],
        ignore_index=True,
    )
    frozen = calibrate_ecology_policy(
        frame,
        calibration_dates=[f"202607{day:02d}" for day in range(1, 9)],
        spec=EcologyPolicySpec(target_candidate_day_rate=0.25),
    )

    assert frozen.score_threshold == 7.0
    assert frozen.calibration_days == 8


def test_active_features_drop_constant_columns() -> None:
    columns = list(V33_LIMIT_ECOLOGY_FEATURE_COLUMNS[:12])
    train = pd.DataFrame(
        {
            column: np.arange(30, dtype=float)
            for column in columns
        }
    )
    calibration = train.copy()
    constant = V33_LIMIT_ECOLOGY_FEATURE_COLUMNS[12]
    train[constant] = 1.0
    calibration[constant] = 1.0

    active = active_ecology_features(train, calibration)

    assert set(columns).issubset(active)
    assert constant not in active
    assert validate_feature_contract(active)


def test_feature_contract_rejects_raw_industry_code() -> None:
    features = tuple(V33_LIMIT_ECOLOGY_FEATURE_COLUMNS[:12]) + (
        "l2_code",
    )

    with pytest.raises(RuntimeError):
        validate_feature_contract(features)


def test_selected_contract_rejects_duplicate_stock_day() -> None:
    selected = scored_rows().iloc[[0, 2]].copy()

    with pytest.raises(RuntimeError):
        validate_selected_contract(selected, policy())


def labeled_rows(days: int) -> pd.DataFrame:
    rows = []
    feature_columns = list(V33_LIMIT_ECOLOGY_FEATURE_COLUMNS[:16])
    for day in range(days):
        for stock in range(4):
            net_return = float((stock - 1.5) * 1.5 + (day % 3) * 0.1)
            row = {
                "trade_date": f"2026{day // 28 + 1:02d}{day % 28 + 1:02d}",
                "signal_slot": "14:20",
                "ts_code": f"{stock:06d}.SZ",
                "net_return_pct": net_return,
                "target_net_positive": float(net_return > 0.0),
                "label_available": True,
                "v23_point_in_time_complete": True,
                "v33_ecology_features_complete": True,
            }
            for offset, column in enumerate(feature_columns):
                row[column] = float(
                    stock + (day % (offset + 2)) / (offset + 2)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_fit_ranker_uses_cleaned_calibration_rows() -> None:
    train = labeled_rows(30)
    calibration = labeled_rows(10)
    calibration.loc[0, "label_available"] = False

    bundle = fit_ecology_ranker(
        train,
        calibration,
        random_seed=17,
        minimum_train_rows=100,
        minimum_calibration_rows=30,
        minimum_train_pair_rows=100,
        minimum_calibration_pair_rows=30,
    )

    assert bundle.train_rows == 120
    assert bundle.calibration_rows == 39
    assert len(bundle.feature_columns) >= 12
