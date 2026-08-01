from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from wp.v3.v34_intraday_path import V34_INTRADAY_PATH_FEATURE_COLUMNS
from wp.v3.v34_path_ranker import (
    DELTA_FEATURES,
    RANK_FEATURES,
    FrozenPathPolicy,
    PathPolicySpec,
    add_path_context_features,
    apply_path_policy,
    labeled_path_rows,
    validate_feature_contract,
    validate_selected_contract,
)


def _base() -> pd.DataFrame:
    rows = []
    for slot_index, slot in enumerate(("14:20", "14:25", "14:30")):
        for stock_index, code in enumerate(("600000.SH", "000001.SZ")):
            row = {
                "trade_date": "20260723",
                "signal_slot": slot,
                "ts_code": code,
                "fold": 1,
                "signal_price": 10.0,
            }
            row.update(
                {
                    column: float(
                        feature_index + slot_index + stock_index / 10.0
                    )
                    for feature_index, column in enumerate(
                        V34_INTRADAY_PATH_FEATURE_COLUMNS
                    )
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_context_features_are_causal_and_cross_sectional() -> None:
    source = _base()
    baseline = add_path_context_features(source)
    changed = source.copy()
    future = changed["signal_slot"].eq("14:30")
    changed.loc[
        future,
        list(V34_INTRADAY_PATH_FEATURE_COLUMNS),
    ] += 1000.0
    replay = add_path_context_features(changed)
    earlier = baseline["signal_slot"].isin(("14:20", "14:25"))
    pdt.assert_frame_equal(
        baseline.loc[
            earlier,
            [*RANK_FEATURES, *DELTA_FEATURES],
        ].reset_index(drop=True),
        replay.loc[
            earlier,
            [*RANK_FEATURES, *DELTA_FEATURES],
        ].reset_index(drop=True),
    )
    assert baseline["v34_candidate_appearance_count"].max() == 3
    assert set(baseline["v34_minutes_since_prior_candidate"]) == {0.0, 5.0}


def test_policy_keeps_first_qualifying_signal_per_stock() -> None:
    frame = add_path_context_features(_base())
    frame["v23_point_in_time_complete"] = True
    frame["v34_path_complete"] = True
    frame["p_round_trip_fill_lower"] = 0.99
    frame["p_severe_loss"] = 0.10
    frame["data_age_seconds"] = 60.0
    frame["v34_p_positive_lower"] = 0.70
    frame["v34_p_margin_lower"] = 0.50
    frame["v34_p_severe_loss_upper"] = 0.10
    frame["v34_expected_net_return_lower_pct"] = 0.50
    frame["v34_positive_model_spread"] = 0.05
    frame["v34_margin_model_spread"] = 0.05
    frame["v34_severe_model_spread"] = 0.05
    frame["v34_expected_return_model_spread_pct"] = 0.10
    frame["v34_path_score"] = np.arange(len(frame), dtype=float)
    policy = FrozenPathPolicy(
        spec=PathPolicySpec(),
        score_threshold=-1.0,
        calibration_start="20260601",
        calibration_end="20260722",
        calibration_days=42,
        eligible_days=42,
    )
    selected = apply_path_policy(frame, policy)
    assert len(selected) == 2
    assert selected["signal_slot"].eq("14:20").all()
    validate_selected_contract(selected, policy)


def test_labeled_rows_require_verified_consistent_complete_outcomes() -> None:
    frame = add_path_context_features(_base().head(3))
    frame["net_return_pct"] = [1.0, -1.0, 1.0]
    frame["target_net_positive"] = [1.0, 0.0, 0.0]
    frame["label_available"] = True
    frame["v23_point_in_time_complete"] = True
    frame["v34_path_complete"] = True
    selected = labeled_path_rows(frame)
    assert len(selected) == 2


def test_feature_contract_rejects_outcome_columns() -> None:
    with pytest.raises(RuntimeError, match="contract violated"):
        validate_feature_contract(
            tuple(
                [
                    *V34_INTRADAY_PATH_FEATURE_COLUMNS,
                    *RANK_FEATURES,
                    "net_return_leak",
                ]
            )
        )
