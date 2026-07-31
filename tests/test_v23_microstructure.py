from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from wp.v3.v23_microstructure import (
    REQUIRED_RESEARCH_SOURCE_COLUMNS,
    fold_test_window,
    load_evaluation_calendar,
    load_v23_research_source,
    selected_outcome_audit,
)
from wp.v3.io import file_sha256
from wp.v3.sharding import (
    SHARD_MANIFEST_NAME,
    SHARD_PREDICTIONS_NAME,
    SHARD_SCHEMA_VERSION,
)
from wp.v3.v23_data import V23_FEATURE_COLUMNS
from wp.v3.v23_microstructure import (
    MODEL_FEATURES,
    MINIMUM_CALIBRATION_ROWS,
    MINIMUM_TRAIN_ROWS,
    MicrostructurePolicySpec,
    apply_microstructure_policy,
    calibrate_microstructure_policy,
    feature_matrix,
    fit_microstructure_gate,
    rolling_microstructure_segments,
)


def model_frame(rows: int, *, seed: int) -> pd.DataFrame:
    random = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=max(rows // 7 + 1, 30))
    slots = ("14:20", "14:25", "14:30", "14:35", "14:40", "14:45", "14:50")
    frame = pd.DataFrame(
        {
            "trade_date": [
                dates[(index // 7) % len(dates)].strftime("%Y%m%d")
                for index in range(rows)
            ],
            "signal_slot": [slots[index % 7] for index in range(rows)],
            "ts_code": [
                f"{600000 + index % 300:06d}.SH" for index in range(rows)
            ],
            "v23_point_in_time_complete": True,
            "p_round_trip_fill_lower": 0.99,
            "p_severe_loss": 0.20,
        }
    )
    for column in V23_FEATURE_COLUMNS:
        frame[column] = random.normal(0.0, 1.0, rows)
    frame["slot_minute"] = (
        frame["signal_slot"].str[-2:].astype(float) - 20.0
    )
    frame["ret_from_prev_close_pct"] = random.normal(3.0, 2.0, rows)
    frame["p_net_positive_lower"] = random.uniform(0.40, 0.70, rows)
    frame["p_conditional_net_positive"] = random.uniform(0.45, 0.75, rows)
    frame["probability_model_spread"] = random.uniform(0.01, 0.10, rows)
    frame["expected_return_model_spread"] = random.uniform(0.1, 0.8, rows)
    frame["expected_utility_lower_pct"] = random.normal(0.1, 0.3, rows)
    frame["selection_score"] = random.normal(0.5, 0.2, rows)
    signal = (
        0.8 * frame[V23_FEATURE_COLUMNS[1]]
        - 0.4 * frame[V23_FEATURE_COLUMNS[2]]
        + 0.2 * frame["p_net_positive_lower"]
    )
    frame["net_return_pct"] = signal + random.normal(0.0, 1.0, rows)
    frame["label_available"] = True
    frame["target_net_positive"] = frame["net_return_pct"].gt(0.0).astype(int)
    return frame


def test_model_feature_contract_has_no_outcome_columns() -> None:
    assert len(MODEL_FEATURES) >= 40
    assert not [
        column
        for column in MODEL_FEATURES
        if any(
            token in column.lower()
            for token in (
                "target",
                "truth",
                "future",
                "gross_return",
                "net_return",
                "t1_",
            )
        )
    ]


def test_rolling_segments_are_strictly_temporal() -> None:
    dates = pd.bdate_range("2024-01-02", periods=350).strftime("%Y%m%d")
    train, calibration = rolling_microstructure_segments(dates)
    assert len(train) == 252
    assert len(calibration) == 42
    assert train[-1] < calibration[0]
    assert calibration[-1] < dates[-1]


def test_research_source_never_replaces_an_unlabeled_leader(
    tmp_path,
) -> None:
    rows = []
    for index, code in enumerate(("600000.SH", "000001.SZ")):
        row = {
            column: np.nan for column in REQUIRED_RESEARCH_SOURCE_COLUMNS
        }
        row.update(
            {
                "trade_date": "20260724",
                "signal_slot": "14:20",
                "ts_code": code,
                "fold": 1,
                "signal_price": 10.0 + index,
                "ret_from_prev_close_pct": 3.0 + index,
                "execution_eligible": True,
                "p_net_positive": 0.60,
                "p_net_positive_lower": 0.55,
                "p_conditional_net_positive": 0.62,
                "p_cross_section_top": 0.58,
                "p_severe_loss": 0.20,
                "p_round_trip_fill_lower": 0.99,
                "probability_model_spread": 0.05,
                "expected_return_model_spread": 0.10,
                "expected_utility_pct": 0.30,
                "expected_utility_lower_pct": 0.20,
                "selection_score": 0.80 - index * 0.10,
                "model_version": "v9",
                "model_fingerprint": "model-1",
                "policy_fingerprint": "policy-1",
                "label_available": bool(index),
                "target_net_positive": 1.0 if index else np.nan,
                "net_return_pct": 1.0 if index else np.nan,
                "test_start": "20260701",
                "test_end": "20260731",
            }
        )
        rows.append(row)
    shard = tmp_path / "fold-1"
    shard.mkdir()
    prediction_path = shard / SHARD_PREDICTIONS_NAME
    pd.DataFrame(rows).to_parquet(prediction_path, index=False)
    prediction_sha = file_sha256(prediction_path)
    manifest = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "expected_folds": [1],
        "produced_folds": [1],
        "prediction_rows": 2,
        "prediction_sha256": prediction_sha,
        "dataset_manifest_sha256": "dataset-1",
    }
    manifest_path = shard / SHARD_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    data_manifest = {
        "source": {
            "schema_version": "wp_v23_v9_leader_source_1",
            "profit_outcomes_read": False,
            "source_integrity": True,
            "dataset_manifest_sha256": "dataset-1",
            "folds": [1],
            "shards": [
                {
                    "manifest": str(manifest_path.relative_to(tmp_path)),
                    "prediction_sha256": prediction_sha,
                }
            ],
        }
    }
    features = pd.DataFrame(
        {
            "trade_date": ["20260724"],
            "signal_slot": ["14:20"],
            "ts_code": ["600000.SH"],
            "fold": [1],
        }
    )

    loaded, source = load_v23_research_source(
        tmp_path,
        evaluation_end="20260724",
        features=features,
        data_manifest=data_manifest,
    )

    assert loaded["ts_code"].tolist() == ["600000.SH"]
    assert not bool(loaded.iloc[0]["label_available"])
    assert pd.isna(loaded.iloc[0]["net_return_pct"])
    assert source["outcome_blind_identity_selection"] is True


def test_calendar_denominator_includes_no_candidate_days() -> None:
    manifest = {
        "trade_calendar": {
            "open_dates": [
                "20260720",
                "20260721",
                "20260722",
                "20260723",
                "20260724",
            ],
            "open_date_count": 5,
        }
    }

    dates = load_evaluation_calendar(
        manifest,
        start_date="20260721",
        end_date="20260724",
    )

    assert dates == ["20260721", "20260722", "20260723", "20260724"]
    assert fold_test_window(
        pd.DataFrame(
            {
                "test_start": ["20260721", "20260721"],
                "test_end": ["20260724", "20260724"],
            }
        )
    ) == ("20260721", "20260724")


def test_feature_matrix_derives_signal_slot_without_future_data() -> None:
    frame = pd.DataFrame(
        {
            "signal_slot": ["14:20", "14:35", "14:50"],
            "selection_score": [0.1, 0.2, 0.3],
        }
    )

    matrix = feature_matrix(
        frame,
        ("slot_minute", "selection_score"),
    )

    assert matrix["slot_minute"].tolist() == [0.0, 15.0, 30.0]


def test_fixed_sample_floor_rejects_incomplete_earliest_history() -> None:
    assert MINIMUM_TRAIN_ROWS == 1_200
    assert MINIMUM_CALIBRATION_ROWS == 200

    with pytest.raises(ValueError, match="847 train rows"):
        fit_microstructure_gate(
            model_frame(847, seed=11),
            model_frame(294, seed=12),
            random_seed=23,
        )


def test_selected_outcome_audit_rejects_missing_truth() -> None:
    selected = pd.DataFrame(
        {
            "trade_date": ["20260723", "20260724"],
            "label_available": [True, False],
            "net_return_pct": [1.0, np.nan],
            "target_net_positive": [1.0, np.nan],
        }
    )

    audit = selected_outcome_audit(selected, total_days=10)

    assert audit["selected_rows"] == 2
    assert audit["selected_days"] == 2
    assert audit["verified_outcome_rows"] == 1
    assert audit["missing_outcome_rows"] == 1
    assert audit["inconsistent_outcome_rows"] == 0
    assert audit["all_selected_outcomes_verified"] is False


def test_fit_predict_and_fixed_policy_are_deterministic() -> None:
    train = model_frame(1_400, seed=1)
    calibration = model_frame(350, seed=2)
    test = model_frame(280, seed=3)
    bundle = fit_microstructure_gate(
        train,
        calibration,
        random_seed=23,
        minimum_train_rows=1_000,
        minimum_calibration_rows=200,
    )
    scored_calibration = bundle.predict(calibration)
    spec = MicrostructurePolicySpec(
        positive_probability_lower_min=0.001,
        margin_probability_lower_min=0.001,
        severe_probability_upper_max=0.999,
        expected_net_return_lower_min_pct=-100.0,
    )
    policy = calibrate_microstructure_policy(
        scored_calibration,
        calibration_dates=calibration["trade_date"].unique(),
        spec=spec,
    )
    first = apply_microstructure_policy(bundle.predict(test), policy)
    second = apply_microstructure_policy(bundle.predict(test), policy)

    assert bundle.feature_columns
    assert first[
        ["trade_date", "signal_slot", "ts_code", "v23_economic_score"]
    ].equals(
        second[
            ["trade_date", "signal_slot", "ts_code", "v23_economic_score"]
        ]
    )
    assert not first.duplicated(["trade_date", "ts_code"]).any()
    if not first.empty:
        assert first.groupby("trade_date").size().max() <= 3
        assert first["signal_slot"].between("14:20", "14:50").all()
