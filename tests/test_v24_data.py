from __future__ import annotations

import json

import numpy as np
import pandas as pd

from wp.v3.io import file_sha256
from wp.v3.sharding import (
    SHARD_MANIFEST_NAME,
    SHARD_PREDICTIONS_NAME,
    SHARD_SCHEMA_VERSION,
)
from wp.v3.v23_data import SOURCE_SELECTION_COLUMNS
from wp.v3.v24_data import (
    SOURCE_CANDIDATES_PER_SLOT,
    V24_DERIVED_SOURCE_FEATURE_COLUMNS,
    load_v24_source_candidates,
)


def test_v24_source_is_fixed_top_five_and_outcome_blind(tmp_path) -> None:
    rows = []
    for index in range(8):
        row = {column: np.nan for column in SOURCE_SELECTION_COLUMNS}
        row.update(
            {
                "trade_date": "20260724",
                "signal_slot": "14:20",
                "ts_code": f"{600000 + index:06d}.SH",
                "fold": 1,
                "signal_price": 10.0 + index,
                "ret_from_prev_close_pct": 2.0 + index * 0.1,
                "execution_eligible": True,
                "data_age_seconds": 30.0,
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
                "selection_score": 0.90 - index * 0.05,
                "model_version": "v9",
                "model_fingerprint": "model-1",
                "policy_fingerprint": "policy-1",
            }
        )
        rows.append(row)
    prediction_path = tmp_path / SHARD_PREDICTIONS_NAME
    pd.DataFrame(rows).to_parquet(prediction_path, index=False)
    manifest = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "expected_folds": [1],
        "produced_folds": [1],
        "prediction_rows": len(rows),
        "prediction_sha256": file_sha256(prediction_path),
        "dataset_manifest_sha256": "dataset-1",
    }
    (tmp_path / SHARD_MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    candidates, source = load_v24_source_candidates(
        tmp_path,
        evaluation_end="20260724",
        top_per_source=8,
        exploration_per_slot=0,
    )

    assert len(candidates) == SOURCE_CANDIDATES_PER_SLOT
    assert candidates["ts_code"].tolist() == [
        "600000.SH",
        "600001.SH",
        "600002.SH",
        "600003.SH",
        "600004.SH",
    ]
    assert candidates["v20_stock_rank_in_slot"].tolist() == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert source["profit_outcomes_read"] is False
    assert source["candidates_per_slot"] == 5
    assert source["source_integrity"] is True


def test_v24_derived_feature_contract_has_no_outcomes() -> None:
    forbidden = (
        "target",
        "label",
        "truth",
        "future",
        "gross_return",
        "net_return",
        "t1_",
        "exit_",
    )
    assert V24_DERIVED_SOURCE_FEATURE_COLUMNS
    assert not [
        column
        for column in V24_DERIVED_SOURCE_FEATURE_COLUMNS
        if any(token in column.lower() for token in forbidden)
    ]
