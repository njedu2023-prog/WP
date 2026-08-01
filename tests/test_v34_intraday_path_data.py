from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from build_wp_v34_intraday_path_data import (
    audit_probe_feature_parity,
    load_v34_probe,
)
from wp.v3.io import file_sha256
from wp.v3.meta_alpha import IDENTITY_COLUMNS
from wp.v3.v34_intraday_path import (
    SCHEMA_VERSION,
    V34_INTRADAY_PATH_FEATURE_COLUMNS,
    V34_QUALITY_COLUMNS,
)


def _features() -> pd.DataFrame:
    rows = []
    for index, trade_date in enumerate(("20250723", "20260723")):
        row = {
            "trade_date": trade_date,
            "signal_slot": "14:20",
            "ts_code": f"{600000 + index:06d}.SH",
            "fold": index + 1,
            "signal_price": 10.0 + index,
            "v34_observed_rows": 201,
            "v34_expected_rows": 201,
            "v34_coverage_ratio": 1.0,
            "v34_latest_time": (
                f"{trade_date[:4]}-{trade_date[4:6]}-"
                f"{trade_date[6:]}T14:20:00"
            ),
            "v34_causal_ok": True,
            "v34_signal_price_error_bps": 0.0,
            "v34_signal_price_parity_ok": True,
            "v34_path_complete": True,
        }
        row.update(
            {
                column: float(index + feature_index / 100.0)
                for feature_index, column in enumerate(
                    V34_INTRADAY_PATH_FEATURE_COLUMNS
                )
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_probe_parity_accepts_exact_reproduction_and_rejects_drift() -> None:
    features = _features()
    audit = audit_probe_feature_parity(features, features.copy())
    assert audit["passed"]
    changed = features.copy()
    changed.loc[0, V34_INTRADAY_PATH_FEATURE_COLUMNS[0]] += 0.001
    rejected = audit_probe_feature_parity(changed, features)
    assert not rejected["passed"]
    assert not rejected["numeric_features_match"]


def test_probe_loader_verifies_manifest_and_digest(tmp_path) -> None:
    frame = _features()
    feature_path = tmp_path / "wp_v34_probe_intraday_path_features.parquet"
    frame.to_parquet(feature_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "profit_outcomes_read": False,
        "full_backfill_authorized": True,
        "probe_dates": sorted(frame["trade_date"].unique()),
        "artifacts": {
            "features": {
                "sha256": file_sha256(feature_path),
            }
        },
    }
    manifest_path = tmp_path / "wp_v34_intraday_path_probe.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded, source = load_v34_probe(tmp_path)
    assert len(loaded) == len(frame)
    assert source["source_integrity"]

    manifest["artifacts"]["features"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        load_v34_probe(tmp_path)


def test_probe_projection_contains_only_frozen_columns() -> None:
    expected = {
        *IDENTITY_COLUMNS,
        "fold",
        "signal_price",
        *V34_QUALITY_COLUMNS,
        *V34_INTRADAY_PATH_FEATURE_COLUMNS,
    }
    assert expected == set(_features().columns)
    assert np.isfinite(
        _features()[list(V34_INTRADAY_PATH_FEATURE_COLUMNS)].to_numpy()
    ).all()
