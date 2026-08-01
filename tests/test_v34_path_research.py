from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_wp_v34_intraday_path import join_path_features
from wp.v3.v34_intraday_path import (
    V34_INTRADAY_PATH_FEATURE_COLUMNS,
    V34_QUALITY_COLUMNS,
)


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20260723", "20260723"],
            "signal_slot": ["14:20", "14:25"],
            "ts_code": ["600000.SH", "000001.SZ"],
            "fold": [1, 1],
            "signal_price": [10.0, 11.0],
            "net_return_pct": [1.0, -1.0],
        }
    )


def _path() -> pd.DataFrame:
    frame = _source()[
        ["trade_date", "signal_slot", "ts_code", "fold", "signal_price"]
    ].copy()
    for column in V34_QUALITY_COLUMNS:
        if column == "v34_latest_time":
            frame[column] = "2026-07-23T14:20:00"
        else:
            frame[column] = True
    for index, column in enumerate(V34_INTRADAY_PATH_FEATURE_COLUMNS):
        frame[column] = np.arange(len(frame), dtype=float) + index
    return frame


def test_join_path_features_preserves_source_outcomes_and_contract() -> None:
    joined = join_path_features(_source(), _path())
    assert len(joined) == 2
    assert joined["net_return_pct"].tolist() == [1.0, -1.0]
    assert set(V34_INTRADAY_PATH_FEATURE_COLUMNS).issubset(joined.columns)


def test_join_path_features_rejects_fold_or_price_drift() -> None:
    changed_fold = _path()
    changed_fold.loc[0, "fold"] = 2
    with pytest.raises(RuntimeError, match="folds"):
        join_path_features(_source(), changed_fold)
    changed_price = _path()
    changed_price.loc[0, "signal_price"] = 10.01
    with pytest.raises(RuntimeError, match="signal prices"):
        join_path_features(_source(), changed_price)
