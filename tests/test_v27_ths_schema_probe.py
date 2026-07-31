from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.probe_wp_v27_ths_schema import (
    THS_FIELDS,
    normalize_snapshot_rank,
    normalize_trade_date,
    ths_schema_record,
)


def ths_frame(*, legacy: bool = False) -> pd.DataFrame:
    rows = []
    for index in range(100):
        score = 20_000_000 - index * 100_000
        rows.append(
            {
                "trade_date": "2023-08-25",
                "data_type": "热股",
                "ts_code": f"{index + 1:06d}.SZ",
                "ts_name": f"S{index}",
                "rank": score if legacy else index + 1,
                "pct_change": 2.0,
                "current_price": 10.0,
                "concept": '["示例题材"]',
                "hot": score if legacy else 100_000 - index,
                "rank_time": f"202308251430{index % 60:02d}",
            }
        )
    return pd.DataFrame(rows, columns=THS_FIELDS.split(","))


def test_trade_date_normalization_handles_historical_formats() -> None:
    assert normalize_trade_date("20230825") == "20230825"
    assert normalize_trade_date("2023-08-25") == "20230825"
    assert normalize_trade_date(20230825.0) == "20230825"


def test_modern_ordinal_rank_is_normalized_without_direction_change() -> None:
    frame = ths_frame()
    normalized, audit = normalize_snapshot_rank(frame)

    assert audit["schema_regime"] == "ordinal_zero_or_one_based"
    assert audit["normalized_rank_valid"]
    assert normalized.iloc[0] == 1.0
    assert normalized.iloc[-1] == 100.0


def test_legacy_score_is_allowed_only_when_hot_field_confirms_it() -> None:
    frame = ths_frame(legacy=True)
    normalized, audit = normalize_snapshot_rank(frame)

    assert audit["schema_regime"] == "legacy_hot_score_descending"
    assert audit["rank_hot_spearman"] == pytest.approx(1.0)
    assert audit["normalized_rank_valid"]
    assert normalized.iloc[0] == 1.0
    assert normalized.iloc[-1] == 100.0


def test_legacy_score_rejects_missing_contemporaneous_hot_confirmation() -> None:
    frame = ths_frame(legacy=True)
    frame["hot"] = np.nan
    _, audit = normalize_snapshot_rank(frame)

    assert audit["schema_regime"] == "unusable"
    assert not audit["normalized_rank_valid"]


def test_schema_record_accepts_causal_legacy_snapshot() -> None:
    record = ths_schema_record(
        ths_frame(legacy=True),
        trade_date="20230825",
    )

    assert record["coverage_pass"]
    assert record["date_consistent"]
    assert record["unique_a_share_codes"] == 100
    assert record["normalized_rank_min"] == 1.0
    assert record["normalized_rank_max"] == 100.0


def test_probe_source_does_not_read_profit_outcomes() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "probe_wp_v27_ths_schema.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "gross_return",
        "net_return",
        "profit_label",
        "target_return",
        "t1_close",
    )

    assert not any(token in source for token in forbidden)
