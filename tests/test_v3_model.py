from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from wp.v3.contracts import V3Config
from wp.v3.dataset import build_supervised_panel
from wp.v3.features import FEATURE_COLUMNS
from wp.v3.model import (
    _deterministic_training_sample,
    predict_bundle,
    train_bundle,
)


def test_temporal_ensemble_trains_and_returns_calibrated_policy_outputs():
    rng = np.random.default_rng(20260727)
    dates = pd.bdate_range("2026-01-01", periods=90)
    rows = []
    for day_index, date in enumerate(dates):
        for stock_index in range(8):
            latent = rng.normal() + 0.02 * day_index + 0.15 * (stock_index % 3)
            signal = 10.0 + stock_index
            t1_return = 0.7 * latent + rng.normal(scale=0.8)
            row = {
                "trade_date": date.strftime("%Y%m%d"),
                "target_trade_date": (date + pd.offsets.BDay(1)).strftime("%Y%m%d"),
                "signal_slot": "14:20",
                "ts_code": f"600{stock_index:03d}.SH",
                "board": "main_board",
                "signal_price": signal,
                "t1_close": signal * (1 + t1_return / 100),
                "listing_days": 500,
                "prev_20d_amount": 300_000_000,
                "slot_amount": 30_000_000,
                "distance_to_up_limit_pct": 3.0,
                "distance_to_down_limit_pct": 12.0,
                "entry_fillable": True,
                "exit_fillable": True,
                "ret_from_prev_close_pct": latent,
                "ret_from_open_pct": latent * 0.7,
                "ret_5m_pct": latent * 0.3,
                "prev_20d_return_pct": latent * 2,
                "market_breadth": 0.5 + 0.1 * np.tanh(latent),
            }
            for feature in FEATURE_COLUMNS:
                row.setdefault(feature, np.nan)
            rows.append(row)
    panel = build_supervised_panel(pd.DataFrame(rows), V3Config())
    base = V3Config()
    config = replace(
        base,
        model=replace(
            base.model,
            calibration_days=20,
            purge_days=2,
            minimum_train_days=40,
            ensemble_windows_days=(30, 60),
            min_train_rows=100,
        ),
    )
    bundle = train_bundle(panel, config)
    prediction = predict_bundle(bundle, panel.tail(32))
    assert prediction["p_net_positive"].between(0, 1).all()
    assert prediction["p_net_positive_lower"].between(0, 1).all()
    assert prediction["expected_net_return_pct"].notna().all()
    assert prediction["passes_policy"].dtype == bool


def test_training_sample_is_capped_and_independent_of_target_values():
    rows = []
    for stock_index in range(20):
        rows.append(
            {
                "trade_date": "20260105",
                "signal_slot": "14:20",
                "ts_code": f"600{stock_index:03d}.SH",
                "target_net_positive": stock_index % 2,
                "net_return_pct": float(stock_index),
            }
        )
    original = pd.DataFrame(rows)
    changed_targets = original.copy()
    changed_targets["target_net_positive"] = 1 - changed_targets["target_net_positive"]
    changed_targets["net_return_pct"] *= -1

    first = _deterministic_training_sample(original, rows_per_slot=7)
    second = _deterministic_training_sample(changed_targets, rows_per_slot=7)

    assert len(first) == 7
    assert first["ts_code"].tolist() == second["ts_code"].tolist()
