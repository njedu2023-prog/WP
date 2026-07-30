from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wp.v3.exit_risk import (
    day_equal_weights,
    exit_failure_target,
    fit_exit_failure_risk,
)


def _risk_frame(rows: int, start: str) -> pd.DataFrame:
    rng = np.random.default_rng(20260730 + rows)
    dates = pd.date_range(start, periods=max(rows // 50, 1), freq="B")
    trade_dates = np.resize(dates.strftime("%Y%m%d").to_numpy(), rows)
    pressure = rng.normal(0.0, 1.0, rows)
    failure = pressure > 1.35
    return pd.DataFrame(
        {
            "trade_date": trade_dates,
            "signal_slot": np.resize(
                np.array(["14:20", "14:30", "14:40", "14:50"]),
                rows,
            ),
            "ts_code": [f"{index:06d}.SH" for index in range(rows)],
            "ret_from_prev_close_pct": pressure * 2.0,
            "distance_to_down_limit_pct": 5.0 - pressure,
            "tail_max_drawdown_pct": -pressure.clip(min=0.0),
            "market_return_pct": rng.normal(0.0, 0.5, rows),
            "exit_fillable": ~failure,
        }
    )


def test_exit_failure_bundle_scores_higher_risk_for_failure_pattern() -> None:
    train = _risk_frame(6_000, "2023-01-02")
    calibration = _risk_frame(1_500, "2024-01-02")
    bundle = fit_exit_failure_risk(
        train,
        calibration,
        random_seed=20260730,
    )
    scored = bundle.predict(calibration)
    target = exit_failure_target(calibration)
    failed_mean = scored.loc[
        target.eq(1),
        "risk_p_exit_failure",
    ].mean()
    safe_mean = scored.loc[
        target.eq(0),
        "risk_p_exit_failure",
    ].mean()
    assert failed_mean > safe_mean
    assert scored["risk_p_exit_failure"].between(0.0, 1.0).all()
    assert scored["risk_failure_rank_pct"].between(0.0, 1.0).all()


def test_day_equal_weights_give_each_day_equal_total() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240102", "20240103"],
        }
    )
    weights = day_equal_weights(frame)
    totals = pd.Series(weights).groupby(frame["trade_date"]).sum()
    assert totals.iloc[0] == pytest.approx(totals.iloc[1])


def test_exit_failure_model_rejects_small_windows() -> None:
    frame = _risk_frame(200, "2024-01-02")
    with pytest.raises(ValueError, match="insufficient rows"):
        fit_exit_failure_risk(
            frame,
            frame,
            random_seed=20260730,
        )
