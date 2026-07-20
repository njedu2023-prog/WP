from pathlib import Path

import pandas as pd

from wp.t1_forecast import build_t1_forecasts, build_training_samples


def _proxy_rows(count: int = 30) -> pd.DataFrame:
    rows = []
    for index in range(count):
        rows.append(
            {
                "backtest_trade_date": f"202601{index + 1:02d}",
                "ts_code": f"600{index:03d}.SH",
                "sector_name": "电力",
                "pct_chg": 8.5 + index % 3 * 0.2,
                "tail_profit_score": 78 + index % 5,
                "risk_penalty_score": 20 + index % 4,
                "amount_ratio_5d": 1.2,
                "next_day_open_pct": -1 + index % 4,
                "next_day_max_pct": 2 + index % 5,
                "next_day_drawdown_pct": -4 + index % 3,
                "next_day_close_pct": -1.5 + index % 5,
            }
        )
    return pd.DataFrame(rows)


def test_training_samples_exclude_future_and_non_ten_percent_boards(tmp_path: Path):
    proxy = _proxy_rows(10)
    proxy.loc[len(proxy)] = {
        **proxy.iloc[0].to_dict(),
        "backtest_trade_date": "20270101",
        "ts_code": "600999.SH",
    }
    proxy.loc[len(proxy)] = {
        **proxy.iloc[0].to_dict(),
        "backtest_trade_date": "20260115",
        "ts_code": "300001.SZ",
    }
    samples = build_training_samples(pd.DataFrame(), tmp_path, "20260201", proxy_samples=proxy)
    assert "20270101" not in set(samples["sample_trade_date"])
    assert "300001.SZ" not in set(samples["ts_code"])


def test_forecast_outputs_consistent_ohlc_quantiles(tmp_path: Path):
    candidate = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "name": "样本",
                "price": 10.0,
                "pct_chg": 8.6,
                "sector_name": "电力",
                "tail_profit_score": 80,
                "risk_penalty_score": 22,
                "amount_ratio_5d": 1.2,
            }
        ]
    )
    result = build_t1_forecasts(
        candidate,
        pd.DataFrame(),
        tmp_path,
        "20260228",
        {"forecast_min_total_samples": 10},
        proxy_samples=_proxy_rows(),
    )
    row = result.table.iloc[0]
    assert row["forecast_mode"] == "日线代理先验"
    for target in ("open", "high", "low", "close"):
        assert row[f"forecast_{target}_q10_pct"] <= row[f"forecast_{target}_q50_pct"]
        assert row[f"forecast_{target}_q50_pct"] <= row[f"forecast_{target}_q90_pct"]
    for suffix in (10, 50, 90):
        assert row[f"forecast_low_q{suffix}_pct"] <= row[f"forecast_open_q{suffix}_pct"]
        assert row[f"forecast_low_q{suffix}_pct"] <= row[f"forecast_close_q{suffix}_pct"]
        assert row[f"forecast_high_q{suffix}_pct"] >= row[f"forecast_open_q{suffix}_pct"]
        assert row[f"forecast_high_q{suffix}_pct"] >= row[f"forecast_close_q{suffix}_pct"]
    assert row["forecast_open_q50_price"] == round(10 * (1 + row["forecast_open_q50_pct"] / 100), 4)
    assert result.summary["manual_decision_support_only"] is True


def test_live_calibration_stops_using_proxy_samples(tmp_path: Path):
    validation = pd.DataFrame(
        [
            {
                "truth_status": "verified",
                "plan_trade_date": f"202601{index + 1:02d}",
                "ts_code": f"601{index:03d}.SH",
                "sector_name": "电力",
                "pct_chg_plan": 8.5,
                "tail_profit_score": 80,
                "risk_penalty_score": 20,
                "amount_ratio_5d": 1.2,
                "return_open_pct": 0.5,
                "return_high_pct": 3.0,
                "return_low_pct": -1.0,
                "return_close_pct": 1.0,
            }
            for index in range(20)
        ]
    )
    candidate = pd.DataFrame(
        [{"ts_code": "600001.SH", "price": 10, "pct_chg": 8.5, "sector_name": "电力", "tail_profit_score": 80, "risk_penalty_score": 20, "amount_ratio_5d": 1.2}]
    )
    result = build_t1_forecasts(
        candidate,
        validation,
        tmp_path,
        "20260228",
        {"forecast_min_total_samples": 10, "forecast_min_live_samples": 10},
        proxy_samples=_proxy_rows(),
    )
    row = result.table.iloc[0]
    assert row["forecast_mode"] == "实时样本校准"
    assert row["forecast_live_sample_count"] == 20
    assert row["forecast_proxy_sample_count"] == 0
