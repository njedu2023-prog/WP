import pandas as pd

from wp.calibration import _load_history, apply_statistical_calibration


def test_statistical_calibration_uses_history(tmp_path):
    trades_dir = tmp_path / "outputs" / "backtests" / "sample"
    trades_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"p_limitup_t1": 40, "label_t1_limitup": 1}] * 20
        + [{"p_limitup_t1": 40, "label_t1_limitup": 0}] * 10
        + [{"p_limitup_t1": 10, "label_t1_limitup": 0}] * 20
    ).to_csv(trades_dir / "trades.csv", index=False)
    frame = pd.DataFrame([{"p_limitup_t1": 40.0}, {"p_limitup_t1": 10.0}])
    out = apply_statistical_calibration(frame, tmp_path, min_samples=30)
    assert out.loc[0, "calibration_sample_count"] >= 30
    assert out.loc[0, "p_limitup_t1"] > out.loc[1, "p_limitup_t1"]
    assert "p_limitup_t1_raw" in out.columns


def test_calibration_is_rank_preserving(tmp_path):
    trades_dir = tmp_path / "outputs" / "backtests" / "sample"
    trades_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"p_limitup_t1": score, "label_t1_limitup": int(index % 25 == 0)} for index, score in enumerate(range(1, 101))]
    ).to_csv(trades_dir / "trades.csv", index=False)
    frame = pd.DataFrame({"p_limitup_t1": [2.0, 4.0, 8.0, 12.0]})
    out = apply_statistical_calibration(frame, tmp_path, min_samples=80)
    assert out["p_limitup_t1"].is_monotonic_increasing
    assert out["calibration_method"].eq("logit_intercept_v1").all()


def test_history_deduplicates_overlapping_windows_and_excludes_eod_proxy(tmp_path):
    for folder in ["first", "second"]:
        trades_dir = tmp_path / "outputs" / "backtests" / folder
        trades_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {"backtest_trade_date": "20260701", "ts_code": "A", "p_limitup_t1_raw": 5, "label_t1_limitup": 0, "model_version": "wp_rule_v2", "backtest_data_mode": "intraday_1420", "calibration_eligible": True},
                {"backtest_trade_date": "20260701", "ts_code": "B", "p_limitup_t1_raw": 8, "label_t1_limitup": 1, "model_version": "wp_rule_v2", "backtest_data_mode": "eod_proxy", "calibration_eligible": False},
            ]
        ).to_csv(trades_dir / "trades.csv", index=False)
    history = _load_history(tmp_path, model_version="wp_rule_v2", before_date="20260702")
    assert history[["backtest_trade_date", "ts_code"]].drop_duplicates().shape[0] == 1
    assert history["ts_code"].tolist() == ["A"]
