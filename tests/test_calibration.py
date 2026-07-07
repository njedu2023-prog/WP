import pandas as pd

from wp.calibration import apply_statistical_calibration


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
