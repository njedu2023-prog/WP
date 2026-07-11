import pandas as pd

from wp.feature_engineering import add_feature_scores
from wp.limitup_probability import add_limitup_probability
from wp.risk_penalty import add_risk_penalty
from wp.scoring_model import add_scores


def test_probability_range():
    df = pd.DataFrame([{"ts_code": "000001.SZ", "name": "A", "open": 10.2, "high": 10.9, "low": 10.1, "close": 10.6, "pre_close": 10, "pct_chg": 8.1, "amount": 200000000, "sector_name": "测试", "pre_day_limitup": 0, "today_limitup": 0}])
    out = add_scores(add_feature_scores(df))
    assert 0 <= out.loc[0, "p_limitup_t1"] <= 100
    assert out.loc[0, "signal_level"] in {"S级", "A级", "B级", "C级", "D级"}


def test_high_open_pullback_raises_risk():
    strong = pd.DataFrame([{"ts_code": "A", "name": "A", "open": 10.2, "high": 10.9, "low": 10.1, "close": 10.85, "pre_close": 10, "pct_chg": 8.5, "amount": 300000000, "amount_ratio_5d": 2, "volume_ratio": 2, "sector_name": "测试", "sector_rank": 5}])
    weak = pd.DataFrame([{"ts_code": "B", "name": "B", "open": 10.8, "high": 10.9, "low": 10.0, "close": 10.15, "pre_close": 10, "pct_chg": 8.2, "amount": 300000000, "amount_ratio_5d": 6, "volume_ratio": 5, "sector_name": "测试", "sector_rank": 5}])
    strong_out = add_scores(add_feature_scores(strong))
    weak_out = add_scores(add_feature_scores(weak))
    assert strong_out.loc[0, "risk_penalty_score"] < weak_out.loc[0, "risk_penalty_score"]
    assert weak_out.loc[0, "high_open_low_walk_flag"] == 1


def test_probability_is_monotonic_and_realistically_bounded():
    frame = pd.DataFrame(
        [
            {"ranking_score": 30, "feature_coverage": 100},
            {"ranking_score": 85, "feature_coverage": 100},
        ]
    )
    out = add_limitup_probability(frame)
    assert 0.2 <= out.loc[0, "p_limitup_t1"] < out.loc[1, "p_limitup_t1"] <= 35


def test_unknown_sector_rank_is_not_treated_as_rear_sector():
    frame = pd.DataFrame(
        [{"pct_chg": 9, "amount": 300000000, "sector_rank": 99, "close_position": 100, "ret_20d": 10, "sector_strength_score": 60, "stock_strength_score": 60}]
    )
    out = add_risk_penalty(frame)
    assert out.loc[0, "risk_rear_sector"] == 0


def test_realtime_fallback_reduces_probability_and_confidence():
    base = {"ts_code": "A", "name": "A", "open": 10.1, "high": 10.9, "low": 10.0, "close": 10.8, "pre_close": 10, "pct_chg": 8.5, "amount": 300000000, "amount_ratio_5d": 2, "volume_ratio": 2, "sector_name": "测试", "sector_rank": 5}
    live = add_scores(add_feature_scores(pd.DataFrame([{**base, "realtime_source": "realtime_quote_rt"}])), calibration_enabled=False)
    fallback = add_scores(add_feature_scores(pd.DataFrame([{**base, "realtime_source": "realtime_quote_fallback"}])), calibration_enabled=False)
    assert fallback.loc[0, "p_limitup_t1"] < live.loc[0, "p_limitup_t1"]
    assert fallback.loc[0, "model_confidence"] < live.loc[0, "model_confidence"]
