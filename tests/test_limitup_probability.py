import pandas as pd

from wp.feature_engineering import add_feature_scores
from wp.scoring_model import add_scores


def test_probability_range():
    df = pd.DataFrame([{"ts_code": "000001.SZ", "name": "A", "close": 10.6, "pre_close": 10, "pct_chg": 6.0, "amount": 200000000, "sector_name": "测试", "pre_day_limitup": 0, "today_limitup": 0}])
    out = add_scores(add_feature_scores(df))
    assert 0 <= out.loc[0, "p_limitup_t1"] <= 100
    assert out.loc[0, "signal_level"] in {"S级", "A级", "B级", "C级", "D级"}
