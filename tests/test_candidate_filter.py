import pandas as pd

from wp.candidate_filter import filter_candidates


def test_candidate_filter_excludes_limitup_and_prev_limitup():
    df = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "name": "A", "close": 10.6, "pre_close": 10, "pct_chg": 6.0, "amount": 200000000, "pre_day_limitup": 0, "today_limitup": 0},
            {"ts_code": "000002.SZ", "name": "B", "close": 10.8, "pre_close": 10, "pct_chg": 8.0, "amount": 200000000, "pre_day_limitup": 1, "today_limitup": 0},
            {"ts_code": "000003.SZ", "name": "C", "close": 11.0, "pre_close": 10, "pct_chg": 10.0, "amount": 200000000, "pre_day_limitup": 0, "today_limitup": 1},
        ]
    )
    out = filter_candidates(df)
    assert out["ts_code"].tolist() == ["000001.SZ"]
