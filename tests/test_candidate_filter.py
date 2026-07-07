import pandas as pd

from wp.candidate_filter import filter_candidates


def test_candidate_filter_excludes_limitup_and_prev_limitup():
    df = pd.DataFrame(
        [
            {"ts_code": "000000.SZ", "name": "Z", "close": 10.8, "pre_close": 10, "pct_chg": 8.0, "amount": 200000000, "pre_day_limitup": 0, "today_limitup": 0},
            {"ts_code": "000001.SZ", "name": "A", "close": 10.81, "pre_close": 10, "pct_chg": 8.1, "amount": 200000000, "pre_day_limitup": 0, "today_limitup": 0},
            {"ts_code": "000002.SZ", "name": "B", "close": 10.9, "pre_close": 10, "pct_chg": 9.0, "amount": 200000000, "pre_day_limitup": 1, "today_limitup": 0},
            {"ts_code": "000003.SZ", "name": "C", "close": 11.0, "pre_close": 10, "pct_chg": 10.0, "amount": 200000000, "pre_day_limitup": 0, "today_limitup": 1},
        ]
    )
    out = filter_candidates(df)
    assert out["ts_code"].tolist() == ["000001.SZ"]


def test_candidate_filter_excludes_new_suspended_delist_and_bad_data():
    df = pd.DataFrame(
        [
            {"ts_code": "000011.SZ", "name": "正常股份", "trade_date": "20260707", "list_date": "20200101", "close": 10.81, "pre_close": 10, "pct_chg": 8.1, "amount": 200000000, "pre_day_limitup": 0, "today_limitup": 0},
            {"ts_code": "000012.SZ", "name": "新股股份", "trade_date": "20260707", "list_date": "20260701", "close": 10.81, "pre_close": 10, "pct_chg": 8.1, "amount": 200000000, "pre_day_limitup": 0, "today_limitup": 0},
            {"ts_code": "000013.SZ", "name": "停牌股份", "trade_date": "20260707", "list_date": "20200101", "close": 10.81, "pre_close": 10, "pct_chg": 8.1, "amount": 0, "pre_day_limitup": 0, "today_limitup": 0},
            {"ts_code": "000014.SZ", "name": "退市股份", "trade_date": "20260707", "list_date": "20200101", "close": 10.81, "pre_close": 10, "pct_chg": 8.1, "amount": 200000000, "pre_day_limitup": 0, "today_limitup": 0},
            {"ts_code": "000015.SZ", "name": "缺字段", "trade_date": "20260707", "list_date": "20200101", "close": 10.81, "pre_close": None, "pct_chg": 8.1, "amount": 200000000, "pre_day_limitup": 0, "today_limitup": 0},
        ]
    )
    out = filter_candidates(df)
    assert out["ts_code"].tolist() == ["000011.SZ"]
