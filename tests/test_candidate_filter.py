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


def test_candidate_filter_parses_text_flags_and_rejects_blank_identity():
    df = pd.DataFrame(
        [
            {"ts_code": "000021.SZ", "name": "正常股份", "close": 10.9, "pre_close": 10, "pct_chg": 9, "amount": 200000000, "pre_day_limitup": "false", "today_limitup": "0"},
            {"ts_code": "000022.SZ", "name": "昨日涨停", "close": 10.9, "pre_close": 10, "pct_chg": 9, "amount": 200000000, "pre_day_limitup": "true", "today_limitup": "false"},
            {"ts_code": "000023.SZ", "name": "今日涨停", "close": 10.9, "pre_close": 10, "pct_chg": 9, "amount": 200000000, "pre_day_limitup": "否", "today_limitup": "是"},
            {"ts_code": "159999.SZ", "name": "", "close": 10.9, "pre_close": 10, "pct_chg": 9, "amount": 200000000, "pre_day_limitup": 0, "today_limitup": 0},
        ]
    )
    out = filter_candidates(df)
    assert out["ts_code"].tolist() == ["000021.SZ"]


def test_candidate_filter_excludes_rules_above_ten_percent_not_current_gain():
    common = {
        "trade_date": "20260717",
        "close": 10.9,
        "pre_close": 10.0,
        "pct_chg": 9.0,
        "amount": 200000000,
        "pre_day_limitup": 0,
        "today_limitup": 0,
    }
    df = pd.DataFrame(
        [
            {**common, "ts_code": "000101.SZ", "name": "主板甲", "today_limit_up_price": 11.0},
            {**common, "ts_code": "600101.SH", "name": "主板乙", "today_limit_up_price": 11.01},
            {**common, "ts_code": "300101.SZ", "name": "创业板", "today_limit_up_price": 12.0},
            {**common, "ts_code": "688101.SH", "name": "科创板", "today_limit_up_price": 12.0},
            {**common, "ts_code": "832101.BJ", "name": "北交所", "today_limit_up_price": 13.0},
            {**common, "ts_code": "000102.SZ", "name": "ST低限", "today_limit_up_price": 10.5},
        ]
    )

    out = filter_candidates(df, exclude_st=False)

    assert out["ts_code"].tolist() == ["000101.SZ", "000102.SZ", "600101.SH"]
    assert out.set_index("ts_code")["limit_rule_pct"].to_dict() == {
        "000101.SZ": 10.0,
        "000102.SZ": 5.0,
        "600101.SH": 10.0,
    }


def test_candidate_filter_does_not_treat_main_board_tick_rounding_as_higher_rule():
    df = pd.DataFrame(
        [
            {
                "ts_code": "000111.SZ",
                "name": "低价主板",
                "close": 0.88,
                "pre_close": 0.81,
                "pct_chg": 8.64,
                "amount": 200000000,
                "today_limit_up_price": 0.90,
                "pre_day_limitup": 0,
                "today_limitup": 0,
            }
        ]
    )

    out = filter_candidates(df)

    assert out["ts_code"].tolist() == ["000111.SZ"]
    assert out.iloc[0]["limit_rule_observed_pct"] > 10.5
    assert out.iloc[0]["limit_rule_pct"] == 10.0
