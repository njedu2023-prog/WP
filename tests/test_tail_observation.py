import pandas as pd

from wp.tail_observation import update_tail_observation


def _health(market_time: str, status: str = "ok") -> dict:
    return {
        "status": status,
        "load_ok": True,
        "data_trade_date": "20260717",
        "market_data_time": market_time,
    }


def _candidate(code: str, score: float, *, eligible: bool = True) -> dict:
    return {
        "ts_code": code,
        "name": f"股票{code[:2]}",
        "price": 10.9,
        "pct_chg": 9.0,
        "sector_name": "测试板块",
        "limit_rule_pct": 10.0,
        "limit_up_pct": 10.0,
        "tail_profit_score": score,
        "tail_profit_eligible": eligible,
        "tail_profit_filter_reason": "风险分过高" if not eligible else "",
        "risk_penalty_score": 20.0,
        "amount_ratio_5d": 1.8,
        "p_limitup_t1": 12.0,
        "wp_score": 80.0,
        "model_confidence": 85.0,
        "pre_day_limitup": 0,
        "today_limitup": 0,
        "is_st": False,
        "suspended_flag": 0,
        "delist_flag": 0,
        "data_quality_flag": 0,
    }


def _plan(code: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": code,
                "confirm_before_buy": "守8%、承接稳",
                "reject_if": "破8%/急回落",
                "buy_reason": "质量领先",
            }
        ]
    )


def test_tail_observation_accumulates_primaries_and_reranks(tmp_path):
    state_path = tmp_path / "wp_tail_observation.csv"
    first = pd.DataFrame([_candidate("000001.SZ", 82.0)])
    result = update_tail_observation(
        first,
        _plan("000001.SZ"),
        first,
        _health("2026-07-17 14:20:00"),
        state_path,
    )
    assert result.table["ts_code"].tolist() == ["000001.SZ"]

    second = pd.DataFrame(
        [
            _candidate("000001.SZ", 86.0),
            _candidate("000002.SZ", 92.0),
        ]
    )
    result = update_tail_observation(
        second,
        _plan("000002.SZ"),
        second,
        _health("2026-07-17 14:25:00"),
        state_path,
    )

    assert result.table["ts_code"].tolist() == ["000002.SZ", "000001.SZ"]
    assert result.table["observation_status"].tolist() == ["当前主票", "观察票"]
    assert result.table["quality_rank"].tolist() == [1, 2]
    assert result.table.iloc[1]["first_seen"] == "2026-07-17 14:20:00"


def test_tail_observation_removes_lost_qualification_after_two_fresh_checks(tmp_path):
    state_path = tmp_path / "wp_tail_observation.csv"
    active = pd.DataFrame([_candidate("000001.SZ", 82.0)])
    update_tail_observation(
        active,
        _plan("000001.SZ"),
        active,
        _health("2026-07-17 14:20:00"),
        state_path,
    )
    invalid = pd.DataFrame([_candidate("000001.SZ", 75.0, eligible=False)])

    first_check = update_tail_observation(
        invalid,
        pd.DataFrame(),
        invalid,
        _health("2026-07-17 14:25:00"),
        state_path,
    )
    assert first_check.table["qualification_status"].tolist() == ["资格复核"]

    second_check = update_tail_observation(
        invalid,
        pd.DataFrame(),
        invalid,
        _health("2026-07-17 14:30:00"),
        state_path,
    )
    assert second_check.table.empty


def test_tail_observation_marks_limitup_instead_of_erasing_history(tmp_path):
    state_path = tmp_path / "wp_tail_observation.csv"
    active = pd.DataFrame([_candidate("000001.SZ", 82.0)])
    update_tail_observation(
        active,
        _plan("000001.SZ"),
        active,
        _health("2026-07-17 14:20:00"),
        state_path,
    )
    sealed = _candidate("000001.SZ", 82.0, eligible=False)
    sealed["today_limitup"] = 1

    result = update_tail_observation(
        pd.DataFrame([sealed]),
        pd.DataFrame(),
        pd.DataFrame([sealed]),
        _health("2026-07-17 14:25:00"),
        state_path,
    )

    assert result.table["observation_status"].tolist() == ["已封板"]
    assert result.table["qualification_reason"].tolist() == ["已涨停，停止新买入"]


def test_tail_observation_preserves_state_during_data_failure(tmp_path):
    state_path = tmp_path / "wp_tail_observation.csv"
    active = pd.DataFrame([_candidate("000001.SZ", 82.0)])
    initial = update_tail_observation(
        active,
        _plan("000001.SZ"),
        active,
        _health("2026-07-17 14:20:00"),
        state_path,
    )

    result = update_tail_observation(
        pd.DataFrame(),
        _plan("000002.SZ"),
        pd.DataFrame(),
        _health("2026-07-17 14:25:00", status="数据异常"),
        state_path,
    )

    assert result.summary["status"] == "preserved_unverified"
    assert result.table.to_dict(orient="records") == initial.table.to_dict(orient="records")


def test_tail_observation_never_adds_above_ten_percent_rule(tmp_path):
    state_path = tmp_path / "wp_tail_observation.csv"
    growth = _candidate("300001.SZ", 95.0)
    growth["limit_rule_pct"] = 20.0
    growth["limit_up_pct"] = 20.0

    result = update_tail_observation(
        pd.DataFrame([growth]),
        _plan("300001.SZ"),
        pd.DataFrame([growth]),
        _health("2026-07-17 14:20:00"),
        state_path,
    )

    assert result.table.empty


def test_tail_observation_recovers_today_primaries_from_validation_history(tmp_path):
    state_path = tmp_path / "wp_tail_observation.csv"
    current = pd.DataFrame(
        [
            _candidate("000001.SZ", 86.0),
            _candidate("000002.SZ", 92.0),
        ]
    )
    history = pd.DataFrame(
        [
            {
                "plan_trade_date": "20260717",
                "plan_time": "2026-07-17 14:25:00",
                "ts_code": "000001.SZ",
                "name": "历史主票",
                "tail_profit_score": 82.0,
            },
            {
                "plan_trade_date": "20260716",
                "plan_time": "2026-07-16 14:25:00",
                "ts_code": "000003.SZ",
                "name": "昨日主票",
                "tail_profit_score": 99.0,
            },
        ]
    )

    result = update_tail_observation(
        current,
        _plan("000002.SZ"),
        current,
        _health("2026-07-17 14:35:00"),
        state_path,
        historical_primaries=history,
    )

    assert result.table["ts_code"].tolist() == ["000002.SZ", "000001.SZ"]
    recovered = result.table.set_index("ts_code").loc["000001.SZ"]
    assert recovered["first_seen"] == "2026-07-17 14:25:00"
    assert recovered["observation_status"] == "观察票"


def test_tail_observation_freezes_after_1450_without_adding_a_new_primary(tmp_path):
    state_path = tmp_path / "wp_tail_observation.csv"
    first = pd.DataFrame([_candidate("000001.SZ", 82.0)])
    initial = update_tail_observation(
        first,
        _plan("000001.SZ"),
        first,
        _health("2026-07-17 14:45:00"),
        state_path,
    )
    later = pd.DataFrame([_candidate("000002.SZ", 99.0)])

    result = update_tail_observation(
        later,
        _plan("000002.SZ"),
        later,
        _health("2026-07-17 14:55:00"),
        state_path,
    )

    assert result.summary["status"] == "frozen_after_tail_window"
    assert result.table.to_dict(orient="records") == initial.table.to_dict(orient="records")


def test_tail_observation_hides_live_pool_at_market_close_without_losing_audit(tmp_path):
    state_path = tmp_path / "wp_tail_observation.csv"
    first = pd.DataFrame([_candidate("000001.SZ", 82.0)])
    update_tail_observation(
        first,
        _plan("000001.SZ"),
        first,
        _health("2026-07-17 14:45:00"),
        state_path,
    )

    result = update_tail_observation(
        first,
        _plan("000001.SZ"),
        first,
        _health("2026-07-17 15:00:00"),
        state_path,
    )

    assert result.summary["status"] == "market_closed"
    assert result.table.empty
    persisted = pd.read_csv(state_path, dtype={"ts_code": str})
    assert persisted["ts_code"].tolist() == ["000001.SZ"]
