from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from wp.v3.contracts import V3Config, session_phase, validate_contract


CN_TZ = ZoneInfo("Asia/Shanghai")


def test_signal_window_and_closed_state_are_immutable():
    config = V3Config()
    assert session_phase(datetime(2026, 7, 27, 14, 29, tzinfo=CN_TZ), config) == "PRE_SIGNAL"
    assert session_phase(datetime(2026, 7, 27, 14, 30, tzinfo=CN_TZ), config) == "SIGNAL"
    assert session_phase(datetime(2026, 7, 27, 14, 31, tzinfo=CN_TZ), config) == "NO_NEW_SIGNAL"
    assert session_phase(datetime(2026, 7, 27, 14, 40, tzinfo=CN_TZ), config) == "FROZEN"
    assert session_phase(datetime(2026, 7, 27, 15, 0, tzinfo=CN_TZ), config) == "CLOSED"


def test_shadow_period_cannot_be_reduced_below_150_days():
    config = V3Config(
        promotion=replace(V3Config().promotion, minimum_shadow_trading_days=149)
    )
    with pytest.raises(ValueError, match="150"):
        validate_contract(config)


def test_cost_contract_uses_35bps_baseline_and_rejects_lower_stress():
    config = V3Config()
    assert config.execution.baseline_all_in_cost_bps == 35.0
    invalid = V3Config(
        execution=replace(
            config.execution,
            stress_cost_bps=(30.0, 50.0),
        )
    )
    with pytest.raises(ValueError, match="baseline all-in"):
        validate_contract(invalid)


def test_exit_order_enters_the_closing_auction_at_1457():
    config = V3Config()
    assert (
        config.execution.exit_order_contract
        == "T+1_14:57_down_limit_sell_for_close_auction"
    )


def test_retrospective_and_live_shadow_statistics_cannot_overlap():
    config = V3Config(
        evidence=replace(
            V3Config().evidence,
            live_shadow_start_date="20260731",
        )
    )
    with pytest.raises(ValueError, match="must end before"):
        validate_contract(config)
