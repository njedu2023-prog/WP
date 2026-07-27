from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from wp.v3.contracts import V3Config, session_phase, validate_contract


CN_TZ = ZoneInfo("Asia/Shanghai")


def test_signal_window_and_closed_state_are_immutable():
    config = V3Config()
    assert session_phase(datetime(2026, 7, 27, 14, 20, tzinfo=CN_TZ), config) == "SIGNAL"
    assert session_phase(datetime(2026, 7, 27, 14, 50, tzinfo=CN_TZ), config) == "SIGNAL"
    assert session_phase(datetime(2026, 7, 27, 14, 51, tzinfo=CN_TZ), config) == "NO_NEW_SIGNAL"
    assert session_phase(datetime(2026, 7, 27, 15, 0, tzinfo=CN_TZ), config) == "CLOSED"


def test_shadow_period_cannot_be_reduced_below_150_days():
    config = V3Config(
        promotion=replace(V3Config().promotion, minimum_shadow_trading_days=149)
    )
    with pytest.raises(ValueError, match="150"):
        validate_contract(config)

