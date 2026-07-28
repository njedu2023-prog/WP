from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from wp.v3.backtest import evaluation_contract_summary, evaluation_window
from wp.v3.contracts import V3Config
from scripts.run_wp_v3_research import _three_year_calendar_summary


def _config() -> V3Config:
    config = V3Config()
    return replace(
        config,
        history=replace(
            config.history,
            start_date="20210101",
            end_date="20261231",
            evaluation_start_date="20230103",
            evaluation_end_date="20260102",
        ),
    )


def test_evaluation_window_excludes_prior_oos_policy_warmup():
    predictions = pd.DataFrame(
        {
            "trade_date": [
                "20221229",
                "20221230",
                "20230103",
                "20260102",
                "20260105",
            ],
            "ts_code": ["600001.SH"] * 5,
        }
    )

    selected = evaluation_window(predictions, _config())

    assert selected["trade_date"].tolist() == ["20230103", "20260102"]


def test_evaluation_summary_reports_warmup_and_evaluation_separately():
    predictions = pd.DataFrame(
        {
            "trade_date": [
                "20221229",
                "20221230",
                "20230103",
                "20230103",
                "20260102",
            ],
            "ts_code": ["600001.SH"] * 5,
        }
    )
    selected = evaluation_window(predictions, _config())

    summary = evaluation_contract_summary(predictions, selected, _config())

    assert summary["evaluation_trade_days"] == 2
    assert summary["evaluation_slot_rows"] == 3
    assert summary["prior_oos_policy_warmup_trade_days"] == 2
    assert summary["prior_oos_policy_warmup_slot_rows"] == 2
    assert summary["all_oos_trade_days"] == 4


def test_evaluation_window_rejects_missing_declared_boundary():
    predictions = pd.DataFrame(
        {
            "trade_date": ["20230104", "20260102"],
            "ts_code": ["600001.SH", "600001.SH"],
        }
    )

    with pytest.raises(RuntimeError, match="complete boundary"):
        evaluation_window(predictions, _config())


def test_research_calendar_requires_three_year_evaluation_not_three_year_panel():
    config = V3Config()
    dates = pd.bdate_range(
        config.history.start_date,
        config.history.end_date,
    )
    calendar = pd.DataFrame({"trade_date": dates.strftime("%Y%m%d")})

    summary = _three_year_calendar_summary(calendar, config)

    assert summary["panel_start_date"] == config.history.start_date
    assert summary["panel_end_date"] == config.history.end_date
    assert (
        summary["evaluation_start_date"]
        == config.history.evaluation_start_date
    )
    assert summary["evaluation_end_date"] == config.history.evaluation_end_date
    assert summary["evaluation_trade_days"] >= 700
    assert summary["panel_trade_days"] > summary["evaluation_trade_days"]
