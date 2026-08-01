from __future__ import annotations

import pandas as pd
import pytest

from wp.v3.contracts import V3Config
from wp.v3.retrospective import hydrate_v40_truth
from wp.v3.v40 import refresh_v40_backtest_summary


def pending(code: str, cohort: str) -> dict[str, object]:
    return {
        "trade_date": "20260731",
        "target_trade_date": "20260803",
        "signal_slot": "14:30",
        "ts_code": code,
        "candidate_cohort": cohort,
        "signal_price": 10.0,
        "entry_benchmark_price": 10.0,
        "entry_price": 10.01,
        "entry_fillable": True,
        "adj_factor": 1.0,
        "label_available": False,
        "net_return_pct": float("nan"),
    }


def truth(
    code: str,
    *,
    close: float,
    down_limit: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": code,
                "close": close,
                "vol": 1_000_000,
                "down_limit": down_limit,
                "adj_factor": 1.0,
            }
        ]
    ).set_index("ts_code")


def test_pending_retrospective_truth_only_updates_outcome() -> None:
    source = pd.DataFrame([pending("600001.SH", "QUALIFIED")])
    hydrated = hydrate_v40_truth(
        source,
        {
            "20260731": truth(
                "600001.SH",
                close=10.0,
                down_limit=9.0,
            ),
            "20260803": truth(
                "600001.SH",
                close=10.5,
                down_limit=9.0,
            ),
        },
        config=V3Config(),
        current_trade_date="20260803",
    )

    assert hydrated.loc[0, "ts_code"] == source.loc[0, "ts_code"]
    assert hydrated.loc[0, "signal_price"] == source.loc[0, "signal_price"]
    assert hydrated.loc[0, "entry_price"] == source.loc[0, "entry_price"]
    assert bool(hydrated.loc[0, "label_available"]) is True
    assert hydrated.loc[0, "net_return_pct"] == pytest.approx(
        (10.5 / 10.01 - 1.0) * 100.0 - 0.25
    )


def test_locked_down_limit_keeps_conservative_nonfill_penalty() -> None:
    source = pd.DataFrame([pending("600001.SH", "OBSERVATION")])
    hydrated = hydrate_v40_truth(
        source,
        {
            "20260731": truth(
                "600001.SH",
                close=10.0,
                down_limit=9.0,
            ),
            "20260803": truth(
                "600001.SH",
                close=9.0,
                down_limit=9.0,
            ),
        },
        config=V3Config(),
        current_trade_date="20260803",
    )

    assert bool(hydrated.loc[0, "exit_fillable"]) is False
    assert hydrated.loc[0, "net_return_pct"] == pytest.approx(-10.0)
    assert hydrated.loc[0, "target_net_positive"] == 0


def test_summary_closes_without_changing_frozen_selection() -> None:
    qualified = pd.DataFrame(
        [
            {
                **pending("600001.SH", "QUALIFIED"),
                "label_available": True,
                "net_return_pct": 1.0,
                "entry_fillable": True,
                "exit_fillable": True,
            }
        ]
    )
    observations = pd.DataFrame(
        [
            {
                **pending(f"00000{index}.SZ", "OBSERVATION"),
                "label_available": True,
                "net_return_pct": -0.1 * index,
                "entry_fillable": True,
                "exit_fillable": True,
            }
            for index in range(1, 6)
        ]
    )
    summary = {
        "status": "INCOMPLETE",
        "evidence_contract": {
            "retrospective_start_date": "20260501",
            "retrospective_end_date": "20260731",
        },
        "source": {
            "source_range_complete": True,
            "truth_range_complete": False,
        },
        "integrity": {"observation_incomplete_days": []},
        "qualified": {},
        "observations": {},
        "monthly": [
            {"month": "202605", "evaluated_trade_days": 20},
            {"month": "202606", "evaluated_trade_days": 20},
            {"month": "202607", "evaluated_trade_days": 23},
        ],
    }

    refreshed = refresh_v40_backtest_summary(
        summary,
        qualified,
        observations,
        V3Config(),
    )

    assert refreshed["status"] == "COMPLETE"
    assert refreshed["source"]["truth_range_complete"] is True
    assert refreshed["qualified"]["metrics"]["events"] == 1
    assert refreshed["observations"]["metrics"]["events"] == 5
    assert [row["month"] for row in refreshed["monthly"]] == [
        "202605",
        "202606",
        "202607",
    ]
