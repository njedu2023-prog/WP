from __future__ import annotations

import numpy as np
import pandas as pd

from wp.v3.alpha_audit import (
    PolicyThresholds,
    _policy_candidates,
    audit_oos_predictions,
    cohort_metrics,
)


def _predictions(days: int = 155) -> pd.DataFrame:
    rows = []
    for day in range(days):
        trade_date = f"2025{1 + day // 28:02d}{1 + day % 28:02d}"
        for slot in ("14:20", "14:25"):
            for index in range(4):
                positive = index == 3
                rows.append(
                    {
                        "trade_date": trade_date,
                        "signal_slot": slot,
                        "ts_code": f"60000{index}.SH",
                        "net_return_pct": 1.0 if positive else -1.0,
                        "target_net_positive": int(positive),
                        "execution_eligible": True,
                        "p_net_positive": 0.70 if positive else 0.30,
                        "p_net_positive_lower": 0.65 if positive else 0.25,
                        "expected_net_return_pct": 0.8 if positive else -0.8,
                        "downside_q10_pct": -1.0 if positive else -6.0,
                        "ranking_score": float(index),
                        "selection_score": float(index),
                        "selection_rank_pct": (index + 1) / 4.0,
                        "slot_minute": 0 if slot == "14:20" else 5,
                        "ret_5m_pct": float(index),
                    }
                )
    return pd.DataFrame(rows)


def test_policy_candidates_lock_first_signal() -> None:
    frame = _predictions(2)
    candidates = _policy_candidates(
        frame,
        PolicyThresholds(0.60, 0.60, 0.20, -2.0, 0.99),
    )
    assert len(candidates) == 2
    assert set(candidates["signal_slot"]) == {"14:20"}
    assert cohort_metrics(candidates)["win_rate"] == 1.0


def test_oos_audit_has_fixed_150_day_lockbox() -> None:
    audit = audit_oos_predictions(_predictions(), lockbox_days=150)
    assert audit["trade_days"] == 155
    assert audit["development"]["trade_days"] == 5
    assert audit["lockbox"]["trade_days"] == 150
    assert audit["lockbox"]["baseline"]["events"] > 0
    assert np.isfinite(audit["lockbox"]["baseline"]["mean_net_return_pct"])
