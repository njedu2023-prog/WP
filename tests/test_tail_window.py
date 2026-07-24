import pandas as pd

from wp.main import _archive_decision_snapshots
from wp.tail_window import (
    TAIL_PHASE_ACTIVE,
    TAIL_PHASE_BEFORE,
    TAIL_PHASE_CLOSED,
    TAIL_PHASE_FROZEN,
    accepts_new_tail_primary,
    tail_window_phase,
)


def test_tail_window_has_one_shared_generation_boundary():
    assert tail_window_phase("2026-07-24 14:19:59") == TAIL_PHASE_BEFORE
    assert tail_window_phase("2026-07-24 14:20:00") == TAIL_PHASE_ACTIVE
    assert tail_window_phase("2026-07-24 14:50:00") == TAIL_PHASE_ACTIVE
    assert tail_window_phase("2026-07-24 14:50:01") == TAIL_PHASE_FROZEN
    assert tail_window_phase("2026-07-24 14:59:59") == TAIL_PHASE_FROZEN
    assert tail_window_phase("2026-07-24 15:00:00") == TAIL_PHASE_CLOSED
    assert not accepts_new_tail_primary("2026-07-24 15:31:00")


def test_decision_snapshot_is_not_archived_after_close(tmp_path):
    paths = _archive_decision_snapshots(
        tmp_path,
        pd.DataFrame([{"ts_code": "000001.SZ"}]),
        pd.DataFrame(),
        pd.DataFrame(),
        {},
        {
            "market_data_time": "2026-07-24 15:31:00",
            "data_trade_date": "20260724",
        },
    )

    assert paths == []
    assert not (tmp_path / "snapshots").exists()
