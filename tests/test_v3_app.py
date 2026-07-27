from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from wp.v3.app import _refresh_data_age


def test_data_age_is_computed_per_symbol_bar_not_global_latest_bar():
    frame = pd.DataFrame(
        {
            "ts_code": ["600001.SH", "600002.SH"],
            "slot_bar_time": [
                "2026-07-27 14:20:00",
                "2026-07-27 14:10:00",
            ],
        }
    )
    result = _refresh_data_age(
        frame,
        current=datetime(2026, 7, 27, 14, 22, tzinfo=ZoneInfo("Asia/Shanghai")),
        market_time=pd.Timestamp("2026-07-27 14:20:00"),
    )
    assert result["data_age_seconds"].tolist() == [120.0, 720.0]
