import pandas as pd

from wp.tail_sampling import update_tail_sampling


def _health(market_time: str) -> dict:
    return {
        "status": "ok",
        "load_ok": True,
        "data_trade_date": "20260724",
        "market_data_time": market_time,
    }


def test_tail_sampling_marks_missing_day_after_close_without_fabricating_records(tmp_path):
    result = update_tail_sampling(
        pd.DataFrame(),
        _health("2026-07-24 15:30:00"),
        tmp_path / "wp_tail_sampling.csv",
    )

    row = result.table.iloc[0]
    assert row["plan_trade_date"] == "20260724"
    assert row["target_trade_date"] == "20260727"
    assert row["sample_status"] == "missing"
    assert row["record_count"] == 0


def test_tail_sampling_keeps_captured_status_after_close(tmp_path):
    path = tmp_path / "wp_tail_sampling.csv"
    validation = pd.DataFrame(
        [
            {
                "plan_trade_date": "20260724",
                "target_trade_date": "20260727",
                "plan_time": "2026-07-24 14:45:00",
            }
        ]
    )
    update_tail_sampling(validation, _health("2026-07-24 14:45:00"), path)
    result = update_tail_sampling(validation, _health("2026-07-24 15:30:00"), path)

    assert result.table.iloc[0]["sample_status"] == "captured"
    assert result.table.iloc[0]["record_count"] == 1
