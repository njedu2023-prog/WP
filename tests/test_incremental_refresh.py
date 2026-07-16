import pandas as pd

from wp.main import should_rebuild_live_report, source_data_hash


def _input(price: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": "20260716",
                "update_time": "20260716 14:35:00",
                "ts_code": "000001.SZ",
                "price": price,
            }
        ]
    )


def test_unchanged_input_skips_report_rebuild():
    frame = _input()
    rebuild, market_time, data_hash = should_rebuild_live_report(
        frame,
        {"generated_at": "2026-07-16 14:36:00"},
        {
            "market_data_time": "2026-07-16 14:35:00",
            "source_data_hash": source_data_hash(frame),
        },
    )

    assert rebuild is False
    assert market_time == "2026-07-16 14:35:00"
    assert data_hash == source_data_hash(frame)


def test_same_time_with_corrected_values_rebuilds_report():
    previous = _input()
    corrected = _input(price=10.2)

    rebuild, market_time, _ = should_rebuild_live_report(
        corrected,
        {"generated_at": "2026-07-16 14:36:30"},
        {
            "market_data_time": "2026-07-16 14:35:00",
            "source_data_hash": source_data_hash(previous),
        },
    )

    assert rebuild is True
    assert market_time == "2026-07-16 14:35:00"


def test_legacy_manifest_uses_market_time_until_hash_is_available():
    rebuild, market_time, _ = should_rebuild_live_report(
        _input(),
        {},
        {"market_data_time": "2026-07-16 14:35:00"},
    )

    assert rebuild is False
    assert market_time == "2026-07-16 14:35:00"


def test_force_rebuild_overrides_unchanged_input():
    frame = _input()
    rebuild, _, _ = should_rebuild_live_report(
        frame,
        {},
        {"source_data_hash": source_data_hash(frame)},
        force=True,
    )

    assert rebuild is True
