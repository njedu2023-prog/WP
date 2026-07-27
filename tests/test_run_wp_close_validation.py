import pandas as pd

from scripts.run_wp_close_validation import _pending_due_count, _truth_state


def test_close_validation_state_tracks_only_truth_changes(tmp_path):
    path = tmp_path / "validation.csv"
    pd.DataFrame(
        [
            {
                "plan_trade_date": "20260720",
                "target_trade_date": "20260721",
                "ts_code": "600001.SH",
                "truth_status": "pending",
                "return_close_pct": "",
            }
        ]
    ).to_csv(path, index=False)

    before = _truth_state(path)
    assert _pending_due_count(path, "20260721") == 1

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.loc[0, "truth_error"] = "truth not ready"
    frame.to_csv(path, index=False)
    assert _truth_state(path) == before

    frame.loc[0, "truth_status"] = "verified"
    frame.loc[0, "return_close_pct"] = "1.25"
    frame.to_csv(path, index=False)
    assert _truth_state(path) != before
    assert _pending_due_count(path, "20260721") == 0
