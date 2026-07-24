import json
from datetime import datetime

import pandas as pd

import wp.buy_validation as buy_validation
from wp.buy_validation import VALIDATION_COLUMNS
from wp.calendar import CN_TZ
from wp.close_validation import run_close_validation


def test_close_validation_settles_pending_record_and_renders_report(tmp_path, monkeypatch):
    output_root = tmp_path / "outputs"
    (output_root / "csv").mkdir(parents=True)
    (output_root / "json").mkdir(parents=True)
    pending = pd.DataFrame(
        [
            {
                "plan_trade_date": "20260715",
                "plan_time": "2026-07-15 14:35:00",
                "market_data_time": "2026-07-15 14:35:00",
                "target_trade_date": "20260716",
                "buy_rank": 1,
                "portfolio_group": "主票",
                "ts_code": "688506.SH",
                "name": "百利天恒",
                "plan_price": 100.0,
                "pct_chg_plan": 8.5,
                "buy_model_version": "tail_profit_v1",
                "actual_trade_date": "20260716",
                "truth_status": "pending",
            }
        ]
    ).reindex(columns=VALIDATION_COLUMNS, fill_value="")
    pending.to_csv(output_root / "csv" / "wp_buy_plan_validation.csv", index=False)
    for name in ("wp_top50.csv", "wp_full_rank.csv", "wp_buy_plan.csv"):
        (output_root / "csv" / name).write_text("", encoding="utf-8")
    (output_root / "json" / "wp_data_healthcheck.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "data_time": "2026-07-16 15:10:00",
                "market_data_time": "2026-07-16 15:10:00",
                "data_trade_date": "20260716",
                "expected_trade_date": "20260716",
                "buy_model_version": "tail_profit_v1",
            }
        ),
        encoding="utf-8",
    )
    source = pd.DataFrame([{"ts_code": "688506.SH", "pre_close": 92.17}])
    target = pd.DataFrame(
        [
            {
                "ts_code": "688506.SH",
                "open": 101.0,
                "high": 110.0,
                "low": 98.0,
                "close": 105.0,
                "pre_close": 100.0,
                "pct_chg": 5.0,
                "up_limit": 120.0,
            }
        ]
    )
    monkeypatch.setattr(
        buy_validation,
        "_fetch_truth_by_date",
        lambda date: (source, "") if date == "20260715" else (target, ""),
    )

    summary = run_close_validation(
        output_root,
        datetime(2026, 7, 16, 15, 20, tzinfo=CN_TZ),
    )

    payload = json.loads((output_root / "json" / "wp_buy_plan_validation.json").read_text(encoding="utf-8"))
    record = payload["records"][0]
    assert record["truth_status"] == "verified"
    assert record["actual_close"] == 105.0
    assert record["return_close_pct"] == 5.0
    assert summary["verified_records"] == 1
    assert summary["cumulative_pct_chg"] == 5.0
    page = (output_root / "html_reports" / "latest.html").read_text(encoding="utf-8")
    assert "百利天恒" in page
    assert "+5.00%" in page
    assert "已验证" in page
    assert "15:00已收盘，停止生成尾盘名单" in page
    assert "15:00已收盘，停止生成尾盘名单和新开仓建议" in page
