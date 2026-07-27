import json

from wp.legacy_history_audit import build_legacy_history_audit, summarize_legacy_history_audit


def _write(path, market_time, action="建议空仓", code=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "market_data_time": market_time,
                "decision": {
                    "action": action,
                    "candidate_code": code,
                    "candidate_name": "甲" if code else "",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_legacy_audit_never_treats_post_close_list_as_trade(tmp_path):
    _write(
        tmp_path / "20260721" / "1445_decision.json",
        "2026-07-21 14:45:00",
        "建议空仓",
    )
    _write(
        tmp_path / "20260722" / "1531_decision.json",
        "2026-07-22 15:31:00",
        "买入",
        "600001.SH",
    )

    table = build_legacy_history_audit(tmp_path)
    day21 = table[table["plan_trade_date"].eq("20260721")].iloc[0]
    day22 = table[table["plan_trade_date"].eq("20260722")].iloc[0]

    assert day21["audit_status"] == "valid_legacy_snapshot"
    assert day21["legacy_action"] == "NO_TRADE"
    assert day22["audit_status"] == "missing_valid_preclose_snapshot"
    assert day22["legacy_action"] == "NO_TRADE"
    assert not bool(day22["formal_strategy_eligible"])
    assert summarize_legacy_history_audit(table)["missing_valid_preclose_days"] == 3
