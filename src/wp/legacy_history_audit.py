from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .tail_window import accepts_new_tail_primary


AUDIT_VERSION = "legacy_tail_history_audit_v1"
AUDIT_START_DATE = "20260721"
AUDIT_END_DATE = "20260724"
AUDIT_COLUMNS = [
    "audit_version",
    "plan_trade_date",
    "snapshot_count",
    "valid_preclose_snapshot_count",
    "invalid_late_snapshot_count",
    "earliest_snapshot_time",
    "latest_snapshot_time",
    "legacy_action",
    "legacy_candidate_code",
    "legacy_candidate_name",
    "audit_status",
    "audit_reason",
    "formal_strategy_eligible",
]


def _read_decision(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _clock(value: object) -> time | None:
    parsed = pd.to_datetime(str(value or ""), errors="coerce")
    return None if pd.isna(parsed) else parsed.to_pydatetime().time()


def build_legacy_history_audit(
    snapshot_root: Path,
    start_date: str = AUDIT_START_DATE,
    end_date: str = AUDIT_END_DATE,
) -> pd.DataFrame:
    rows: list[dict] = []
    for day in pd.date_range(start_date, end_date, freq="D"):
        trade_date = day.strftime("%Y%m%d")
        snapshots: list[dict] = []
        for path in sorted((snapshot_root / trade_date).glob("*_decision.json")):
            payload = _read_decision(path)
            market_time = str(payload.get("market_data_time") or "")
            decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
            snapshots.append(
                {
                    "market_data_time": market_time,
                    "clock": _clock(market_time),
                    "decision": decision,
                }
            )
        valid = [
            item
            for item in snapshots
            if item["clock"] is not None
            and accepts_new_tail_primary(f"{trade_date} {item['clock']}")
        ]
        chosen = valid[-1] if valid else None
        if chosen is None:
            action = "NO_TRADE"
            code = ""
            name = ""
            status = "missing_valid_preclose_snapshot"
            reason = "没有14:20-14:50可交易时点快照；盘后名单无效，按未交易记账"
        else:
            decision = chosen["decision"]
            raw_action = str(decision.get("action") or "")
            code = str(decision.get("candidate_code") or "")
            name = str(decision.get("candidate_name") or "")
            action = "BUY" if raw_action == "买入" and code else "NO_TRADE"
            status = "valid_legacy_snapshot"
            reason = (
                "旧版盘中决策可审计，但模型合同不同，只保留为研究历史"
                if action == "BUY"
                else "旧版盘中正式意见为空仓；只保留为研究历史"
            )
        times = [item["market_data_time"] for item in snapshots if item["market_data_time"]]
        rows.append(
            {
                "audit_version": AUDIT_VERSION,
                "plan_trade_date": trade_date,
                "snapshot_count": len(snapshots),
                "valid_preclose_snapshot_count": len(valid),
                "invalid_late_snapshot_count": len(snapshots) - len(valid),
                "earliest_snapshot_time": min(times) if times else "",
                "latest_snapshot_time": max(times) if times else "",
                "legacy_action": action,
                "legacy_candidate_code": code,
                "legacy_candidate_name": name,
                "audit_status": status,
                "audit_reason": reason,
                "formal_strategy_eligible": False,
            }
        )
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def summarize_legacy_history_audit(table: pd.DataFrame) -> dict:
    if table is None or table.empty:
        return {
            "version": AUDIT_VERSION,
            "days": 0,
            "valid_legacy_days": 0,
            "missing_valid_preclose_days": 0,
        }
    statuses = table["audit_status"].fillna("").astype(str)
    return {
        "version": AUDIT_VERSION,
        "days": int(len(table)),
        "valid_legacy_days": int(statuses.eq("valid_legacy_snapshot").sum()),
        "missing_valid_preclose_days": int(statuses.eq("missing_valid_preclose_snapshot").sum()),
        "formal_strategy_eligible_days": 0,
    }
