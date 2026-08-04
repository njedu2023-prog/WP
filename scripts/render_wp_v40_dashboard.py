from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from wp.v3.contracts import load_v3_config
from wp.v3.dashboard import render_v3_dashboard
from wp.v3.ledger import load_shadow_ledger
from wp.v3.registry import load_registry


ROOT = Path(__file__).resolve().parents[1]
CN_TZ = ZoneInfo("Asia/Shanghai")


def main() -> int:
    output = ROOT / "outputs"
    config = load_v3_config(ROOT / "config" / "wp_v3.yml")
    manifest = _read_json(output / "json" / "wp_manifest.json")
    now = datetime.now(CN_TZ)
    manifest.setdefault("source_trade_date", now.strftime("%Y%m%d"))
    manifest.setdefault("session_phase", "CLOSED")
    manifest.setdefault("live_display_allowed", False)
    manifest.setdefault("buy_plan_count", 0)
    manifest.setdefault("health_status", "ok")
    manifest.setdefault("v3_state", "SHADOW_OBSERVATION")
    manifest.setdefault(
        "v3_message",
        "固定 14:00 模型已冻结，等待下一交易日真实影子运行。",
    )
    predictions = _read_csv(
        output / "csv" / "wp_v3_live_predictions.csv"
    )
    render_v3_dashboard(
        output / "html_reports" / "latest.html",
        manifest=manifest,
        predictions=predictions,
        ledger=load_shadow_ledger(
            output / "json" / "wp_v3_candidate_ledger.json"
        ),
        registry=load_registry(
            output / "json" / "wp_model_registry_v3.json"
        ),
        config=config,
        replay=_read_json(
            output / "json" / "wp_v3_historical_replay.json"
        ),
        legacy_audit=_read_json(
            output / "json" / "wp_legacy_history_audit.json"
        ),
        retrospective=_read_json(
            output / "json" / "wp_v41_backtest_202605_202607.json"
        ),
        research_seed=_read_json(
            ROOT / "config" / "wp_v15_frozen_shadow_candidate.json"
        ),
    )
    return 0


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype={"ts_code": str})
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


if __name__ == "__main__":
    raise SystemExit(main())
