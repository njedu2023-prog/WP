from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from wp.v3.backtest import walk_forward_backtest
from wp.v3.contracts import load_v3_config
from wp.v3.dashboard import render_v3_dashboard
from wp.v3.dataset import audit_panel
from wp.v3.diagnostics import diagnostics_tables
from wp.v3.history import load_panel_partitions
from wp.v3.ledger import empty_shadow_ledger
from wp.v3.model import bundle_metadata, save_bundle, train_bundle
from wp.v3.policy import policy_selection_from_dict
from wp.v3.registry import (
    apply_promotion_decision,
    load_registry,
    model_record,
    register_research_model,
    save_registry,
)
from wp.v3.sharding import load_walk_forward_shards


ROOT = Path(__file__).resolve().parents[1]
CN_TZ = ZoneInfo("Asia/Shanghai")


def _strict_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full nested walk-forward research and register a WP V5 shadow model."
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "wp_v3.yml"))
    parser.add_argument(
        "--panel-dir",
        default=str(ROOT / "artifacts" / "wp_v3_history" / "panel"),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "wp_v3_research"))
    parser.add_argument(
        "--registry",
        default=str(ROOT / "outputs" / "json" / "wp_model_registry_v3.json"),
    )
    parser.add_argument(
        "--shard-dir",
        help="Directory containing complete deterministic walk-forward shards.",
    )
    parser.add_argument(
        "--dataset-manifest",
        default=str(
            ROOT
            / "artifacts"
            / "wp_v3_history"
            / "wp_v3_dataset_manifest.json"
        ),
    )
    parser.add_argument("--max-folds", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_v3_config(args.config)
    panel = load_panel_partitions(args.panel_dir)
    audit = audit_panel(panel)
    if audit.trade_days < 700:
        raise RuntimeError(
            f"three-year research contract requires at least 700 covered trade days; "
            f"received {audit.trade_days}"
        )
    start = pd.Timestamp(str(panel["trade_date"].min()))
    end = pd.Timestamp(str(panel["trade_date"].max()))
    if (end - start).days < 1_000:
        raise RuntimeError(
            f"dataset covers only {(end - start).days} calendar days; three years are required"
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.shard_dir:
        if args.max_folds is not None:
            raise ValueError("--max-folds cannot be used with --shard-dir")
        print(
            f"[wp-v5] validating and aggregating walk-forward shards "
            f"from {args.shard_dir}",
            flush=True,
        )
        backtest = load_walk_forward_shards(
            args.shard_dir,
            panel=panel,
            config=config,
            dataset_manifest_path=args.dataset_manifest,
        )
    else:
        backtest = walk_forward_backtest(
            panel,
            config,
            max_folds=args.max_folds,
        )
    print(
        f"[wp-v5] OOS aggregation complete folds={len(backtest.folds)} "
        f"rows={len(backtest.predictions):,} "
        f"candidates={len(backtest.candidates):,}",
        flush=True,
    )
    bundle = train_bundle(
        panel,
        config,
        policy_selection=policy_selection_from_dict(backtest.final_policy),
    )
    model_path = output / "models" / f"{bundle.fingerprint}.joblib"
    save_bundle(bundle, model_path)
    metadata = bundle_metadata(bundle)

    backtest_path = output / "wp_v3_backtest.json"
    backtest_path.write_text(
        _strict_json(backtest.summary()) + "\n",
        encoding="utf-8",
    )
    metadata_path = output / "wp_v3_model_metadata.json"
    metadata_path.write_text(
        _strict_json(metadata) + "\n",
        encoding="utf-8",
    )
    candidate_path = output / "wp_v3_oos_candidates.csv"
    backtest.candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    diagnostics = backtest.metrics.get("diagnostics", {})
    (output / "wp_v3_prediction_diagnostics.json").write_text(
        _strict_json(diagnostics) + "\n",
        encoding="utf-8",
    )
    for name, table in diagnostics_tables(diagnostics).items():
        table.to_csv(
            output / f"wp_v3_diagnostic_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    replay = _historical_replay(backtest.predictions, backtest.candidates)
    replay_json_path = ROOT / "outputs" / "json" / "wp_v3_historical_replay.json"
    replay_csv_path = ROOT / "outputs" / "csv" / "wp_v3_historical_replay.csv"
    replay_json_path.parent.mkdir(parents=True, exist_ok=True)
    replay_csv_path.parent.mkdir(parents=True, exist_ok=True)
    replay_json_path.write_text(
        _strict_json(replay) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(replay["candidates"]).to_csv(
        replay_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    registry = load_registry(args.registry)
    register_research_model(
        registry,
        metadata=metadata,
        backtest=backtest.metrics,
        artifact_path=str(model_path.relative_to(ROOT)),
    )
    decision = apply_promotion_decision(
        registry,
        bundle.fingerprint,
        config,
        authorize=False,
    )
    registered = model_record(registry, bundle.fingerprint) or {}
    save_registry(registry, args.registry)

    summary = {
        "schema_version": "wp_v5_research_summary_1",
        "dataset": asdict(audit),
        "date_start": str(panel["trade_date"].min()),
        "date_end": str(panel["trade_date"].max()),
        "model": metadata,
        "backtest": backtest.metrics,
        "promotion": asdict(decision),
        "deployment_state": registered.get("status", "RESEARCH"),
        "minimum_shadow_trading_days": config.promotion.minimum_shadow_trading_days,
    }
    (output / "wp_v3_research_summary.json").write_text(
        _strict_json(summary) + "\n",
        encoding="utf-8",
    )
    _render_research_audit_dashboard(
        output=output,
        config=config,
        registry=registry,
        model_fingerprint=bundle.fingerprint,
        deployment_state=str(registered.get("status") or "RESEARCH"),
        research_start=str(panel["trade_date"].min()),
        research_end=str(panel["trade_date"].max()),
    )
    print(_strict_json(summary))
    return 0


def _historical_replay(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict:
    candidate_records = []
    for row in candidates.to_dict(orient="records"):
        record = dict(row)
        record.update(
            {
                "record_type": "RECONSTRUCTED_OOS",
                "status": "RECONSTRUCTED_OOS_QUALIFIED",
                "first_signal_time": row.get("signal_slot"),
                "first_signal_price": row.get("signal_price"),
                "truth_status": "verified",
                "net_positive": bool(float(row.get("net_return_pct", -999)) > 0),
            }
        )
        candidate_records.append(record)
    by_date: dict[str, list[dict]] = {}
    for record in candidate_records:
        by_date.setdefault(str(record.get("trade_date")), []).append(record)
    days = []
    for trade_date in sorted(predictions["trade_date"].astype(str).unique()):
        records = by_date.get(trade_date, [])
        target_dates = predictions.loc[
            predictions["trade_date"].astype(str).eq(trade_date),
            "target_trade_date",
        ].astype(str)
        days.append(
            {
                "trade_date": trade_date,
                "target_trade_date": target_dates.iloc[0] if not target_dates.empty else None,
                "status": "QUALIFIED" if records else "NO_SIGNAL",
                "candidate_count": len(records),
            }
        )
    focus = [
        day
        for day in days
        if "20260721" <= day["trade_date"] <= "20260724"
    ]
    return {
        "schema_version": "wp_v3_historical_replay_1",
        "record_semantics": (
            "Chronological out-of-sample reconstruction; these records were not "
            "published live and must not be represented as user fills."
        ),
        "days": days,
        "focus_20260721_20260724": focus,
        "candidates": candidate_records,
    }


def _render_research_audit_dashboard(
    *,
    output: Path,
    config,
    registry: dict,
    model_fingerprint: str,
    deployment_state: str,
    research_start: str,
    research_end: str,
) -> None:
    current = datetime.now(CN_TZ)
    revision = current.strftime("%Y-%m-%d %H:%M:%S")
    model = model_record(registry, model_fingerprint) or {}
    state = (
        deployment_state
        if deployment_state in {"SHADOW", "SHADOW_OBSERVATION"}
        else "MODEL_NOT_READY"
    )
    manifest = {
        "schema_version": "wp_manifest_v3",
        "latest_update": revision,
        "report_revision": revision,
        "source_trade_date": current.strftime("%Y%m%d"),
        "target_trade_date": None,
        "signal_slot": None,
        "market_data_time": None,
        "source_mode": "three_year_research",
        "source_repository": "njedu2023-prog/WP",
        "session_phase": "CLOSED",
        "buy_plan_count": 0,
        "shadow_qualified_count": 0,
        "health_status": "research_ready",
        "manual_execution_only": True,
        "order_routing_enabled": False,
        "v3_state": state,
        "v3_model_version": model.get("model_version"),
        "v3_model_fingerprint": model_fingerprint,
        "v3_policy_fingerprint": model.get("policy_fingerprint"),
        "v3_formal_authorization": False,
        "v3_qualified_count": 0,
        "v3_message": (
            "Three-year research is registered. The policy is in the mandatory "
            "150-trading-day shadow period."
            if state == "SHADOW"
            else (
                "Three-year research failed its promotion gate. The frozen model "
                "is designated for forward observation only."
                if state == "SHADOW_OBSERVATION"
                else "Three-year research is registered but no model is shadow-designated."
            )
        ),
        "research_start": research_start,
        "research_end": research_end,
    }
    render_v3_dashboard(
        output / "wp_v3_research_report.html",
        manifest=manifest,
        predictions=pd.DataFrame(),
        ledger=empty_shadow_ledger(),
        registry=registry,
        config=config,
        replay=json.loads(
            (ROOT / "outputs" / "json" / "wp_v3_historical_replay.json").read_text(
                encoding="utf-8"
            )
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
