from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import tushare as ts

from wp.v3.history import TushareHistoryClient
from wp.v3.io import (
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
)
from wp.v3.v30_limit_event import (
    CURRENT_DAY_FORBIDDEN_COLUMNS,
    KPL_FIELDS,
    KPL_TAGS,
    SCHEMA_VERSION,
    SIGNAL_SLOTS,
    attach_candidate_event_state,
    audit_kpl_frame,
    build_causal_event_projection,
    normalize_kpl_frame,
)


SOURCE_V24_DATA_RUN_ID = 30_635_569_735
V24_SCHEMA_VERSION = "wp_v24_point_in_time_features_1"
PROBE_DATES = (
    "20230825",
    "20231229",
    "20240315",
    "20240927",
    "20250115",
    "20250723",
    "20260115",
    "20260723",
)
TAG_CACHE_KEYS = {"涨停": "up", "炸板": "failed", "跌停": "down"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe outcome-blind V30 limit-event data."
    )
    parser.add_argument("--v24-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for the V30 data probe")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest, candidates = load_v24_source(args.v24_data_dir)
    previous_dates = previous_trade_dates(
        manifest["trade_calendar"]["open_dates"],
        PROBE_DATES,
    )
    sample_candidates = candidates.loc[
        candidates["trade_date"].astype(str).isin(PROBE_DATES),
        ["trade_date", "signal_slot", "ts_code"],
    ].copy()
    if sample_candidates.empty:
        raise RuntimeError("V30 probe dates have no immutable V24 candidates")

    client = TushareHistoryClient(
        ts.pro_api(token),
        output / "cache",
        page_size=2_000,
        requests_per_minute=120,
        attempts=2,
    )
    query_records: list[dict[str, Any]] = []
    normalized_by_date: dict[str, list[pd.DataFrame]] = {}
    for role, dates in (
        ("current", PROBE_DATES),
        ("previous", tuple(previous_dates[date] for date in PROBE_DATES)),
    ):
        for trade_date in dates:
            frames: list[pd.DataFrame] = []
            for tag in KPL_TAGS:
                frame, record = fetch_event_category(
                    client,
                    trade_date=trade_date,
                    tag=tag,
                    role=role,
                )
                query_records.append(record)
                if frame is not None:
                    frames.append(frame)
            normalized_by_date.setdefault(trade_date, []).extend(frames)

    projections: list[pd.DataFrame] = []
    stock_events: list[pd.DataFrame] = []
    date_records: list[dict[str, Any]] = []
    for trade_date in PROBE_DATES:
        frames = normalized_by_date.get(trade_date, [])
        try:
            projection, stocks = build_causal_event_projection(
                frames,
                trade_date=trade_date,
            )
            touch_or_down = (
                projection["market_limit_hit_count"]
                + projection["market_limit_down_count"]
            )
            date_pass = bool(
                len(projection) == len(SIGNAL_SLOTS)
                and projection[
                    ["trade_date", "signal_slot"]
                ].duplicated().sum()
                == 0
                and int(touch_or_down.iloc[-1]) >= 5
            )
            projections.append(projection)
            stock_events.append(stocks)
            date_records.append(
                {
                    "trade_date": trade_date,
                    "projection_rows": int(len(projection)),
                    "limit_hits_by_1450": int(
                        projection["market_limit_hit_count"].iloc[-1]
                    ),
                    "limit_opens_by_1450": int(
                        projection["market_limit_open_count"].iloc[-1]
                    ),
                    "limit_downs_by_1450": int(
                        projection["market_limit_down_count"].iloc[-1]
                    ),
                    "coverage_pass": date_pass,
                }
            )
        except Exception as error:
            date_records.append(
                {
                    "trade_date": trade_date,
                    "projection_rows": 0,
                    "coverage_pass": False,
                    "error": str(error)[:500],
                }
            )

    projection_frame = (
        pd.concat(projections, ignore_index=True)
        if projections
        else pd.DataFrame()
    )
    stock_event_frame = (
        pd.concat(stock_events, ignore_index=True)
        if stock_events
        else pd.DataFrame()
    )
    candidate_state = attach_candidate_event_state(
        sample_candidates,
        stock_event_frame,
        projection_frame,
    )
    market_columns = [
        column
        for column in projection_frame.columns
        if column not in {"trade_date", "signal_slot"}
    ]
    context_complete = bool(
        len(projection_frame) == len(PROBE_DATES) * len(SIGNAL_SLOTS)
        and not projection_frame.duplicated(
            ["trade_date", "signal_slot"]
        ).any()
        and projection_frame[market_columns].notna().all(axis=None)
    )
    candidates_complete = bool(
        len(candidate_state) == len(sample_candidates)
        and candidate_state[
            [
                "market_limit_hit_count",
                "market_limit_open_count",
                "market_limit_down_count",
                "candidate_limit_hit_before_signal",
                "candidate_limit_open_before_signal",
                "candidate_limit_down_before_signal",
            ]
        ].notna().all(axis=None)
    )
    candidate_limit_hits = int(
        candidate_state["candidate_limit_hit_before_signal"].sum()
    )
    expected_query_count = len(PROBE_DATES) * len(KPL_TAGS) * 2
    queries_pass = bool(
        len(query_records) == expected_query_count
        and all(record.get("coverage_pass") for record in query_records)
    )
    no_forbidden_output = not any(
        column in projection_frame.columns
        or column in candidate_state.columns
        for column in CURRENT_DAY_FORBIDDEN_COLUMNS
    )
    authorized = bool(
        queries_pass
        and all(record["coverage_pass"] for record in date_records)
        and context_complete
        and candidates_complete
        and candidate_limit_hits >= 1
        and no_forbidden_output
    )

    projection_path = atomic_write_csv(
        projection_frame,
        output / "wp_v30_probe_market_projection.csv",
    )
    candidate_path = atomic_write_csv(
        candidate_state,
        output / "wp_v30_probe_candidate_state.csv",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "profit_outcomes_read": False,
        "selection_used_profit_outcomes": False,
        "source_runs": {
            "v24_data_run_id": SOURCE_V24_DATA_RUN_ID,
        },
        "probe_dates": list(PROBE_DATES),
        "previous_trade_dates": previous_dates,
        "signal_slots": list(SIGNAL_SLOTS),
        "query_contract": {
            "api_name": "kpl_list",
            "categories": list(KPL_TAGS),
            "fields": KPL_FIELDS.split(","),
            "expected_query_count": expected_query_count,
            "current_day_final_category_used_as_feature": False,
            "current_day_end_state_fields_used": False,
            "historical_times_discretized_to_live_five_minute_resolution": True,
            "live_source": "archived_all_market_rt_min_bars_plus_stk_limit",
        },
        "query_records": query_records,
        "date_projection_records": date_records,
        "coverage": {
            "sample_candidate_rows": int(len(sample_candidates)),
            "projected_candidate_rows": int(len(candidate_state)),
            "market_context_rows": int(len(projection_frame)),
            "candidate_limit_hit_rows": candidate_limit_hits,
            "query_contract_passed": queries_pass,
            "market_context_complete": context_complete,
            "candidate_projection_complete": candidates_complete,
            "no_forbidden_current_day_output": no_forbidden_output,
        },
        "artifacts": {
            "market_projection": artifact_record(projection_path),
            "candidate_state": artifact_record(candidate_path),
        },
        "selected_source_family": (
            "kpl_limit_event_tape" if authorized else None
        ),
        "full_backfill_authorized": authorized,
        "model_research_authorized": False,
        "next_gate": (
            "full_three_year_outcome_blind_event_build"
            if authorized
            else "close_v30_data_direction"
        ),
    }
    atomic_write_json(output / "wp_v30_limit_event_probe.json", payload)
    print(
        "WP_V30_PROBE_RESULT="
        + json.dumps(
            {
                "probe_dates": len(PROBE_DATES),
                "queries": len(query_records),
                "query_contract_passed": queries_pass,
                "date_projection_passes": sum(
                    bool(record["coverage_pass"])
                    for record in date_records
                ),
                "market_context_rows": int(len(projection_frame)),
                "sample_candidate_rows": int(len(sample_candidates)),
                "candidate_limit_hit_rows": candidate_limit_hits,
                "full_backfill_authorized": authorized,
                "next_gate": payload["next_gate"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    if not authorized:
        failed_queries = [
            {
                key: record.get(key)
                for key in (
                    "status",
                    "role",
                    "trade_date",
                    "requested_tag",
                    "rows",
                    "relevant_time_coverage",
                    "supplied_relevant_time_parse_rate",
                    "all_supplied_times_parseable",
                    "times_within_session",
                    "open_not_before_first_touch",
                    "coverage_pass",
                    "error",
                )
                if key in record
            }
            for record in query_records
            if not record.get("coverage_pass")
        ]
        overlap_by_date = (
            candidate_state.groupby("trade_date", sort=True)
            .agg(
                candidate_rows=("ts_code", "size"),
                candidate_codes=("ts_code", "nunique"),
                causal_limit_hits=(
                    "candidate_limit_hit_before_signal",
                    "sum",
                ),
                causal_limit_opens=(
                    "candidate_limit_open_before_signal",
                    "sum",
                ),
            )
            .reset_index()
            .to_dict(orient="records")
        )
        print(
            "WP_V30_PROBE_FAILURES="
            + json.dumps(
                {
                    "failed_queries": failed_queries,
                    "date_projection_records": date_records,
                    "candidate_overlap_by_date": overlap_by_date,
                    "failed_gates": {
                        "query_contract": not queries_pass,
                        "date_projection": not all(
                            record["coverage_pass"]
                            for record in date_records
                        ),
                        "market_context": not context_complete,
                        "candidate_projection": not candidates_complete,
                        "candidate_limit_event_observed": (
                            candidate_limit_hits < 1
                        ),
                        "forbidden_output": not no_forbidden_output,
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
            flush=True,
        )
        raise RuntimeError("V30 limit-event source failed frozen probe gates")
    return 0


def load_v24_source(
    root: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    manifests = list(Path(root).rglob("wp_v24_data_manifest.json"))
    candidates = list(
        Path(root).rglob("wp_v24_outcome_blind_candidate_index.parquet")
    )
    if len(manifests) != 1 or len(candidates) != 1:
        raise RuntimeError("V30 expected one immutable V24 source artifact")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != V24_SCHEMA_VERSION:
        raise RuntimeError("V30 V24 source schema mismatch")
    if manifest.get("profit_outcomes_read") is not False:
        raise RuntimeError("V30 V24 source is not outcome blind")
    expected = str(
        manifest["artifacts"]["candidate_index"].get("sha256") or ""
    )
    if not expected or file_sha256(candidates[0]) != expected:
        raise RuntimeError("V30 V24 candidate digest mismatch")
    frame = pd.read_parquet(candidates[0])
    return manifest, frame


def previous_trade_dates(
    open_dates: list[str],
    probe_dates: tuple[str, ...],
) -> dict[str, str]:
    ordered = sorted(str(value) for value in open_dates)
    mapping: dict[str, str] = {}
    for date in probe_dates:
        previous = [value for value in ordered if value < date]
        if not previous:
            raise RuntimeError(f"no previous trade date for {date}")
        mapping[date] = previous[-1]
    return mapping


def fetch_event_category(
    client: TushareHistoryClient,
    *,
    trade_date: str,
    tag: str,
    role: str,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    try:
        frame = client.query(
            "kpl_list",
            cache_key=(
                f"{trade_date}_{TAG_CACHE_KEYS[tag]}_{role}_v30_probe"
            ),
            paged=True,
            trade_date=trade_date,
            tag=tag,
            fields=KPL_FIELDS,
        )
        record = audit_kpl_frame(
            frame,
            trade_date=trade_date,
            requested_tag=tag,
        )
        record["role"] = role
        normalized = normalize_kpl_frame(
            frame,
            trade_date=trade_date,
            requested_tag=tag,
        )
        return normalized, record
    except Exception as error:
        return None, {
            "status": "error",
            "role": role,
            "trade_date": trade_date,
            "requested_tag": tag,
            "rows": 0,
            "coverage_pass": False,
            "error": str(error)[:500],
        }


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
