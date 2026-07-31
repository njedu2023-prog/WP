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
from wp.v3.io import atomic_write_csv, atomic_write_json, file_sha256
from wp.v3.v31_public_event import (
    LOOKBACK_TRADE_DAYS,
    PROBE_DATES,
    SCHEMA_VERSION,
    SOURCE_SPECS,
    audit_event_frame,
    build_candidate_event_presence,
    build_lookback_map,
    causal_dates_valid,
    normalize_event_frame,
)


SOURCE_V24_DATA_RUN_ID = 30_635_569_735
V24_SCHEMA_VERSION = "wp_v24_point_in_time_features_1"
MIN_NONEMPTY_SOURCE_DATES = 4
MIN_SOURCE_MATCHES = 2
MIN_SOURCE_TARGET_DATES = 2
MIN_ADMITTED_SOURCES = 2
MIN_FAMILY_MATCHES = 10
MIN_FAMILY_TARGET_DATES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe outcome-blind V31 public-event data."
    )
    parser.add_argument("--v24-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for the V31 data probe")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest, candidates = load_v24_source(args.v24_data_dir)
    lookback_map = build_lookback_map(
        manifest["trade_calendar"]["open_dates"],
        PROBE_DATES,
    )
    sample_candidates = candidates.loc[
        candidates["trade_date"].astype(str).isin(PROBE_DATES),
        ["trade_date", "ts_code"],
    ].drop_duplicates()
    if sample_candidates.empty:
        raise RuntimeError("V31 probe dates have no immutable V24 candidates")

    client = TushareHistoryClient(
        ts.pro_api(token),
        output / "cache",
        page_size=2_000,
        requests_per_minute=120,
        attempts=2,
    )
    required_dates = sorted(
        {
            date
            for dates in lookback_map.values()
            for date in dates
        }
    )
    query_records: list[dict[str, Any]] = []
    events_by_source: dict[str, pd.DataFrame] = {}
    for source, spec in SOURCE_SPECS.items():
        frames: list[pd.DataFrame] = []
        for requested_date in required_dates:
            frame, record = fetch_source_date(
                client,
                source=source,
                requested_date=requested_date,
            )
            query_records.append(record)
            if frame is not None:
                frames.append(frame)
        events_by_source[source] = (
            pd.concat(frames, ignore_index=True).drop_duplicates()
            if frames
            else pd.DataFrame(
                columns=[
                    *spec["fields"].split(","),
                    "event_date",
                    "event_source",
                ]
            )
        )

    presence = build_candidate_event_presence(
        sample_candidates,
        events_by_source,
        lookback_map,
    )
    source_metrics: dict[str, dict[str, Any]] = {}
    admitted_sources: list[str] = []
    for source in SOURCE_SPECS:
        records = [
            record
            for record in query_records
            if record["source"] == source
        ]
        match_column = f"event_{source}"
        matched = presence.loc[presence[match_column]]
        query_contract = bool(
            len(records) == len(required_dates)
            and all(record.get("coverage_pass") for record in records)
        )
        metric = {
            "query_contract_passed": query_contract,
            "required_query_dates": len(required_dates),
            "nonempty_query_dates": sum(
                int(record.get("rows", 0)) > 0 for record in records
            ),
            "candidate_identity_matches": int(len(matched)),
            "matched_target_dates": int(matched["trade_date"].nunique()),
        }
        metric["admitted"] = bool(
            metric["query_contract_passed"]
            and metric["nonempty_query_dates"]
            >= MIN_NONEMPTY_SOURCE_DATES
            and metric["candidate_identity_matches"] >= MIN_SOURCE_MATCHES
            and metric["matched_target_dates"]
            >= MIN_SOURCE_TARGET_DATES
        )
        source_metrics[source] = metric
        if metric["admitted"]:
            admitted_sources.append(source)

    admitted_columns = [
        f"event_{source}" for source in admitted_sources
    ]
    if admitted_columns:
        admitted_union = presence[admitted_columns].any(axis=1)
    else:
        admitted_union = pd.Series(False, index=presence.index)
    presence["event_admitted_union"] = admitted_union
    family_matches = presence.loc[presence["event_admitted_union"]]
    presence_complete = bool(
        len(presence) == len(sample_candidates)
        and not presence.duplicated(["trade_date", "ts_code"]).any()
        and presence[
            [f"event_{source}" for source in SOURCE_SPECS]
        ].notna().all(axis=None)
    )
    causal_valid = causal_dates_valid(events_by_source, lookback_map)
    authorized = bool(
        len(admitted_sources) >= MIN_ADMITTED_SOURCES
        and len(family_matches) >= MIN_FAMILY_MATCHES
        and family_matches["trade_date"].nunique()
        >= MIN_FAMILY_TARGET_DATES
        and presence_complete
        and causal_valid
    )

    presence_path = atomic_write_csv(
        presence,
        output / "wp_v31_probe_candidate_event_presence.csv",
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
        "lookback_trade_days": LOOKBACK_TRADE_DAYS,
        "lookback_map": lookback_map,
        "required_query_dates": required_dates,
        "query_records": query_records,
        "source_metrics": source_metrics,
        "admitted_sources": admitted_sources,
        "family_coverage": {
            "candidate_identities": int(len(presence)),
            "matched_candidate_identities": int(len(family_matches)),
            "matched_target_dates": int(
                family_matches["trade_date"].nunique()
            ),
            "candidate_presence_complete": presence_complete,
            "causal_dates_valid": causal_valid,
        },
        "frozen_gates": {
            "minimum_nonempty_source_dates": MIN_NONEMPTY_SOURCE_DATES,
            "minimum_source_matches": MIN_SOURCE_MATCHES,
            "minimum_source_target_dates": MIN_SOURCE_TARGET_DATES,
            "minimum_admitted_sources": MIN_ADMITTED_SOURCES,
            "minimum_family_matches": MIN_FAMILY_MATCHES,
            "minimum_family_target_dates": MIN_FAMILY_TARGET_DATES,
        },
        "artifacts": {
            "candidate_event_presence": artifact_record(presence_path),
        },
        "full_backfill_authorized": authorized,
        "model_research_authorized": False,
        "next_gate": (
            "full_three_year_outcome_blind_public_event_build"
            if authorized
            else "close_v31_data_direction"
        ),
    }
    atomic_write_json(output / "wp_v31_public_event_probe.json", payload)
    result = {
        "probe_dates": len(PROBE_DATES),
        "required_query_dates": len(required_dates),
        "queries": len(query_records),
        "admitted_sources": admitted_sources,
        "candidate_identities": int(len(presence)),
        "matched_candidate_identities": int(len(family_matches)),
        "matched_target_dates": int(
            family_matches["trade_date"].nunique()
        ),
        "full_backfill_authorized": authorized,
        "next_gate": payload["next_gate"],
    }
    print(
        "WP_V31_PROBE_RESULT="
        + json.dumps(
            result,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    print(
        "WP_V31_SOURCE_METRICS="
        + json.dumps(
            source_metrics,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    if not authorized:
        failed_queries = [
            record
            for record in query_records
            if not record.get("coverage_pass")
        ]
        print(
            "WP_V31_PROBE_FAILURES="
            + json.dumps(
                {
                    "failed_query_count": len(failed_queries),
                    "failed_queries": failed_queries[:20],
                    "presence_complete": presence_complete,
                    "causal_dates_valid": causal_valid,
                },
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
            flush=True,
        )
        raise RuntimeError("V31 public-event source failed frozen probe gates")
    return 0


def fetch_source_date(
    client: TushareHistoryClient,
    *,
    source: str,
    requested_date: str,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    spec = SOURCE_SPECS[source]
    try:
        frame = client.query(
            source,
            cache_key=f"{requested_date}_{source}_v31_probe",
            paged=True,
            fields=spec["fields"],
            **{spec["date_arg"]: requested_date},
        )
        record = audit_event_frame(
            frame,
            source=source,
            requested_date=requested_date,
        )
        normalized = normalize_event_frame(frame, source=source)
        return normalized, record
    except Exception as error:
        return None, {
            "status": "error",
            "source": source,
            "requested_date": requested_date,
            "rows": 0,
            "coverage_pass": False,
            "error": str(error)[:500],
        }


def load_v24_source(
    root: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    manifests = list(Path(root).rglob("wp_v24_data_manifest.json"))
    candidates = list(
        Path(root).rglob("wp_v24_outcome_blind_candidate_index.parquet")
    )
    if len(manifests) != 1 or len(candidates) != 1:
        raise RuntimeError("V31 expected one immutable V24 source artifact")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != V24_SCHEMA_VERSION:
        raise RuntimeError("V31 V24 source schema mismatch")
    if manifest.get("profit_outcomes_read") is not False:
        raise RuntimeError("V31 V24 source is not outcome blind")
    expected = str(
        manifest["artifacts"]["candidate_index"].get("sha256") or ""
    )
    if not expected or file_sha256(candidates[0]) != expected:
        raise RuntimeError("V31 V24 candidate digest mismatch")
    return manifest, pd.read_parquet(candidates[0])


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


if __name__ == "__main__":
    raise SystemExit(main())

