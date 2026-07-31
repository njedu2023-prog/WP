from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import tushare as ts

from build_wp_v23_point_in_time_data import (
    fetch_cross_section_partitions,
    file_artifact,
    parallel_map,
    read_partitions,
)
from wp.v3.contracts import load_v3_config
from wp.v3.history import TushareHistoryClient
from wp.v3.io import atomic_write_json, atomic_write_parquet, file_sha256
from wp.v3.v25_positioning import (
    CYQ_RAW_COLUMNS,
    MARGIN_RAW_COLUMNS,
    SCHEMA_VERSION,
    TOP_LIST_RAW_COLUMNS,
    V25_FEATURE_COLUMNS,
    attach_candidate_signal_price,
    attach_positioning_features,
    positioning_coverage_audit,
    previous_date_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build outcome-blind V25 prior-positioning features."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for V25 backfill")
    config = load_v3_config(args.config)
    source_root = Path(args.source_dir)
    output = Path(args.output_dir)
    raw_root = output / "raw"
    output.mkdir(parents=True, exist_ok=True)
    client = TushareHistoryClient(
        ts.pro_api(token),
        args.cache_dir,
        page_size=config.history.tushare_page_size,
        requests_per_minute=config.history.tushare_requests_per_minute,
        attempts=6,
    )

    source_path = find_one(
        source_root,
        "wp_v24_point_in_time_features.parquet",
    )
    candidate_path = find_one(
        source_root,
        "wp_v24_outcome_blind_candidate_index.parquet",
    )
    manifest_path = find_one(source_root, "wp_v24_data_manifest.json")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_source_manifest(source_manifest)
    validate_source_artifact(source_manifest, "features", source_path)
    validate_source_artifact(
        source_manifest,
        "candidate_index",
        candidate_path,
    )
    source = pd.read_parquet(source_path)
    candidate_index = pd.read_parquet(candidate_path)
    validate_outcome_blind(source)
    validate_outcome_blind(candidate_index)
    source = attach_candidate_signal_price(source, candidate_index)
    open_dates = [
        str(value)
        for value in (source_manifest.get("trade_calendar") or {}).get(
            "open_dates",
            [],
        )
    ]
    if not open_dates:
        raise RuntimeError("V25 source has no immutable trade calendar")
    print(
        f"[wp-v25-data] outcome-blind source rows={len(source):,}",
        flush=True,
    )

    cyq_path, cyq_failures = fetch_required_cyq(
        client,
        source,
        output_dir=raw_root / "previous_cyq",
        workers=args.workers,
    )
    margin_requirements = required_margin_codes(source, open_dates=open_dates)
    margin_paths, margin_failures = fetch_cross_section_partitions(
        client,
        api_name="margin_detail",
        fields=",".join(MARGIN_RAW_COLUMNS),
        required=margin_requirements,
        output_dir=raw_root / "previous_margin",
        file_prefix="wp_v25_margin",
        workers=args.workers,
    )
    top_requirements = required_previous_codes(source)
    top_paths, top_failures = fetch_cross_section_partitions(
        client,
        api_name="top_list",
        fields=",".join(TOP_LIST_RAW_COLUMNS),
        required=top_requirements,
        output_dir=raw_root / "previous_top_list",
        file_prefix="wp_v25_top_list",
        workers=args.workers,
    )

    cyq = pd.read_parquet(cyq_path)
    margin = read_partitions(margin_paths)
    top_list = read_partitions(top_paths)
    features = attach_positioning_features(
        source,
        cyq,
        margin,
        top_list,
        open_dates=open_dates,
    )
    audit = positioning_coverage_audit(features)
    source_identity = ["trade_date", "signal_slot", "ts_code"]
    if len(features) != len(source):
        raise RuntimeError("V25 feature build changed source row count")
    if features.duplicated(source_identity).any():
        raise RuntimeError("V25 feature build duplicated candidate identities")

    feature_path = atomic_write_parquet(
        features,
        output / "wp_v25_positioning_features.parquet",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "profit_outcomes_read": False,
        "future_information_allowed": False,
        "source": {
            "v24_manifest_sha256": file_sha256(manifest_path),
            "v24_features_sha256": file_sha256(source_path),
            "v24_candidate_index_sha256": file_sha256(candidate_path),
            "v24_schema_version": source_manifest.get("schema_version"),
            "candidate_rows": int(len(source)),
            "trade_dates": int(source["trade_date"].astype(str).nunique()),
        },
        "protocol": {
            "cyq_use": "immediately_previous_A_share_trade_day_only",
            "margin_use": "T_minus_1_and_T_minus_2_trade_days_only",
            "top_list_use": "immediately_previous_A_share_trade_day_only",
            "outcome_driven_row_dropping": False,
            "candidate_identity_mutation": False,
        },
        "requirements": {
            "candidate_rows": int(len(source)),
            "cyq_codes": int(source["ts_code"].astype(str).nunique()),
            "margin_dates": int(len(margin_requirements)),
            "top_list_dates": int(len(top_requirements)),
        },
        "query_failures": {
            "previous_cyq": cyq_failures,
            "previous_margin": margin_failures,
            "previous_top_list": top_failures,
        },
        "coverage_audit": audit,
        "feature_columns": list(V25_FEATURE_COLUMNS),
        "artifacts": {
            "features": file_artifact(feature_path),
            "cyq_required": file_artifact(cyq_path),
            "margin_partitions": [
                file_artifact(path) for path in margin_paths
            ],
            "top_list_partitions": [
                file_artifact(path) for path in top_paths
            ],
        },
        "v25_model_research_authorized": bool(
            audit["coverage_passed"]
            and not cyq_failures
            and not margin_failures
            and not top_failures
        ),
    }
    atomic_write_json(output / "wp_v25_data_manifest.json", manifest)
    print(
        "WP_V25_DATA_RESULT="
        + json.dumps(
            {
                "candidate_rows": int(len(source)),
                "requirements": manifest["requirements"],
                "coverage_audit": audit,
                "failure_counts": {
                    key: len(value)
                    for key, value in manifest["query_failures"].items()
                },
                "v25_model_research_authorized": (
                    manifest["v25_model_research_authorized"]
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    if not manifest["v25_model_research_authorized"]:
        raise RuntimeError(
            "V25 positioning dataset failed the frozen coverage contract"
        )
    return 0


def fetch_required_cyq(
    client: TushareHistoryClient,
    source: pd.DataFrame,
    *,
    output_dir: Path,
    workers: int,
) -> tuple[Path, list[dict[str, str]]]:
    required = {
        str(code): tuple(
            sorted(group["v23_prev_trade_date"].astype(str).unique())
        )
        for code, group in source.groupby("ts_code", sort=True)
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    def fetch(
        task: tuple[str, tuple[str, ...]],
    ) -> tuple[pd.DataFrame, dict[str, str] | None]:
        ts_code, dates = task
        try:
            frame = client.query(
                "cyq_perf",
                cache_key=f"{ts_code.replace('.', '_')}_v25",
                paged=True,
                ts_code=ts_code,
                start_date=min(dates),
                end_date=max(dates),
                fields=",".join(CYQ_RAW_COLUMNS),
            )
            frame["trade_date"] = frame["trade_date"].astype(str)
            frame["ts_code"] = frame["ts_code"].astype(str)
            return frame.loc[frame["trade_date"].isin(dates)].copy(), None
        except Exception as error:
            return pd.DataFrame(), {
                "ts_code": ts_code,
                "error": str(error)[:500],
            }

    rows = parallel_map(fetch, sorted(required.items()), workers=workers)
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for index, (frame, failure) in enumerate(rows, start=1):
        if failure:
            failures.append(failure)
        elif not frame.empty:
            frames.append(frame)
        if index % 100 == 0:
            print(
                f"[wp-v25-data] cyq codes={index}/{len(required)} "
                f"failures={len(failures)}",
                flush=True,
            )
    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=CYQ_RAW_COLUMNS)
    )
    path = atomic_write_parquet(
        combined,
        output_dir / "wp_v25_required_cyq.parquet",
    )
    return path, failures


def required_previous_codes(
    source: pd.DataFrame,
) -> dict[str, tuple[str, ...]]:
    return {
        str(date): tuple(sorted(group["ts_code"].astype(str).unique()))
        for date, group in source.groupby("v23_prev_trade_date", sort=True)
    }


def required_margin_codes(
    source: pd.DataFrame,
    *,
    open_dates: list[str],
) -> dict[str, tuple[str, ...]]:
    mapping = previous_date_map(open_dates)
    requirements: dict[str, set[str]] = {}
    for row in source[["v23_prev_trade_date", "ts_code"]].itertuples(
        index=False
    ):
        previous = str(row.v23_prev_trade_date)
        code = str(row.ts_code)
        requirements.setdefault(previous, set()).add(code)
        previous_two = mapping.get(previous)
        if previous_two:
            requirements.setdefault(previous_two, set()).add(code)
    return {
        date: tuple(sorted(codes))
        for date, codes in sorted(requirements.items())
    }


def validate_source_manifest(manifest: dict[str, Any]) -> None:
    if not manifest.get("v24_model_research_authorized"):
        raise RuntimeError("V25 source V24 dataset was not authorized")
    if manifest.get("profit_outcomes_read") is not False:
        raise RuntimeError("V25 source manifest is not outcome-blind")
    if manifest.get("future_information_allowed") is not False:
        raise RuntimeError("V25 source permits future information")


def validate_source_artifact(
    manifest: dict[str, Any],
    artifact_name: str,
    path: Path,
) -> None:
    artifact = (manifest.get("artifacts") or {}).get(artifact_name) or {}
    expected = str(artifact.get("sha256") or "")
    if not expected:
        raise RuntimeError(
            f"V25 source manifest has no {artifact_name} digest"
        )
    if file_sha256(path) != expected:
        raise RuntimeError(
            f"V25 source {artifact_name} digest mismatch"
        )


def validate_outcome_blind(frame: pd.DataFrame) -> None:
    forbidden = (
        "target",
        "label",
        "truth",
        "future",
        "gross_return",
        "net_return",
        "t1_",
        "exit_",
    )
    contaminated = [
        column
        for column in frame.columns
        if any(token in column.lower() for token in forbidden)
    ]
    if contaminated:
        raise RuntimeError(
            f"V25 source contains profit outcomes: {contaminated}"
        )


def find_one(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one {name} under {root}, found {len(matches)}"
        )
    return matches[0]


if __name__ == "__main__":
    raise SystemExit(main())
