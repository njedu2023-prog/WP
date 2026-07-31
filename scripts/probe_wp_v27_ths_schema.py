from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tushare as ts

try:
    from scripts.probe_wp_v26_crowd_confirmation_data import parse_rank_time
except ModuleNotFoundError:
    from probe_wp_v26_crowd_confirmation_data import parse_rank_time
from wp.v3.history import TushareHistoryClient
from wp.v3.io import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATES = (
    "20230728",
    "20230825",
    "20230928",
    "20231031",
    "20231130",
    "20231229",
    "20240315",
    "20240927",
    "20250115",
    "20250723",
    "20260115",
    "20260723",
)
THS_FIELDS = (
    "trade_date,data_type,ts_code,ts_name,rank,pct_change,current_price,"
    "concept,hot,rank_time"
)
A_SHARE_CODE = re.compile(r"^\d{6}\.(?:SH|SZ|BJ)$")


def main() -> int:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for the V27 schema probe")
    output = Path(
        os.getenv(
            "WP_V27_PROBE_OUTPUT",
            str(ROOT / "artifacts" / "wp_v27_schema_probe"),
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    client = TushareHistoryClient(
        ts.pro_api(token),
        output / "cache",
        page_size=2_000,
        requests_per_minute=120,
        attempts=2,
    )

    probes: list[dict[str, Any]] = []
    for trade_date in SAMPLE_DATES:
        try:
            frame = client.query(
                "ths_hot",
                cache_key=f"{trade_date}_v27_schema_probe",
                paged=True,
                trade_date=trade_date,
                market="热股",
                is_new="N",
                fields=THS_FIELDS,
            )
            probes.append(
                ths_schema_record(frame, trade_date=trade_date)
            )
        except Exception as error:
            probes.append(
                {
                    "trade_date": trade_date,
                    "status": "error",
                    "rows": 0,
                    "coverage_pass": False,
                    "error": str(error)[:500],
                }
            )

    passed = bool(probes and all(row["coverage_pass"] for row in probes))
    payload = {
        "schema_version": "wp_v27_ths_schema_probe_1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "profit_outcomes_read": False,
        "selection_used_profit_outcomes": False,
        "probe_dates": list(SAMPLE_DATES),
        "passed_dates": sum(bool(row["coverage_pass"]) for row in probes),
        "probes": probes,
        "v27_full_attention_backfill_authorized": passed,
        "model_research_authorized": False,
        "next_gate": (
            "full_three_year_outcome_blind_attention_coverage_audit"
            if passed
            else "stop_v27_schema_direction"
        ),
    }
    atomic_write_json(output / "wp_v27_ths_schema_probe.json", payload)
    print(
        "WP_V27_SCHEMA_PROBE_RESULT="
        + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def normalize_trade_date(value: Any) -> str:
    digits = re.sub(r"\D", "", str(value))
    return digits[:8] if len(digits) >= 8 else ""


def select_tail_batch(
    frame: pd.DataFrame,
    *,
    trade_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = set(THS_FIELDS.split(","))
    if not required.issubset(frame.columns):
        return pd.DataFrame(), {
            "schema_ok": False,
            "batch_selection": "missing_schema",
        }
    data = frame.loc[:, THS_FIELDS.split(",")].copy()
    data["normalized_trade_date"] = [
        normalize_trade_date(value) for value in data["trade_date"]
    ]
    data["ts_code"] = data["ts_code"].fillna("").astype(str).str.strip()
    parsed = pd.to_datetime(
        [parse_rank_time(value, trade_date) for value in data["rank_time"]],
        errors="coerce",
    )
    data["rank_timestamp"] = pd.Series(
        parsed,
        index=data.index,
        dtype="datetime64[ns]",
    )
    data["snapshot_minute"] = data["rank_timestamp"].dt.floor("min")
    lower = pd.Timestamp(f"{trade_date} 13:20:00")
    upper = pd.Timestamp(f"{trade_date} 14:50:00")
    tail = data.loc[
        data["snapshot_minute"].between(lower, upper, inclusive="both")
    ].copy()
    if tail.empty:
        return tail, {
            "schema_ok": True,
            "batch_selection": "no_tail_batch",
        }
    minute = tail["snapshot_minute"].max()
    snapshot = tail.loc[tail["snapshot_minute"].eq(minute)].copy()
    selection = "minute_bucket"
    if snapshot["ts_code"].duplicated().any():
        exact_batches = [
            batch
            for _, batch in snapshot.groupby("rank_timestamp", sort=True)
            if batch["ts_code"].nunique() >= 50
            and not batch["ts_code"].duplicated().any()
        ]
        if exact_batches:
            snapshot = exact_batches[-1].copy()
            selection = "latest_complete_exact_timestamp"
    snapshot = snapshot.loc[snapshot["ts_code"].str.match(A_SHARE_CODE)].copy()
    return snapshot, {
        "schema_ok": True,
        "batch_selection": selection,
        "snapshot_minute": minute.isoformat(),
    }


def normalize_snapshot_rank(
    snapshot: pd.DataFrame,
) -> tuple[pd.Series, dict[str, Any]]:
    raw_rank = pd.to_numeric(snapshot["rank"], errors="coerce")
    hot = pd.to_numeric(snapshot["hot"], errors="coerce")
    rank_coverage = float(raw_rank.notna().mean()) if len(snapshot) else 0.0
    rank_unique_ratio = (
        float(raw_rank.nunique() / len(snapshot)) if len(snapshot) else 0.0
    )
    hot_coverage = float(hot.notna().mean()) if len(snapshot) else 0.0
    raw_min = float(raw_rank.min()) if raw_rank.notna().any() else None
    raw_max = float(raw_rank.max()) if raw_rank.notna().any() else None
    paired = pd.DataFrame({"rank": raw_rank, "hot": hot}).dropna()
    rank_hot_spearman = (
        float(paired["rank"].corr(paired["hot"], method="spearman"))
        if len(paired) >= 3
        else None
    )

    ordinal = bool(
        rank_coverage >= 0.95
        and rank_unique_ratio >= 0.95
        and raw_min in (0.0, 1.0)
        and raw_max is not None
        and raw_max <= 2_000.0
    )
    legacy_score = bool(
        rank_coverage >= 0.95
        and rank_unique_ratio >= 0.95
        and raw_min is not None
        and raw_min > 2_000.0
        and hot_coverage >= 0.95
        and rank_hot_spearman is not None
        and rank_hot_spearman >= 0.99
    )
    normalized = pd.Series(np.nan, index=snapshot.index, dtype=float)
    if ordinal:
        normalized = raw_rank - float(raw_min) + 1.0
        regime = "ordinal_zero_or_one_based"
    elif legacy_score:
        normalized = raw_rank.rank(method="first", ascending=False)
        regime = "legacy_hot_score_descending"
    else:
        regime = "unusable"

    normalized_coverage = (
        float(normalized.notna().mean()) if len(snapshot) else 0.0
    )
    normalized_unique_ratio = (
        float(normalized.nunique() / len(snapshot)) if len(snapshot) else 0.0
    )
    normalized_valid = bool(
        normalized_coverage >= 0.95
        and normalized_unique_ratio >= 0.95
        and normalized.dropna().ge(1.0).all()
        and normalized.dropna().le(float(len(snapshot))).all()
    )
    return normalized, {
        "schema_regime": regime,
        "raw_rank_coverage": rank_coverage,
        "raw_rank_unique_ratio": rank_unique_ratio,
        "raw_rank_min": raw_min,
        "raw_rank_max": raw_max,
        "hot_coverage": hot_coverage,
        "rank_hot_spearman": rank_hot_spearman,
        "normalized_rank_coverage": normalized_coverage,
        "normalized_rank_unique_ratio": normalized_unique_ratio,
        "normalized_rank_valid": normalized_valid,
    }


def ths_schema_record(
    frame: pd.DataFrame,
    *,
    trade_date: str,
) -> dict[str, Any]:
    snapshot, batch = select_tail_batch(frame, trade_date=trade_date)
    if snapshot.empty:
        return {
            "trade_date": trade_date,
            "status": "ok",
            "rows": int(len(frame)),
            **batch,
            "coverage_pass": False,
        }
    normalized_rank, rank_audit = normalize_snapshot_rank(snapshot)
    duplicate_codes = bool(snapshot["ts_code"].duplicated().any())
    unique_codes = int(snapshot["ts_code"].nunique())
    date_consistent = bool(
        snapshot["normalized_trade_date"].eq(trade_date).all()
        and snapshot["rank_timestamp"]
        .dt.strftime("%Y%m%d")
        .eq(trade_date)
        .all()
    )
    concept_coverage = float(
        snapshot["concept"].fillna("").astype(str).str.strip().ne("").mean()
    )
    passed = bool(
        batch["schema_ok"]
        and unique_codes >= 50
        and not duplicate_codes
        and date_consistent
        and rank_audit["normalized_rank_valid"]
        and rank_audit["schema_regime"] != "unusable"
    )
    return {
        "trade_date": trade_date,
        "status": "ok",
        "rows": int(len(frame)),
        **batch,
        "snapshot_rows": int(len(snapshot)),
        "unique_a_share_codes": unique_codes,
        "duplicate_codes": duplicate_codes,
        "date_consistent": date_consistent,
        "concept_coverage": concept_coverage,
        **rank_audit,
        "normalized_rank_min": (
            float(normalized_rank.min())
            if normalized_rank.notna().any()
            else None
        ),
        "normalized_rank_max": (
            float(normalized_rank.max())
            if normalized_rank.notna().any()
            else None
        ),
        "coverage_pass": passed,
    }


if __name__ == "__main__":
    raise SystemExit(main())
