from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import CN_TZ, V3Config, policy_fingerprint
from .io import (
    atomic_write_json,
    atomic_write_parquet,
    canonical_digest,
    file_sha256,
    stable_frame_digest,
)


EVIDENCE_SCHEMA_VERSION = "wp_slot_decision_evidence_1"


def archive_signal_evidence(
    output_root: str | Path,
    *,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    source_manifest: dict[str, Any],
    inference_manifest: dict[str, Any],
    config: V3Config,
) -> dict[str, Any]:
    trade_date = str(source_manifest.get("trade_date") or "")
    signal_slot = str(source_manifest.get("signal_slot") or "")
    if len(trade_date) != 8 or not trade_date.isdigit():
        raise ValueError("signal evidence requires an eight-digit trade_date")
    if signal_slot not in config.strategy.signal_slots:
        raise ValueError("signal evidence can only archive a configured signal slot")
    if features.empty or predictions.empty:
        raise ValueError("signal evidence cannot archive an empty decision universe")

    feature_codes = set(features["ts_code"].astype(str))
    prediction_codes = set(predictions["ts_code"].astype(str))
    if feature_codes != prediction_codes:
        raise ValueError("feature and prediction universes differ")
    if len(feature_codes) != len(features) or len(prediction_codes) != len(predictions):
        raise ValueError("signal evidence requires one row per stock")

    feature_digest = stable_frame_digest(features)
    prediction_digest = stable_frame_digest(predictions)
    source_manifest_digest = canonical_digest(
        {
            key: source_manifest.get(key)
            for key in (
                "schema_version",
                "trade_date",
                "target_trade_date",
                "signal_slot",
                "market_data_time",
                "latest_bar_slot",
                "row_count",
                "fresh_row_count",
                "eligible_count",
                "feature_version",
                "expected_symbols",
                "open_universe_coverage",
                "tail_universe_coverage",
                "capture_contract",
            )
        }
    )
    identity = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "trade_date": trade_date,
        "signal_slot": signal_slot,
        "feature_version": config.model.feature_version,
        "contract_fingerprint": policy_fingerprint(config),
        "model_fingerprint": inference_manifest.get("v3_model_fingerprint"),
        "policy_fingerprint": inference_manifest.get("v3_policy_fingerprint"),
        "source_manifest_digest": source_manifest_digest,
        "feature_frame_digest": feature_digest,
        "prediction_frame_digest": prediction_digest,
    }
    evidence_digest = canonical_digest(identity)
    slot_name = signal_slot.replace(":", "")
    evidence_dir = (
        Path(output_root)
        / "audit"
        / trade_date[:4]
        / trade_date
        / slot_name
    )
    manifest_path = evidence_dir / "manifest.json"
    if manifest_path.exists():
        existing = _read_manifest(manifest_path)
        _verify_existing_evidence(
            existing,
            evidence_dir=evidence_dir,
            expected_digest=evidence_digest,
        )
        return existing

    evidence_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            dir=evidence_dir.parent,
            prefix=f".{slot_name}.",
            suffix=".tmp",
        )
    )
    try:
        universe_path = temporary_dir / "decision_universe.parquet"
        atomic_write_parquet(predictions, universe_path)
        state_counts = {
            str(key): int(value)
            for key, value in predictions.get(
                "candidate_state",
                pd.Series("UNKNOWN", index=predictions.index),
            )
            .fillna("UNKNOWN")
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        }
        rejection_counts = _rejection_counts(predictions)
        captured_at = (
            source_manifest.get("capture_completed_at")
            or source_manifest.get("market_data_time")
            or datetime.now(CN_TZ).isoformat()
        )
        manifest = {
            **identity,
            "evidence_digest": evidence_digest,
            "captured_at": str(captured_at),
            "target_trade_date": source_manifest.get("target_trade_date"),
            "market_data_time": source_manifest.get("market_data_time"),
            "latest_bar_slot": source_manifest.get("latest_bar_slot"),
            "row_count": int(len(predictions)),
            "eligible_count": int(
                predictions.get(
                    "execution_eligible",
                    pd.Series(False, index=predictions.index),
                )
                .fillna(False)
                .astype(bool)
                .sum()
            ),
            "qualified_count": int(
                predictions.get(
                    "passes_policy",
                    pd.Series(False, index=predictions.index),
                )
                .fillna(False)
                .astype(bool)
                .sum()
            ),
            "candidate_state_counts": state_counts,
            "rejection_reason_counts": rejection_counts,
            "inference_state": inference_manifest.get("v3_state"),
            "formal_authorization": bool(
                inference_manifest.get("v3_formal_authorization", False)
            ),
            "columns": list(predictions.columns.astype(str)),
            "files": {
                "decision_universe.parquet": {
                    "sha256": file_sha256(universe_path),
                    "bytes": universe_path.stat().st_size,
                }
            },
        }
        atomic_write_json(temporary_dir / "manifest.json", manifest)
        os.replace(temporary_dir, evidence_dir)
        return manifest
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def verify_signal_evidence(path: str | Path) -> dict[str, Any]:
    evidence_dir = Path(path)
    manifest = _read_manifest(evidence_dir / "manifest.json")
    _verify_existing_evidence(
        manifest,
        evidence_dir=evidence_dir,
        expected_digest=str(manifest.get("evidence_digest") or ""),
    )
    return manifest


def _verify_existing_evidence(
    manifest: dict[str, Any],
    *,
    evidence_dir: Path,
    expected_digest: str,
) -> None:
    if manifest.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported signal evidence under {evidence_dir}")
    if not expected_digest or manifest.get("evidence_digest") != expected_digest:
        raise RuntimeError(
            f"immutable signal evidence conflict under {evidence_dir}"
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError(f"signal evidence file index is missing under {evidence_dir}")
    for name, metadata in files.items():
        target = evidence_dir / str(name)
        expected_sha = str((metadata or {}).get("sha256") or "")
        if (
            not target.is_file()
            or not expected_sha
            or file_sha256(target) != expected_sha
        ):
            raise RuntimeError(f"signal evidence file failed verification: {target}")


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read signal evidence manifest {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid signal evidence manifest {path}")
    return payload


def _rejection_counts(predictions: pd.DataFrame) -> dict[str, int]:
    if "rejection_reasons" not in predictions:
        return {}
    counts: Counter[str] = Counter()
    for value in predictions["rejection_reasons"].fillna("").astype(str):
        for reason in (item for item in value.split("|") if item):
            counts[reason] += 1
    return dict(sorted(counts.items()))
