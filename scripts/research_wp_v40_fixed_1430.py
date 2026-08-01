from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from wp.v3.contracts import load_v3_config, policy_fingerprint
from wp.v3.exit_risk import fit_exit_failure_risk
from wp.v3.features import FEATURE_COLUMNS
from wp.v3.history import load_panel_partitions
from wp.v3.io import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
)
from wp.v3.meta_alpha import fit_meta_alpha, prune_candidate_universe
from wp.v3.model import load_bundle
from wp.v3.registry import (
    load_registry,
    register_research_model,
    save_registry,
)
from wp.v3.sharding import (
    AGGREGATE_PREDICTION_COLUMNS,
    SHARD_MANIFEST_NAME,
    SHARD_PREDICTIONS_NAME,
    SHARD_SCHEMA_VERSION,
)
from wp.v3.v40 import (
    V40Policy,
    attach_v40_policy_gates,
    evaluate_v40_fixed_1430,
    v40_historical_gate,
)
from wp.v3.v40_model import (
    META_CALIBRATION_DAYS,
    META_TRAIN_DAYS,
    PURGE_DAYS,
    RISK_CALIBRATION_DAYS,
    RISK_TRAIN_DAYS,
    save_v40_bundle,
    train_v40_bundle,
    v40_bundle_metadata,
)


IDENTITY = ["trade_date", "signal_slot", "ts_code"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build strict nested-OOS V40 evidence and a forward-only "
            "fixed-14:30 deployable shadow bundle."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--registry",
        default="outputs/json/wp_model_registry_v3.json",
    )
    parser.add_argument("--source-run-id")
    parser.add_argument("--start-date", default="20260501")
    parser.add_argument("--end-date", default="20260731")
    parser.add_argument("--top-per-score", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_v3_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = load_verified_shards(args.shard_dir, config=config)
    frontier = prune_candidate_universe(
        raw,
        top_per_score=args.top_per_score,
        require_label=False,
    )
    print(
        "[wp-v40] aggregate source "
        f"raw_rows={len(raw):,} "
        f"raw_folds={raw['fold'].nunique()} "
        f"execution_eligible={int(_boolean(raw['execution_eligible']).sum()):,} "
        f"labelled={int(_boolean(raw['label_available']).sum()):,} "
        f"frontier_rows={len(frontier):,} "
        f"frontier_dates={frontier['trade_date'].nunique() if not frontier.empty else 0} "
        f"frontier_folds={frontier['fold'].nunique() if not frontier.empty else 0}",
        flush=True,
    )
    if frontier.empty:
        raise RuntimeError(
            "V40 candidate frontier is empty after execution pruning"
        )
    frontier = attach_original_features(
        frontier,
        args.panel_dir,
        start_date=config.history.evaluation_start_date,
        end_date=config.history.evaluation_end_date,
    )
    scored_frames: list[pd.DataFrame] = []
    fold_audit: list[dict[str, Any]] = []
    for fold in sorted(
        pd.to_numeric(frontier["fold"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    ):
        current = pd.to_numeric(
            frontier["fold"],
            errors="coerce",
        ).eq(fold)
        test = frontier.loc[current].copy()
        test_start = str(test["trade_date"].min())
        history = frontier.loc[
            ~current
            & frontier["trade_date"].astype(str).lt(test_start)
            & _boolean(frontier["label_available"])
            & _boolean(frontier["execution_eligible"])
        ].copy()
        dates = sorted(history["trade_date"].astype(str).unique())
        meta_segments = rolling_segments(
            dates,
            train_days=META_TRAIN_DAYS,
            calibration_days=META_CALIBRATION_DAYS,
        )
        risk_segments = rolling_segments(
            dates,
            train_days=RISK_TRAIN_DAYS,
            calibration_days=RISK_CALIBRATION_DAYS,
        )
        base_row = {
            "fold": int(fold),
            "test_start": test_start,
            "test_end": str(test["trade_date"].max()),
            "test_rows": int(len(test)),
        }
        if meta_segments is None or risk_segments is None:
            audit = {
                **base_row,
                "scored": False,
                "reason": "insufficient_strictly_prior_history",
                "history_rows": int(len(history)),
                "history_dates": int(len(dates)),
            }
            fold_audit.append(audit)
            print(
                "[wp-v40] skipped "
                + json.dumps(
                    audit,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            continue
        meta_train_dates, meta_calibration_dates = meta_segments
        risk_train_dates, risk_calibration_dates = risk_segments
        meta_train = history.loc[
            history["trade_date"].isin(meta_train_dates)
        ].copy()
        meta_calibration = history.loc[
            history["trade_date"].isin(meta_calibration_dates)
        ].copy()
        risk_history = history.loc[
            _boolean(history["entry_fillable"])
        ].copy()
        risk_train = risk_history.loc[
            risk_history["trade_date"].isin(risk_train_dates)
        ].copy()
        risk_calibration = risk_history.loc[
            risk_history["trade_date"].isin(risk_calibration_dates)
        ].copy()
        try:
            meta = fit_meta_alpha(
                meta_train,
                meta_calibration,
                random_seed=config.model.random_seed + fold * 101,
            )
            risk = fit_exit_failure_risk(
                risk_train,
                risk_calibration,
                random_seed=config.model.random_seed + fold * 139,
            )
        except ValueError as error:
            audit = {
                **base_row,
                "scored": False,
                "reason": str(error),
                "history_rows": int(len(history)),
                "history_dates": int(len(dates)),
                "meta_train_rows": int(len(meta_train)),
                "meta_calibration_rows": int(len(meta_calibration)),
                "risk_train_rows": int(len(risk_train)),
                "risk_calibration_rows": int(len(risk_calibration)),
                "risk_train_failures": int(
                    (~_boolean(risk_train["exit_fillable"])).sum()
                ),
                "risk_calibration_failures": int(
                    (~_boolean(risk_calibration["exit_fillable"])).sum()
                ),
            }
            fold_audit.append(audit)
            print(
                "[wp-v40] skipped "
                + json.dumps(
                    audit,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            continue
        scored = risk.predict(meta.predict(test))
        scored["v40_outer_fold"] = int(fold)
        scored = attach_v40_policy_gates(
            scored,
            V40Policy(
                observation_count=config.strategy.observation_count
            ),
        )
        scored_frames.append(scored)
        fold_audit.append(
            {
                **base_row,
                "scored": True,
                "reason": "strictly_prior_meta_and_exit_risk_models",
                "meta_train_start": meta_train_dates[0],
                "meta_train_end": meta_train_dates[-1],
                "meta_calibration_start": meta_calibration_dates[0],
                "meta_calibration_end": meta_calibration_dates[-1],
                "risk_train_start": risk_train_dates[0],
                "risk_train_end": risk_train_dates[-1],
                "risk_calibration_start": risk_calibration_dates[0],
                "risk_calibration_end": risk_calibration_dates[-1],
            }
        )
        print(
            f"[wp-v40] fold={fold} test={base_row['test_start']}.."
            f"{base_row['test_end']} rows={len(test):,}",
            flush=True,
        )
    if not scored_frames:
        raise RuntimeError(
            "V40 produced no strictly out-of-sample fold; audit="
            + json.dumps(
                fold_audit,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    scored_all = pd.concat(scored_frames, ignore_index=True)
    expected_trade_dates = (
        load_panel_partitions(
            args.panel_dir,
            columns=["trade_date"],
            start_date=args.start_date,
            end_date=args.end_date,
        )["trade_date"]
        .astype(str)
        .unique()
        .tolist()
    )
    result = evaluate_v40_fixed_1430(
        scored_all,
        config,
        start_date=args.start_date,
        end_date=args.end_date,
        source_run_id=args.source_run_id,
        expected_trade_dates=expected_trade_dates,
    )
    summary = {
        **result.summary,
        "folds": fold_audit,
        "source": {
            **result.summary["source"],
            "verified_shard_rows": int(len(raw)),
            "pruned_frontier_rows": int(len(frontier)),
            "scored_oos_rows": int(len(scored_all)),
            "top_per_base_score": int(args.top_per_score),
        },
    }
    backtest_gate = v40_historical_gate(summary, config)
    summary["backtest_gate"] = backtest_gate

    base_bundle = load_bundle(args.base_model)
    deployable = train_v40_bundle(
        base_bundle=base_bundle,
        oos_frontier=frontier,
        config=config,
        top_per_base_score=args.top_per_score,
    )
    model_path = (
        output / "models" / f"{deployable.fingerprint}.joblib"
    )
    save_v40_bundle(deployable, model_path)
    metadata = v40_bundle_metadata(deployable)
    summary["model"] = metadata
    atomic_write_json(output / "wp_v40_model_metadata.json", metadata)
    atomic_write_parquet(
        scored_all,
        output / "wp_v40_scored_oos_frontier.parquet",
    )
    atomic_write_json(
        output / "wp_v40_backtest_202605_202607.json",
        summary,
    )
    atomic_write_csv(
        result.qualified,
        output / "wp_v40_backtest_qualified_202605_202607.csv",
    )
    atomic_write_csv(
        result.observations,
        output / "wp_v40_backtest_observations_202605_202607.csv",
    )

    outputs_json = Path("outputs/json")
    outputs_csv = Path("outputs/csv")
    outputs_json.mkdir(parents=True, exist_ok=True)
    outputs_csv.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        outputs_json / "wp_v40_backtest_202605_202607.json",
        summary,
    )
    atomic_write_csv(
        result.qualified,
        outputs_csv / "wp_v40_backtest_qualified_202605_202607.csv",
    )
    atomic_write_csv(
        result.observations,
        outputs_csv
        / "wp_v40_backtest_observations_202605_202607.csv",
    )
    registry = load_registry(args.registry)
    record = register_research_model(
        registry,
        metadata=metadata,
        backtest={
            "backtest_gate": backtest_gate,
            "metrics": summary["qualified"]["metrics"],
            "contract": summary["policy"],
            "evidence_status": summary["status"],
        },
        artifact_path=model_path.as_posix(),
    )
    record["shadow"]["evidence_scope"] = "exact_model"
    record["shadow"]["model_fingerprint"] = deployable.fingerprint
    record["shadow"]["started_trade_date"] = (
        config.evidence.live_shadow_start_date
    )
    save_registry(registry, args.registry)
    marker = {
        "status": summary["status"],
        "backtest_gate": backtest_gate,
        "qualified": summary["qualified"],
        "observations": summary["observations"],
        "model": metadata,
        "registry_status": record["status"],
    }
    print(
        "WP_V40_RESEARCH_RESULT="
        + json.dumps(
            marker,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def load_verified_shards(
    path: str | Path,
    *,
    config: Any,
) -> pd.DataFrame:
    root = Path(path)
    manifests = sorted(root.rglob(SHARD_MANIFEST_NAME))
    if not manifests:
        raise FileNotFoundError(f"no V9 shard manifests under {root}")
    expected_policy = policy_fingerprint(config)
    frames: list[pd.DataFrame] = []
    seen_folds: set[int] = set()
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SHARD_SCHEMA_VERSION:
            raise RuntimeError(f"invalid shard schema: {manifest_path}")
        if manifest.get("policy_fingerprint") != expected_policy:
            raise RuntimeError(f"shard policy mismatch: {manifest_path}")
        prediction_path = manifest_path.with_name(SHARD_PREDICTIONS_NAME)
        expected_sha = str(manifest.get("prediction_sha256") or "")
        if sha256(prediction_path) != expected_sha:
            raise RuntimeError(f"shard digest mismatch: {prediction_path}")
        frame = pd.read_parquet(prediction_path)
        available = [
            column
            for column in AGGREGATE_PREDICTION_COLUMNS
            if column in frame
        ]
        frame = frame.loc[:, available].copy()
        folds = set(
            pd.to_numeric(frame["fold"], errors="raise").astype(int).unique()
        )
        if seen_folds & folds:
            raise RuntimeError(
                f"duplicate V9 folds: {sorted(seen_folds & folds)}"
            )
        seen_folds.update(folds)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    for column in IDENTITY:
        result[column] = result[column].astype(str)
    if result.duplicated(IDENTITY, keep=False).any():
        raise RuntimeError("V9 OOS shards contain duplicate identities")
    return result.sort_values(
        ["fold", *IDENTITY],
        kind="stable",
    ).reset_index(drop=True)


def attach_original_features(
    frontier: pd.DataFrame,
    panel_dir: str | Path,
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    missing_features = [
        column for column in FEATURE_COLUMNS if column not in frontier
    ]
    if not missing_features:
        return frontier
    requested = frontier.loc[:, IDENTITY].drop_duplicates()
    panel_columns = list(dict.fromkeys([*IDENTITY, *missing_features]))
    panel = load_panel_partitions(
        panel_dir,
        columns=panel_columns,
        start_date=start_date,
        end_date=end_date,
    )
    for column in IDENTITY:
        panel[column] = panel[column].astype(str)
    duplicated = panel.duplicated(IDENTITY, keep=False)
    if duplicated.any():
        examples = (
            panel.loc[duplicated, IDENTITY]
            .head(5)
            .to_dict(orient="records")
        )
        raise RuntimeError(
            "V40 causal feature panel contains duplicate identities: "
            f"{examples}"
        )
    matched = requested.merge(
        panel,
        on=IDENTITY,
        how="left",
        validate="one_to_one",
    )
    if len(matched) != len(requested):
        raise RuntimeError("V40 feature join changed identity count")
    completely_missing = [
        column
        for column in missing_features
        if matched[column].notna().sum() == 0
    ]
    if len(completely_missing) == len(missing_features):
        raise RuntimeError("V40 matched no original causal features")
    return frontier.merge(
        matched,
        on=IDENTITY,
        how="left",
        validate="one_to_one",
    )


def rolling_segments(
    prior_dates: list[str],
    *,
    train_days: int,
    calibration_days: int,
) -> tuple[list[str], list[str]] | None:
    needed = train_days + PURGE_DAYS + calibration_days
    if len(prior_dates) < needed:
        return None
    selected = prior_dates[-needed:]
    train = selected[:train_days]
    calibration = selected[train_days + PURGE_DAYS :]
    return train, calibration


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _boolean(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y", "qualified", "pass"})
    )


if __name__ == "__main__":
    raise SystemExit(main())
