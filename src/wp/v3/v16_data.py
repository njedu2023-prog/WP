from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .io import (
    atomic_write_json,
    atomic_write_parquet,
    canonical_digest,
    file_sha256,
)


SCHEMA_VERSION = "wp_v16_t1_full_day_5m_1"
IDENTITY_COLUMNS = ("trade_date", "signal_slot", "ts_code")
PAIR_COLUMNS = ("target_trade_date", "ts_code")
MINUTE_COLUMNS = (
    "target_trade_date",
    "ts_code",
    "trade_time",
    "bar_slot",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
)
DEFAULT_MINIMUM_BARS = 44
DEFAULT_MINIMUM_PAIR_COVERAGE = 0.98


def candidate_exit_pairs(
    frontier: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Map each immutable T-day candidate identity to its T+1 symbol/date pair."""
    missing_frontier = sorted(set(IDENTITY_COLUMNS) - set(frontier.columns))
    missing_panel = sorted(
        set((*IDENTITY_COLUMNS, "target_trade_date")) - set(panel.columns)
    )
    if missing_frontier:
        raise ValueError(f"frontier missing identity columns: {missing_frontier}")
    if missing_panel:
        raise ValueError(f"panel missing exit-pair columns: {missing_panel}")

    identities = frontier.loc[:, IDENTITY_COLUMNS].copy()
    mapping = panel.loc[
        :,
        [*IDENTITY_COLUMNS, "target_trade_date"],
    ].copy()
    for frame in (identities, mapping):
        for column in IDENTITY_COLUMNS:
            frame[column] = frame[column].astype(str)
    mapping["target_trade_date"] = _date_text(mapping["target_trade_date"])
    if mapping.duplicated(list(IDENTITY_COLUMNS), keep=False).any():
        raise ValueError("panel contains duplicate T-day candidate identities")

    joined = identities.merge(
        mapping,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    missing = joined["_merge"].ne("both") | joined["target_trade_date"].eq("")
    if missing.any():
        examples = joined.loc[
            missing,
            list(IDENTITY_COLUMNS),
        ].head(5).to_dict(orient="records")
        raise RuntimeError(
            f"T+1 date mapping missing for {int(missing.sum())} candidates: "
            f"{examples}"
        )
    return (
        joined.loc[:, PAIR_COLUMNS]
        .drop_duplicates()
        .sort_values(list(PAIR_COLUMNS), kind="stable")
        .reset_index(drop=True)
    )


def normalize_full_day_minutes(
    raw_minutes: pd.DataFrame,
    required_pairs: pd.DataFrame,
    *,
    minimum_bars: int = DEFAULT_MINIMUM_BARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize full T+1 five-minute paths and produce pair-level quality."""
    if minimum_bars < 1:
        raise ValueError("minimum_bars must be positive")
    missing_pairs = sorted(set(PAIR_COLUMNS) - set(required_pairs.columns))
    if missing_pairs:
        raise ValueError(f"required pairs missing columns: {missing_pairs}")
    required = required_pairs.loc[:, PAIR_COLUMNS].copy()
    required["target_trade_date"] = _date_text(required["target_trade_date"])
    required["ts_code"] = required["ts_code"].astype(str)
    required = required.drop_duplicates().reset_index(drop=True)
    if required.empty:
        raise ValueError("required T+1 pairs are empty")

    minute_required = {
        "ts_code",
        "trade_time",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
    }
    missing_minutes = sorted(minute_required - set(raw_minutes.columns))
    if missing_minutes:
        raise ValueError(f"raw minutes missing columns: {missing_minutes}")

    minutes = raw_minutes.loc[:, sorted(minute_required)].copy()
    minutes["ts_code"] = minutes["ts_code"].astype(str)
    minutes["trade_time"] = pd.to_datetime(
        minutes["trade_time"],
        errors="coerce",
    )
    minutes = minutes.dropna(subset=["trade_time", "ts_code"])
    minutes["target_trade_date"] = minutes["trade_time"].dt.strftime("%Y%m%d")
    minutes["bar_slot"] = minutes["trade_time"].dt.strftime("%H:%M")
    for column in ("open", "high", "low", "close", "vol", "amount"):
        minutes[column] = pd.to_numeric(minutes[column], errors="coerce")
    minutes = minutes.merge(
        required.assign(_required_pair=True),
        on=list(PAIR_COLUMNS),
        how="inner",
        validate="many_to_one",
    ).drop(columns="_required_pair")
    minutes = minutes.loc[
        minutes["bar_slot"].between("09:30", "15:00", inclusive="both")
    ].copy()
    minutes.sort_values(
        ["target_trade_date", "ts_code", "trade_time"],
        kind="stable",
        inplace=True,
    )
    minutes.drop_duplicates(
        ["target_trade_date", "ts_code", "trade_time"],
        keep="last",
        inplace=True,
    )

    valid_bar = (
        minutes["open"].gt(0)
        & minutes["high"].gt(0)
        & minutes["low"].gt(0)
        & minutes["close"].gt(0)
        & minutes["vol"].ge(0)
        & minutes["amount"].ge(0)
    )
    observed = (
        minutes.assign(_valid=valid_bar.astype(int))
        .groupby(list(PAIR_COLUMNS), sort=False)
        .agg(
            bars=("trade_time", "size"),
            valid_bars=("_valid", "sum"),
            first_slot=("bar_slot", "min"),
            last_slot=("bar_slot", "max"),
        )
        .reset_index()
    )
    quality = required.merge(
        observed,
        on=list(PAIR_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    quality["bars"] = pd.to_numeric(quality["bars"], errors="coerce").fillna(0)
    quality["valid_bars"] = (
        pd.to_numeric(quality["valid_bars"], errors="coerce").fillna(0)
    )
    quality["has_close_bar"] = quality["last_slot"].astype(str).eq("15:00")
    quality["covered"] = (
        quality["valid_bars"].ge(minimum_bars)
        & quality["has_close_bar"]
    )
    normalized = minutes.reindex(columns=MINUTE_COLUMNS).reset_index(drop=True)
    return normalized, quality


def write_exit_path_dataset(
    minutes: pd.DataFrame,
    quality: pd.DataFrame,
    output_dir: str | Path,
    *,
    source: dict[str, Any],
    minimum_pair_coverage: float = DEFAULT_MINIMUM_PAIR_COVERAGE,
    minimum_bars: int = DEFAULT_MINIMUM_BARS,
) -> dict[str, Any]:
    if not 0.0 < minimum_pair_coverage <= 1.0:
        raise ValueError("minimum_pair_coverage must be in (0, 1]")
    if quality.empty:
        raise ValueError("exit-path quality is empty")
    required = int(len(quality))
    covered = int(quality["covered"].fillna(False).astype(bool).sum())
    coverage = covered / required
    if coverage < minimum_pair_coverage:
        examples = quality.loc[
            ~quality["covered"].fillna(False).astype(bool),
            [*PAIR_COLUMNS, "bars", "valid_bars", "last_slot"],
        ].head(8).to_dict(orient="records")
        raise RuntimeError(
            f"T+1 full-day pair coverage {coverage:.2%} is below "
            f"{minimum_pair_coverage:.2%}; examples={examples}"
        )

    output = Path(output_dir)
    partition_dir = output / "minute"
    partition_dir.mkdir(parents=True, exist_ok=True)
    clean = minutes.copy()
    clean["target_trade_date"] = _date_text(clean["target_trade_date"])
    clean = clean.loc[
        clean.set_index(list(PAIR_COLUMNS)).index.isin(
            quality.loc[
                quality["covered"].fillna(False).astype(bool),
                list(PAIR_COLUMNS),
            ].set_index(list(PAIR_COLUMNS)).index
        )
    ].copy()
    partitions: list[dict[str, Any]] = []
    for month, frame in clean.groupby(
        clean["target_trade_date"].str[:6],
        sort=True,
    ):
        path = atomic_write_parquet(
            frame.reset_index(drop=True),
            partition_dir / f"wp_v16_t1_minutes_{month}.parquet",
        )
        partitions.append(
            {
                "month": str(month),
                "path": path.name,
                "rows": int(len(frame)),
                "pairs": int(
                    frame.loc[:, PAIR_COLUMNS].drop_duplicates().shape[0]
                ),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )

    quality_path = atomic_write_parquet(
        quality.sort_values(list(PAIR_COLUMNS), kind="stable").reset_index(
            drop=True
        ),
        output / "wp_v16_t1_path_quality.parquet",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "source_digest": canonical_digest(source),
        "contract": {
            "bar_frequency": "5min",
            "session": "09:30-15:00",
            "minimum_valid_bars_per_pair": minimum_bars,
            "minimum_pair_coverage": minimum_pair_coverage,
            "future_information_allowed_in_t_day_entry_model": False,
            "permitted_use": (
                "T+1 exit research and truth labels after the T-day candidate "
                "identity has been frozen"
            ),
        },
        "required_pairs": required,
        "covered_pairs": covered,
        "pair_coverage": coverage,
        "rows": int(len(clean)),
        "trade_dates": int(clean["target_trade_date"].nunique()),
        "symbols": int(clean["ts_code"].nunique()),
        "quality_sha256": file_sha256(quality_path),
        "partitions": partitions,
    }
    manifest["dataset_fingerprint"] = canonical_digest(
        {
            "schema_version": manifest["schema_version"],
            "source_digest": manifest["source_digest"],
            "contract": manifest["contract"],
            "required_pairs": required,
            "covered_pairs": covered,
            "partitions": [
                {
                    "month": item["month"],
                    "rows": item["rows"],
                    "pairs": item["pairs"],
                    "sha256": item["sha256"],
                }
                for item in partitions
            ],
        }
    )
    atomic_write_json(output / "wp_v16_t1_path_manifest.json", manifest)
    return manifest


def load_exit_path_partitions(
    path: str | Path,
    *,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    root = Path(path)
    files = sorted((root / "minute").glob("wp_v16_t1_minutes_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no V16 exit-path partitions under {root}")
    frames = [pd.read_parquet(file, columns=columns) for file in files]
    return pd.concat(frames, ignore_index=True)


def _date_text(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.replace("-", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
        .where(lambda series: series.str.fullmatch(r"\d{8}"), "")
    )
