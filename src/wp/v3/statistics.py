from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClusteredInterval:
    clusters: int
    observations: int
    win_rate_lower: float
    win_rate_upper: float
    mean_return_lower_pct: float
    mean_return_upper_pct: float


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = proportion + z**2 / (2 * total)
    margin = z * sqrt(
        (proportion * (1 - proportion) + z**2 / (4 * total)) / total
    )
    return (centre - margin) / denominator, (centre + margin) / denominator


def day_clustered_intervals(
    frame: pd.DataFrame,
    *,
    date_column: str = "trade_date",
    return_column: str = "net_return_pct",
    samples: int = 4_000,
    seed: int = 20_260_727,
    block_days: int = 5,
) -> ClusteredInterval:
    clean = frame.reindex(columns=[date_column, return_column]).copy()
    clean[date_column] = clean[date_column].astype(str)
    clean[return_column] = pd.to_numeric(clean[return_column], errors="coerce")
    clean = clean.dropna(subset=[return_column])
    if clean.empty:
        return ClusteredInterval(0, 0, 0.0, 0.0, float("nan"), float("nan"))

    clusters = (
        clean.assign(
            _wins=clean[return_column].gt(0).astype(int),
            _count=1,
        )
        .groupby(date_column, sort=True)
        .agg(
            wins=("_wins", "sum"),
            observations=("_count", "sum"),
            return_sum=(return_column, "sum"),
        )
    )
    rng = np.random.default_rng(seed)
    block_length = max(1, min(int(block_days), len(clusters)))
    blocks_per_sample = int(np.ceil(len(clusters) / block_length))
    starts = rng.integers(
        0,
        len(clusters),
        size=(samples, blocks_per_sample),
        endpoint=False,
    )
    offsets = np.arange(block_length, dtype=int)
    choices = (
        (starts[:, :, None] + offsets[None, None, :])
        % len(clusters)
    ).reshape(samples, -1)[:, : len(clusters)]
    wins = clusters["wins"].to_numpy(dtype=float)[choices].sum(axis=1)
    observations = (
        clusters["observations"].to_numpy(dtype=float)[choices].sum(axis=1)
    )
    return_sum = (
        clusters["return_sum"].to_numpy(dtype=float)[choices].sum(axis=1)
    )
    win_rates = np.divide(
        wins,
        observations,
        out=np.zeros_like(wins),
        where=observations > 0,
    )
    mean_returns = np.divide(
        return_sum,
        observations,
        out=np.full_like(return_sum, np.nan),
        where=observations > 0,
    )
    return ClusteredInterval(
        clusters=int(len(clusters)),
        observations=int(len(clean)),
        win_rate_lower=float(np.quantile(win_rates, 0.025)),
        win_rate_upper=float(np.quantile(win_rates, 0.975)),
        mean_return_lower_pct=float(np.nanquantile(mean_returns, 0.025)),
        mean_return_upper_pct=float(np.nanquantile(mean_returns, 0.975)),
    )


def clustered_binary_lower(
    target: np.ndarray,
    dates: np.ndarray,
    *,
    samples: int = 2_000,
    seed: int = 20_260_727,
) -> tuple[int, float]:
    target_values = np.asarray(target, dtype=int)
    date_values = np.asarray(dates).astype(str)
    if len(target_values) == 0:
        return 0, 0.0
    proxy = pd.DataFrame(
        {
            "trade_date": date_values,
            "net_return_pct": np.where(target_values > 0, 1.0, -1.0),
        }
    )
    interval = day_clustered_intervals(
        proxy,
        samples=samples,
        seed=seed,
    )
    return interval.clusters, interval.win_rate_lower
