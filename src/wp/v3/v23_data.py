from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .io import file_sha256
from .meta_alpha import IDENTITY_COLUMNS
from .sharding import (
    SHARD_MANIFEST_NAME,
    SHARD_PREDICTIONS_NAME,
    SHARD_SCHEMA_VERSION,
)
from .v19_recall import build_recall_frontier
from .v22_market_license import build_market_slot_leaders


SCHEMA_VERSION = "wp_v23_point_in_time_features_1"
MINIMUM_MINUTE_COVERAGE = 0.90
MINIMUM_DATASET_COVERAGE = 0.98

AUCTION_FIELDS = (
    "ts_code,trade_date,close,open,high,low,vol,amount,vwap"
)
MONEYFLOW_FIELDS = (
    "ts_code,trade_date,buy_sm_amount,sell_sm_amount,"
    "buy_md_amount,sell_md_amount,buy_lg_amount,sell_lg_amount,"
    "buy_elg_amount,sell_elg_amount,net_mf_amount"
)

SOURCE_SELECTION_COLUMNS = (
    *IDENTITY_COLUMNS,
    "fold",
    "signal_price",
    "ret_from_prev_close_pct",
    "execution_eligible",
    "data_age_seconds",
    "p_net_positive",
    "p_net_positive_lower",
    "p_conditional_net_positive",
    "p_cross_section_top",
    "p_severe_loss",
    "p_round_trip_fill_lower",
    "probability_model_spread",
    "expected_return_model_spread",
    "expected_utility_pct",
    "expected_utility_lower_pct",
    "selection_score",
    "model_version",
    "model_fingerprint",
    "policy_fingerprint",
)

MINUTE_FEATURE_COLUMNS = (
    "v23_m1_coverage_ratio",
    "v23_m1_zero_amount_share",
    "v23_m1_ret_1m_pct",
    "v23_m1_ret_3m_pct",
    "v23_m1_ret_5m_pct",
    "v23_m1_ret_10m_pct",
    "v23_m1_ret_from_1400_pct",
    "v23_m1_realized_volatility_pct",
    "v23_m1_downside_volatility_pct",
    "v23_m1_return_skew",
    "v23_m1_return_autocorr1",
    "v23_m1_directional_efficiency",
    "v23_m1_max_drawdown_pct",
    "v23_m1_rebound_from_low_pct",
    "v23_m1_reversal_from_high_pct",
    "v23_m1_up_minute_share",
    "v23_m1_down_minute_share",
    "v23_m1_max_up_streak",
    "v23_m1_max_down_streak",
    "v23_m1_signed_amount_imbalance",
    "v23_m1_amihud_proxy",
    "v23_m1_amount_concentration",
    "v23_m1_last3_amount_share",
    "v23_m1_last5_amount_share",
    "v23_m1_amount_acceleration",
    "v23_m1_vwap_gap_pct",
    "v23_m1_close_position",
    "v23_m1_upper_wick_pressure",
    "v23_m1_lower_wick_pressure",
    "v23_m1_high_break_count",
    "v23_m1_low_break_count",
    "v23_m1_last5_vs_prior5_return_pct",
)

AUCTION_FEATURE_COLUMNS = (
    "v23_auction_gap_from_prev_close_pct",
    "v23_auction_return_pct",
    "v23_auction_range_pct",
    "v23_auction_close_position",
    "v23_auction_vwap_gap_pct",
    "v23_auction_amount_log",
    "v23_auction_volume_log",
)

MONEYFLOW_FEATURE_COLUMNS = (
    "v23_prev_mf_net_share",
    "v23_prev_mf_large_share",
    "v23_prev_mf_medium_share",
    "v23_prev_mf_small_share",
    "v23_prev_mf_institution_retail_spread",
    "v23_prev_mf_gross_amount_log",
)

QUALITY_COLUMNS = (
    "v23_minute_observed_rows",
    "v23_minute_expected_rows",
    "v23_minute_latest_time",
    "v23_minute_causal_ok",
    "v23_auction_available",
    "v23_moneyflow_available",
    "v23_prev_trade_date",
    "v23_point_in_time_complete",
)

V23_FEATURE_COLUMNS = (
    *MINUTE_FEATURE_COLUMNS,
    *AUCTION_FEATURE_COLUMNS,
    *MONEYFLOW_FEATURE_COLUMNS,
)

FORBIDDEN_SOURCE_TOKENS = (
    "target",
    "label",
    "truth",
    "future",
    "gross_return",
    "net_return",
    "t1_",
    "exit_",
)


def load_v23_source_leaders(
    shard_dir: str | Path,
    *,
    evaluation_end: str,
    top_per_source: int,
    exploration_per_slot: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the immutable V9 leader index without reading profit outcomes."""
    root = Path(shard_dir)
    manifests = sorted(root.rglob(SHARD_MANIFEST_NAME))
    if not manifests:
        raise FileNotFoundError(f"no V9 shard manifests under {root}")
    contaminated = [
        column
        for column in SOURCE_SELECTION_COLUMNS
        if any(token in column.lower() for token in FORBIDDEN_SOURCE_TOKENS)
    ]
    if contaminated:
        raise RuntimeError(
            f"V23 source projection contains outcomes: {contaminated}"
        )

    frames: list[pd.DataFrame] = []
    source_rows = 0
    expected_folds: set[int] = set()
    produced_folds: set[int] = set()
    source_shards: list[dict[str, Any]] = []
    dataset_digests: set[str] = set()
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SHARD_SCHEMA_VERSION:
            raise RuntimeError(f"invalid V9 shard schema: {manifest_path}")
        expected = {int(value) for value in manifest["expected_folds"]}
        produced = {int(value) for value in manifest["produced_folds"]}
        if not expected or produced != expected:
            raise RuntimeError(
                f"V9 shard fold mismatch: expected={expected} produced={produced}"
            )
        expected_folds.update(expected)
        dataset_digests.add(
            str(manifest.get("dataset_manifest_sha256") or "")
        )

        prediction_path = manifest_path.with_name(SHARD_PREDICTIONS_NAME)
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        actual_sha = file_sha256(prediction_path)
        if actual_sha != str(manifest.get("prediction_sha256") or ""):
            raise RuntimeError(
                f"V9 prediction digest mismatch: {prediction_path}"
            )
        available = set(pq.read_schema(prediction_path).names)
        missing = sorted(set(SOURCE_SELECTION_COLUMNS) - available)
        if missing:
            raise RuntimeError(
                f"V9 source projection missing {missing}: {prediction_path}"
            )
        frame = pq.read_table(
            prediction_path,
            columns=list(SOURCE_SELECTION_COLUMNS),
        ).to_pandas()
        if len(frame) != int(manifest.get("prediction_rows", -1)):
            raise RuntimeError(
                f"V9 prediction row count mismatch: {prediction_path}"
            )
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame = frame.loc[
            frame["trade_date"].le(str(evaluation_end))
        ].copy()
        folds = {
            int(value)
            for value in pd.to_numeric(frame["fold"], errors="coerce")
            .dropna()
            .astype(int)
        }
        overlap = produced_folds.intersection(folds)
        if overlap:
            raise RuntimeError(f"duplicate V9 folds: {sorted(overlap)}")
        produced_folds.update(folds)
        source_rows += len(frame)
        frontier = build_recall_frontier(
            frame,
            top_per_source=top_per_source,
            exploration_per_slot=exploration_per_slot,
            require_label=False,
        )
        leaders = build_market_slot_leaders(frontier)
        frames.append(leaders)
        source_shards.append(
            {
                "manifest": str(manifest_path.relative_to(root)),
                "prediction_sha256": actual_sha,
                "folds": sorted(folds),
                "source_rows": int(len(frame)),
                "frontier_rows": int(len(frontier)),
                "leader_rows": int(len(leaders)),
            }
        )

    if not frames:
        raise RuntimeError("V23 source contains no leader rows")
    if produced_folds != expected_folds:
        raise RuntimeError(
            "V23 source folds incomplete: "
            f"produced={sorted(produced_folds)} "
            f"expected={sorted(expected_folds)}"
        )
    if len(dataset_digests - {""}) != 1 or "" in dataset_digests:
        raise RuntimeError("V23 source dataset digest is inconsistent")

    result = pd.concat(frames, ignore_index=True)
    result.sort_values(["fold", *IDENTITY_COLUMNS], kind="stable", inplace=True)
    result.reset_index(drop=True, inplace=True)
    if result.duplicated(list(IDENTITY_COLUMNS), keep=False).any():
        raise RuntimeError("V23 source leaders contain duplicate identities")
    retained = (
        *IDENTITY_COLUMNS,
        "fold",
        "signal_price",
        "ret_from_prev_close_pct",
        "selection_score",
        "model_version",
        "model_fingerprint",
        "policy_fingerprint",
    )
    result = result.reindex(columns=retained)
    return result, {
        "schema_version": "wp_v23_v9_leader_source_1",
        "profit_outcomes_read": False,
        "source_rows": int(source_rows),
        "leader_rows": int(len(result)),
        "folds": sorted(produced_folds),
        "dataset_manifest_sha256": next(
            iter(dataset_digests - {""})
        ),
        "shards": source_shards,
        "source_integrity": True,
    }


def required_stock_months(
    leaders: pd.DataFrame,
) -> dict[str, dict[str, tuple[str, ...]]]:
    required = _leader_identity(leaders)
    required["month"] = required["trade_date"].str[:6]
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for (month, ts_code), group in required.groupby(
        ["month", "ts_code"],
        sort=True,
    ):
        result.setdefault(str(month), {})[str(ts_code)] = tuple(
            sorted(group["trade_date"].astype(str).unique())
        )
    return result


def required_codes_by_date(
    leaders: pd.DataFrame,
) -> dict[str, tuple[str, ...]]:
    required = _leader_identity(leaders)
    return {
        str(trade_date): tuple(sorted(group["ts_code"].astype(str).unique()))
        for trade_date, group in required.groupby("trade_date", sort=True)
    }


def attach_previous_trade_dates(
    leaders: pd.DataFrame,
    open_trade_dates: Iterable[str],
) -> pd.DataFrame:
    result = leaders.copy()
    dates = sorted(set(map(str, open_trade_dates)))
    previous = {date: dates[index - 1] for index, date in enumerate(dates) if index}
    result["v23_prev_trade_date"] = result["trade_date"].astype(str).map(
        previous
    )
    if result["v23_prev_trade_date"].isna().any():
        missing = sorted(
            result.loc[
                result["v23_prev_trade_date"].isna(),
                "trade_date",
            ]
            .astype(str)
            .unique()
        )
        raise RuntimeError(
            f"V23 trade calendar lacks previous dates for {missing[:5]}"
        )
    if not (
        result["v23_prev_trade_date"].astype(str)
        < result["trade_date"].astype(str)
    ).all():
        raise RuntimeError("V23 previous-day map is not strictly causal")
    return result


def normalize_one_minute(frame: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "ts_code",
        "trade_time",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
    )
    result = frame.reindex(columns=columns).copy()
    result["ts_code"] = result["ts_code"].astype(str)
    result["trade_time"] = pd.to_datetime(
        result["trade_time"],
        errors="coerce",
    )
    result = result.dropna(subset=["ts_code", "trade_time"])
    result["trade_date"] = result["trade_time"].dt.strftime("%Y%m%d")
    for column in ("open", "high", "low", "close", "vol", "amount"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.loc[
        result["trade_time"].dt.strftime("%H:%M").between("13:55", "15:00")
    ].copy()
    result.sort_values(["ts_code", "trade_time"], kind="stable", inplace=True)
    result.drop_duplicates(["ts_code", "trade_time"], keep="last", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def build_minute_features(
    leaders: pd.DataFrame,
    minutes: pd.DataFrame,
) -> pd.DataFrame:
    required = _leader_identity(leaders)
    normalized = normalize_one_minute(minutes)
    grouped = {
        (str(ts_code), str(trade_date)): group.reset_index(drop=True)
        for (ts_code, trade_date), group in normalized.groupby(
            ["ts_code", "trade_date"],
            sort=False,
        )
    }
    rows: list[dict[str, Any]] = []
    for record in required.to_dict(orient="records"):
        trade_date = str(record["trade_date"])
        signal_slot = str(record["signal_slot"])
        signal_time = pd.Timestamp(
            f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]} "
            f"{signal_slot}:00"
        )
        source = grouped.get(
            (str(record["ts_code"]), trade_date),
            normalized.head(0),
        )
        bars = source.loc[source["trade_time"].le(signal_time)].copy()
        bars = bars.loc[
            bars["trade_time"].dt.strftime("%H:%M").ge("13:55")
        ]
        expected = int(
            (signal_time - signal_time.normalize().replace(
                hour=13,
                minute=55,
            )).total_seconds()
            // 60
            + 1
        )
        latest = bars["trade_time"].max() if not bars.empty else pd.NaT
        causal = bool(pd.isna(latest) or latest <= signal_time)
        values = _minute_feature_values(bars)
        rows.append(
            {
                **record,
                "v23_minute_observed_rows": int(len(bars)),
                "v23_minute_expected_rows": expected,
                "v23_minute_latest_time": (
                    latest.isoformat() if not pd.isna(latest) else None
                ),
                "v23_minute_causal_ok": causal,
                **values,
                "v23_m1_coverage_ratio": min(
                    len(bars) / max(expected, 1),
                    1.0,
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not result["v23_minute_causal_ok"].all():
        raise RuntimeError("V23 minute features crossed a signal timestamp")
    return result


def build_auction_features(
    leaders: pd.DataFrame,
    auctions: pd.DataFrame,
) -> pd.DataFrame:
    source = leaders.loc[
        :,
        [
            *IDENTITY_COLUMNS,
            "signal_price",
            "ret_from_prev_close_pct",
        ],
    ].copy()
    auction = auctions.reindex(columns=AUCTION_FIELDS.split(",")).copy()
    auction["trade_date"] = auction["trade_date"].astype(str)
    auction["ts_code"] = auction["ts_code"].astype(str)
    if auction.duplicated(["trade_date", "ts_code"], keep=False).any():
        raise RuntimeError("V23 auction source has duplicate stock-date rows")
    for column in AUCTION_FIELDS.split(",")[2:]:
        auction[column] = pd.to_numeric(auction[column], errors="coerce")
    auction.rename(
        columns={
            column: f"_auction_{column}"
            for column in AUCTION_FIELDS.split(",")[2:]
        },
        inplace=True,
    )
    result = source.merge(
        auction,
        on=["trade_date", "ts_code"],
        how="left",
        validate="many_to_one",
    )
    prior_close = _numeric(result, "signal_price") / (
        1.0 + _numeric(result, "ret_from_prev_close_pct") / 100.0
    ).replace(0.0, np.nan)
    open_price = _numeric(result, "_auction_open")
    close = _numeric(result, "_auction_close")
    high = _numeric(result, "_auction_high")
    low = _numeric(result, "_auction_low")
    vwap = _numeric(result, "_auction_vwap")
    spread = (high - low).replace(0.0, np.nan)
    result["v23_auction_gap_from_prev_close_pct"] = (
        open_price / prior_close.replace(0.0, np.nan) - 1.0
    ) * 100.0
    result["v23_auction_return_pct"] = (
        close / open_price.replace(0.0, np.nan) - 1.0
    ) * 100.0
    result["v23_auction_range_pct"] = (
        high / low.replace(0.0, np.nan) - 1.0
    ) * 100.0
    result["v23_auction_close_position"] = (close - low) / spread
    result["v23_auction_vwap_gap_pct"] = (
        close / vwap.replace(0.0, np.nan) - 1.0
    ) * 100.0
    result["v23_auction_amount_log"] = np.log1p(
        _numeric(result, "_auction_amount").clip(lower=0.0)
    )
    result["v23_auction_volume_log"] = np.log1p(
        _numeric(result, "_auction_vol").clip(lower=0.0)
    )
    result["v23_auction_available"] = (
        close.gt(0.0)
        & open_price.gt(0.0)
        & high.ge(low)
        & _numeric(result, "_auction_amount").ge(0.0)
    )
    return result.loc[
        :,
        [*IDENTITY_COLUMNS, "v23_auction_available", *AUCTION_FEATURE_COLUMNS],
    ]


def build_moneyflow_features(
    leaders: pd.DataFrame,
    moneyflow: pd.DataFrame,
) -> pd.DataFrame:
    source = leaders.loc[
        :,
        [*IDENTITY_COLUMNS, "v23_prev_trade_date"],
    ].copy()
    flow = moneyflow.reindex(columns=MONEYFLOW_FIELDS.split(",")).copy()
    flow["trade_date"] = flow["trade_date"].astype(str)
    flow["ts_code"] = flow["ts_code"].astype(str)
    if flow.duplicated(["trade_date", "ts_code"], keep=False).any():
        raise RuntimeError("V23 money-flow source has duplicate stock-date rows")
    numeric_columns = MONEYFLOW_FIELDS.split(",")[2:]
    for column in numeric_columns:
        flow[column] = pd.to_numeric(flow[column], errors="coerce")
    flow.rename(
        columns={
            "trade_date": "v23_prev_trade_date",
            **{column: f"_mf_{column}" for column in numeric_columns},
        },
        inplace=True,
    )
    result = source.merge(
        flow,
        on=["v23_prev_trade_date", "ts_code"],
        how="left",
        validate="many_to_one",
    )
    gross = sum(
        _numeric(result, f"_mf_{column}").abs()
        for column in (
            "buy_sm_amount",
            "sell_sm_amount",
            "buy_md_amount",
            "sell_md_amount",
            "buy_lg_amount",
            "sell_lg_amount",
            "buy_elg_amount",
            "sell_elg_amount",
        )
    )
    denominator = gross.replace(0.0, np.nan)
    small = (
        _numeric(result, "_mf_buy_sm_amount")
        - _numeric(result, "_mf_sell_sm_amount")
    )
    medium = (
        _numeric(result, "_mf_buy_md_amount")
        - _numeric(result, "_mf_sell_md_amount")
    )
    large = (
        _numeric(result, "_mf_buy_lg_amount")
        - _numeric(result, "_mf_sell_lg_amount")
        + _numeric(result, "_mf_buy_elg_amount")
        - _numeric(result, "_mf_sell_elg_amount")
    )
    result["v23_prev_mf_net_share"] = (
        _numeric(result, "_mf_net_mf_amount") / denominator
    )
    result["v23_prev_mf_large_share"] = large / denominator
    result["v23_prev_mf_medium_share"] = medium / denominator
    result["v23_prev_mf_small_share"] = small / denominator
    result["v23_prev_mf_institution_retail_spread"] = (
        large - small
    ) / denominator
    result["v23_prev_mf_gross_amount_log"] = np.log1p(
        gross.clip(lower=0.0)
    )
    result["v23_moneyflow_available"] = (
        denominator.gt(0.0)
        & _numeric(result, "_mf_net_mf_amount").notna()
    )
    if not (
        result["v23_prev_trade_date"].astype(str)
        < result["trade_date"].astype(str)
    ).all():
        raise RuntimeError("V23 money-flow join crossed the signal date")
    return result.loc[
        :,
        [
            *IDENTITY_COLUMNS,
            "v23_prev_trade_date",
            "v23_moneyflow_available",
            *MONEYFLOW_FEATURE_COLUMNS,
        ],
    ]


def assemble_v23_feature_frame(
    leaders: pd.DataFrame,
    minute_features: pd.DataFrame,
    auction_features: pd.DataFrame,
    moneyflow_features: pd.DataFrame,
) -> pd.DataFrame:
    source = leaders.loc[:, [*IDENTITY_COLUMNS, "fold"]].copy()
    result = source.merge(
        minute_features,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    result = result.merge(
        auction_features,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    result = result.merge(
        moneyflow_features,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    result["v23_point_in_time_complete"] = (
        _numeric(result, "v23_m1_coverage_ratio").ge(
            MINIMUM_MINUTE_COVERAGE
        )
        & _boolean(result, "v23_minute_causal_ok")
        & _boolean(result, "v23_auction_available")
        & _boolean(result, "v23_moneyflow_available")
    )
    result.sort_values(["fold", *IDENTITY_COLUMNS], kind="stable", inplace=True)
    result.reset_index(drop=True, inplace=True)
    if result.duplicated(list(IDENTITY_COLUMNS), keep=False).any():
        raise RuntimeError("V23 feature frame contains duplicate identities")
    return result.reindex(
        columns=(
            *IDENTITY_COLUMNS,
            "fold",
            *V23_FEATURE_COLUMNS,
            *QUALITY_COLUMNS,
        )
    )


def feature_coverage_audit(features: pd.DataFrame) -> dict[str, Any]:
    rows = len(features)
    minute_ready = (
        _numeric(features, "v23_m1_coverage_ratio").ge(
            MINIMUM_MINUTE_COVERAGE
        )
        & _boolean(features, "v23_minute_causal_ok")
    )
    auction_ready = _boolean(features, "v23_auction_available")
    moneyflow_ready = _boolean(features, "v23_moneyflow_available")
    complete = _boolean(features, "v23_point_in_time_complete")
    feature_coverage = {
        column: float(
            pd.to_numeric(features[column], errors="coerce").notna().mean()
        )
        for column in V23_FEATURE_COLUMNS
    }
    return {
        "leader_rows": int(rows),
        "minute_ready_rows": int(minute_ready.sum()),
        "minute_coverage_rate": float(minute_ready.mean()) if rows else 0.0,
        "auction_ready_rows": int(auction_ready.sum()),
        "auction_coverage_rate": (
            float(auction_ready.mean()) if rows else 0.0
        ),
        "moneyflow_ready_rows": int(moneyflow_ready.sum()),
        "moneyflow_coverage_rate": (
            float(moneyflow_ready.mean()) if rows else 0.0
        ),
        "complete_rows": int(complete.sum()),
        "complete_coverage_rate": float(complete.mean()) if rows else 0.0,
        "feature_non_null_coverage": feature_coverage,
        "minimum_required_dataset_coverage": MINIMUM_DATASET_COVERAGE,
        "coverage_passed": bool(
            rows
            and float(minute_ready.mean()) >= MINIMUM_DATASET_COVERAGE
            and float(auction_ready.mean()) >= MINIMUM_DATASET_COVERAGE
            and float(moneyflow_ready.mean()) >= MINIMUM_DATASET_COVERAGE
            and float(complete.mean()) >= MINIMUM_DATASET_COVERAGE
        ),
    }


def _minute_feature_values(bars: pd.DataFrame) -> dict[str, float]:
    empty = {column: float("nan") for column in MINUTE_FEATURE_COLUMNS}
    if bars.empty:
        return empty
    close = _numeric(bars, "close")
    high = _numeric(bars, "high")
    low = _numeric(bars, "low")
    open_price = _numeric(bars, "open")
    amount = _numeric(bars, "amount").fillna(0.0).clip(lower=0.0)
    volume = _numeric(bars, "vol").fillna(0.0).clip(lower=0.0)
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan)
    finite_returns = returns.dropna()
    total_amount = float(amount.sum())
    amount_share = (
        amount / total_amount
        if total_amount > 0.0
        else pd.Series(0.0, index=amount.index)
    )
    price_changes = close.diff()
    direction_denominator = float(price_changes.abs().sum())
    cumulative_high = close.cummax()
    drawdown = close / cumulative_high.replace(0.0, np.nan) - 1.0
    close_range = float(high.max() - low.min())
    volume_total = float(volume.sum())
    weighted_price = (
        float((close * volume).sum() / volume_total)
        if volume_total > 0.0
        else float(close.mean())
    )
    previous_high = high.shift(1).cummax()
    previous_low = low.shift(1).cummin()
    body_top = pd.concat([open_price, close], axis=1).max(axis=1)
    body_bottom = pd.concat([open_price, close], axis=1).min(axis=1)
    price_scale = close.abs().replace(0.0, np.nan)

    values = {
        "v23_m1_zero_amount_share": float(amount.le(0.0).mean()),
        "v23_m1_ret_1m_pct": _window_return(close, 1),
        "v23_m1_ret_3m_pct": _window_return(close, 3),
        "v23_m1_ret_5m_pct": _window_return(close, 5),
        "v23_m1_ret_10m_pct": _window_return(close, 10),
        "v23_m1_ret_from_1400_pct": _return_from_time(bars, "14:00"),
        "v23_m1_realized_volatility_pct": (
            float(finite_returns.std(ddof=0) * 100.0)
            if len(finite_returns)
            else float("nan")
        ),
        "v23_m1_downside_volatility_pct": (
            float(
                finite_returns.loc[finite_returns.lt(0.0)].std(ddof=0)
                * 100.0
            )
            if finite_returns.lt(0.0).sum() >= 2
            else 0.0
        ),
        "v23_m1_return_skew": (
            float(finite_returns.skew())
            if len(finite_returns) >= 3
            else float("nan")
        ),
        "v23_m1_return_autocorr1": (
            float(finite_returns.autocorr(lag=1))
            if len(finite_returns) >= 4
            else float("nan")
        ),
        "v23_m1_directional_efficiency": (
            float((close.iloc[-1] - close.iloc[0]) / direction_denominator)
            if direction_denominator > 0.0
            else 0.0
        ),
        "v23_m1_max_drawdown_pct": float(drawdown.min() * 100.0),
        "v23_m1_rebound_from_low_pct": (
            float((close.iloc[-1] / low.min() - 1.0) * 100.0)
            if float(low.min()) > 0.0
            else float("nan")
        ),
        "v23_m1_reversal_from_high_pct": (
            float((close.iloc[-1] / high.max() - 1.0) * 100.0)
            if float(high.max()) > 0.0
            else float("nan")
        ),
        "v23_m1_up_minute_share": float(finite_returns.gt(0.0).mean()),
        "v23_m1_down_minute_share": float(finite_returns.lt(0.0).mean()),
        "v23_m1_max_up_streak": float(
            _maximum_streak(finite_returns.gt(0.0))
        ),
        "v23_m1_max_down_streak": float(
            _maximum_streak(finite_returns.lt(0.0))
        ),
        "v23_m1_signed_amount_imbalance": (
            float((np.sign(returns.fillna(0.0)) * amount).sum() / total_amount)
            if total_amount > 0.0
            else float("nan")
        ),
        "v23_m1_amihud_proxy": (
            float(
                (
                    finite_returns.abs()
                    / amount.reindex(finite_returns.index).replace(0.0, np.nan)
                ).mean()
                * 1e8
            )
            if len(finite_returns)
            else float("nan")
        ),
        "v23_m1_amount_concentration": float((amount_share**2).sum()),
        "v23_m1_last3_amount_share": float(amount_share.tail(3).sum()),
        "v23_m1_last5_amount_share": float(amount_share.tail(5).sum()),
        "v23_m1_amount_acceleration": _amount_acceleration(amount),
        "v23_m1_vwap_gap_pct": (
            float((close.iloc[-1] / weighted_price - 1.0) * 100.0)
            if weighted_price > 0.0
            else float("nan")
        ),
        "v23_m1_close_position": (
            float((close.iloc[-1] - low.min()) / close_range)
            if close_range > 0.0
            else 0.5
        ),
        "v23_m1_upper_wick_pressure": float(
            ((high - body_top).clip(lower=0.0) / price_scale).mean()
            * 100.0
        ),
        "v23_m1_lower_wick_pressure": float(
            ((body_bottom - low).clip(lower=0.0) / price_scale).mean()
            * 100.0
        ),
        "v23_m1_high_break_count": float(
            high.gt(previous_high).fillna(False).sum()
        ),
        "v23_m1_low_break_count": float(
            low.lt(previous_low).fillna(False).sum()
        ),
        "v23_m1_last5_vs_prior5_return_pct": _last_vs_prior_return(close),
    }
    return {**empty, **values}


def _leader_identity(leaders: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(IDENTITY_COLUMNS) - set(leaders.columns))
    if missing:
        raise ValueError(f"V23 leaders missing identity columns: {missing}")
    result = leaders.loc[:, list(IDENTITY_COLUMNS)].copy()
    for column in IDENTITY_COLUMNS:
        result[column] = result[column].astype(str)
    if result.duplicated(list(IDENTITY_COLUMNS), keep=False).any():
        raise RuntimeError("V23 leaders contain duplicate identities")
    return result


def _window_return(close: pd.Series, minutes: int) -> float:
    if len(close) <= minutes:
        return float("nan")
    base = float(close.iloc[-minutes - 1])
    return (
        float((close.iloc[-1] / base - 1.0) * 100.0)
        if base > 0.0
        else float("nan")
    )


def _return_from_time(bars: pd.DataFrame, time_text: str) -> float:
    base = bars.loc[
        bars["trade_time"].dt.strftime("%H:%M").eq(time_text),
        "close",
    ]
    if base.empty:
        return float("nan")
    value = float(pd.to_numeric(base, errors="coerce").iloc[-1])
    close = float(pd.to_numeric(bars["close"], errors="coerce").iloc[-1])
    return (close / value - 1.0) * 100.0 if value > 0.0 else float("nan")


def _maximum_streak(mask: pd.Series) -> int:
    maximum = current = 0
    for value in mask.fillna(False).astype(bool):
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def _amount_acceleration(amount: pd.Series) -> float:
    if len(amount) < 10:
        return float("nan")
    prior = float(amount.iloc[-10:-5].mean())
    latest = float(amount.iloc[-5:].mean())
    return latest / prior if prior > 0.0 else float("nan")


def _last_vs_prior_return(close: pd.Series) -> float:
    if len(close) < 11:
        return float("nan")
    last = float(close.iloc[-1] / close.iloc[-6] - 1.0)
    prior = float(close.iloc[-6] / close.iloc[-11] - 1.0)
    return (last - prior) * 100.0


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )
