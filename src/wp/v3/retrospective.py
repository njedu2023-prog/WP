from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import V3Config
from .io import atomic_write_csv, atomic_write_json
from .registry import load_registry, model_record, save_registry
from .v40 import refresh_v40_backtest_summary, v40_historical_gate


QUALIFIED_CSV = "wp_v40_backtest_qualified_202605_202607.csv"
OBSERVATION_CSV = "wp_v40_backtest_observations_202605_202607.csv"
SUMMARY_JSON = "wp_v40_backtest_202605_202607.json"


def finalize_v40_retrospective(
    output_root: str | Path,
    *,
    current_trade_date: str,
    config: V3Config,
    pro: Any | None,
) -> dict[str, Any]:
    output = Path(output_root)
    summary_path = output / "json" / SUMMARY_JSON
    qualified_path = output / "csv" / QUALIFIED_CSV
    observation_path = output / "csv" / OBSERVATION_CSV
    if not (
        summary_path.exists()
        and qualified_path.exists()
        and observation_path.exists()
    ):
        return {
            "available": False,
            "changed": False,
            "pending_due_count": 0,
            "pending_total_count": 0,
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    qualified = _read_csv(qualified_path)
    observations = _read_csv(observation_path)
    frames = [qualified, observations]
    pending_due = _pending_rows(
        frames,
        current_trade_date=current_trade_date,
        due_only=True,
    )
    pending_total_before = len(
        _pending_rows(
            frames,
            current_trade_date=current_trade_date,
            due_only=False,
        )
    )
    if pending_due.empty:
        return {
            "available": True,
            "changed": False,
            "pending_due_count": 0,
            "pending_total_count": pending_total_before,
            "status": summary.get("status"),
        }
    if pro is None:
        raise RuntimeError(
            "TUSHARE_TOKEN is required to close pending V40 retrospective truth"
        )
    truth_dates = sorted(
        {
            str(value)
            for value in pending_due["target_trade_date"]
            if str(value).isdigit()
        }
    )
    entry_dates = sorted(
        {
            str(value)
            for value in pending_due["trade_date"]
            if str(value).isdigit()
        }
    )
    truth = {
        date: _fetch_truth(pro, date)
        for date in sorted(set(truth_dates + entry_dates))
    }
    qualified = hydrate_v40_truth(
        qualified,
        truth,
        config=config,
        current_trade_date=current_trade_date,
    )
    observations = hydrate_v40_truth(
        observations,
        truth,
        config=config,
        current_trade_date=current_trade_date,
    )
    refreshed = refresh_v40_backtest_summary(
        summary,
        qualified,
        observations,
        config,
    )
    refreshed["backtest_gate"] = v40_historical_gate(refreshed, config)
    atomic_write_csv(qualified, qualified_path)
    atomic_write_csv(observations, observation_path)
    atomic_write_json(summary_path, refreshed)
    _refresh_registry(output, refreshed)
    pending_due_after = len(
        _pending_rows(
            [qualified, observations],
            current_trade_date=current_trade_date,
            due_only=True,
        )
    )
    pending_total_after = len(
        _pending_rows(
            [qualified, observations],
            current_trade_date=current_trade_date,
            due_only=False,
        )
    )
    return {
        "available": True,
        "changed": True,
        "pending_due_count": pending_due_after,
        "pending_total_count": pending_total_after,
        "status": refreshed.get("status"),
    }


def hydrate_v40_truth(
    frame: pd.DataFrame,
    truth_by_date: dict[str, pd.DataFrame],
    *,
    config: V3Config,
    current_trade_date: str,
) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result
    for index, row in result.iterrows():
        if _as_bool(row.get("label_available")):
            continue
        target_date = str(row.get("target_trade_date") or "")
        trade_date = str(row.get("trade_date") or "")
        if (
            not target_date.isdigit()
            or target_date > current_trade_date
            or target_date not in truth_by_date
        ):
            continue
        code = str(row.get("ts_code") or "")
        target_truth = truth_by_date[target_date]
        if code not in target_truth.index:
            continue
        target = target_truth.loc[code]
        entry_truth = truth_by_date.get(trade_date)
        entry_adj = _positive(row.get("adj_factor"))
        if not entry_adj and entry_truth is not None and code in entry_truth.index:
            entry_adj = _positive(entry_truth.loc[code].get("adj_factor"))
        target_adj = _positive(target.get("adj_factor"))
        close = _positive(target.get("close"))
        volume = _positive(target.get("vol"))
        down_limit = _positive(target.get("down_limit"))
        if not entry_adj or not target_adj or not close:
            continue
        entry_fillable = _as_bool(row.get("entry_fillable"))
        exit_fillable = bool(
            volume
            and (
                down_limit is None
                or close > down_limit * 1.0001
            )
        )
        entry_price = _positive(row.get("entry_price"))
        if not entry_price:
            benchmark = _positive(row.get("entry_benchmark_price"))
            if benchmark:
                entry_price = benchmark * (
                    1.0
                    + config.execution.entry_slippage_bps / 10_000.0
                )
        total_return_close = close * target_adj / entry_adj
        gross = (
            (total_return_close / entry_price - 1.0) * 100.0
            if entry_price
            else None
        )
        if not entry_fillable:
            net = 0.0
        elif not exit_fillable:
            net = config.execution.non_fill_penalty_pct
        elif gross is not None:
            net = gross - config.execution.round_trip_cost_bps / 100.0
        else:
            continue
        result.at[index, "t1_close"] = close
        result.at[index, "t1_adj_factor"] = target_adj
        result.at[index, "t1_total_return_close"] = total_return_close
        result.at[index, "entry_price"] = entry_price
        result.at[index, "exit_fillable"] = exit_fillable
        result.at[index, "execution_success"] = (
            entry_fillable and exit_fillable
        )
        result.at[index, "gross_return_pct"] = gross
        result.at[index, "net_return_pct"] = net
        result.at[index, "target_net_positive"] = int(
            entry_fillable and exit_fillable and net > 0
        )
        result.at[index, "target_entry_fillable"] = int(entry_fillable)
        result.at[index, "target_exit_fillable"] = (
            int(exit_fillable) if entry_fillable else np.nan
        )
        result.at[index, "label_available"] = True
        result.at[index, "target_market_truth_available"] = True
    return result


def _pending_rows(
    frames: list[pd.DataFrame],
    *,
    current_trade_date: str,
    due_only: bool,
) -> pd.DataFrame:
    available = []
    for frame in frames:
        if frame.empty:
            continue
        labels = frame.get(
            "label_available",
            pd.Series(False, index=frame.index),
        ).map(_as_bool)
        pending = frame.loc[~labels].copy()
        if due_only:
            targets = pending.get(
                "target_trade_date",
                pd.Series("", index=pending.index),
            ).astype(str)
            pending = pending.loc[
                targets.str.fullmatch(r"\d{8}", na=False)
                & targets.le(current_trade_date)
            ]
        available.append(pending)
    return (
        pd.concat(available, ignore_index=True)
        if available
        else pd.DataFrame()
    )


def _fetch_truth(pro: Any, trade_date: str) -> pd.DataFrame:
    daily = pro.daily(
        trade_date=trade_date,
        fields="ts_code,trade_date,open,high,low,close,vol,amount",
    )
    limits = pro.stk_limit(
        trade_date=trade_date,
        fields="trade_date,ts_code,up_limit,down_limit",
    )
    adjustments = pro.adj_factor(
        trade_date=trade_date,
        fields="ts_code,trade_date,adj_factor",
    )
    if len(daily) <= 1_000 or len(limits) <= 1_000 or len(adjustments) <= 1_000:
        raise RuntimeError(
            f"V40 retrospective truth for {trade_date} is incomplete: "
            f"daily={len(daily)} limits={len(limits)} adj={len(adjustments)}"
        )
    truth = (
        adjustments.merge(
            daily,
            on=["trade_date", "ts_code"],
            how="left",
        )
        .merge(limits, on=["trade_date", "ts_code"], how="left")
    )
    for column in (
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "up_limit",
        "down_limit",
        "adj_factor",
    ):
        truth[column] = pd.to_numeric(truth[column], errors="coerce")
    return truth.drop_duplicates("ts_code").set_index("ts_code")


def _refresh_registry(output: Path, summary: dict[str, Any]) -> None:
    fingerprint = str(
        summary.get("model", {}).get("fingerprint") or ""
    )
    registry_path = output / "json" / "wp_model_registry_v3.json"
    if not fingerprint or not registry_path.exists():
        return
    registry = load_registry(registry_path)
    record = model_record(registry, fingerprint)
    if record is None:
        return
    updated = deepcopy(record.get("backtest", {}))
    updated.update(
        {
            "backtest_gate": summary.get("backtest_gate", {}),
            "metrics": summary.get("qualified", {}).get("metrics", {}),
            "contract": summary.get("policy", {}),
            "evidence_status": summary.get("status"),
        }
    )
    record["backtest"] = updated
    if (
        bool(summary.get("backtest_gate", {}).get("passed"))
        and registry.get("shadow_model_fingerprint") == fingerprint
        and record.get("status") == "SHADOW_OBSERVATION"
    ):
        record["status"] = "SHADOW"
    save_registry(registry, registry_path)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path,
            keep_default_na=False,
            na_values=["", "null", "None", "<NA>"],
            dtype={
                "trade_date": str,
                "target_trade_date": str,
                "ts_code": str,
            },
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "qualified",
            "pass",
        }
    try:
        return bool(value) if pd.notna(value) else False
    except (TypeError, ValueError):
        return False


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None
