from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .calendar import next_trading_day_str
from .tail_window import (
    TAIL_PHASE_ACTIVE,
    TAIL_PHASE_CLOSED,
    parse_market_datetime,
    tail_window_phase,
)


TAIL_SAMPLING_COLUMNS = [
    "plan_trade_date",
    "target_trade_date",
    "sample_status",
    "record_count",
    "first_sample_time",
    "last_sample_time",
    "note",
    "updated_at",
]


@dataclass
class TailSamplingResult:
    table: pd.DataFrame
    summary: dict


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=TAIL_SAMPLING_COLUMNS)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty()
    try:
        frame = pd.read_csv(
            path,
            keep_default_na=False,
            dtype={"plan_trade_date": str, "target_trade_date": str},
        )
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return _empty()
    for column in TAIL_SAMPLING_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[TAIL_SAMPLING_COLUMNS].copy()


def _write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _date_text(value: object) -> str:
    return str(value or "").strip().replace("-", "").removesuffix(".0")


def _validation_rows(validation: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if validation is None or validation.empty or "plan_trade_date" not in validation.columns:
        return pd.DataFrame()
    dates = validation["plan_trade_date"].map(_date_text)
    return validation[dates.eq(trade_date)].copy()


def _sample_times(rows: pd.DataFrame) -> tuple[str, str]:
    if rows.empty or "plan_time" not in rows.columns:
        return "", ""
    parsed = pd.to_datetime(rows["plan_time"], errors="coerce").dropna().sort_values()
    if parsed.empty:
        return "", ""
    return (
        parsed.iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
        parsed.iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
    )


def update_tail_sampling(
    validation: pd.DataFrame,
    health: dict,
    path: str | Path,
) -> TailSamplingResult:
    """Record whether each trade day had at least one valid tail-window sample."""
    audit_path = Path(path)
    table = _read(audit_path)
    market_time_text = str(health.get("market_data_time") or health.get("data_time") or "")
    market_time = parse_market_datetime(market_time_text)
    trade_date = _date_text(
        health.get("data_trade_date") or health.get("source_trade_date")
    )
    phase = tail_window_phase(market_time_text)
    reliable = (
        health.get("status") in {"ok", "无符合条件股票"}
        and bool(health.get("load_ok", True))
        and len(trade_date) == 8
        and market_time is not None
    )

    if reliable and phase in {TAIL_PHASE_ACTIVE, TAIL_PHASE_CLOSED}:
        rows = _validation_rows(validation, trade_date)
        first_sample, last_sample = _sample_times(rows)
        existing_match = table["plan_trade_date"].map(_date_text).eq(trade_date)
        existing_row = table[existing_match].iloc[-1] if existing_match.any() else None
        captured_before = (
            existing_row is not None
            and str(existing_row.get("sample_status") or "") == "captured"
        )

        if phase == TAIL_PHASE_ACTIVE or captured_before or not rows.empty:
            status = "captured"
            note = "已取得14:20-14:50合法窗口样本"
        else:
            status = "missing"
            note = "当日未取得14:20-14:50合法窗口快照；不使用15:00后数据回补"

        target_dates = (
            rows.get("target_trade_date", pd.Series(dtype="object"))
            .map(_date_text)
            .loc[lambda values: values.str.fullmatch(r"\d{8}")]
        )
        target_trade_date = (
            str(target_dates.iloc[0])
            if not target_dates.empty
            else next_trading_day_str(trade_date)
        )
        row = {
            "plan_trade_date": trade_date,
            "target_trade_date": target_trade_date,
            "sample_status": status,
            "record_count": int(len(rows)),
            "first_sample_time": first_sample,
            "last_sample_time": last_sample,
            "note": note,
            "updated_at": market_time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        table = table[~existing_match].copy()
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
        table = table.sort_values("plan_trade_date", kind="mergesort").reset_index(drop=True)
        _write(audit_path, table)

    status_values = table.get("sample_status", pd.Series(dtype="object")).astype(str)
    return TailSamplingResult(
        table=table,
        summary={
            "day_count": int(len(table)),
            "captured_day_count": int(status_values.eq("captured").sum()),
            "missing_day_count": int(status_values.eq("missing").sum()),
            "days": table.to_dict(orient="records"),
        },
    )
