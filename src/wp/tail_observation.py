from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path

import pandas as pd


TAIL_OBSERVATION_START = time(14, 20)
INVALID_GRACE_RUNS = 2

OBSERVATION_COLUMNS = [
    "observation_trade_date",
    "quality_rank",
    "previous_quality_rank",
    "rank_change",
    "observation_status",
    "qualification_status",
    "qualification_reason",
    "invalid_count",
    "entry_order",
    "first_seen",
    "last_seen",
    "ts_code",
    "name",
    "price",
    "pct_chg",
    "sector_name",
    "limit_rule_pct",
    "tail_profit_score",
    "peak_tail_profit_score",
    "risk_penalty_score",
    "amount_ratio_5d",
    "p_limitup_t1",
    "wp_score",
    "model_confidence",
    "confirm_before_buy",
    "reject_if",
    "buy_reason",
]

LIVE_VALUE_COLUMNS = [
    "name",
    "price",
    "pct_chg",
    "sector_name",
    "limit_rule_pct",
    "tail_profit_score",
    "risk_penalty_score",
    "amount_ratio_5d",
    "p_limitup_t1",
    "wp_score",
    "model_confidence",
]


@dataclass
class TailObservationResult:
    table: pd.DataFrame
    summary: dict


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=OBSERVATION_COLUMNS)


def _read_state(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty()
    try:
        frame = pd.read_csv(
            path,
            dtype={"observation_trade_date": str, "ts_code": str},
            keep_default_na=False,
        )
    except Exception:
        return _empty()
    for column in OBSERVATION_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[OBSERVATION_COLUMNS].copy()


def _write_state(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _parse_market_time(value: object) -> datetime | None:
    parsed = pd.to_datetime(str(value or "").strip(), errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _number(value: object, default: float = 0.0) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else float(default)


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype="bool")
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(False)
    return values.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _dedupe(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "ts_code" not in frame.columns:
        return frame.copy()
    out = frame.copy()
    out["ts_code"] = out["ts_code"].fillna("").astype(str).str.strip()
    return out[out["ts_code"].ne("")].drop_duplicates("ts_code", keep="last")


def _summary(frame: pd.DataFrame, status: str) -> dict:
    qualification = frame.get("qualification_status", pd.Series(dtype="object"))
    qualification = qualification.fillna("").astype(str)
    return {
        "count": int(len(frame)),
        "active_count": int(qualification.eq("合格").sum()),
        "sealed_count": int(qualification.eq("已封板").sum()),
        "review_count": int(qualification.eq("资格复核").sum()),
        "status": status,
    }


def _copy_live_values(target: dict, source: pd.Series) -> None:
    for column in LIVE_VALUE_COLUMNS:
        if column in source.index:
            target[column] = source.get(column, target.get(column, ""))


def _critical_disqualification(row: pd.Series | None, max_limit_up_pct: float) -> str:
    if row is None:
        return ""
    limit_rule_pct = _number(row.get("limit_rule_pct"), 0.0)
    inferred_limit_pct = _number(row.get("limit_up_pct"), limit_rule_pct)
    if limit_rule_pct > max_limit_up_pct or inferred_limit_pct > max_limit_up_pct:
        return f"涨跌停规则超过{max_limit_up_pct:g}%"
    if bool(row.get("is_st", False)):
        return "ST股票"
    if int(_number(row.get("suspended_flag"))) == 1:
        return "停牌或无有效成交"
    if int(_number(row.get("delist_flag"))) == 1:
        return "退市风险"
    if int(_number(row.get("data_quality_flag"))) == 1:
        return "关键行情字段异常"
    if int(_number(row.get("pre_day_limitup"))) == 1:
        return "昨日涨停"
    return ""


def _rank_change(previous: object, current: int) -> str:
    prior = int(_number(previous, 0))
    if prior <= 0:
        return "新"
    delta = prior - current
    if delta > 0:
        return f"升{delta}"
    if delta < 0:
        return f"降{abs(delta)}"
    return "持平"


def _finalize(frame: pd.DataFrame, primary_codes: set[str]) -> pd.DataFrame:
    if frame.empty:
        return _empty()
    out = frame.copy()
    status_priority = {"合格": 0, "已封板": 1, "资格复核": 2}
    out["_status_priority"] = out["qualification_status"].map(status_priority).fillna(3)
    out["_tail_score"] = pd.to_numeric(out["tail_profit_score"], errors="coerce").fillna(0.0)
    out["_risk_score"] = pd.to_numeric(out["risk_penalty_score"], errors="coerce").fillna(100.0)
    out["_entry_order"] = pd.to_numeric(out["entry_order"], errors="coerce").fillna(999999)
    out = out.sort_values(
        ["_status_priority", "_tail_score", "_risk_score", "_entry_order", "ts_code"],
        ascending=[True, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    previous = out["quality_rank"].tolist()
    out["previous_quality_rank"] = previous
    out["quality_rank"] = range(1, len(out) + 1)
    out["rank_change"] = [
        _rank_change(prior, current)
        for prior, current in zip(previous, out["quality_rank"], strict=False)
    ]
    out["observation_status"] = "观察票"
    out.loc[out["qualification_status"].eq("已封板"), "observation_status"] = "已封板"
    out.loc[out["qualification_status"].eq("资格复核"), "observation_status"] = "资格复核"
    current_primary = out["ts_code"].isin(primary_codes) & out["qualification_status"].eq("合格")
    out.loc[current_primary, "observation_status"] = "当前主票"
    out = out.drop(columns=["_status_priority", "_tail_score", "_risk_score", "_entry_order"])
    for column in OBSERVATION_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    return out[OBSERVATION_COLUMNS].copy()


def update_tail_observation(
    ranked_input: pd.DataFrame,
    buy_plan: pd.DataFrame,
    market_universe: pd.DataFrame,
    health: dict,
    state_path: str | Path,
    *,
    max_limit_up_pct: float = 10.0,
    invalid_grace_runs: int = INVALID_GRACE_RUNS,
) -> TailObservationResult:
    """Persist every post-14:20 primary while its intrinsic qualification remains valid."""
    path = Path(state_path)
    existing = _dedupe(_read_state(path))
    market_time_text = str(health.get("market_data_time") or health.get("data_time") or "")
    market_time = _parse_market_time(market_time_text)
    trade_date = str(health.get("data_trade_date") or health.get("source_trade_date") or "").replace("-", "")
    data_reliable = (
        health.get("status") in {"ok", "无符合条件股票"}
        and bool(health.get("load_ok", True))
        and len(trade_date) == 8
    )

    if not data_reliable or market_time is None:
        return TailObservationResult(
            existing,
            _summary(existing, "preserved_unverified"),
        )

    if not existing.empty:
        existing = existing[existing["observation_trade_date"].astype(str).eq(trade_date)].copy()

    if market_time.time() < TAIL_OBSERVATION_START:
        empty = _empty()
        _write_state(path, empty)
        return TailObservationResult(empty, _summary(empty, "before_tail_window"))

    current = _dedupe(ranked_input)
    current_codes = set(current.get("ts_code", pd.Series(dtype="object")).astype(str))
    if current.empty:
        eligible = current.copy()
    else:
        eligible = current[_bool_series(current, "tail_profit_eligible")].copy()
    eligible_by_code = {str(row["ts_code"]): row for _, row in eligible.iterrows()}

    universe = _dedupe(market_universe)
    universe_by_code = {str(row["ts_code"]): row for _, row in universe.iterrows()}
    primary_codes = {
        str(code).strip()
        for code in buy_plan.get("ts_code", pd.Series(dtype="object")).tolist()
        if str(code).strip()
    }
    existing_by_code = {str(row["ts_code"]): row.to_dict() for _, row in existing.iterrows()}
    next_rows: list[dict] = []

    for code, stored in existing_by_code.items():
        current_row = eligible_by_code.get(code)
        universe_row = universe_by_code.get(code)
        critical_reason = _critical_disqualification(universe_row, max_limit_up_pct)
        if critical_reason:
            continue
        if universe_row is not None and int(_number(universe_row.get("today_limitup"))) == 1:
            _copy_live_values(stored, universe_row)
            stored["last_seen"] = market_time_text
            stored["invalid_count"] = 0
            stored["qualification_status"] = "已封板"
            stored["qualification_reason"] = "已涨停，停止新买入"
            next_rows.append(stored)
            continue
        if current_row is not None:
            _copy_live_values(stored, current_row)
            stored["last_seen"] = market_time_text
            stored["invalid_count"] = 0
            stored["qualification_status"] = "合格"
            stored["qualification_reason"] = ""
            stored["peak_tail_profit_score"] = max(
                _number(stored.get("peak_tail_profit_score")),
                _number(current_row.get("tail_profit_score")),
            )
            next_rows.append(stored)
            continue

        invalid_count = int(_number(stored.get("invalid_count"))) + 1
        if invalid_count < max(int(invalid_grace_runs), 1):
            if universe_row is not None:
                _copy_live_values(stored, universe_row)
            stored["last_seen"] = market_time_text
            stored["invalid_count"] = invalid_count
            stored["qualification_status"] = "资格复核"
            reason = "不再满足当前内在资格"
            if code in current_codes:
                match = current[current["ts_code"].eq(code)].iloc[-1]
                reason = _text(match.get("tail_profit_filter_reason")) or reason
            stored["qualification_reason"] = reason
            next_rows.append(stored)

    max_entry_order = max(
        [int(_number(row.get("entry_order"))) for row in existing_by_code.values()] or [0]
    )
    plan_by_code = {
        str(row["ts_code"]): row
        for _, row in _dedupe(buy_plan).iterrows()
    }
    for code in sorted(primary_codes):
        if code in existing_by_code or code not in eligible_by_code:
            continue
        universe_row = universe_by_code.get(code)
        if _critical_disqualification(universe_row, max_limit_up_pct):
            continue
        if universe_row is not None and int(_number(universe_row.get("today_limitup"))) == 1:
            continue
        source = eligible_by_code[code]
        plan_row = plan_by_code.get(code)
        max_entry_order += 1
        row = {column: "" for column in OBSERVATION_COLUMNS}
        row.update(
            {
                "observation_trade_date": trade_date,
                "quality_rank": "",
                "observation_status": "当前主票",
                "qualification_status": "合格",
                "qualification_reason": "",
                "invalid_count": 0,
                "entry_order": max_entry_order,
                "first_seen": market_time_text,
                "last_seen": market_time_text,
                "ts_code": code,
                "peak_tail_profit_score": _number(source.get("tail_profit_score")),
                "confirm_before_buy": "守8%、承接稳",
                "reject_if": "破8%/急回落/板块转弱",
                "buy_reason": "",
            }
        )
        _copy_live_values(row, source)
        if plan_row is not None:
            row["confirm_before_buy"] = _text(plan_row.get("confirm_before_buy")) or row["confirm_before_buy"]
            row["reject_if"] = _text(plan_row.get("reject_if")) or row["reject_if"]
            row["buy_reason"] = _text(plan_row.get("buy_reason"))
        next_rows.append(row)

    result = _finalize(pd.DataFrame(next_rows), primary_codes)
    _write_state(path, result)
    return TailObservationResult(result, _summary(result, "updated"))
