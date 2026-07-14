from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .tail_profit_model import TAIL_PROFIT_MODEL_VERSION, add_tail_profit_scores


DEFAULT_BUY_CONFIG = {
    "buy_max_count": 1,
    "buy_max_sector_positions": 1,
}


BUY_COLUMNS = [
    "buy_rank",
    "portfolio_group",
    "ts_code",
    "name",
    "rank",
    "price",
    "pct_chg",
    "sector_name",
    "tail_profit_score",
    "tail_profit_model_version",
    "amount_ratio_5d",
    "p_limitup_t1",
    "wp_score",
    "decision_score",
    "model_confidence",
    "risk_penalty_score",
    "acceptance_score",
    "momentum_score",
    "close_position",
    "intraday_pullback_pct",
    "intraday_vwap_position",
    "confirm_before_buy",
    "reject_if",
    "buy_reason",
]


DECISION_COLUMNS = BUY_COLUMNS + ["buy_flag", "decision_action", "skip_reason"]


@dataclass
class BuyDecisionResult:
    buy_plan: pd.DataFrame
    decision_table: pd.DataFrame
    summary: dict


def _num(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce").fillna(default)


def _txt(frame: pd.DataFrame, name: str, default: str = "") -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="object")
    return frame[name].fillna(default).astype(str)


def _sort_for_buy(frame: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [
        "tail_profit_score",
        "pct_chg",
        "capital_score",
        "sector_strength_score",
        "risk_penalty_score",
        "amount",
    ]
    for col in sort_cols:
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame["ts_code"] = _txt(frame, "ts_code")
    return frame.sort_values(
        sort_cols + ["ts_code"],
        ascending=[False, True, False, False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _row_float(row: pd.Series, name: str, default: float = 0.0) -> float:
    value = pd.to_numeric(pd.Series([row.get(name, default)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else float(default)


def _reason(row: pd.Series) -> str:
    reasons = ["涨幅不过热"]
    if _row_float(row, "tail_rank_capital") >= 0.70:
        reasons.append("资金强")
    if _row_float(row, "tail_rank_sector") >= 0.70:
        reasons.append("板块强")
    if len(reasons) < 3 and _row_float(row, "risk_penalty_score", 100) <= 25:
        reasons.append("风险低")
    return "、".join(reasons[:3])


def _confirm_text() -> str:
    return "守8%、承接稳"


def _reject_text() -> str:
    return "破8%/急回落/板块转弱"


def build_buy_decision(ranked_input: pd.DataFrame, config: dict | None = None) -> BuyDecisionResult:
    cfg = DEFAULT_BUY_CONFIG.copy()
    cfg.update({key: value for key, value in (config or {}).items() if key in cfg})
    for key in DEFAULT_BUY_CONFIG:
        cfg[key] = float(cfg[key])
    # tail_profit_v1 was validated as a single-position model. Keep the
    # production constraint fixed even if an older config requests more.
    max_count = 1
    max_sector_positions = 1

    if ranked_input.empty:
        empty_plan = pd.DataFrame(columns=BUY_COLUMNS)
        empty_decision = pd.DataFrame(columns=DECISION_COLUMNS)
        return BuyDecisionResult(empty_plan, empty_decision, {"buy_count": 0})

    tail_columns = {
        "tail_profit_score",
        "tail_profit_eligible",
        "tail_profit_filter_reason",
        "tail_profit_model_version",
        "tail_rank_capital",
        "tail_rank_sector",
    }
    out = ranked_input.copy() if tail_columns.issubset(ranked_input.columns) else add_tail_profit_scores(ranked_input, config)
    if "rank" not in out.columns:
        out = _sort_for_buy(out)
        out["rank"] = out.index + 1

    out["sector_name"] = _txt(out, "sector_name", "未知板块")
    out["p_limitup_t1"] = _num(out, "p_limitup_t1")
    out["wp_score"] = _num(out, "wp_score")
    out["sector_strength_score"] = _num(out, "sector_strength_score")
    out["stock_strength_score"] = _num(out, "stock_strength_score")
    out["acceptance_score"] = _num(out, "acceptance_score")
    out["momentum_score"] = _num(out, "momentum_score")
    out["capital_score"] = _num(out, "capital_score")
    out["model_confidence"] = _num(out, "model_confidence")
    out["risk_penalty_score"] = _num(out, "risk_penalty_score")
    out["tail_profit_score"] = _num(out, "tail_profit_score")
    out["amount_ratio_5d"] = _num(out, "amount_ratio_5d")
    out["pct_chg"] = _num(out, "pct_chg")
    out["price"] = _num(out, "price")
    out["close_position"] = _num(out, "close_position", 50)
    out["intraday_pullback_pct"] = _num(out, "intraday_pullback_pct")
    out["intraday_vwap_position"] = _num(out, "intraday_vwap_position")
    out["pre_day_limitup"] = _num(out, "pre_day_limitup").astype(int)
    out["today_limitup"] = _num(out, "today_limitup").astype(int)
    out["decision_score"] = out["tail_profit_score"]
    out["skip_reason"] = _txt(out, "tail_profit_filter_reason")
    out["buy_eligible"] = out.get("tail_profit_eligible", False).fillna(False).astype(bool)
    out = _sort_for_buy(out)

    selected_indexes: list[int] = []
    sector_counts: dict[str, int] = {}
    for idx, row in out.iterrows():
        if not bool(row["buy_eligible"]):
            continue
        sector = str(row["sector_name"])
        if sector_counts.get(sector, 0) >= max_sector_positions:
            continue
        selected_indexes.append(idx)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected_indexes) >= max_count:
            break

    out["buy_flag"] = 0
    out["decision_action"] = "跳过"
    out.loc[out["buy_eligible"], "decision_action"] = "备选观察"
    if selected_indexes:
        out.loc[selected_indexes, "buy_flag"] = 1
        out.loc[selected_indexes, "decision_action"] = "买入观察"

    selected = out.loc[selected_indexes].copy()
    if not selected.empty:
        selected["buy_rank"] = range(1, len(selected) + 1)
        selected["portfolio_group"] = "主票"
        selected["confirm_before_buy"] = _confirm_text()
        selected["reject_if"] = _reject_text()
        selected["buy_reason"] = selected.apply(_reason, axis=1)
    for col in BUY_COLUMNS:
        if col not in selected.columns:
            selected[col] = ""
    buy_plan = selected[BUY_COLUMNS].copy()

    decision = out.copy()
    decision["buy_rank"] = pd.Series([""] * len(decision), index=decision.index, dtype="object")
    if selected_indexes:
        decision.loc[selected_indexes, "buy_rank"] = list(range(1, len(selected_indexes) + 1))
    decision["portfolio_group"] = pd.Series([""] * len(decision), index=decision.index, dtype="object")
    decision.loc[selected_indexes, "portfolio_group"] = buy_plan["portfolio_group"].values
    decision["confirm_before_buy"] = _confirm_text()
    decision["reject_if"] = _reject_text()
    decision["buy_reason"] = decision.apply(_reason, axis=1)
    for col in DECISION_COLUMNS:
        if col not in decision.columns:
            decision[col] = ""
    decision_table = decision[DECISION_COLUMNS].copy()
    summary = {
        "buy_count": int(len(buy_plan)),
        "max_buy_count": max_count,
        "max_sector_positions": max_sector_positions,
        "buy_model_version": TAIL_PROFIT_MODEL_VERSION,
        "selection_rule": "14:35截面选最多1支；无合格标的则空仓，14:50前人工确认。",
    }
    return BuyDecisionResult(buy_plan, decision_table, summary)
