from __future__ import annotations

import numpy as np
import pandas as pd


MARKET_REGIME_MODEL_VERSION = "market_regime_v1"

DEFAULT_REGIME_CONFIG = {
    "regime_allow_score": 55.0,
    "regime_avoid_score": 40.0,
    "regime_min_universe_count": 200,
}


def _num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _is_main_board(code: object) -> bool:
    text = str(code or "").strip()
    raw = text.split(".", 1)[0]
    return raw.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))


def assess_market_regime(
    raw: pd.DataFrame,
    candidates: pd.DataFrame,
    config: dict | None = None,
) -> dict:
    cfg = DEFAULT_REGIME_CONFIG.copy()
    cfg.update({key: value for key, value in (config or {}).items() if key in cfg})
    if raw is None or raw.empty or "ts_code" not in raw.columns:
        return {
            "model_version": MARKET_REGIME_MODEL_VERSION,
            "state": "\u6570\u636e\u4e0d\u8db3",
            "score": 0.0,
            "manual_action": "\u5efa\u8bae\u7a7a\u4ed3",
            "reason": "\u7f3a\u5c11\u53ef\u7528\u5e02\u573a\u5e7f\u5ea6\u6570\u636e",
            "manual_decision_support_only": True,
        }

    universe = raw[raw["ts_code"].map(_is_main_board)].copy()
    suspended = _num(universe, "suspended_flag", 0).fillna(0).astype(int)
    quality = _num(universe, "data_quality_flag", 0).fillna(0).astype(int)
    price = _num(universe, "price")
    amount = _num(universe, "amount")
    universe = universe[suspended.eq(0) & quality.eq(0) & price.gt(0) & amount.gt(0)].copy()
    pct = _num(universe, "pct_chg")
    valid = pct.notna()
    pct = pct[valid]
    universe = universe.loc[pct.index]
    universe_count = int(len(universe))
    if universe_count < int(cfg["regime_min_universe_count"]):
        return {
            "model_version": MARKET_REGIME_MODEL_VERSION,
            "state": "\u6570\u636e\u4e0d\u8db3",
            "score": 0.0,
            "universe_count": universe_count,
            "manual_action": "\u5efa\u8bae\u7a7a\u4ed3",
            "reason": "\u4e3b\u677f\u6709\u6548\u884c\u60c5\u6837\u672c\u4e0d\u8db3",
            "manual_decision_support_only": True,
        }

    breadth_up_rate = float(pct.gt(0).mean() * 100)
    breadth_strong_rate = float(pct.ge(3).mean() * 100)
    breadth_weak_rate = float(pct.le(-3).mean() * 100)
    today_limitup = _num(universe, "today_limitup", 0).fillna(0).astype(int).eq(1)
    limit_up_count = int((today_limitup | pct.ge(9.5)).sum())
    limit_down_count = int(pct.le(-9.5).sum())
    candidate_count = int(len(candidates))
    candidate_sector_count = int(candidates.get("sector_name", pd.Series(dtype="object")).fillna("").astype(str).replace("", np.nan).nunique())

    score = 50.0
    score += (breadth_up_rate - 50.0) * 0.55
    score += min(breadth_strong_rate, 12.0) * 0.9
    score -= min(breadth_weak_rate, 12.0) * 0.8
    score += min(limit_up_count, 30) * 0.45
    score -= min(limit_down_count, 30) * 1.1
    score += min(candidate_count, 8) * 1.2
    if candidate_count and candidate_sector_count <= 1:
        score -= 4.0
    score = float(np.clip(score, 0, 100))

    allow_score = float(cfg["regime_allow_score"])
    avoid_score = float(cfg["regime_avoid_score"])
    if score >= allow_score:
        state = "\u5141\u8bb8\u5bfb\u627e\u673a\u4f1a"
        action = "\u53ef\u8bc4\u4f30\u9ad8\u8d28\u91cf\u4e3b\u7968"
    elif score >= avoid_score:
        state = "\u8c28\u614e"
        action = "\u53ea\u63a5\u53d7\u9ad8\u7f6e\u4fe1\u5ea6\u4e3b\u7968"
    else:
        state = "\u56de\u907f"
        action = "\u5efa\u8bae\u7a7a\u4ed3"

    reasons: list[str] = []
    reasons.append(f"\u4e3b\u677f\u4e0a\u6da8{breadth_up_rate:.1f}%")
    reasons.append(f"\u5f3a\u52bf{breadth_strong_rate:.1f}%")
    reasons.append(f"\u6da8\u505c{limit_up_count}\u53ea/\u8dcc\u505c{limit_down_count}\u53ea")
    reasons.append(f"\u5c3e\u76d8\u5019\u9009{candidate_count}\u53ea")
    return {
        "model_version": MARKET_REGIME_MODEL_VERSION,
        "state": state,
        "score": round(score, 2),
        "manual_action": action,
        "reason": "\uff1b".join(reasons),
        "universe_count": universe_count,
        "breadth_up_rate": round(breadth_up_rate, 2),
        "breadth_strong_rate": round(breadth_strong_rate, 2),
        "breadth_weak_rate": round(breadth_weak_rate, 2),
        "limit_up_count": limit_up_count,
        "limit_down_count": limit_down_count,
        "candidate_count": candidate_count,
        "candidate_sector_count": candidate_sector_count,
        "manual_decision_support_only": True,
    }
