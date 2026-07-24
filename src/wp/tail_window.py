from __future__ import annotations

from datetime import datetime, time

import pandas as pd


TAIL_WINDOW_START = time(14, 20)
TAIL_WINDOW_END = time(14, 50)
MARKET_CLOSE = time(15, 0)

TAIL_PHASE_UNKNOWN = "unknown"
TAIL_PHASE_BEFORE = "before_tail_window"
TAIL_PHASE_ACTIVE = "tail_window_active"
TAIL_PHASE_FROZEN = "tail_window_frozen"
TAIL_PHASE_CLOSED = "market_closed"


def parse_market_datetime(value: object) -> datetime | None:
    parsed = pd.to_datetime(str(value or "").strip(), errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def tail_window_phase(value: object) -> str:
    parsed = parse_market_datetime(value)
    if parsed is None:
        return TAIL_PHASE_UNKNOWN
    clock = parsed.time()
    if clock < TAIL_WINDOW_START:
        return TAIL_PHASE_BEFORE
    if clock <= TAIL_WINDOW_END:
        return TAIL_PHASE_ACTIVE
    if clock < MARKET_CLOSE:
        return TAIL_PHASE_FROZEN
    return TAIL_PHASE_CLOSED


def accepts_new_tail_primary(value: object) -> bool:
    return tail_window_phase(value) == TAIL_PHASE_ACTIVE
