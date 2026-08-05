from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data" / "v3" / "cache"

KEEP_DATE_COUNTS = {
    "daily": 50,
    "daily_basic": 50,
    "adj_factor": 50,
    "stk_limit": 5,
    "stock_basic": 3,
    "index_member_all": 3,
    "trade_cal": 4,
}
DATE_PATTERN = re.compile(r"20\d{6}")


def cache_file_date(path: Path) -> str:
    dates = DATE_PATTERN.findall(path.name)
    return max(dates, default="")


def prune_directory(path: Path, *, keep_date_count: int) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    dated = [(item, cache_file_date(item)) for item in files]
    keep_dates = set(
        sorted({date for _, date in dated if date})[-keep_date_count:]
    )
    removed = 0
    for item, date in dated:
        if date and date not in keep_dates:
            item.unlink()
            removed += 1
    return len(files) - removed, removed


def main() -> int:
    total_kept = 0
    total_removed = 0
    for directory, keep_date_count in KEEP_DATE_COUNTS.items():
        kept, removed = prune_directory(
            CACHE_ROOT / directory,
            keep_date_count=keep_date_count,
        )
        total_kept += kept
        total_removed += removed
        print(
            f"runtime cache {directory}: kept={kept} removed={removed}",
            flush=True,
        )
    print(
        f"runtime cache total: kept={total_kept} removed={total_removed}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
