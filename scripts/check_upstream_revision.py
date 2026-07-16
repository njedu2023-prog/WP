from __future__ import annotations

import base64
import json
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from zoneinfo import ZoneInfo

try:
    from scripts.http_retry import request_json
except ModuleNotFoundError:  # Executed as python scripts/check_upstream_revision.py.
    from http_retry import request_json


ROOT = Path(__file__).resolve().parents[1]
LOCAL_MANIFEST = ROOT / "outputs" / "json" / "wp_manifest.json"
UPSTREAM_API = os.environ.get(
    "WP_UPSTREAM_MANIFEST_API",
    "https://api.github.com/repos/njedu2023-prog/a-share-top3-data/contents/data/wp/latest/wp_manifest.json?ref=main",
)
CN_TZ = ZoneInfo("Asia/Shanghai")
SCHEDULE_GRACE_SECONDS = int(os.environ.get("WP_SCHEDULE_GRACE_SECONDS", "600"))


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_upstream_manifest() -> dict[str, Any]:
    payload = request_json(
        UPSTREAM_API,
        user_agent="WP-direct-source-gate",
    )
    content = "".join(str(payload.get("content", "")).split())
    if payload.get("encoding") != "base64" or not content:
        raise RuntimeError("Unsupported upstream manifest payload.")
    return json.loads(base64.b64decode(content).decode("utf-8-sig"))


def _slots(start: datetime, end: datetime, minutes: int) -> list[datetime]:
    values: list[datetime] = []
    current = start
    while current <= end:
        values.append(current)
        current += timedelta(minutes=minutes)
    return values


def scheduled_slots(day: date) -> list[datetime]:
    def at(hour: int, minute: int) -> datetime:
        return datetime.combine(day, time(hour, minute), CN_TZ)

    return sorted(
        set(
            [
                *_slots(at(9, 25), at(11, 35), 10),
                *_slots(at(12, 55), at(14, 15), 10),
                *_slots(at(14, 20), at(14, 55), 5),
                at(15, 5),
                at(15, 10),
            ]
        )
    )


def latest_due_slot(current: datetime) -> datetime | None:
    local = current.astimezone(CN_TZ)
    morning_start = datetime.combine(local.date(), time(9, 25), CN_TZ)
    morning_end = datetime.combine(local.date(), time(11, 35), CN_TZ)
    afternoon_start = datetime.combine(local.date(), time(12, 55), CN_TZ)
    afternoon_end = datetime.combine(local.date(), time(15, 10), CN_TZ)
    grace = timedelta(seconds=SCHEDULE_GRACE_SECONDS)
    in_window = (
        morning_start <= local <= morning_end + grace
        or afternoon_start <= local <= afternoon_end + grace
    )
    if not in_window:
        return None
    due = [slot for slot in scheduled_slots(local.date()) if slot <= local]
    return due[-1] if due else None


def parse_datetime(value: Any) -> datetime | None:
    text_value = str(value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text_value[:19], fmt).replace(tzinfo=CN_TZ)
        except ValueError:
            continue
    return None


def _local_coverage(local: dict[str, Any]) -> datetime | None:
    for name in ("source_scheduled_slot", "source_generated_at", "market_data_time", "wp_run_time"):
        parsed = parse_datetime(local.get(name))
        if parsed is not None:
            return parsed
    return None


def resolve_decision(
    event_name: str,
    upstream: dict[str, Any],
    local: dict[str, Any],
    current: datetime | None = None,
) -> tuple[bool, str]:
    if event_name in {"push", "workflow_dispatch"}:
        return True, f"explicit {event_name} run"

    if event_name == "repository_dispatch":
        if upstream.get("status") != "ok":
            return False, f"upstream status is {upstream.get('status')!r}"
        upstream_revision = str(upstream.get("generated_at") or "").strip()
        local_revision = str(local.get("source_generated_at") or "").strip()
        if not upstream_revision:
            return False, "upstream generated_at is missing"
        if upstream_revision == local_revision:
            return False, "upstream revision already processed"
        return True, f"repository dispatch has upstream revision {upstream_revision}"

    now = current or datetime.now(CN_TZ)
    target = latest_due_slot(now)
    if target is None:
        return False, "outside A-share data window"
    coverage = _local_coverage(local)
    local_trade_date = str(
        local.get("source_trade_date") or local.get("data_trade_date") or ""
    ).replace("-", "")
    if not local_trade_date and coverage is not None:
        local_trade_date = coverage.strftime("%Y%m%d")
    if local_trade_date == target.strftime("%Y%m%d") and coverage is not None and coverage >= target:
        return False, f"target slot {target:%H:%M} already covered"
    return True, f"direct source target slot {target:%Y-%m-%d %H:%M:%S} is due"


def write_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT", "").strip()
    if output_path:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def main() -> None:
    event_name = os.environ.get("GITHUB_EVENT_NAME", "schedule").strip()
    local = read_json(LOCAL_MANIFEST)
    current = datetime.now(CN_TZ)
    upstream: dict[str, Any] = {}
    upstream_error = ""
    if event_name == "repository_dispatch":
        try:
            upstream = read_upstream_manifest()
        except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            upstream_error = str(exc)
    should_run, reason = resolve_decision(event_name, upstream, local, current)

    target = latest_due_slot(current)
    upstream_revision = str(upstream.get("generated_at") or "")
    write_output("should_run", str(should_run).lower())
    write_output("target_slot", target.strftime("%Y-%m-%d %H:%M:%S") if target else "")
    write_output("upstream_revision", upstream_revision)
    write_output("reason", reason)
    print(
        json.dumps(
            {
                "event_name": event_name,
                "should_run": should_run,
                "target_slot": target.strftime("%Y-%m-%d %H:%M:%S") if target else "",
                "upstream_revision": upstream_revision,
                "local_revision": local.get("source_generated_at", ""),
                "local_source_mode": local.get("source_mode", ""),
                "upstream_read_error": upstream_error,
                "reason": reason,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
