from __future__ import annotations

import base64
import json
import os
import sys
from datetime import datetime, time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from zoneinfo import ZoneInfo

try:
    from scripts.check_upstream_revision import latest_due_slot
    from scripts.http_retry import request_json
except ModuleNotFoundError:  # Executed as python scripts/monitor_wp_system.py.
    from check_upstream_revision import latest_due_slot
    from http_retry import request_json


CN_TZ = ZoneInfo("Asia/Shanghai")
WP_REPO = os.environ.get("WP_MONITOR_WP_REPO", "njedu2023-prog/WP")
WP_WORKFLOW = os.environ.get("WP_MONITOR_WP_WORKFLOW", "run_wp_10min.yml")
WP_MANIFEST = "outputs/json/wp_manifest.json"
PAGES_MANIFEST_URL = os.environ.get(
    "WP_MONITOR_PAGES_MANIFEST_URL",
    "https://njedu2023-prog.github.io/WP/outputs/json/wp_manifest.json",
)
MAX_PAGE_LAG_MIN = float(os.environ.get("WP_MONITOR_MAX_PAGE_LAG_MIN", "10"))
ACCEPTED_HEALTH_STATUSES = {"ok", "无符合条件股票"}
ACTIVE_RUN_STATUSES = {"queued", "in_progress", "waiting", "pending"}


def now_cn() -> datetime:
    return datetime.now(CN_TZ)


def in_trade_window(current: datetime) -> bool:
    return (
        time(9, 25) <= current.time() <= time(11, 45)
        or time(12, 55) <= current.time() <= time(15, 10)
    )


def in_close_finalize_window(current: datetime) -> bool:
    return time(15, 10) < current.time() <= time(15, 35)


def in_monitor_window(current: datetime) -> bool:
    return in_trade_window(current) or in_close_finalize_window(current)


def parse_dt(value: Any) -> datetime | None:
    text_value = str(value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text_value[:19], fmt).replace(tzinfo=CN_TZ)
        except ValueError:
            continue
    return None


def target_slot(current: datetime) -> datetime | None:
    if in_close_finalize_window(current):
        return datetime.combine(current.date(), time(15, 10), CN_TZ)
    return latest_due_slot(current)


def read_github_file(repo: str, path: str, token: str = "") -> dict[str, Any]:
    encoded = quote(path, safe="/")
    payload = request_json(
        f"https://api.github.com/repos/{repo}/contents/{encoded}?ref=main",
        token=token,
        user_agent="WP-direct-system-monitor",
    )
    content = "".join(str(payload.get("content", "")).split())
    if payload.get("encoding") != "base64" or not content:
        raise RuntimeError(f"Unsupported GitHub contents payload for {repo}/{path}")
    return json.loads(base64.b64decode(content).decode("utf-8-sig"))


def read_public_json(url: str) -> dict[str, Any]:
    separator = "&" if "?" in url else "?"
    return request_json(
        f"{url}{separator}monitor_cachebust={int(now_cn().timestamp())}",
        user_agent="WP-direct-system-monitor",
    )


def is_trade_day(token: str, day: str) -> bool:
    payload = request_json(
        "https://api.tushare.pro",
        method="POST",
        payload={
            "api_name": "trade_cal",
            "token": token,
            "params": {"exchange": "SSE", "start_date": day, "end_date": day},
            "fields": "cal_date,is_open",
        },
        user_agent="WP-direct-system-monitor",
    )
    if int(payload.get("code", -1)) != 0:
        raise RuntimeError(f"Tushare trade_cal failed: {payload.get('msg')}")
    fields = payload.get("data", {}).get("fields", [])
    items = payload.get("data", {}).get("items", [])
    if not items or "is_open" not in fields:
        return False
    return int(items[0][fields.index("is_open")]) == 1


def workflow_runs(repo: str, workflow: str, token: str) -> list[dict[str, Any]]:
    payload = request_json(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/runs?per_page=10",
        token=token,
        user_agent="WP-direct-system-monitor",
    )
    return list(payload.get("workflow_runs", []))


def workflow_active(runs: list[dict[str, Any]]) -> bool:
    return any(run.get("status") in ACTIVE_RUN_STATUSES for run in runs)


def dispatch_workflow(repo: str, workflow: str, token: str) -> None:
    request_json(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches",
        token=token,
        method="POST",
        payload={
            "ref": "main",
            "inputs": {"mode": "live", "force_rebuild": "true"},
        },
        user_agent="WP-direct-system-monitor",
    )


def source_coverage(manifest: dict[str, Any]) -> datetime | None:
    for field in (
        "source_scheduled_slot",
        "scheduled_slot",
        "market_data_time",
        "source_generated_at",
    ):
        parsed = parse_dt(manifest.get(field))
        if parsed is not None:
            return parsed
    return None


def wp_health(manifest: dict[str, Any], slot: datetime) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    health = str(manifest.get("health_status") or "").strip()
    source_mode = str(manifest.get("source_mode") or "").strip()
    source_trade_date = str(manifest.get("source_trade_date") or "").replace("-", "")
    coverage = source_coverage(manifest)
    if health not in ACCEPTED_HEALTH_STATUSES:
        reasons.append(f"health_status={health or 'missing'}")
    if source_mode != "direct_tushare":
        reasons.append(f"source_mode={source_mode or 'missing'}")
    if source_trade_date != slot.strftime("%Y%m%d"):
        reasons.append(f"source_trade_date={source_trade_date or 'missing'}")
    if coverage is None or coverage < slot:
        reasons.append(
            f"coverage={coverage.strftime('%Y-%m-%d %H:%M:%S') if coverage else 'missing'}"
        )
    if bool(manifest.get("direct_fallback_used")):
        reasons.append("direct_fallback_used=true")
    return not reasons, reasons


def pages_health(
    wp_manifest: dict[str, Any],
    pages_manifest: dict[str, Any],
) -> tuple[bool, str]:
    wp_revision = str(wp_manifest.get("report_revision") or "").strip()
    page_revision = str(pages_manifest.get("report_revision") or "").strip()
    if wp_revision and page_revision == wp_revision:
        return True, "report revisions match"
    wp_time = parse_dt(wp_manifest.get("wp_run_time") or wp_revision)
    page_time = parse_dt(pages_manifest.get("wp_run_time") or page_revision)
    if wp_time is None or page_time is None:
        return False, "report revision or run time is missing"
    lag = (wp_time - page_time).total_seconds() / 60
    return lag <= MAX_PAGE_LAG_MIN, f"Pages lag={lag:.1f} min"


def repair_or_wait(
    *,
    reason: str,
    runs: list[dict[str, Any]],
    token: str,
) -> int:
    if workflow_active(runs):
        print(f"::warning::{reason}; WP workflow is already active, wait for completion.")
        return 0
    latest = runs[0] if runs else {}
    if latest.get("conclusion") in {"failure", "cancelled", "timed_out"}:
        print(
            "::warning::Latest WP workflow ended "
            f"{latest.get('conclusion')}: {latest.get('html_url', '')}"
        )
    if not token:
        print(f"::error::{reason}; GITHUB_TOKEN is unavailable, cannot self-heal.")
        return 1
    try:
        dispatch_workflow(WP_REPO, WP_WORKFLOW, token)
    except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"::error::{reason}; cannot dispatch WP repair: {exc}")
        return 1
    print(f"::warning::{reason}; dispatched a forced direct-source WP repair.")
    return 0


def monitor(current: datetime | None = None) -> int:
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    tushare_token = os.environ.get("TUSHARE_TOKEN", "").strip()
    current = current or now_cn()
    today = current.strftime("%Y%m%d")
    print(f"WP direct-system monitor at {current:%Y-%m-%d %H:%M:%S} Asia/Shanghai")

    if not in_monitor_window(current):
        print("Outside A-share monitoring window; no action.")
        return 0
    if tushare_token:
        try:
            if not is_trade_day(tushare_token, today):
                print(f"{today} is not an A-share trading day; no action.")
                return 0
        except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"::warning::Cannot verify Tushare calendar; continue with manifest checks: {exc}")

    slot = target_slot(current)
    if slot is None:
        print("No due market-data slot; no action.")
        return 0

    try:
        wp_manifest = read_github_file(WP_REPO, WP_MANIFEST, token=github_token)
        runs = workflow_runs(WP_REPO, WP_WORKFLOW, github_token)
    except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"::error::Cannot read WP production state: {exc}")
        return 1

    wp_ok, reasons = wp_health(wp_manifest, slot)
    coverage = source_coverage(wp_manifest)
    print(
        json.dumps(
            {
                "target_slot": slot.strftime("%Y-%m-%d %H:%M:%S"),
                "source_mode": wp_manifest.get("source_mode"),
                "source_trade_date": wp_manifest.get("source_trade_date"),
                "coverage": coverage.strftime("%Y-%m-%d %H:%M:%S") if coverage else "",
                "wp_run_time": wp_manifest.get("wp_run_time"),
                "report_revision": wp_manifest.get("report_revision"),
                "active_workflow": workflow_active(runs),
            },
            ensure_ascii=False,
        )
    )
    if not wp_ok:
        return repair_or_wait(
            reason=f"WP direct data does not cover {slot:%H:%M}: {', '.join(reasons)}",
            runs=runs,
            token=github_token,
        )

    try:
        pages_manifest = read_public_json(PAGES_MANIFEST_URL)
    except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        pages_manifest = {}
        print(f"::warning::Cannot read Pages manifest: {exc}")
    pages_ok, pages_reason = pages_health(wp_manifest, pages_manifest)
    if not pages_ok:
        return repair_or_wait(
            reason=f"GitHub Pages is behind WP: {pages_reason}",
            runs=runs,
            token=github_token,
        )

    print(f"Monitor passed: direct WP covers {slot:%H:%M}; {pages_reason}.")
    return 0


if __name__ == "__main__":
    sys.exit(monitor())
