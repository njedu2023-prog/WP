from __future__ import annotations

import base64
import json
import os
import sys
from datetime import datetime, time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


CN_TZ = ZoneInfo("Asia/Shanghai")
UPSTREAM_REPO = os.environ.get("WP_MONITOR_UPSTREAM_REPO", "njedu2023-prog/a-share-top3-data")
WP_REPO = os.environ.get("WP_MONITOR_WP_REPO", "njedu2023-prog/WP")
UPSTREAM_WORKFLOW = os.environ.get("WP_MONITOR_UPSTREAM_WORKFLOW", "run_wp_data_10min.yml")
WP_WORKFLOW = os.environ.get("WP_MONITOR_WP_WORKFLOW", "run_wp_10min.yml")
UPSTREAM_MANIFEST = "data/wp/latest/wp_manifest.json"
WP_MANIFEST = "outputs/json/wp_manifest.json"
PAGES_MANIFEST_URL = os.environ.get(
    "WP_MONITOR_PAGES_MANIFEST_URL",
    "https://njedu2023-prog.github.io/WP/outputs/json/wp_manifest.json",
)
MAX_UPSTREAM_AGE_MIN = float(os.environ.get("WP_MONITOR_MAX_UPSTREAM_AGE_MIN", "25"))
MAX_WP_AGE_MIN = float(os.environ.get("WP_MONITOR_MAX_WP_AGE_MIN", "25"))
MAX_PAGE_LAG_MIN = float(os.environ.get("WP_MONITOR_MAX_PAGE_LAG_MIN", "20"))
SESSION_GRACE_MIN = float(os.environ.get("WP_MONITOR_SESSION_GRACE_MIN", "25"))


def now_cn() -> datetime:
    return datetime.now(CN_TZ).replace(tzinfo=None)


def in_trade_window(now: datetime) -> bool:
    return time(9, 25) <= now.time() <= time(11, 35) or time(12, 55) <= now.time() <= time(15, 10)


def in_close_finalize_window(now: datetime) -> bool:
    # The upstream 15:10 slot can finish after the normal trading-window monitor.
    return time(15, 10) < now.time() <= time(15, 35)


def in_monitor_window(now: datetime) -> bool:
    return in_trade_window(now) or in_close_finalize_window(now)


def session_start(now: datetime) -> datetime | None:
    if time(9, 25) <= now.time() <= time(11, 35):
        return datetime.combine(now.date(), time(9, 25))
    if time(12, 55) <= now.time() <= time(15, 10):
        return datetime.combine(now.date(), time(12, 55))
    return None


def parse_dt(value: Any) -> datetime | None:
    text_value = str(value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text_value[:19], fmt)
        except ValueError:
            continue
    return None


def request_json(
    url: str,
    token: str = "",
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "WP-system-monitor",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=data, method=method, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return json.loads(body) if body else {}


def read_github_file(repo: str, path: str, token: str = "") -> dict[str, Any]:
    encoded = quote(path, safe="/")
    payload = request_json(f"https://api.github.com/repos/{repo}/contents/{encoded}?ref=main", token=token)
    content = "".join(str(payload.get("content", "")).split())
    if payload.get("encoding") != "base64" or not content:
        raise RuntimeError(f"Unsupported GitHub contents payload for {repo}/{path}")
    return json.loads(base64.b64decode(content).decode("utf-8-sig"))


def read_public_json(url: str) -> dict[str, Any]:
    separator = "&" if "?" in url else "?"
    return request_json(f"{url}{separator}monitor_cachebust={int(now_cn().timestamp())}")


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
    )
    if int(payload.get("code", -1)) != 0:
        raise RuntimeError(f"Tushare trade_cal failed: {payload.get('msg')}")
    fields = payload.get("data", {}).get("fields", [])
    items = payload.get("data", {}).get("items", [])
    if not items or "is_open" not in fields:
        return False
    return int(items[0][fields.index("is_open")]) == 1


def age_minutes(now: datetime, value: Any) -> float | None:
    parsed = parse_dt(value)
    if parsed is None:
        return None
    return (now - parsed).total_seconds() / 60


def workflow_active(repo: str, workflow: str, token: str) -> bool:
    payload = request_json(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/runs?per_page=10",
        token=token,
    )
    return any(run.get("status") in {"queued", "in_progress", "waiting", "pending"} for run in payload.get("workflow_runs", []))


def dispatch_workflow(repo: str, workflow: str, token: str, inputs: dict[str, str]) -> None:
    request_json(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches",
        token=token,
        method="POST",
        payload={"ref": "main", "inputs": inputs},
    )


def add_check(errors: list[str], warnings: list[str], ok: bool, message: str, warn_only: bool = False) -> None:
    if ok:
        print(f"[ok] {message}")
    elif warn_only:
        warnings.append(message)
        print(f"::warning::{message}")
    else:
        errors.append(message)
        print(f"::error::{message}")


def within_start_grace(now: datetime) -> bool:
    start = session_start(now)
    return bool(start and (now - start).total_seconds() / 60 < SESSION_GRACE_MIN)


def attempt_repairs(
    upstream_fresh: bool,
    wp_fresh: bool,
    grace: bool,
    github_token: str,
    repair_token: str,
) -> None:
    if grace:
        return
    if not upstream_fresh:
        if not repair_token:
            print("::warning::Upstream repair token is not configured; the upstream watchdog remains the fallback.")
            return
        try:
            if workflow_active(UPSTREAM_REPO, UPSTREAM_WORKFLOW, repair_token):
                print("::warning::Upstream is stale, but an upstream workflow is already active.")
            else:
                dispatch_workflow(UPSTREAM_REPO, UPSTREAM_WORKFLOW, repair_token, {"mode": "due"})
                print("::warning::Upstream is stale; dispatched an upstream repair.")
        except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"::error::Cannot dispatch upstream repair: {exc}")
        return
    if wp_fresh:
        return
    try:
        if workflow_active(WP_REPO, WP_WORKFLOW, github_token):
            print("::warning::WP is stale, but a WP workflow is already active.")
        else:
            dispatch_workflow(WP_REPO, WP_WORKFLOW, github_token, {"mode": "live"})
            print("::warning::WP is stale while upstream is fresh; dispatched a WP repair.")
    except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"::error::Cannot dispatch WP repair: {exc}")


def monitor() -> int:
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    repair_token = os.environ.get("WP_REPAIR_TOKEN", "").strip()
    tushare_token = os.environ.get("TUSHARE_TOKEN", "").strip()
    current = now_cn()
    today = current.strftime("%Y%m%d")
    print(f"WP system monitor at {current:%Y-%m-%d %H:%M:%S} Asia/Shanghai")

    if not in_monitor_window(current):
        print("Outside A-share trading or close-finalization window; no monitoring action.")
        return 0
    if tushare_token:
        try:
            if not is_trade_day(tushare_token, today):
                print(f"{today} is not an A-share trading day; no monitoring action.")
                return 0
        except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"::warning::Cannot verify Tushare calendar; continue with manifest checks: {exc}")
    else:
        print("::warning::TUSHARE_TOKEN is not configured; continue with manifest checks.")

    errors: list[str] = []
    warnings: list[str] = []
    grace = within_start_grace(current)

    try:
        upstream = read_github_file(UPSTREAM_REPO, UPSTREAM_MANIFEST, token=repair_token)
        wp = read_github_file(WP_REPO, WP_MANIFEST, token=github_token)
    except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"::error::Cannot read system manifests: {exc}")
        return 1

    try:
        pages = read_public_json(PAGES_MANIFEST_URL)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        pages = {}
        warnings.append(f"Cannot read GitHub Pages manifest: {exc}")
        print(f"::warning::Cannot read GitHub Pages manifest: {exc}")

    upstream_time = parse_dt(upstream.get("generated_at"))
    wp_time = parse_dt(wp.get("wp_run_time") or wp.get("latest_update"))
    page_time = parse_dt(pages.get("wp_run_time") or pages.get("latest_update")) if pages else None
    upstream_age = age_minutes(current, upstream.get("generated_at"))
    wp_age = age_minutes(current, wp.get("wp_run_time") or wp.get("latest_update"))
    page_age = age_minutes(current, pages.get("wp_run_time") or pages.get("latest_update")) if pages else None
    wp_upstream_lag = (upstream_time - wp_time).total_seconds() / 60 if upstream_time and wp_time else None
    page_lag = (wp_time - page_time).total_seconds() / 60 if wp_time and page_time else None
    upstream_fresh = (
        upstream.get("status") == "ok"
        and str(upstream.get("source_trade_date")) == today
        and upstream_age is not None
        and upstream_age <= MAX_UPSTREAM_AGE_MIN
    )
    wp_fresh = (
        wp.get("health_status") == "ok"
        and wp_age is not None
        and wp_age <= MAX_WP_AGE_MIN
        and wp_upstream_lag is not None
        and wp_upstream_lag <= 0
    )

    print(json.dumps({"upstream": upstream, "wp": wp, "pages": pages}, ensure_ascii=False, indent=2)[:6000])
    add_check(errors, warnings, upstream.get("status") == "ok", f"upstream status is {upstream.get('status')!r}", grace)
    add_check(
        errors,
        warnings,
        str(upstream.get("source_trade_date")) == today,
        f"upstream trade date {upstream.get('source_trade_date')} matches today {today}",
        grace,
    )
    add_check(
        errors,
        warnings,
        upstream_age is not None and upstream_age <= MAX_UPSTREAM_AGE_MIN,
        f"upstream generated_at age {upstream_age if upstream_age is not None else 'unknown'} min <= {MAX_UPSTREAM_AGE_MIN}",
        grace,
    )
    add_check(errors, warnings, wp.get("health_status") == "ok", f"WP health_status is {wp.get('health_status')!r}", grace)
    add_check(
        errors,
        warnings,
        wp_age is not None and wp_age <= MAX_WP_AGE_MIN,
        f"WP run age {wp_age if wp_age is not None else 'unknown'} min <= {MAX_WP_AGE_MIN}",
        grace,
    )
    add_check(
        errors,
        warnings,
        wp_upstream_lag is not None and wp_upstream_lag <= 0,
        f"WP processed latest upstream generation; lag={wp_upstream_lag if wp_upstream_lag is not None else 'unknown'} min",
        grace,
    )
    add_check(
        errors,
        warnings,
        str(wp.get("market_data_time", "")).startswith(current.strftime("%Y-%m-%d")),
        f"WP market_data_time {wp.get('market_data_time')} is today",
        grace,
    )
    if pages:
        add_check(
            errors,
            warnings,
            page_age is not None and page_age <= MAX_WP_AGE_MIN + MAX_PAGE_LAG_MIN,
            f"Pages run age {page_age if page_age is not None else 'unknown'} min within allowed lag",
            warn_only=True,
        )
        add_check(
            errors,
            warnings,
            page_lag is not None and page_lag <= MAX_PAGE_LAG_MIN,
            f"Pages lag {page_lag if page_lag is not None else 'unknown'} min <= {MAX_PAGE_LAG_MIN}",
            warn_only=True,
        )

    attempt_repairs(upstream_fresh, wp_fresh, grace, github_token, repair_token)
    if warnings:
        print(f"Warnings: {len(warnings)}")
    if errors:
        print(f"Monitor found {len(errors)} health error(s).")
        return 1
    print("Monitor passed.")
    return 0


if __name__ == "__main__":
    sys.exit(monitor())
