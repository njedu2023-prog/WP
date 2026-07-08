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
UPSTREAM_MANIFEST = "data/wp/latest/wp_manifest.json"
WP_MANIFEST = "outputs/json/wp_manifest.json"
PAGES_MANIFEST_URL = os.environ.get(
    "WP_MONITOR_PAGES_MANIFEST_URL",
    "https://njedu2023-prog.github.io/WP/outputs/json/wp_manifest.json",
)
MAX_UPSTREAM_AGE_MIN = float(os.environ.get("WP_MONITOR_MAX_UPSTREAM_AGE_MIN", "25"))
MAX_WP_AGE_MIN = float(os.environ.get("WP_MONITOR_MAX_WP_AGE_MIN", "25"))
MAX_PAGE_LAG_MIN = float(os.environ.get("WP_MONITOR_MAX_PAGE_LAG_MIN", "20"))
SESSION_GRACE_MIN = float(os.environ.get("WP_MONITOR_SESSION_GRACE_MIN", "20"))


def now_cn() -> datetime:
    return datetime.now(CN_TZ).replace(tzinfo=None)


def in_trade_window(now: datetime) -> bool:
    return time(9, 25) <= now.time() <= time(11, 35) or time(12, 55) <= now.time() <= time(15, 10)


def session_start(now: datetime) -> datetime | None:
    if time(9, 25) <= now.time() <= time(11, 35):
        return datetime.combine(now.date(), time(9, 25))
    if time(12, 55) <= now.time() <= time(15, 10):
        return datetime.combine(now.date(), time(12, 55))
    return None


def parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    return None


def request_json(url: str, token: str = "", timeout: int = 30) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "WP-system-monitor",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def read_github_file(repo: str, path: str, token: str = "") -> dict:
    encoded = quote(path, safe="/")
    url = f"https://api.github.com/repos/{repo}/contents/{encoded}?ref=main"
    payload = request_json(url, token=token)
    content = "".join(str(payload.get("content", "")).split())
    if payload.get("encoding") != "base64" or not content:
        raise RuntimeError(f"Unsupported GitHub contents payload for {repo}/{path}")
    return json.loads(base64.b64decode(content).decode("utf-8-sig"))


def read_public_json(url: str) -> dict:
    separator = "&" if "?" in url else "?"
    return request_json(f"{url}{separator}monitor_cachebust={int(now_cn().timestamp())}")


def age_minutes(now: datetime, value: Any) -> float | None:
    parsed = parse_dt(value)
    if parsed is None:
        return None
    return (now - parsed).total_seconds() / 60


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


def monitor() -> int:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    now = now_cn()
    today = now.strftime("%Y%m%d")
    print(f"WP system monitor at {now:%Y-%m-%d %H:%M:%S} Asia/Shanghai")

    if not in_trade_window(now):
        print("Outside A-share trading window; monitor passes without freshness enforcement.")
        return 0

    errors: list[str] = []
    warnings: list[str] = []
    grace = within_start_grace(now)

    try:
        upstream = read_github_file(UPSTREAM_REPO, UPSTREAM_MANIFEST)
    except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"::error::Cannot read upstream manifest: {exc}")
        return 1

    try:
        wp = read_github_file(WP_REPO, WP_MANIFEST, token=token)
    except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"::error::Cannot read WP manifest: {exc}")
        return 1

    try:
        pages = read_public_json(PAGES_MANIFEST_URL)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        pages = {}
        warnings.append(f"Cannot read GitHub Pages manifest: {exc}")
        print(f"::warning::Cannot read GitHub Pages manifest: {exc}")

    upstream_age = age_minutes(now, upstream.get("generated_at"))
    wp_age = age_minutes(now, wp.get("wp_run_time") or wp.get("latest_update"))
    page_age = age_minutes(now, pages.get("wp_run_time") or pages.get("latest_update")) if pages else None
    wp_time = parse_dt(wp.get("wp_run_time") or wp.get("latest_update"))
    page_time = parse_dt(pages.get("wp_run_time") or pages.get("latest_update")) if pages else None
    page_lag = (wp_time - page_time).total_seconds() / 60 if wp_time and page_time else None

    print(json.dumps({"upstream": upstream, "wp": wp, "pages": pages}, ensure_ascii=False, indent=2)[:6000])

    add_check(
        errors,
        warnings,
        upstream.get("status") == "ok",
        f"upstream status is {upstream.get('status')!r}",
        warn_only=grace,
    )
    add_check(
        errors,
        warnings,
        str(upstream.get("source_trade_date")) == today,
        f"upstream trade date {upstream.get('source_trade_date')} matches today {today}",
        warn_only=grace,
    )
    add_check(
        errors,
        warnings,
        upstream_age is not None and upstream_age <= MAX_UPSTREAM_AGE_MIN,
        f"upstream generated_at age {upstream_age if upstream_age is not None else 'unknown'} min <= {MAX_UPSTREAM_AGE_MIN}",
        warn_only=grace,
    )
    add_check(
        errors,
        warnings,
        wp.get("health_status") == "ok",
        f"WP health_status is {wp.get('health_status')!r}",
        warn_only=grace,
    )
    add_check(
        errors,
        warnings,
        wp_age is not None and wp_age <= MAX_WP_AGE_MIN,
        f"WP run age {wp_age if wp_age is not None else 'unknown'} min <= {MAX_WP_AGE_MIN}",
        warn_only=grace,
    )
    add_check(
        errors,
        warnings,
        str(wp.get("market_data_time", "")).startswith(now.strftime("%Y-%m-%d")),
        f"WP market_data_time {wp.get('market_data_time')} is today",
        warn_only=grace,
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

    if warnings:
        print(f"Warnings: {len(warnings)}")
    if errors:
        print(f"Monitor failed with {len(errors)} error(s).")
        return 1
    print("Monitor passed.")
    return 0


if __name__ == "__main__":
    sys.exit(monitor())
