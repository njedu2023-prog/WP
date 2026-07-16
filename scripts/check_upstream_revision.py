from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
LOCAL_MANIFEST = ROOT / "outputs" / "json" / "wp_manifest.json"
UPSTREAM_API = os.environ.get(
    "WP_UPSTREAM_MANIFEST_API",
    "https://api.github.com/repos/njedu2023-prog/a-share-top3-data/contents/data/wp/latest/wp_manifest.json?ref=main",
)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_upstream_manifest() -> dict[str, Any]:
    request = Request(
        UPSTREAM_API,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "WP-upstream-revision-gate",
        },
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = "".join(str(payload.get("content", "")).split())
    if payload.get("encoding") != "base64" or not content:
        raise RuntimeError("Unsupported upstream manifest payload.")
    return json.loads(base64.b64decode(content).decode("utf-8-sig"))


def resolve_decision(event_name: str, upstream: dict[str, Any], local: dict[str, Any]) -> tuple[bool, str]:
    if event_name in {"push", "workflow_dispatch"}:
        return True, f"explicit {event_name} run"
    if upstream.get("status") != "ok":
        return False, f"upstream status is {upstream.get('status')!r}"

    upstream_revision = str(upstream.get("generated_at") or "").strip()
    local_revision = str(local.get("source_generated_at") or "").strip()
    if not upstream_revision:
        return False, "upstream generated_at is missing"
    if upstream_revision == local_revision:
        return False, "upstream revision already processed"
    return True, f"new upstream revision {upstream_revision} (local {local_revision or 'missing'})"


def write_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT", "").strip()
    if output_path:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def main() -> None:
    event_name = os.environ.get("GITHUB_EVENT_NAME", "schedule").strip()
    local = read_json(LOCAL_MANIFEST)
    try:
        upstream = read_upstream_manifest()
        should_run, reason = resolve_decision(event_name, upstream, local)
    except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        upstream = {}
        should_run = event_name in {"push", "workflow_dispatch"}
        reason = f"cannot read upstream manifest: {exc}"

    upstream_revision = str(upstream.get("generated_at") or "")
    write_output("should_run", str(should_run).lower())
    write_output("upstream_revision", upstream_revision)
    write_output("reason", reason)
    print(
        json.dumps(
            {
                "event_name": event_name,
                "should_run": should_run,
                "upstream_revision": upstream_revision,
                "local_revision": local.get("source_generated_at", ""),
                "reason": reason,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
