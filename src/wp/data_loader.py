from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from urllib.parse import quote, urlparse
import base64
import json
import os

import pandas as pd

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal runners
    requests = None
from urllib.request import Request, urlopen


@dataclass
class LoadResult:
    frame: pd.DataFrame
    source: str
    ok: bool
    error: str = ""
    fallback_used: bool = False
    metadata: dict | None = None


def _read_manifest_for_source(source: str, timeout: int = 20) -> dict:
    try:
        if source.startswith("http://") or source.startswith("https://"):
            manifest_url = source.rsplit("/", 1)[0] + "/wp_manifest.json"
            return json.loads(_read_remote_text(manifest_url, timeout=timeout))
        manifest_path = Path(source).parent / "wp_manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "WP-data-loader",
    }
    token = os.environ.get("UPSTREAM_GITHUB_TOKEN", "").strip() or os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _contents_api_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if parsed.netloc != "raw.githubusercontent.com":
        return ""
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 4:
        return ""
    owner, repo = parts[0], parts[1]
    if len(parts) >= 6 and parts[2] == "refs" and parts[3] == "heads":
        branch = parts[4]
        rel_path = "/".join(parts[5:])
    else:
        branch = parts[2]
        rel_path = "/".join(parts[3:])
    return f"https://api.github.com/repos/{owner}/{repo}/contents/{quote(rel_path, safe='/')}?ref={quote(branch, safe='')}"


def _read_github_contents(api_url: str, timeout: int) -> str:
    if requests is not None:
        response = requests.get(api_url, headers=_github_headers(), timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    else:
        req = Request(api_url, headers=_github_headers())
        with urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    content = "".join(str(payload.get("content", "")).split())
    encoding = payload.get("encoding", "")
    if encoding != "base64" or not content:
        raise ValueError(f"Unsupported GitHub contents payload: encoding={encoding!r}")
    return base64.b64decode(content).decode("utf-8-sig")


def _read_remote_text(source: str, timeout: int = 20) -> str:
    api_url = _contents_api_url(source)
    if api_url:
        try:
            return _read_github_contents(api_url, timeout)
        except Exception:
            pass
    headers = {"User-Agent": "WP-data-loader"}
    if requests is not None:
        response = requests.get(source, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    req = Request(source, headers=headers)
    with urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8-sig")


def read_rank_input(source: str, cache_path: str | Path | None = None, timeout: int = 20) -> LoadResult:
    metadata = _read_manifest_for_source(source, timeout=timeout)
    try:
        if source.startswith("http://") or source.startswith("https://"):
            text = _read_remote_text(source, timeout=timeout)
            if cache_path:
                cache_file = Path(cache_path)
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(text, encoding="utf-8")
            return LoadResult(pd.read_csv(StringIO(text)), source, True, metadata=metadata)
        return LoadResult(pd.read_csv(source), source, True, metadata=metadata)
    except Exception as exc:
        if cache_path and Path(cache_path).exists():
            try:
                return LoadResult(pd.read_csv(cache_path), str(cache_path), True, str(exc), True, metadata=metadata)
            except Exception as cache_exc:
                return LoadResult(pd.DataFrame(), source, False, f"{exc}; cache failed: {cache_exc}", True, metadata=metadata)
        return LoadResult(pd.DataFrame(), source, False, str(exc), False, metadata=metadata)
