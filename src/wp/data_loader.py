from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import json

import pandas as pd

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal runners
    requests = None
from urllib.request import urlopen


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
            if requests is not None:
                response = requests.get(manifest_url, timeout=timeout)
                response.raise_for_status()
                return response.json()
            with urlopen(manifest_url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        manifest_path = Path(source).parent / "wp_manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def read_rank_input(source: str, cache_path: str | Path | None = None, timeout: int = 20) -> LoadResult:
    metadata = _read_manifest_for_source(source, timeout=timeout)
    try:
        if source.startswith("http://") or source.startswith("https://"):
            if requests is not None:
                response = requests.get(source, timeout=timeout)
                response.raise_for_status()
                text = response.text
            else:
                with urlopen(source, timeout=timeout) as response:
                    text = response.read().decode("utf-8")
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
