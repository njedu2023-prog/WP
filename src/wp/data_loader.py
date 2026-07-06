from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

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


def read_rank_input(source: str, cache_path: str | Path | None = None, timeout: int = 20) -> LoadResult:
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
            return LoadResult(pd.read_csv(StringIO(text)), source, True)
        return LoadResult(pd.read_csv(source), source, True)
    except Exception as exc:
        if cache_path and Path(cache_path).exists():
            try:
                return LoadResult(pd.read_csv(cache_path), str(cache_path), True, str(exc), True)
            except Exception as cache_exc:
                return LoadResult(pd.DataFrame(), source, False, f"{exc}; cache failed: {cache_exc}", True)
        return LoadResult(pd.DataFrame(), source, False, str(exc), False)
