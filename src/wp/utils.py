from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal runners
    yaml = None


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_yaml(path: str | Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return default or {}
    with file_path.open("r", encoding="utf-8") as handle:
        if yaml is not None:
            return yaml.safe_load(handle) or {}
        data: dict[str, Any] = {}
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            if value.lower() in {"true", "false"}:
                parsed: Any = value.lower() == "true"
            else:
                try:
                    parsed = float(value) if "." in value else int(value)
                except ValueError:
                    parsed = value
            data[key.strip()] = parsed
        return data or (default or {})


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=file_path.parent,
        prefix=f".{file_path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, file_path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(
            (_json_safe(item) for item in value),
            key=lambda item: json.dumps(
                item,
                ensure_ascii=True,
                sort_keys=True,
                default=str,
            ),
        )
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    normalized = {str(col).lower(): col for col in df.columns}
    for name in names:
        if name in df.columns:
            return name
        key = name.lower()
        if key in normalized:
            return normalized[key]
    return None


def numeric_series(df: pd.DataFrame, names: list[str], default: float = 0.0) -> pd.Series:
    col = first_existing(df, names)
    if col is None:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def text_series(df: pd.DataFrame, names: list[str], default: str = "") -> pd.Series:
    col = first_existing(df, names)
    if col is None:
        return pd.Series(default, index=df.index, dtype="object")
    return df[col].fillna(default).astype(str)


def clip(series: pd.Series, low: float = 0.0, high: float = 100.0) -> pd.Series:
    return pd.Series(pd.to_numeric(series, errors="coerce")).fillna(0).clip(low, high)


def sigmoid_to_percent(value: pd.Series | float, center: float = 50.0, scale: float = 12.0):
    return 100.0 / (1.0 + math.e ** 0) if False else 100.0 / (1.0 + pd.Series(value).rsub(center).div(scale).map(math.exp))


def safe_percent(value: float) -> float:
    if value != value:
        return 0.0
    return max(0.0, min(100.0, float(value)))
