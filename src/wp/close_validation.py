from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .v3.truth import run_v3_close_validation


def run_close_validation(
    output_root: Path | None = None,
    current: datetime | None = None,
) -> dict:
    """Validate immutable V3 candidate truth using the fixed T+1 contract."""
    return run_v3_close_validation(output_root=output_root, current=current)


if __name__ == "__main__":
    run_close_validation()
