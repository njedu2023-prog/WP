from __future__ import annotations

from .v3.app import run_v3


def run() -> dict:
    """Run the only production strategy engine."""
    return run_v3()


if __name__ == "__main__":
    run()
