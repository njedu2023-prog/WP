from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wp.v3.alpha_audit import audit_oos_predictions
from wp.v3.sharding import SHARD_PREDICTIONS_NAME


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit immutable WP V4 chronological OOS predictions."
    )
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "artifacts"
            / "wp_v4_alpha_audit"
            / "wp_v4_oos_alpha_audit.json"
        ),
    )
    parser.add_argument("--lockbox-days", type=int, default=150)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(Path(args.shard_dir).rglob(SHARD_PREDICTIONS_NAME))
    if not paths:
        raise FileNotFoundError(
            f"no {SHARD_PREDICTIONS_NAME} files under {args.shard_dir}"
        )
    predictions = pd.concat(
        [pd.read_parquet(path) for path in paths],
        ignore_index=True,
    )
    audit = audit_oos_predictions(
        predictions,
        lockbox_days=args.lockbox_days,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        audit,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    target.write_text(encoded + "\n", encoding="utf-8")
    print("WP_ALPHA_AUDIT_JSON=" + json.dumps(audit, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
