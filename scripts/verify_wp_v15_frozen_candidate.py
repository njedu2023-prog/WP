from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from wp.v3.contracts import load_v3_config
from wp.v3.frozen_candidate import (
    load_frozen_candidate,
    verify_runtime_contract,
)
from wp.v3.io import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the immutable WP V15 frozen candidate record."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_frozen_candidate(args.manifest)
    config = load_v3_config(args.config)
    runtime = verify_runtime_contract(manifest, config)
    result = {
        "schema_version": "wp_v15_freeze_verification_1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "candidate_id": manifest["candidate_id"],
            "status": manifest["status"],
            "freeze_manifest_sha256": manifest["integrity"][
                "freeze_manifest_sha256"
            ],
            "production_authorized": manifest["production_authorized"],
            "shadow_status": manifest["shadow_contract"]["status"],
        },
        "runtime": runtime,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output, result)
    print(
        "WP_V15_FREEZE_VERIFICATION_RESULT="
        + json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
