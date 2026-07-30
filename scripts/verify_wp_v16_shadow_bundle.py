from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from wp.v3.io import file_sha256
from wp.v3.v16_research import EXIT_CONTRACT_ID, SCHEMA_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the integrity and non-production status of V16."
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_path = Path(args.bundle)
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payload: dict[str, Any] = joblib.load(bundle_path)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert summary["schema_version"] == SCHEMA_VERSION
    assert payload["research_only"] is True
    assert payload["production_authorized"] is False
    assert summary["research_only"] is True
    assert summary["production_authorized"] is False
    assert payload["exit_contract_id"] == EXIT_CONTRACT_ID
    assert len(payload["specialists"]) >= 4
    assert all(
        bundle.feature_columns and bundle.fit_rows > 0
        for bundle in payload["specialists"]
    )
    expected = (
        summary["artifacts"]["shadow_model"]["sha256"]
    )
    actual = file_sha256(bundle_path)
    assert actual == expected, f"bundle digest mismatch: {actual} != {expected}"
    print(
        json.dumps(
            {
                "verified": True,
                "schema_version": SCHEMA_VERSION,
                "sha256": actual,
                "specialists": len(payload["specialists"]),
                "policy_id": (
                    payload["candidate_policy"].policy_id
                    if payload.get("candidate_policy") is not None
                    else None
                ),
                "production_authorized": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
