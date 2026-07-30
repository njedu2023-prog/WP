from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from wp.v3.io import atomic_write_csv, atomic_write_json
from wp.v3.v16_funnel import build_funnel, rejection_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute every immutable V15 scored-frontier row to the first "
            "fixed gate that prevented it from becoming a candidate."
        )
    )
    parser.add_argument("--v15-source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.v15_source_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frontier_path = _one(
        source,
        "wp_v15_forward_scored_meta_frontier.parquet",
    )
    summary_path = _one(
        source,
        "wp_v15_forward_risk_validation_summary.json",
    )
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = str(source_summary.get("scored_meta_frontier_sha256") or "")
    actual = digest(frontier_path)
    if not expected or expected != actual:
        raise RuntimeError(
            f"V15 scored-frontier digest mismatch: {actual} != {expected}"
        )
    frontier = pd.read_parquet(frontier_path)
    funnel, attributed = build_funnel(
        frontier,
        max_candidates_per_day=3,
    )
    rejections = rejection_summary(attributed)
    selected = attributed.loc[attributed["funnel_selected"]].copy()
    summary = {
        "schema_version": "wp_v16_candidate_funnel_1",
        "research_only": True,
        "production_model_changed": False,
        "v15_frozen_candidate_changed": False,
        "source": {
            "v15_schema_version": source_summary.get("schema_version"),
            "v15_scored_frontier_sha256": actual,
            "rows": int(len(frontier)),
            "trade_days": int(frontier["trade_date"].nunique()),
        },
        "selected": {
            "rows": int(len(selected)),
            "trade_days": int(selected["trade_date"].nunique()),
            "symbols": int(selected["ts_code"].nunique()),
        },
        "funnel": funnel.to_dict(orient="records"),
        "rejections": rejections.to_dict(orient="records"),
    }
    atomic_write_csv(funnel, output / "wp_v16_candidate_funnel.csv")
    atomic_write_csv(rejections, output / "wp_v16_candidate_rejections.csv")
    atomic_write_csv(
        attributed.loc[
            :,
            [
                column
                for column in (
                    "trade_date",
                    "signal_slot",
                    "ts_code",
                    "name",
                    "fold",
                    "first_rejection_stage",
                    "funnel_selected",
                    "meta_p_positive",
                    "meta_expected_net_return_pct",
                    "meta_p_severe_loss",
                    "p_round_trip_fill_lower",
                    "meta_rank_pct",
                    "risk_failure_rank_pct",
                    "net_return_pct",
                )
                if column in attributed
            ],
        ],
        output / "wp_v16_candidate_attribution.csv",
    )
    atomic_write_json(
        output / "wp_v16_candidate_funnel_summary.json",
        summary,
    )
    print(
        "WP_V16_FUNNEL_RESULT="
        + json.dumps(
            summary,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def _one(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {name} under {root}, found {len(matches)}"
        )
    return matches[0]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
