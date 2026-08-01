from __future__ import annotations

import argparse
from pathlib import Path

from wp.v3.contracts import load_v3_config
from wp.v3.history import load_panel_partitions
from wp.v3.io import atomic_write_json
from wp.v3.model import bundle_metadata, save_bundle, train_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen V9 causal base scorer used inside WP V40. "
            "This command does not select or publish candidates."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_v3_config(args.config)
    panel = load_panel_partitions(args.panel_dir)
    bundle = train_bundle(
        panel,
        config,
        model_version="wpv40-base-v9",
    )
    output = Path(args.output_dir)
    model_path = output / "models" / f"{bundle.fingerprint}.joblib"
    save_bundle(bundle, model_path)
    metadata = bundle_metadata(bundle)
    metadata["artifact_path"] = model_path.as_posix()
    atomic_write_json(output / "wp_v40_base_model_metadata.json", metadata)
    print(
        "WP_V40_BASE_MODEL="
        f"{model_path.as_posix()} fingerprint={bundle.fingerprint}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
