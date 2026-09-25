#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_yaml, save_yaml
from protein_metadata import build_poi_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build UniProt metadata and sequences for conditional POIs."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )

    args = parser.parse_args()

    cfg = load_yaml(args.config)
    uniprot_cfg = cfg["uniprot"]

    output_csv = Path(cfg["output_csv"])
    stats_json = Path(cfg["stats_json"])

    stats = build_poi_metadata(
        input_jsonls=cfg["input_jsonls"],
        output_csv=output_csv,
        stats_json=stats_json,
        target_uniprot_field=cfg["target_uniprot_field"],
        poi_field=cfg["poi_field"],
        base_url=uniprot_cfg["base_url"],
        timeout_seconds=uniprot_cfg["timeout_seconds"],
        max_retries=uniprot_cfg["max_retries"],
        sleep_seconds=uniprot_cfg["sleep_seconds"],
        fail_on_missing_sequence=cfg["fail_on_missing_sequence"],
    )

    save_yaml(
        cfg,
        output_csv.parent / "build_poi_metadata.resolved.yaml",
    )

    print(f"[OK] POI metadata saved to: {output_csv}")
    print(f"[OK] Stats saved to: {stats_json}")
    print(
        f"[OK] UniProt IDs={stats['num_unique_uniprots']} "
        f"POIs={stats['num_unique_pois']} "
        f"resolved={stats['num_ok']} "
        f"not_found={stats['num_not_found']} "
        f"missing_sequences={stats['num_missing_sequence']}"
    )
    print(
        "[OK] Sequence lengths: "
        f"min={stats['min_sequence_length']} "
        f"median={stats['median_sequence_length']} "
        f"max={stats['max_sequence_length']}"
    )


if __name__ == "__main__":
    main()