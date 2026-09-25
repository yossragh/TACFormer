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
from protein_embeddings import build_poi_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build mean-pooled ESM-2 POI embeddings from "
            "poi_metadata.csv."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )

    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_cfg = cfg["model"]

    output_pt = Path(cfg["output_pt"])
    stats_json = Path(cfg["stats_json"])

    stats = build_poi_embeddings(
        metadata_csv=cfg["metadata_csv"],
        output_pt=output_pt,
        stats_json=stats_json,
        model_name_or_path=model_cfg["name_or_path"],
        uniprot_field=cfg["uniprot_field"],
        poi_field=cfg["poi_field"],
        sequence_field=cfg["sequence_field"],
        device=model_cfg["device"],
        dtype=model_cfg["dtype"],
        batch_size=int(model_cfg["batch_size"]),
        max_amino_acids_per_chunk=int(
            model_cfg["max_amino_acids_per_chunk"]
        ),
        chunk_long_sequences=bool(
            model_cfg["chunk_long_sequences"]
        ),
        chunk_overlap=int(
            model_cfg["chunk_overlap"]
        ),
        local_files_only=bool(
            model_cfg["local_files_only"]
        ),
        pooling=cfg["pooling"],
    )

    save_yaml(
        cfg,
        output_pt.parent / "build_poi_embeddings.resolved.yaml",
    )

    print(f"[OK] POI embeddings saved to: {output_pt}")
    print(f"[OK] Stats saved to: {stats_json}")

    print(
        f"[OK] num_embeddings={stats['num_embeddings']} "
        f"embedding_dim={stats['embedding_dim']} "
        f"dtype={stats['dtype']}"
    )

    print(
        "[OK] Sequence lengths: "
        f"min={stats['min_sequence_length']} "
        f"median={stats['median_sequence_length']} "
        f"max={stats['max_sequence_length']}"
    )

    print(
        "[OK] Long-sequence handling: "
        f"chunked={stats['num_chunked_sequences']} "
        f"max_chunks={stats['max_num_chunks']} "
        f"chunk_size={stats['max_amino_acids_per_chunk']} "
        f"overlap={stats['chunk_overlap']} "
        f"stride={stats['chunk_stride']}"
    )


if __name__ == "__main__":
    main()