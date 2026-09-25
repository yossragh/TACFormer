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
from prepare import prepare_dataset
from utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a cleaned PROTAC CSV into train/val/test JSONL files."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )

    args = parser.parse_args()

    cfg = load_yaml(args.config)
    split_cfg = cfg["split"]

    output_dir = Path(cfg["output_dir"])

    stats = prepare_dataset(
        input_csv=cfg["input_csv"],
        output_dir=output_dir,
        smiles_column=cfg["smiles_column"],
        keep_columns=cfg.get("keep_columns", []),
        split_method=split_cfg["method"],
        seed=split_cfg["seed"],
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
        poi_column=split_cfg.get("poi_column"),
        top_n_eval_pois=split_cfg.get("top_n_eval_pois"),
    )

    write_json(
        stats,
        output_dir / "stats.json",
    )

    save_yaml(
        cfg,
        output_dir / "prepare_data.resolved.yaml",
    )

    print(f"[OK] Output dir: {output_dir}")
    print(
        f"[OK] Split '{stats['split_method']}': "
        f"train={stats['train_records']} "
        f"val={stats['val_records']} "
        f"test={stats['test_records']}"
    )

    conditional_stats = stats.get("conditional_split")

    if conditional_stats:
        print(
            f"[OK] Eval POIs: {conditional_stats['top_n_eval_pois']} "
            f"({conditional_stats['num_eval_poi_records']} records), "
            f"{conditional_stats['num_train_only_pois']} train-only POIs "
            f"({conditional_stats['num_train_only_poi_records']} records)"
        )

    print(
        f"[OK] Stats: "
        f"{output_dir / 'stats.json'}"
    )


if __name__ == "__main__":
    main()