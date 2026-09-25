#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyze_warhead_diversity import run_analysis
from config import apply_cli_overrides, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze per-POI generated PROTAC warhead uniqueness, "
            "diversity, and POI-specific novelty."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Optional YAML overrides in key=value form.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cfg = load_yaml(args.config)

    if args.override:
        cfg = apply_cli_overrides(
            cfg,
            args.override,
        )

    run_analysis(
        cfg=cfg,
        project_root=PROJECT_ROOT,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
