#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import apply_cli_overrides, load_yaml, save_yaml
from tokenizer import SmilesTokenizer
from utils import ensure_dir, read_jsonl, write_json


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0

    values = sorted(values)
    idx = int(round((len(values) - 1) * p))
    idx = max(0, min(idx, len(values) - 1))

    return int(values[idx])


def length_stats(values: list[int]) -> dict[str, int]:
    return {
        "min": min(values) if values else 0,
        "median": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else 0,
    }


def check_dataset(
    tokenizer: SmilesTokenizer,
    dataset_cfg: dict[str, Any],
    max_seq_len: int,
) -> dict[str, Any]:
    path = Path(dataset_cfg["path"])
    smiles_field = dataset_cfg.get("smiles_field", "smiles")

    records = read_jsonl(path)

    num_smiles = 0
    num_missing_smiles = 0
    num_tokenization_errors = 0

    num_tokens = 0
    num_unknown_tokens = 0
    num_smiles_with_unknowns = 0

    token_lengths: list[int] = []
    encoded_lengths: list[int] = []

    unknown_examples: list[dict[str, Any]] = []
    error_examples: list[dict[str, Any]] = []
    long_examples: list[dict[str, Any]] = []

    for row_index, rec in enumerate(records):
        smiles = rec.get(smiles_field)

        if not isinstance(smiles, str) or not smiles.strip():
            num_missing_smiles += 1
            continue

        num_smiles += 1

        try:
            tokens = tokenizer.tokenize(smiles)
        except ValueError as exc:
            num_tokenization_errors += 1

            if len(error_examples) < 20:
                error_examples.append(
                    {
                        "row_index": row_index,
                        "smiles": smiles,
                        "error": str(exc),
                    }
                )

            continue

        unknown_tokens = [
            tok for tok in tokens
            if tok not in tokenizer.token_to_id
        ]

        if unknown_tokens:
            num_smiles_with_unknowns += 1
            num_unknown_tokens += len(unknown_tokens)

            if len(unknown_examples) < 20:
                unknown_examples.append(
                    {
                        "row_index": row_index,
                        "smiles": smiles,
                        "unknown_tokens": unknown_tokens,
                    }
                )

        token_len = len(tokens)
        encoded_len = len(tokenizer.encode(smiles, add_bos=True, add_eos=True))

        token_lengths.append(token_len)
        encoded_lengths.append(encoded_len)
        num_tokens += token_len

        if encoded_len > max_seq_len and len(long_examples) < 20:
            long_examples.append(
                {
                    "row_index": row_index,
                    "smiles": smiles,
                    "encoded_len": encoded_len,
                    "max_seq_len": max_seq_len,
                }
            )

    num_over_max_seq_len = sum(1 for x in encoded_lengths if x > max_seq_len)

    return {
        "name": dataset_cfg["name"],
        "path": str(path),
        "smiles_field": smiles_field,

        "num_records": len(records),
        "num_smiles": num_smiles,
        "num_missing_smiles": num_missing_smiles,
        "num_tokenization_errors": num_tokenization_errors,

        "num_tokens": num_tokens,
        "num_unknown_tokens": num_unknown_tokens,
        "num_smiles_with_unknowns": num_smiles_with_unknowns,

        "unknown_token_rate": (
            num_unknown_tokens / num_tokens
            if num_tokens > 0
            else 0.0
        ),
        "smiles_with_unknown_rate": (
            num_smiles_with_unknowns / num_smiles
            if num_smiles > 0
            else 0.0
        ),
        "tokenization_error_rate": (
            num_tokenization_errors / num_smiles
            if num_smiles > 0
            else 0.0
        ),

        "token_length": length_stats(token_lengths),
        "encoded_length_with_bos_eos": length_stats(encoded_lengths),

        "max_seq_len": max_seq_len,
        "num_over_max_seq_len": num_over_max_seq_len,
        "over_max_seq_len_rate": (
            num_over_max_seq_len / num_smiles
            if num_smiles > 0
            else 0.0
        ),

        "unknown_examples": unknown_examples,
        "tokenization_error_examples": error_examples,
        "long_sequence_examples": long_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check existing tokenizer coverage on JSONL SMILES datasets."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = apply_cli_overrides(load_yaml(args.config), args.override)
    check_cfg = cfg["tokenizer_coverage_check"]

    output_dir = ensure_dir(check_cfg["output_dir"])
    save_yaml(cfg, output_dir / "check_tokenizer_coverage.resolved.yaml")

    tokenizer_path = check_cfg["tokenizer_path"]
    max_seq_len = int(check_cfg.get("max_seq_len", 512))

    tokenizer = SmilesTokenizer.load(tokenizer_path)

    print(f"[INFO] tokenizer_path: {tokenizer_path}")
    print(f"[INFO] vocab_size: {tokenizer.vocab_size}")
    print(f"[INFO] max_seq_len: {max_seq_len}")

    dataset_reports = []

    for dataset_cfg in check_cfg["datasets"]:
        print(f"[INFO] Checking {dataset_cfg['name']}: {dataset_cfg['path']}")

        report = check_dataset(
            tokenizer=tokenizer,
            dataset_cfg=dataset_cfg,
            max_seq_len=max_seq_len,
        )

        dataset_reports.append(report)

        print(
            f"[INFO] {dataset_cfg['name']} "
            f"unknown_token_rate={report['unknown_token_rate']:.6f} "
            f"smiles_with_unknown_rate={report['smiles_with_unknown_rate']:.6f} "
            f"tokenization_error_rate={report['tokenization_error_rate']:.6f} "
            f"p95_encoded_len={report['encoded_length_with_bos_eos']['p95']} "
            f"max_encoded_len={report['encoded_length_with_bos_eos']['max']} "
            f"over_max_seq_len={report['num_over_max_seq_len']}"
        )

    global_num_smiles = sum(r["num_smiles"] for r in dataset_reports)
    global_num_tokens = sum(r["num_tokens"] for r in dataset_reports)
    global_num_unknown_tokens = sum(r["num_unknown_tokens"] for r in dataset_reports)
    global_num_smiles_with_unknowns = sum(r["num_smiles_with_unknowns"] for r in dataset_reports)
    global_num_tokenization_errors = sum(r["num_tokenization_errors"] for r in dataset_reports)
    global_num_over_max_seq_len = sum(r["num_over_max_seq_len"] for r in dataset_reports)

    keep_current_tokenizer = (
        global_num_unknown_tokens == 0
        and global_num_tokenization_errors == 0
    )

    final_report = {
        "tokenizer_path": tokenizer_path,
        "vocab_size": tokenizer.vocab_size,
        "max_seq_len": max_seq_len,
        "global_summary": {
            "num_smiles": global_num_smiles,
            "num_tokens": global_num_tokens,
            "num_unknown_tokens": global_num_unknown_tokens,
            "num_smiles_with_unknowns": global_num_smiles_with_unknowns,
            "num_tokenization_errors": global_num_tokenization_errors,
            "num_over_max_seq_len": global_num_over_max_seq_len,
            "unknown_token_rate": (
                global_num_unknown_tokens / global_num_tokens
                if global_num_tokens > 0
                else 0.0
            ),
            "smiles_with_unknown_rate": (
                global_num_smiles_with_unknowns / global_num_smiles
                if global_num_smiles > 0
                else 0.0
            ),
            "tokenization_error_rate": (
                global_num_tokenization_errors / global_num_smiles
                if global_num_smiles > 0
                else 0.0
            ),
            "over_max_seq_len_rate": (
                global_num_over_max_seq_len / global_num_smiles
                if global_num_smiles > 0
                else 0.0
            ),
        },
        "decision": {
            "keep_current_tokenizer": keep_current_tokenizer,
            "reason": (
                "No unknown tokens and no tokenization errors found."
                if keep_current_tokenizer
                else "Unknown tokens and/or tokenization errors were found."
            ),
        },
        "datasets": dataset_reports,
    }

    write_json(final_report, output_dir / "tokenizer_coverage_report.json")

    print("[DONE] Tokenizer coverage check finished.")
    print(f"[INFO] Report: {output_dir / 'tokenizer_coverage_report.json'}")
    print(f"[INFO] keep_current_tokenizer={keep_current_tokenizer}")


if __name__ == "__main__":
    main()