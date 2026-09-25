#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import apply_cli_overrides, load_yaml, save_yaml
from utils import ensure_dir, read_jsonl, write_json
from tokenizer import SmilesTokenizer


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    idx = int(round((len(values) - 1) * p))
    idx = max(0, min(idx, len(values) - 1))
    return int(values[idx])


def split_unknown_stats(tokenizer: SmilesTokenizer, records: list[dict], smiles_field: str) -> dict:
    total_smiles = len(records)
    total_tokens = 0
    total_unk = 0
    smiles_with_unk = 0

    for rec in records:
        smiles = rec[smiles_field]
        toks = tokenizer.tokenize(smiles)
        unk_count = sum(1 for tok in toks if tok not in tokenizer.token_to_id)
        total_tokens += len(toks)
        total_unk += unk_count
        if unk_count > 0:
            smiles_with_unk += 1

    return {
        "num_smiles": total_smiles,
        "num_tokens": total_tokens,
        "num_unknown_tokens": total_unk,
        "num_smiles_with_unknowns": smiles_with_unk,
        "unknown_token_rate": (total_unk / total_tokens) if total_tokens > 0 else 0.0,
        "smiles_with_unknown_rate": (smiles_with_unk / total_smiles) if total_smiles > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a regex-based SMILES tokenizer from the train split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = apply_cli_overrides(load_yaml(args.config), args.override)
    output_dir = ensure_dir(cfg["output_dir"])
    smiles_field = cfg.get("smiles_field", "smiles")

    train_records = read_jsonl(cfg["train_jsonl"])
    train_smiles = [r[smiles_field] for r in train_records]

    tokenizer = SmilesTokenizer.build_from_smiles(
        train_smiles,
        min_freq=cfg.get("min_freq", 1),
    )
    tokenizer.save(output_dir / "tokenizer.json")

    token_lengths = [len(tokenizer.tokenize(s)) for s in train_smiles]
    token_counter = Counter()
    for s in train_smiles:
        token_counter.update(tokenizer.tokenize(s))

    stats = {
        "train_jsonl": cfg["train_jsonl"],
        "vocab_size": tokenizer.vocab_size,
        "min_freq": cfg.get("min_freq", 1),
        "num_train_smiles": len(train_smiles),
        "token_length": {
            "min": min(token_lengths) if token_lengths else 0,
            "median": percentile(token_lengths, 0.50),
            "p90": percentile(token_lengths, 0.90),
            "p95": percentile(token_lengths, 0.95),
            "p99": percentile(token_lengths, 0.99),
            "max": max(token_lengths) if token_lengths else 0,
        },
        "top_tokens": token_counter.most_common(100),
        "coverage": {
            "train": split_unknown_stats(tokenizer, train_records, smiles_field),
        },
    }

    if cfg.get("val_jsonl"):
        val_records = read_jsonl(cfg["val_jsonl"])
        stats["coverage"]["val"] = split_unknown_stats(tokenizer, val_records, smiles_field)

    if cfg.get("test_jsonl"):
        test_records = read_jsonl(cfg["test_jsonl"])
        stats["coverage"]["test"] = split_unknown_stats(tokenizer, test_records, smiles_field)

    write_json(stats, output_dir / "tokenizer_stats.json")
    save_yaml(cfg, output_dir / "build_tokenizer.resolved.yaml")

    print(f"[OK] tokenizer saved to: {output_dir / 'tokenizer.json'}")
    print(f"[OK] tokenizer stats saved to: {output_dir / 'tokenizer_stats.json'}")
    print(f"[OK] vocab_size={stats['vocab_size']}")
    print(f"[OK] token_length={stats['token_length']}")


if __name__ == "__main__":
    main()