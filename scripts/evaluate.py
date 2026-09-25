#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemistry import (
    canonicalize_from_mol,
    compute_mol_properties,
    internal_diversity_from_mols,
    mol_from_smiles,
    murcko_scaffold_from_mol,
    suppress_rdkit_warnings,
)
from config import apply_cli_overrides, load_yaml, save_yaml
from utils import ensure_dir, read_jsonl, write_json, write_jsonl


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_reference_paths(eval_cfg: dict[str, Any]) -> list[str]:

    if eval_cfg.get("reference_jsonls"):
        return [
            str(path)
            for path in eval_cfg["reference_jsonls"]
            if path is not None and str(path).strip()
        ]

    paths: list[str] = []

    if eval_cfg.get("reference_train_jsonl"):
        paths.append(str(eval_cfg["reference_train_jsonl"]))

    if eval_cfg.get("reference_val_jsonl"):
        paths.append(str(eval_cfg["reference_val_jsonl"]))

    return paths


def load_reference_sets(
    paths: list[str],
    smiles_field: str,
) -> tuple[set[str], set[str]]:
    """
    Load reference molecules and reference scaffolds.

    Used for:
    - exact novelty
    - exact reference matching
    - scaffold overlap

    Parse-once logic:
        reference SMILES -> Mol -> canonical SMILES + Murcko scaffold
    """
    reference_set: set[str] = set()
    reference_scaffold_set: set[str] = set()

    for path in paths:
        if path is None or str(path).strip() == "":
            continue

        records = read_jsonl(path)

        for rec in records:
            if smiles_field not in rec:
                raise KeyError(
                    f"Reference file {path} does not contain field '{smiles_field}'. "
                    f"Available keys: {list(rec.keys())}"
                )

            mol = mol_from_smiles(rec[smiles_field])
            if mol is None:
                continue

            canonical = canonicalize_from_mol(mol)
            if canonical is not None:
                reference_set.add(canonical)

            scaffold = murcko_scaffold_from_mol(mol)
            if scaffold is not None:
                reference_scaffold_set.add(scaffold)

    return reference_set, reference_scaffold_set


def summarize_properties(
    annotated_records: list[dict[str, Any]],
) -> dict[str, dict[str, float]] | None:
    """
    Summarize molecular properties for unique valid molecules only.

    We use only first-occurrence valid molecules so duplicated generated molecules
    do not bias the property summary.
    """
    property_rows = [
        rec["properties"]
        for rec in annotated_records
        if rec.get("is_valid")
        and rec.get("is_first_occurrence_valid")
        and isinstance(rec.get("properties"), dict)
    ]

    if not property_rows:
        return None

    keys = sorted({key for row in property_rows for key in row.keys()})
    summary: dict[str, dict[str, float]] = {}

    for key in keys:
        values = [
            row[key]
            for row in property_rows
            if key in row and row[key] is not None
        ]

        if not values:
            continue

        arr = np.asarray(values, dtype=float)

        summary[key] = {
            "mean": float(np.mean(arr)),
            "std":    float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "p10":    float(np.percentile(arr, 10)),
            "p25":    float(np.percentile(arr, 25)),
            "p75":    float(np.percentile(arr, 75)),
            "p90":    float(np.percentile(arr, 90)),
            "max": float(np.max(arr)),
        }

    return summary


def write_smi(path: str | Path, smiles_list: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")


def annotate_generated_records(
    generated_records: list[dict[str, Any]],
    generated_smiles_field: str,
    reference_set: set[str],
    reference_scaffold_set: set[str],
) -> tuple[list[dict[str, Any]], list[str], list[Any]]:
    """
    Annotate every generated record.
    for each record, we add fields for:

        raw_smiles -> mol
        mol -> canonical SMILES
        mol -> Murcko scaffold
        mol -> molecular properties

    Returns:
        annotated_records
        valid_canonical_all
        unique_valid_mols_ordered
    """
    if len(generated_records) == 0:
        raise ValueError("Generated JSONL is empty. Nothing to evaluate.")

    temp_records: list[dict[str, Any]] = []
    valid_canonical_all: list[str] = []

    # Store one Mol per unique valid canonical SMILES.
    unique_mol_by_canonical: dict[str, Any] = {}

    iterator = tqdm(
        generated_records,
        desc="Annotating generated SMILES",
        unit="mol",
    )

    for rec in iterator:
        if generated_smiles_field not in rec:
            raise KeyError(
                f"Generated record does not contain field '{generated_smiles_field}'. "
                f"Available keys: {list(rec.keys())}"
            )

        raw_smiles = rec[generated_smiles_field]
        mol = mol_from_smiles(raw_smiles)

        if mol is None:
            canonical_smiles = None
            is_valid = False
            scaffold = None
            properties = None
        else:
            canonical_smiles = canonicalize_from_mol(mol)
            is_valid = canonical_smiles is not None

            if is_valid:
                scaffold = murcko_scaffold_from_mol(mol)
                properties = compute_mol_properties(mol)

                valid_canonical_all.append(canonical_smiles)

                if canonical_smiles not in unique_mol_by_canonical:
                    unique_mol_by_canonical[canonical_smiles] = mol
            else:
                scaffold = None
                properties = None

        temp_records.append(
            {
                **rec,
                "canonical_smiles": canonical_smiles,
                "is_valid": is_valid,
                "murcko_scaffold": scaffold,
                "properties": properties,
            }
        )

    canonical_counts = Counter(valid_canonical_all)
    seen_valid: set[str] = set()
    annotated_records: list[dict[str, Any]] = []

    for rec in temp_records:
        canonical_smiles = rec["canonical_smiles"]
        is_valid = rec["is_valid"]
        scaffold = rec["murcko_scaffold"]

        if is_valid:
            canonical_count = canonical_counts[canonical_smiles]

            is_first_occurrence = canonical_smiles not in seen_valid
            seen_valid.add(canonical_smiles)

            is_duplicate_generated = canonical_count > 1
            is_reference_match = canonical_smiles in reference_set
            is_novel = canonical_smiles not in reference_set

            scaffold_in_reference = (
                scaffold is not None
                and scaffold in reference_scaffold_set
            )
        else:
            canonical_count = 0
            is_first_occurrence = False
            is_duplicate_generated = False
            is_reference_match = False
            is_novel = False
            scaffold_in_reference = False

        annotated_records.append(
            {
                **rec,
                "canonical_count_in_generated": canonical_count,
                "is_first_occurrence_valid": is_first_occurrence,
                "is_duplicate_generated": is_duplicate_generated,
                "is_reference_match": is_reference_match,
                "is_novel": is_novel,
                "scaffold_in_reference": scaffold_in_reference,
            }
        )

    unique_valid_ordered = list(dict.fromkeys(valid_canonical_all))
    unique_valid_mols_ordered = [
        unique_mol_by_canonical[smiles]
        for smiles in unique_valid_ordered
    ]

    return annotated_records, valid_canonical_all, unique_valid_mols_ordered


def compute_metrics(
    annotated_records: list[dict[str, Any]],
    valid_canonical_all: list[str],
    unique_valid_mols_ordered: list[Any],
    reference_set: set[str],
    reference_scaffold_set: set[str],
    fp_radius: int,
    fp_n_bits: int,
    max_diversity_samples: int,
    seed: int,
) -> dict[str, Any]:
    num_generated = len(annotated_records)
    num_valid = len(valid_canonical_all)
    num_invalid = num_generated - num_valid

    unique_valid_ordered = list(dict.fromkeys(valid_canonical_all))
    unique_valid_set = set(unique_valid_ordered)

    duplicate_valid_count = num_valid - len(unique_valid_set)

    unique_novel_valid = [
        smiles for smiles in unique_valid_ordered if smiles not in reference_set
    ]

    valid_reference_matches_all = [
        smiles for smiles in valid_canonical_all if smiles in reference_set
    ]

    unique_reference_matches = {
        smiles for smiles in unique_valid_set if smiles in reference_set
    }

    valid_scaffolds = [
        rec["murcko_scaffold"]
        for rec in annotated_records
        if rec["is_valid"] and rec["murcko_scaffold"] is not None
    ]

    unique_valid_scaffolds = list(dict.fromkeys(valid_scaffolds))

    unique_reference_matching_scaffolds = {
        scaffold
        for scaffold in unique_valid_scaffolds
        if scaffold in reference_scaffold_set
    }

    unique_novel_scaffolds = [
        scaffold
        for scaffold in unique_valid_scaffolds
        if scaffold not in reference_scaffold_set
    ]

    int_div = internal_diversity_from_mols(
        mols=unique_valid_mols_ordered,
        radius=fp_radius,
        n_bits=fp_n_bits,
        max_samples=max_diversity_samples,
        seed=seed,
    )

    property_summary = summarize_properties(annotated_records)

    metrics = {
    # Total number of raw SMILES strings generated by the model.
    "num_generated": num_generated,

    # Number of generated SMILES that RDKit successfully parsed.
    "num_valid": num_valid,

    # Number of generated SMILES that RDKit could not parse.
    "num_invalid": num_invalid,

    # Fraction of generated SMILES that are valid.
    # Formula: num_valid / num_generated
    "validity": num_valid / max(1, num_generated),


    # Number of unique valid molecules after RDKit canonicalization.
    # This removes duplicate molecules written with different SMILES strings.
    "num_unique_valid": len(unique_valid_set),

    # Fraction of valid molecules that are unique.
    # Formula: num_unique_valid / num_valid
    "uniqueness_among_valid": len(unique_valid_set) / max(1, num_valid),

    # Fraction of all generated strings that are both valid and unique.
    # Formula: num_unique_valid / num_generated
    "unique_valid_fraction_of_generated": len(unique_valid_set) / max(1, num_generated),


    # Number of repeated valid molecules after canonicalization.
    # Example: if the same valid molecule appears 3 times, it contributes 2 duplicates.
    "duplicate_valid_count": duplicate_valid_count,

    # Fraction of valid molecules that are duplicates.
    # Formula: duplicate_valid_count / num_valid
    "duplicate_rate_among_valid": duplicate_valid_count / max(1, num_valid),


    # Number of valid generated molecules that exactly match the reference set.
    # This counts duplicates if the same reference molecule is generated multiple times.
    "num_reference_matches_valid": len(valid_reference_matches_all),

    # Fraction of valid generated molecules that exactly match the reference set.
    # Formula: num_reference_matches_valid / num_valid
    "reference_match_rate_among_valid": len(valid_reference_matches_all) / max(1, num_valid),

    # Number of unique valid generated molecules that exactly match the reference set.
    "num_reference_matches_unique_valid": len(unique_reference_matches),

    # Fraction of unique valid generated molecules that exactly match the reference set.
    # Formula: num_reference_matches_unique_valid / num_unique_valid
    "reference_match_rate_among_unique_valid": len(unique_reference_matches) / max(1, len(unique_valid_set)),


    # Number of unique valid generated molecules not found in the reference set.
    # This is exact canonical-SMILES novelty.
    "num_novel_unique_valid": len(unique_novel_valid),

    # Fraction of unique valid generated molecules that are novel.
    # Formula: num_novel_unique_valid / num_unique_valid
    "novelty_among_unique_valid": len(unique_novel_valid) / max(1, len(unique_valid_set)),


    # Number of Murcko scaffolds extracted from valid generated molecules.
    # This counts duplicates.
    "num_valid_scaffolds": len(valid_scaffolds),

    # Number of unique Murcko scaffolds among valid generated molecules.
    "num_unique_valid_scaffolds": len(unique_valid_scaffolds),

    # Number of unique generated scaffolds that already exist in the reference scaffold set.
    "num_reference_matching_unique_scaffolds": len(unique_reference_matching_scaffolds),

    # Fraction of unique generated scaffolds already present in the reference scaffold set.
    # Formula: num_reference_matching_unique_scaffolds / num_unique_valid_scaffolds
    "scaffold_reference_overlap_rate": len(unique_reference_matching_scaffolds) / max(1, len(unique_valid_scaffolds)),

    # Number of unique generated scaffolds not found in the reference scaffold set.
    "num_novel_unique_scaffolds": len(unique_novel_scaffolds),

    # Fraction of unique generated scaffolds that are novel.
    # Formula: num_novel_unique_scaffolds / num_unique_valid_scaffolds
    "scaffold_novelty_rate": len(unique_novel_scaffolds) / max(1, len(unique_valid_scaffolds)),


    # Mean pairwise Tanimoto distance among unique valid generated molecules.
    # Higher value means generated molecules are more chemically diverse.
    "internal_diversity_unique_valid": int_div,


    # Number of unique canonical molecules in the reference set.
    # Usually this is the training set size after canonicalization.
    "reference_set_size": len(reference_set),

    # Number of unique Murcko scaffolds in the reference set.
    "reference_scaffold_set_size": len(reference_scaffold_set),


    # Fingerprint settings used for internal diversity calculation.
    "fingerprint": {
        "type": "Morgan",
        "radius": fp_radius,
        "n_bits": fp_n_bits,
        "diversity_max_samples": max_diversity_samples,
    },


    # Summary statistics of molecular properties for unique valid generated molecules.
    # Example properties: MW, QED, TPSA, LogP, rotatable bonds, rings, SA score if available.
    "property_summary_unique_valid": property_summary,
}

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated unconditional PROTAC SMILES."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--override", nargs="*", default=[], help="Optional key=value overrides.")
    args = parser.parse_args()

    cfg = apply_cli_overrides(load_yaml(args.config), args.override)
    eval_cfg = cfg["evaluation"]

    output_dir = ensure_dir(eval_cfg["output_dir"])
    save_yaml(cfg, output_dir / "evaluate.resolved.yaml")

    seed = eval_cfg.get("seed", 42)
    set_seed(seed)

    if eval_cfg.get("suppress_rdkit_warnings", True):
        suppress_rdkit_warnings()

    generated_jsonl = eval_cfg["generated_jsonl"]
    generated_smiles_field = eval_cfg.get("generated_smiles_field", "raw_smiles")

    reference_paths = get_reference_paths(eval_cfg)
    if len(reference_paths) == 0:
        raise ValueError(
            "No reference files provided. Add evaluation.reference_jsonls "
            "or evaluation.reference_train_jsonl to the YAML."
        )

    reference_smiles_field = eval_cfg.get("reference_smiles_field", "smiles")

    fp_radius = eval_cfg.get("fp_radius", 2)
    fp_n_bits = eval_cfg.get("fp_n_bits", 2048)
    max_diversity_samples = eval_cfg.get("max_diversity_samples", 1000)
    max_invalid_examples = eval_cfg.get("max_invalid_examples", 50)

    print(f"[INFO] Generated JSONL: {generated_jsonl}")
    print(f"[INFO] Generated SMILES field: {generated_smiles_field}")
    print(f"[INFO] Reference files: {reference_paths}")
    print(f"[INFO] Reference SMILES field: {reference_smiles_field}")
    print(f"[INFO] Output dir: {output_dir}")

    generated_records = read_jsonl(generated_jsonl)

    reference_set, reference_scaffold_set = load_reference_sets(
        paths=reference_paths,
        smiles_field=reference_smiles_field,
    )

    annotated_records, valid_canonical_all, unique_valid_mols_ordered = annotate_generated_records(
        generated_records=generated_records,
        generated_smiles_field=generated_smiles_field,
        reference_set=reference_set,
        reference_scaffold_set=reference_scaffold_set,
    )

    metrics = compute_metrics(
        annotated_records=annotated_records,
        valid_canonical_all=valid_canonical_all,
        unique_valid_mols_ordered=unique_valid_mols_ordered,
        reference_set=reference_set,
        reference_scaffold_set=reference_scaffold_set,
        fp_radius=fp_radius,
        fp_n_bits=fp_n_bits,
        max_diversity_samples=max_diversity_samples,
        seed=seed,
    )

    unique_valid_ordered = list(dict.fromkeys(valid_canonical_all))

    unique_novel_valid_ordered = [
        smiles for smiles in unique_valid_ordered if smiles not in reference_set
    ]

    invalid_examples = [
        rec for rec in annotated_records if not rec["is_valid"]
    ][:max_invalid_examples]

    write_json(metrics, output_dir / "metrics.json")
    write_jsonl(annotated_records, output_dir / "annotated_generated.jsonl")

    write_smi(output_dir / "valid_all.smi", valid_canonical_all)
    write_smi(output_dir / "valid_unique.smi", unique_valid_ordered)
    write_smi(output_dir / "valid_unique_novel.smi", unique_novel_valid_ordered)
    write_jsonl(invalid_examples, output_dir / "invalid_examples.jsonl")

    print("[DONE] Evaluation finished.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()