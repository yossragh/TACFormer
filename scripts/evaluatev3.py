#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  #new
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemistry import (
    canonicalize_from_mol,
    compute_mol_properties,
    get_morgan_fingerprints_from_mols,
    internal_diversity_from_mols,
    mol_from_smiles,
    murcko_scaffold_from_mol,
    suppress_rdkit_warnings,
)
from config import apply_cli_overrides, load_yaml, save_yaml
from utils import (
    ensure_dir,
    read_jsonl,
    write_json,
    write_jsonl,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_reference_paths(eval_cfg: dict[str, Any]) -> list[str]:
    #new
    if eval_cfg.get("reference_train_jsonl"):
        #new
        return [
            #new
            str(eval_cfg["reference_train_jsonl"])
            #new
        ]

    #new
    return []


def load_reference_sets(
    paths: list[str],
    smiles_field: str,
) -> tuple[set[str], set[str]]:
    """
    Load exact TRAIN SMILES and their Murcko scaffolds.

    TRAIN JSONL already contains the finalized canonical SMILES written by
    prepare.py, so exact novelty/reference matching uses those strings
    directly. RDKit parsing is only needed here for scaffold computation.
    """
    reference_set: set[str] = set()
    reference_scaffold_set: set[str] = set()

    for path in paths:
        if path is None or str(path).strip() == "":
            continue

        records = read_jsonl(path)

        for record in records:
            if smiles_field not in record:
                raise KeyError(
                    f"Reference file {path} does not contain field "
                    f"'{smiles_field}'. Available keys: {list(record.keys())}"
                )

            smiles = str(
                record[smiles_field]
            ).strip()

            reference_set.add(
                smiles
            )

            mol = mol_from_smiles(
                smiles
            )

            scaffold = murcko_scaffold_from_mol(
                mol
            )

            if scaffold is not None:
                reference_scaffold_set.add(
                    scaffold
                )

    return reference_set, reference_scaffold_set


def summarize_properties(
    annotated_records: list[dict[str, Any]],
) -> dict[str, dict[str, float]] | None:
    """
    Summarize molecular properties for unique valid molecules only.
    """
    property_rows = [
        record["properties"]
        for record in annotated_records
        if record.get("is_valid")
        and record.get("is_first_occurrence_valid")
        and isinstance(record.get("properties"), dict)
    ]

    if not property_rows:
        return None

    keys = sorted(
        {
            key
            for row in property_rows
            for key in row.keys()
        }
    )

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
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
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
    Annotate every generated record with validity, canonical SMILES,
    scaffold, novelty, duplication, and molecular properties.
    """
    if len(generated_records) == 0:
        raise ValueError(
            "Generated JSONL is empty. Nothing to evaluate."
        )

    temp_records: list[dict[str, Any]] = []
    valid_canonical_all: list[str] = []
    unique_mol_by_canonical: dict[str, Any] = {}

    iterator = tqdm(
        generated_records,
        desc="Annotating generated SMILES",
        unit="mol",
    )

    for record in iterator:
        if generated_smiles_field not in record:
            raise KeyError(
                f"Generated record does not contain field "
                f"'{generated_smiles_field}'. "
                f"Available keys: {list(record.keys())}"
            )

        raw_smiles = record[generated_smiles_field]
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
                **record,
                "canonical_smiles": canonical_smiles,
                "is_valid": is_valid,
                "murcko_scaffold": scaffold,
                "properties": properties,
            }
        )

    canonical_counts = Counter(valid_canonical_all)
    seen_valid: set[str] = set()
    annotated_records: list[dict[str, Any]] = []

    for record in temp_records:
        canonical_smiles = record["canonical_smiles"]
        is_valid = record["is_valid"]
        scaffold = record["murcko_scaffold"]

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
                **record,
                "canonical_count_in_generated": canonical_count,
                "is_first_occurrence_valid": is_first_occurrence,
                "is_duplicate_generated": is_duplicate_generated,
                "is_reference_match": is_reference_match,
                "is_novel": is_novel,
                "scaffold_in_reference": scaffold_in_reference,
            }
        )

    unique_valid_ordered = list(
        dict.fromkeys(valid_canonical_all)
    )

    unique_valid_mols_ordered = [
        unique_mol_by_canonical[smiles]
        for smiles in unique_valid_ordered
    ]

    return (
        annotated_records,
        valid_canonical_all,
        unique_valid_mols_ordered,
    )


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

    unique_valid_ordered = list(
        dict.fromkeys(valid_canonical_all)
    )

    unique_valid_set = set(unique_valid_ordered)
    duplicate_valid_count = num_valid - len(unique_valid_set)

    unique_novel_valid = [
        smiles
        for smiles in unique_valid_ordered
        if smiles not in reference_set
    ]

    valid_reference_matches_all = [
        smiles
        for smiles in valid_canonical_all
        if smiles in reference_set
    ]

    unique_reference_matches = {
        smiles
        for smiles in unique_valid_set
        if smiles in reference_set
    }

    valid_scaffolds = [
        record["murcko_scaffold"]
        for record in annotated_records
        if record["is_valid"]
        and record["murcko_scaffold"] is not None
    ]

    unique_valid_scaffolds = list(
        dict.fromkeys(valid_scaffolds)
    )

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

    internal_diversity = internal_diversity_from_mols(
        mols=unique_valid_mols_ordered,
        radius=fp_radius,
        n_bits=fp_n_bits,
        max_samples=max_diversity_samples,
        seed=seed,
    )

    property_summary = summarize_properties(
        annotated_records
    )

    metrics = {
        "num_generated": num_generated,
        "num_valid": num_valid,
        "num_invalid": num_invalid,
        "validity": num_valid / max(1, num_generated),
        "num_unique_valid": len(unique_valid_set),
        "uniqueness_among_valid": (
            len(unique_valid_set)
            / max(1, num_valid)
        ),
        "unique_valid_fraction_of_generated": (
            len(unique_valid_set)
            / max(1, num_generated)
        ),
        "duplicate_valid_count": duplicate_valid_count,
        "duplicate_rate_among_valid": (
            duplicate_valid_count
            / max(1, num_valid)
        ),
        "num_reference_matches_valid": (
            len(valid_reference_matches_all)
        ),
        "reference_match_rate_among_valid": (
            len(valid_reference_matches_all)
            / max(1, num_valid)
        ),
        "num_reference_matches_unique_valid": (
            len(unique_reference_matches)
        ),
        "reference_match_rate_among_unique_valid": (
            len(unique_reference_matches)
            / max(1, len(unique_valid_set))
        ),
        "num_novel_unique_valid": len(unique_novel_valid),
        "novelty_among_unique_valid": (
            len(unique_novel_valid)
            / max(1, len(unique_valid_set))
        ),
        "num_valid_scaffolds": len(valid_scaffolds),
        "num_unique_valid_scaffolds": (
            len(unique_valid_scaffolds)
        ),
        "num_reference_matching_unique_scaffolds": (
            len(unique_reference_matching_scaffolds)
        ),
        "scaffold_reference_overlap_rate": (
            len(unique_reference_matching_scaffolds)
            / max(1, len(unique_valid_scaffolds))
        ),
        "num_novel_unique_scaffolds": (
            len(unique_novel_scaffolds)
        ),
        "scaffold_novelty_rate": (
            len(unique_novel_scaffolds)
            / max(1, len(unique_valid_scaffolds))
        ),
        "internal_diversity_unique_valid": internal_diversity,
        "reference_set_size": len(reference_set),
        "reference_scaffold_set_size": len(reference_scaffold_set),
        "fingerprint": {
            "type": "Morgan",
            "radius": fp_radius,
            "n_bits": fp_n_bits,
            "diversity_max_samples": max_diversity_samples,
        },
        "property_summary_unique_valid": property_summary,
    }

    return metrics


def build_condition_nll_examples(
    records: list[dict[str, Any]],
    smiles_field: str,
    target_uniprot_field: str,
    records_per_poi: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Build the fixed validation panel used for the
    correct-vs-all-other-POIs NLL guardrail.

    For a molecule associated with multiple POIs,
    every associated POI is excluded from the
    wrong-condition set.
    """
    rng = random.Random(seed)

    records_by_target: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    targets_by_smiles: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for record in records:
        smiles = str(
            record[smiles_field]
        ).strip()

        target_uniprot = str(
            record[target_uniprot_field]
        ).strip()

        records_by_target[
            target_uniprot
        ].append(
            record
        )

        targets_by_smiles[
            smiles
        ].add(
            target_uniprot
        )

    all_targets = sorted(
        records_by_target
    )

    examples: list[
        dict[str, Any]
    ] = []

    for target_uniprot in all_targets:

        selected_records = rng.sample(
            records_by_target[target_uniprot],
            records_per_poi,
        )

        for record in selected_records:

            smiles = str(
                record[smiles_field]
            ).strip()

            wrong_target_uniprots = [
                candidate
                for candidate in all_targets
                if candidate
                not in targets_by_smiles[smiles]
            ]

            examples.append(
                {
                    "smiles": smiles,
                    "target_uniprot": target_uniprot,
                    "wrong_target_uniprots": (
                        wrong_target_uniprots
                    ),
                }
            )

    return examples

def _encode_smiles_for_lm(
    smiles: str,
    tokenizer: Any,
    max_seq_len: int,
) -> list[int]:
    ids = tokenizer.encode(
        smiles,
        add_bos=True,
        add_eos=True,
    )

    ids = ids[:max_seq_len]

    if (
        len(ids) == max_seq_len
        and ids[-1] != tokenizer.eos_id
    ):
        ids[-1] = tokenizer.eos_id

    return ids


def _make_lm_batch(
    encoded_ids: list[list[int]],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(
        len(ids) - 1
        for ids in encoded_ids
    )

    input_ids = torch.full(
        (len(encoded_ids), max_length),
        pad_id,
        dtype=torch.long,
        device=device,
    )

    labels = torch.full(
        (len(encoded_ids), max_length),
        -100,
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.zeros(
        (len(encoded_ids), max_length),
        dtype=torch.long,
        device=device,
    )

    for row_index, ids in enumerate(encoded_ids):
        sequence_length = len(ids) - 1

        input_ids[row_index, :sequence_length] = torch.tensor(
            ids[:-1],
            dtype=torch.long,
            device=device,
        )

        labels[row_index, :sequence_length] = torch.tensor(
            ids[1:],
            dtype=torch.long,
            device=device,
        )

        attention_mask[row_index, :sequence_length] = 1

    return input_ids, labels, attention_mask


def _sequence_nll_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute length-normalized next-token NLL independently
    for every sequence in a batch.
    """
    token_nll = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)

    token_mask = labels.ne(-100)

    return (
        (token_nll * token_mask).sum(dim=1)
        / token_mask.sum(dim=1).clamp_min(1)
    )


def compute_condition_nll_metrics(
    model: Any,
    tokenizer: Any,
    nll_examples: list[dict[str, Any]],
    poi_embeddings_by_uniprot: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
    max_seq_len: int | None = None,
    return_per_poi: bool = False,
) -> dict[str, Any]:
    """
    Compute the lightweight condition-use guardrail.

    delta_nll =
        mean(NLL under all valid wrong POIs)
        - NLL under correct POI

    Positive delta_nll means the model assigns the real molecule
    higher likelihood under its correct POI condition.

    Different molecules may have different numbers of valid wrong
    conditions because molecules can be associated with multiple POIs.
    """
    if max_seq_len is None:
        max_seq_len = int(
            model.config.max_seq_len
        )

    correct_nll_values: list[float] = []
    wrong_nll_values: list[float] = []
    target_uniprots: list[str] = []

    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():

            for batch_start in range(
                0,
                len(nll_examples),
                batch_size,
            ):
                batch = nll_examples[
                    batch_start:
                    batch_start + batch_size
                ]

                encoded_ids = [
                    _encode_smiles_for_lm(
                        smiles=record["smiles"],
                        tokenizer=tokenizer,
                        max_seq_len=max_seq_len,
                    )
                    for record in batch
                ]

                # --------------------------------------------------
                # Correct condition
                # --------------------------------------------------
                (
                    input_ids,
                    labels,
                    attention_mask,
                ) = _make_lm_batch(
                    encoded_ids=encoded_ids,
                    pad_id=tokenizer.pad_id,
                    device=device,
                )

                correct_poi_embeddings = torch.stack(
                    [
                        poi_embeddings_by_uniprot[
                            record["target_uniprot"]
                        ]
                        for record in batch
                    ],
                    dim=0,
                ).float().to(
                    device
                )

                correct_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    poi_embeddings=correct_poi_embeddings,
                )

                correct_nll = _sequence_nll_from_logits(
                    logits=correct_outputs.logits,
                    labels=labels,
                )

                # --------------------------------------------------
                # All valid wrong conditions.
                #
                # Flatten:
                #   molecule 0 -> wrong A, B, C, ...
                #   molecule 1 -> wrong A, C, ...
                #
                # This safely supports different numbers of wrong
                # conditions for multi-POI molecules.
                # --------------------------------------------------
                wrong_pairs: list[
                    tuple[int, str]
                ] = []

                for local_index, record in enumerate(
                    batch
                ):
                    wrong_targets = record[
                        "wrong_target_uniprots"
                    ]

                    if not wrong_targets:
                        raise ValueError(
                            "NLL guardrail example has no "
                            "valid wrong POI conditions."
                        )

                    for wrong_target in wrong_targets:
                        wrong_pairs.append(
                            (
                                local_index,
                                wrong_target,
                            )
                        )

                wrong_nll_sum = torch.zeros(
                    len(batch),
                    dtype=torch.float32,
                    device=device,
                )

                wrong_nll_count = torch.zeros(
                    len(batch),
                    dtype=torch.float32,
                    device=device,
                )

                # Process flattened wrong-condition pairs in chunks
                # so using all 20 POIs does not create one huge batch.
                for wrong_start in range(
                    0,
                    len(wrong_pairs),
                    batch_size,
                ):
                    wrong_chunk = wrong_pairs[
                        wrong_start:
                        wrong_start + batch_size
                    ]

                    wrong_encoded_ids = [
                        encoded_ids[local_index]
                        for local_index, _
                        in wrong_chunk
                    ]

                    (
                        wrong_input_ids,
                        wrong_labels,
                        wrong_attention_mask,
                    ) = _make_lm_batch(
                        encoded_ids=wrong_encoded_ids,
                        pad_id=tokenizer.pad_id,
                        device=device,
                    )

                    wrong_poi_embeddings = torch.stack(
                        [
                            poi_embeddings_by_uniprot[
                                wrong_target
                            ]
                            for _, wrong_target
                            in wrong_chunk
                        ],
                        dim=0,
                    ).float().to(
                        device
                    )

                    wrong_outputs = model(
                        input_ids=wrong_input_ids,
                        attention_mask=wrong_attention_mask,
                        poi_embeddings=wrong_poi_embeddings,
                    )

                    wrong_chunk_nll = (
                        _sequence_nll_from_logits(
                            logits=wrong_outputs.logits,
                            labels=wrong_labels,
                        )
                    )

                    owner_indices = torch.tensor(
                        [
                            local_index
                            for local_index, _
                            in wrong_chunk
                        ],
                        dtype=torch.long,
                        device=device,
                    )

                    wrong_nll_sum.index_add_(
                        0,
                        owner_indices,
                        wrong_chunk_nll,
                    )

                    wrong_nll_count.index_add_(
                        0,
                        owner_indices,
                        torch.ones_like(
                            wrong_chunk_nll
                        ),
                    )

                wrong_nll = (
                    wrong_nll_sum
                    / wrong_nll_count.clamp_min(1.0)
                )

                correct_nll_values.extend(
                    correct_nll.detach().cpu().tolist()
                )

                wrong_nll_values.extend(
                    wrong_nll.detach().cpu().tolist()
                )

                target_uniprots.extend(
                    record["target_uniprot"]
                    for record in batch
                )

    finally:
        model.train(
            was_training
        )

    correct_nll_array = np.asarray(
        correct_nll_values,
        dtype=float,
    )

    wrong_nll_array = np.asarray(
        wrong_nll_values,
        dtype=float,
    )

    delta_nll_array = (
        wrong_nll_array
        - correct_nll_array
    )

    metrics: dict[str, Any] = {
        "condition_nll_correct": float(
            np.mean(
                correct_nll_array
            )
        ),
        "condition_nll_wrong": float(
            np.mean(
                wrong_nll_array
            )
        ),
        "condition_nll_gap": float(
            np.mean(
                delta_nll_array
            )
        ),
        "condition_nll_fraction_positive": float(
            np.mean(
                delta_nll_array > 0.0
            )
        ),
    }

    if return_per_poi:
        target_array = np.asarray(
            target_uniprots,
            dtype=object,
        )

        metrics["per_poi"] = {
            target_uniprot: {
                "condition_nll_gap": float(
                    np.mean(
                        delta_nll_array[
                            target_array
                            == target_uniprot
                        ]
                    )
                )
            }
            for target_uniprot
            in sorted(
                set(target_uniprots)
            )
        }

    return metrics

def build_poi_knn_reference_pool(
    records: list[dict[str, Any]],
    smiles_field: str,
    target_uniprot_field: str,
    fp_radius: int,
    fp_n_bits: int,
) -> dict[str, Any]:
    """
    Build the fixed real validation reference pool used for generated
    k-NN requested-POI purity.

    VAL JSONL already contains finalized canonical SMILES, so those stored
    strings are used directly as molecule identities. RDKit parsing is only
    needed to create the fingerprints used by the k-NN metric.

    Each unique canonical molecule appears once in chemical space.
    If a molecule is associated with multiple POIs, all associated
    POIs are stored with that single reference molecule.
    """
    mol_by_canonical: dict[str, Chem.Mol] = {}
    targets_by_canonical: dict[str, set[str]] = defaultdict(set)

    for record in records:
        canonical = str(
            record[smiles_field]
        ).strip()

        if canonical not in mol_by_canonical:
            mol_by_canonical[canonical] = mol_from_smiles(
                canonical
            )

        targets_by_canonical[canonical].add(
            str(record[target_uniprot_field]).strip()
        )

    canonical_smiles = list(mol_by_canonical)

    reference_mols = [
        mol_by_canonical[canonical]
        for canonical in canonical_smiles
    ]

    fingerprints = get_morgan_fingerprints_from_mols(
        mols=reference_mols,
        radius=fp_radius,
        n_bits=fp_n_bits,
    )

    if not fingerprints:
        raise ValueError(
            "No valid molecules found for POI k-NN reference pool."
        )

    target_uniprots_by_molecule = [
        sorted(targets_by_canonical[canonical])
        for canonical in canonical_smiles
    ]

    return {
        "fingerprints": fingerprints,
        "target_uniprots_by_molecule": target_uniprots_by_molecule,
        "canonical_smiles": canonical_smiles,
        "fp_radius": fp_radius,
        "fp_n_bits": fp_n_bits,
    }


def compute_generated_poi_knn_purity(
    generated_records: list[dict[str, Any]],
    reference_pool: dict[str, Any],
    generated_smiles_field: str,
    target_uniprot_field: str,
    k: int = 5,
    max_valid_per_poi: int = 25,
    return_per_poi: bool = False,
) -> dict[str, Any]:
    """
    Compute the lightweight generated-chemistry
    condition-use guardrail during training.

    For each POI, select up to max_valid_per_poi
    UNIQUE valid generated molecules.

    For each selected molecule, find its k nearest
    real validation molecules and calculate the
    fraction for which the requested POI is one of
    the known associated POIs.

    Generated molecules are deduplicated within each
    requested POI using canonical SMILES.
    """

    reference_fingerprints = reference_pool[
        "fingerprints"
    ]

    reference_target_sets = reference_pool[
        "target_uniprots_by_molecule"
    ]

    # ------------------------------------------------------------
    # Collect up to max_valid_per_poi UNIQUE valid molecules
    # independently for each requested POI.
    # ------------------------------------------------------------
    generated_by_poi: dict[
        str,
        list[Chem.Mol],
    ] = defaultdict(list)

    seen_canonical_by_poi: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for record in generated_records:

        target_uniprot = str(
            record[target_uniprot_field]
        ).strip()

        # Once this POI already has enough unique valid
        # molecules, no more queries are needed.
        if (
            len(
                generated_by_poi[
                    target_uniprot
                ]
            )
            >= max_valid_per_poi
        ):
            continue

        mol = mol_from_smiles(
            record[
                generated_smiles_field
            ]
        )

        if mol is None:
            continue

        canonical_smiles = (
            canonicalize_from_mol(
                mol
            )
        )

        if canonical_smiles is None:
            continue

        # POI-local deduplication:
        #
        # AR -> X, X, X
        # contributes X only once.
        #
        # But:
        # AR   -> X
        # ESR1 -> X
        #
        # may each contribute once because these are
        # different conditioning experiments.
        if (
            canonical_smiles
            in seen_canonical_by_poi[
                target_uniprot
            ]
        ):
            continue

        seen_canonical_by_poi[
            target_uniprot
        ].add(
            canonical_smiles
        )

        generated_by_poi[
            target_uniprot
        ].append(
            mol
        )

    # ------------------------------------------------------------
    # Flatten the POI-specific query sets.
    # ------------------------------------------------------------
    query_mols: list[
        Chem.Mol
    ] = []

    query_targets: list[
        str
    ] = []

    for target_uniprot in sorted(
        generated_by_poi
    ):
        for mol in generated_by_poi[
            target_uniprot
        ]:
            query_mols.append(
                mol
            )

            query_targets.append(
                target_uniprot
            )

    # ------------------------------------------------------------
    # Nothing valid to evaluate.
    # ------------------------------------------------------------
    if not query_mols:
        return {
            "generated_poi_knn_purity_mean": None,
            "generated_poi_knn_k": int(k),
            "generated_poi_knn_num_queries": 0,
            "generated_poi_knn_num_reference_molecules": (
                len(
                    reference_fingerprints
                )
            ),
        }

    # ------------------------------------------------------------
    # Fingerprint generated queries.
    # ------------------------------------------------------------
    query_fingerprints = (
        get_morgan_fingerprints_from_mols(
            mols=query_mols,
            radius=(
                reference_pool[
                    "fp_radius"
                ]
            ),
            n_bits=(
                reference_pool[
                    "fp_n_bits"
                ]
            ),
        )
    )

    effective_k = min(
        k,
        len(
            reference_fingerprints
        ),
    )

    purity_values: list[
        float
    ] = []

    purity_values_by_poi: dict[
        str,
        list[float],
    ] = defaultdict(list)

    # ------------------------------------------------------------
    # Compute requested-POI purity.
    # ------------------------------------------------------------
    for (
        query_fingerprint,
        target_uniprot,
    ) in zip(
        query_fingerprints,
        query_targets,
    ):

        similarities = (
            DataStructs.BulkTanimotoSimilarity(
                query_fingerprint,
                reference_fingerprints,
            )
        )

        top_indices = np.argsort(
            np.asarray(
                similarities,
                dtype=float,
            )
        )[
            -effective_k:
        ][::-1]

        purity = (
            sum(
                target_uniprot
                in reference_target_sets[
                    int(index)
                ]
                for index
                in top_indices
            )
            / effective_k
        )

        purity_values.append(
            float(purity)
        )

        purity_values_by_poi[
            target_uniprot
        ].append(
            float(purity)
        )

    # ------------------------------------------------------------
    # Pooled metrics.
    # ------------------------------------------------------------
    metrics: dict[
        str,
        Any,
    ] = {
        "generated_poi_knn_purity_mean": float(
            np.mean(
                np.asarray(
                    purity_values,
                    dtype=float,
                )
            )
        ),
        "generated_poi_knn_k": int(
            effective_k
        ),
        "generated_poi_knn_num_queries": (
            len(
                purity_values
            )
        ),
        "generated_poi_knn_num_reference_molecules": (
            len(
                reference_fingerprints
            )
        ),
    }

    # ------------------------------------------------------------
    # Optional per-POI monitoring.
    # ------------------------------------------------------------
    if return_per_poi:

        metrics["per_poi"] = {
            target_uniprot: {
                "generated_poi_knn_purity_mean": float(
                    np.mean(
                        np.asarray(
                            values,
                            dtype=float,
                        )
                    )
                )
            }
            for (
                target_uniprot,
                values,
            )
            in sorted(
                purity_values_by_poi.items()
            )
        }

    return metrics

#new
# ============================================================
# Final standalone conditional-fidelity evaluation
# ============================================================


#new
def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


#new
def _safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


#new
def _safe_std(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.std(np.asarray(values, dtype=float)))


#new
def _compute_per_poi_generation_metrics(
    annotated_generated_records: list[dict[str, Any]],
    mol_by_canonical: dict[str, Chem.Mol],
    target_uniprot_field: str,
    poi_name_lookup: dict[str, str],
    reference_set: set[str],
    reference_scaffold_set: set[str],
    fp_radius: int,
    fp_n_bits: int,
    max_diversity_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Compute generation metrics independently within each requested POI.

    Global annotation already parsed/canonicalized generated molecules and
    computed scaffolds/properties. Here we reuse those results and recompute
    only POI-local occurrence/uniqueness information.
    """
    records_by_target: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in annotated_generated_records:
        if target_uniprot_field not in record:
            raise KeyError(
                "Conditional generated record is missing field "
                f"{target_uniprot_field!r}."
            )

        target_uniprot = str(
            record[target_uniprot_field]
        ).strip()

        records_by_target[target_uniprot].append(
            record
        )

    rows: list[dict[str, Any]] = []

    for target_uniprot, target_records in records_by_target.items():
        valid_canonical_all = [
            record["canonical_smiles"]
            for record in target_records
            if record.get("is_valid")
        ]

        canonical_counts = Counter(
            valid_canonical_all
        )

        seen_valid: set[str] = set()
        annotated_target_records: list[dict[str, Any]] = []

        for record in target_records:
            target_record = dict(
                record
            )

            if record.get("is_valid"):
                canonical = record[
                    "canonical_smiles"
                ]

                target_record[
                    "canonical_count_in_generated"
                ] = canonical_counts[
                    canonical
                ]

                target_record[
                    "is_first_occurrence_valid"
                ] = canonical not in seen_valid

                target_record[
                    "is_duplicate_generated"
                ] = canonical_counts[
                    canonical
                ] > 1

                seen_valid.add(
                    canonical
                )

            annotated_target_records.append(
                target_record
            )

        unique_valid_ordered = list(
            dict.fromkeys(
                valid_canonical_all
            )
        )

        unique_valid_mols_ordered = [
            mol_by_canonical[canonical]
            for canonical in unique_valid_ordered
        ]

        target_metrics = compute_metrics(
            annotated_records=annotated_target_records,
            valid_canonical_all=valid_canonical_all,
            unique_valid_mols_ordered=unique_valid_mols_ordered,
            reference_set=reference_set,
            reference_scaffold_set=reference_scaffold_set,
            fp_radius=fp_radius,
            fp_n_bits=fp_n_bits,
            max_diversity_samples=max_diversity_samples,
            seed=seed,
        )

        rows.append(
            {
                "target_uniprot": target_uniprot,
                "poi_norm": poi_name_lookup.get(
                    target_uniprot,
                    target_uniprot,
                ),
                **target_metrics,
            }
        )

    return rows


#new
def _summarize_per_poi_generation_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Summarize Level-2 generation quality across POIs in two ways.

    macro:
        Every POI contributes equally. We report mean and SD across POIs.

    weighted:
        POI-local numerators and denominators are summed before taking the
        ratio. This preserves the per-POI definition of duplication:
        the same canonical molecule appearing once under POI A and once under
        POI B is not collapsed as a duplicate across conditions.

        Examples:
            weighted validity =
                sum_p num_valid_p / sum_p num_generated_p

            weighted uniqueness_among_valid =
                sum_p num_unique_valid_p / sum_p num_valid_p

    Internal diversity has no single count-ratio denominator, so it remains
    summarized through the per-POI macro mean/SD rather than being forced into
    the weighted ratio summary.
    """
    if not rows:
        return {
            "num_pois": 0,
            "macro": {},
            "weighted": {},
        }

    scalar_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in {
                "target_uniprot",
                "poi_norm",
            }
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
    )

    macro: dict[str, Any] = {}

    for key in scalar_keys:
        values = [
            float(row[key])
            for row in rows
            if isinstance(
                row.get(key),
                (int, float),
            )
            and not isinstance(
                row.get(key),
                bool,
            )
        ]

        macro[key] = {
            "mean": _safe_mean(values),
            "std": _safe_std(values),
        }

    def ratio_of_sums(
        numerator_key: str,
        denominator_key: str,
    ) -> float | None:
        numerator = sum(
            float(row[numerator_key])
            for row in rows
            if isinstance(
                row.get(numerator_key),
                (int, float),
            )
            and not isinstance(
                row.get(numerator_key),
                bool,
            )
        )

        denominator = sum(
            float(row[denominator_key])
            for row in rows
            if isinstance(
                row.get(denominator_key),
                (int, float),
            )
            and not isinstance(
                row.get(denominator_key),
                bool,
            )
        )

        if denominator <= 0:
            return None

        return float(
            numerator
            / denominator
        )

    weighted = {
        "validity": ratio_of_sums(
            "num_valid",
            "num_generated",
        ),
        "uniqueness_among_valid": ratio_of_sums(
            "num_unique_valid",
            "num_valid",
        ),
        "unique_valid_fraction_of_generated": ratio_of_sums(
            "num_unique_valid",
            "num_generated",
        ),
        "duplicate_rate_among_valid": ratio_of_sums(
            "duplicate_valid_count",
            "num_valid",
        ),
        "reference_match_rate_among_valid": ratio_of_sums(
            "num_reference_matches_valid",
            "num_valid",
        ),
        "reference_match_rate_among_unique_valid": ratio_of_sums(
            "num_reference_matches_unique_valid",
            "num_unique_valid",
        ),
        "novelty_among_unique_valid": ratio_of_sums(
            "num_novel_unique_valid",
            "num_unique_valid",
        ),
        "scaffold_reference_overlap_rate": ratio_of_sums(
            "num_reference_matching_unique_scaffolds",
            "num_unique_valid_scaffolds",
        ),
        "scaffold_novelty_rate": ratio_of_sums(
            "num_novel_unique_scaffolds",
            "num_unique_valid_scaffolds",
        ),
    }

    totals = {
        key: int(
            sum(
                int(row[key])
                for row in rows
                if isinstance(
                    row.get(key),
                    (int, float),
                )
                and not isinstance(
                    row.get(key),
                    bool,
                )
            )
        )
        for key in [
            "num_generated",
            "num_valid",
            "num_invalid",
            "num_unique_valid",
            "duplicate_valid_count",
            "num_reference_matches_valid",
            "num_reference_matches_unique_valid",
            "num_novel_unique_valid",
            "num_unique_valid_scaffolds",
            "num_reference_matching_unique_scaffolds",
            "num_novel_unique_scaffolds",
        ]
    }

    return {
        "num_pois": len(rows),
        "aggregation": {
            "macro": "equal_weight_per_poi",
            "weighted": (
                "ratio_of_summed_per_poi_numerators_and_denominators; "
                "cross_poi_duplicates_are_not_collapsed"
            ),
        },
        "totals": totals,
        "macro": macro,
        "weighted": weighted,
    }


#new
def _build_test_reference_pool(
    records: list[dict[str, Any]],
    smiles_field: str,
    target_uniprot_field: str,
    fp_radius: int,
    fp_n_bits: int,
) -> dict[str, Any]:
    """
    Build the held-out TEST chemistry reference in one pass.

    TEST JSONL already contains finalized clean SMILES. Those stored strings
    are used directly as molecule identities. The same object is reused by
    multi-condition NLL, real-test k-NN calibration, and generated
    conditional-fidelity evaluation.
    """
    mol_by_canonical: dict[str, Chem.Mol] = {}
    targets_by_canonical: dict[
        str,
        set[str],
    ] = defaultdict(set)
    record_counts_by_target: Counter = Counter()

    for record in records:
        canonical = str(
            record[smiles_field]
        ).strip()

        target_uniprot = str(
            record[target_uniprot_field]
        ).strip()

        record_counts_by_target[
            target_uniprot
        ] += 1

        targets_by_canonical[
            canonical
        ].add(
            target_uniprot
        )

        if canonical not in mol_by_canonical:
            mol_by_canonical[
                canonical
            ] = mol_from_smiles(
                canonical
            )

    canonical_smiles = list(
        mol_by_canonical
    )

    reference_mols = [
        mol_by_canonical[canonical]
        for canonical in canonical_smiles
    ]

    fingerprints = get_morgan_fingerprints_from_mols(
        mols=reference_mols,
        radius=fp_radius,
        n_bits=fp_n_bits,
    )

    target_uniprots_by_molecule = [
        sorted(
            targets_by_canonical[
                canonical
            ]
        )
        for canonical in canonical_smiles
    ]

    return {
        "canonical_smiles": canonical_smiles,
        "mol_by_canonical": mol_by_canonical,
        "fingerprints": fingerprints,
        "targets_by_canonical": targets_by_canonical,
        "target_uniprots_by_molecule": target_uniprots_by_molecule,
        "record_counts_by_target": record_counts_by_target,
        "fp_radius": fp_radius,
        "fp_n_bits": fp_n_bits,
    }


#new
def compute_full_condition_nll_metrics(
    model: Any,
    tokenizer: Any,
    test_reference_pool: dict[str, Any],
    poi_embeddings_by_uniprot: dict[str, torch.Tensor],
    target_uniprots: list[str],
    batch_size: int,
    device: torch.device,
    max_seq_len: int | None = None,
) -> dict[str, Any]:
    """
    Final multi-condition NLL evaluation on unique held-out test
    molecules.

    Each unique stored test molecule is scored under every requested
    condition. For a valid association (x, p), all other known valid POIs
    for x are excluded from the wrong-condition set and from the ranking
    competitors for that association.
    """
    if max_seq_len is None:
        max_seq_len = int(
            model.config.max_seq_len
        )

    test_index = test_reference_pool

    target_set = set(
        target_uniprots
    )

    scoring_canonicals = [
        canonical
        for canonical
        in test_index["canonical_smiles"]
        if test_index[
            "targets_by_canonical"
        ][canonical]
        & target_set
    ]

    if not scoring_canonicals:
        raise ValueError(
            "No held-out test molecules are associated with the requested "
            "conditional-evaluation POIs."
        )

    encoded_by_canonical = {
        canonical: _encode_smiles_for_lm(
            smiles=canonical,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        for canonical in scoring_canonicals
    }

    scoring_rows = [
        (canonical, target_uniprot)
        for canonical in scoring_canonicals
        for target_uniprot in target_uniprots
    ]

    nll_by_canonical: dict[
        str,
        dict[str, float],
    ] = {
        canonical: {}
        for canonical in scoring_canonicals
    }

    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            for start in tqdm(
                range(
                    0,
                    len(scoring_rows),
                    batch_size,
                ),
                desc="Full condition-NLL",
                unit="batch",
            ):
                batch_rows = scoring_rows[
                    start:start + batch_size
                ]

                encoded_ids = [
                    encoded_by_canonical[
                        canonical
                    ]
                    for canonical, _
                    in batch_rows
                ]

                (
                    input_ids,
                    labels,
                    attention_mask,
                ) = _make_lm_batch(
                    encoded_ids=encoded_ids,
                    pad_id=tokenizer.pad_id,
                    device=device,
                )

                batch_poi_embeddings = torch.stack(
                    [
                        poi_embeddings_by_uniprot[
                            target_uniprot
                        ]
                        for _, target_uniprot
                        in batch_rows
                    ],
                    dim=0,
                ).float().to(
                    device
                )

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    poi_embeddings=(
                        batch_poi_embeddings
                    ),
                )

                nll_values = (
                    _sequence_nll_from_logits(
                        logits=outputs.logits,
                        labels=labels,
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )

                for (
                    canonical,
                    condition_target,
                ), nll in zip(
                    batch_rows,
                    nll_values,
                ):
                    nll_by_canonical[
                        canonical
                    ][condition_target] = float(
                        nll
                    )
    finally:
        model.train(
            was_training
        )

    association_rows: list[
        dict[str, Any]
    ] = []
    rows_by_target: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for canonical in scoring_canonicals:
        valid_targets_all = test_index[
            "targets_by_canonical"
        ][canonical]

        valid_targets = [
            target
            for target in target_uniprots
            if target in valid_targets_all
        ]

        wrong_targets = [
            target
            for target in target_uniprots
            if target not in valid_targets_all
        ]

        if not wrong_targets:
            continue

        for target_uniprot in valid_targets:
            correct_nll = nll_by_canonical[
                canonical
            ][target_uniprot]

            wrong_values = [
                nll_by_canonical[
                    canonical
                ][wrong_target]
                for wrong_target
                in wrong_targets
            ]

            wrong_mean = float(
                np.mean(
                    np.asarray(
                        wrong_values,
                        dtype=float,
                    )
                )
            )

            gap = (
                wrong_mean
                - correct_nll
            )

            ranking_targets = [
                target_uniprot,
                *wrong_targets,
            ]

            ranking_targets = sorted(
                ranking_targets,
                key=lambda target: (
                    nll_by_canonical[
                        canonical
                    ][target],
                    target,
                ),
            )

            rank = (
                ranking_targets.index(
                    target_uniprot
                )
                + 1
            )

            row = {
                "target_uniprot": (
                    target_uniprot
                ),
                "correct_nll": float(
                    correct_nll
                ),
                "wrong_nll_mean": (
                    wrong_mean
                ),
                "gap": float(gap),
                "is_positive": bool(
                    gap > 0.0
                ),
                "top1": bool(
                    rank == 1
                ),
                "top5": bool(
                    rank <= min(
                        5,
                        len(ranking_targets),
                    )
                ),
            }

            association_rows.append(
                row
            )
            rows_by_target[
                target_uniprot
            ].append(
                row
            )

    def summarize_rows(
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        gaps = [
            float(row["gap"])
            for row in rows
        ]

        return {
            "num_associations": len(rows),
            "condition_nll_correct_mean": (
                _safe_mean(
                    [
                        float(row["correct_nll"])
                        for row in rows
                    ]
                )
            ),
            "condition_nll_wrong_mean": (
                _safe_mean(
                    [
                        float(row["wrong_nll_mean"])
                        for row in rows
                    ]
                )
            ),
            "condition_nll_gap_mean": (
                _safe_mean(gaps)
            ),
            "condition_nll_gap_median": (
                _safe_median(gaps)
            ),
            "condition_nll_fraction_positive": (
                _safe_mean(
                    [
                        float(row["is_positive"])
                        for row in rows
                    ]
                )
            ),
            "nll_top1_accuracy": (
                _safe_mean(
                    [
                        float(row["top1"])
                        for row in rows
                    ]
                )
            ),
            "nll_top5_accuracy": (
                _safe_mean(
                    [
                        float(row["top5"])
                        for row in rows
                    ]
                )
            ),
        }

    per_poi: dict[str, Any] = {}

    for target_uniprot in target_uniprots:
        target_rows = rows_by_target[
            target_uniprot
        ]

        target_summary = summarize_rows(
            target_rows
        )
        target_summary[
            "num_test_records"
        ] = int(
            test_index[
                "record_counts_by_target"
            ][target_uniprot]
        )
        target_summary[
            "num_test_unique_molecules"
        ] = len(
            {
                canonical
                for canonical
                in scoring_canonicals
                if target_uniprot
                in test_index[
                    "targets_by_canonical"
                ][canonical]
            }
        )

        per_poi[
            target_uniprot
        ] = target_summary

    macro_metric_names = [
        "condition_nll_correct_mean",
        "condition_nll_wrong_mean",
        "condition_nll_gap_mean",
        "condition_nll_gap_median",
        "condition_nll_fraction_positive",
        "nll_top1_accuracy",
        "nll_top5_accuracy",
    ]

    macro = {
        metric_name: _safe_mean(
            [
                float(
                    per_poi[target][
                        metric_name
                    ]
                )
                for target
                in target_uniprots
                if per_poi[target].get(
                    metric_name
                )
                is not None
            ]
        )
        for metric_name in macro_metric_names
    }

    return {
        "num_conditions": len(
            target_uniprots
        ),
        "num_unique_test_molecules_scored": len(
            scoring_canonicals
        ),
        "num_test_associations_scored": len(
            association_rows
        ),
        "pooled": summarize_rows(
            association_rows
        ),
        "macro": macro,
        "per_poi": per_poi,
    }


#new
def compute_real_test_knn_calibration(
    reference_pool: dict[str, Any],
    target_uniprots: list[str],
    k_values: list[int],
) -> dict[str, Any]:
    """
    Compute the same POI-purity metric on the real held-out test
    chemistry, excluding each query molecule from its own neighbor set.
    """
    fingerprints = reference_pool[
        "fingerprints"
    ]
    target_sets = [
        set(targets)
        for targets in reference_pool[
            "target_uniprots_by_molecule"
        ]
    ]

    if len(fingerprints) <= 1:
        raise ValueError(
            "Real-test k-NN calibration requires at least two unique "
            "test molecules."
        )

    k_values = sorted(
        {
            int(k)
            for k in k_values
        }
    )

    if any(k <= 0 for k in k_values):
        raise ValueError(
            "All k-NN k values must be > 0."
        )

    values_by_target: dict[
        str,
        dict[int, list[float]],
    ] = {
        target: {
            k: []
            for k in k_values
        }
        for target in target_uniprots
    }

    pooled_values: dict[
        int,
        list[float],
    ] = {
        k: []
        for k in k_values
    }

    association_count = 0

    for query_index, query_fp in enumerate(
        tqdm(
            fingerprints,
            desc="Real-test k-NN calibration",
            unit="mol",
        )
    ):
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                query_fp,
                fingerprints,
            ),
            dtype=float,
        )

        similarities[
            query_index
        ] = -np.inf

        ranked_indices = np.argsort(
            similarities
        )[
            ::-1
        ]

        valid_targets = [
            target
            for target in target_uniprots
            if target
            in target_sets[query_index]
        ]

        if not valid_targets:
            continue

        for target_uniprot in valid_targets:
            association_count += 1

            for k in k_values:
                effective_k = min(
                    k,
                    len(fingerprints) - 1,
                )
                top_indices = ranked_indices[
                    :effective_k
                ]

                purity = (
                    sum(
                        target_uniprot
                        in target_sets[
                            int(index)
                        ]
                        for index
                        in top_indices
                    )
                    / effective_k
                )

                values_by_target[
                    target_uniprot
                ][k].append(
                    float(purity)
                )
                pooled_values[k].append(
                    float(purity)
                )

    per_poi = {
        target: {
            "num_query_associations": (
                len(
                    values_by_target[
                        target
                    ][k_values[0]]
                )
                if k_values
                else 0
            ),
            **{
                f"knn_purity_k{k}": (
                    _safe_mean(
                        values_by_target[
                            target
                        ][k]
                    )
                )
                for k in k_values
            },
        }
        for target in target_uniprots
    }

    pooled = {
        f"knn_purity_k{k}": (
            _safe_mean(
                pooled_values[k]
            )
        )
        for k in k_values
    }

    macro = {
        f"knn_purity_k{k}": (
            _safe_mean(
                [
                    float(
                        per_poi[target][
                            f"knn_purity_k{k}"
                        ]
                    )
                    for target
                    in target_uniprots
                    if per_poi[target][
                        f"knn_purity_k{k}"
                    ]
                    is not None
                ]
            )
        )
        for k in k_values
    }

    return {
        "num_reference_molecules": len(
            fingerprints
        ),
        "num_query_associations": (
            association_count
        ),
        "pooled": pooled,
        "macro": macro,
        "per_poi": per_poi,
    }


#new
def compute_final_generated_fidelity(
    annotated_generated_records: list[dict[str, Any]],
    reference_pool: dict[str, Any],
    target_uniprots: list[str],
    target_uniprot_field: str,
    k_values: list[int],
    local_top_m: int,
) -> dict[str, Any]:
    """
    Final generated-chemistry conditional-fidelity evaluation.

    The primary generated set is unique by (requested POI, canonical SMILES),
    valid, and novel-to-training. Repeated generations of the same canonical
    molecule under the same requested POI therefore contribute once, while
    the same canonical molecule generated under different POIs can contribute
    once to each conditioning experiment.

    Exact matches to held-out TEST chemistry remain part of the TEST
    neighborhood. If a unique train-novel generated molecule exactly matches
    a held-out TEST molecule, that Tanimoto-1.0 neighbor is retained as a
    legitimate fidelity signal.

    One similarity scan per unique valid train-novel POI/molecule pair is
    reused for:
      - generated requested-POI k-NN purity for all requested k values,
      - local requested-vs-competing POI scores,
      - local Top-1/Top-5 recovery,
      - the POI-by-POI alignment matrix.
    """
    k_values = sorted(
        {
            int(k)
            for k in k_values
        }
    )

    if not k_values:
        raise ValueError(
            "conditional_fidelity.knn_k_values cannot be empty."
        )

    if any(k <= 0 for k in k_values):
        raise ValueError(
            "All k-NN k values must be > 0."
        )

    if local_top_m <= 0:
        raise ValueError(
            "conditional_fidelity.local_top_m must be > 0."
        )

    reference_fingerprints = reference_pool[
        "fingerprints"
    ]
    reference_target_sets = [
        set(targets)
        for targets in reference_pool[
            "target_uniprots_by_molecule"
        ]
    ]
    target_indices: dict[
        str,
        np.ndarray,
    ] = {}

    for target_uniprot in target_uniprots:
        indices = [
            index
            for index, labels
            in enumerate(
                reference_target_sets
            )
            if target_uniprot in labels
        ]

        if not indices:
            raise ValueError(
                "Held-out test reference contains no valid molecules for "
                f"requested POI {target_uniprot}."
            )

        target_indices[
            target_uniprot
        ] = np.asarray(
            indices,
            dtype=int,
        )

    query_records: list[
        dict[str, Any]
    ] = []
    query_mols: list[
        Chem.Mol
    ] = []

    target_set = set(
        target_uniprots
    )
    seen_query_keys: set[
        tuple[str, str]
    ] = set()

    for record in annotated_generated_records:
        if not record.get(
            "is_valid"
        ):
            continue

        if not record.get(
            "is_novel",
            False,
        ):
            continue

        target_uniprot = str(
            record[
                target_uniprot_field
            ]
        ).strip()

        if target_uniprot not in target_set:
            continue

        canonical_smiles = str(
            record[
                "canonical_smiles"
            ]
        )
        query_key = (
            target_uniprot,
            canonical_smiles,
        )

        if query_key in seen_query_keys:
            continue

        seen_query_keys.add(
            query_key
        )

        mol = mol_from_smiles(
            canonical_smiles
        )

        if mol is None:
            continue

        query_records.append(
            record
        )
        query_mols.append(
            mol
        )

    if not query_mols:
        raise ValueError(
            "No unique valid train-novel generated POI/molecule pairs are "
            "available for final conditional-fidelity evaluation."
        )

    query_fingerprints = (
        get_morgan_fingerprints_from_mols(
            mols=query_mols,
            radius=reference_pool[
                "fp_radius"
            ],
            n_bits=reference_pool[
                "fp_n_bits"
            ],
        )
    )

    purity_values: dict[
        str,
        dict[int, list[float]],
    ] = {
        target: {
            k: []
            for k in k_values
        }
        for target in target_uniprots
    }

    local_margins: dict[
        str,
        list[float],
    ] = {
        target: []
        for target in target_uniprots
    }

    requested_scores: dict[
        str,
        list[float],
    ] = {
        target: []
        for target in target_uniprots
    }

    strongest_competitor_scores: dict[
        str,
        list[float],
    ] = {
        target: []
        for target in target_uniprots
    }

    local_top1: dict[
        str,
        list[float],
    ] = {
        target: []
        for target in target_uniprots
    }

    local_top5: dict[
        str,
        list[float],
    ] = {
        target: []
        for target in target_uniprots
    }

    matrix_sums: dict[
        str,
        dict[str, float],
    ] = {
        requested: {
            reference_target: 0.0
            for reference_target
            in target_uniprots
        }
        for requested
        in target_uniprots
    }

    matrix_counts: dict[
        str,
        int,
    ] = {
        target: 0
        for target in target_uniprots
    }

    pooled_purity_values: dict[
        int,
        list[float],
    ] = {
        k: []
        for k in k_values
    }

    pooled_margins: list[float] = []
    pooled_top1: list[float] = []
    pooled_top5: list[float] = []

    for query_fp, record in tqdm(
        zip(
            query_fingerprints,
            query_records,
        ),
        total=len(query_records),
        desc="Generated conditional fidelity",
        unit="mol",
    ):
        requested_target = str(
            record[
                target_uniprot_field
            ]
        ).strip()
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                query_fp,
                reference_fingerprints,
            ),
            dtype=float,
        )

        ranked_indices = np.argsort(
            similarities
        )[
            ::-1
        ]

        purity_by_k: dict[
            int,
            float,
        ] = {}

        for k in k_values:
            effective_k = min(
                k,
                len(reference_fingerprints),
            )
            top_indices = ranked_indices[
                :effective_k
            ]

            purity_by_k[k] = float(
                sum(
                    requested_target
                    in reference_target_sets[
                        int(index)
                    ]
                    for index
                    in top_indices
                )
                / effective_k
            )
            #new
            record[
                f"knn_purity_k{k}"
            ] = purity_by_k[k]

        local_scores: dict[
            str,
            float,
        ] = {}

        for reference_target in target_uniprots:
            reference_indices = target_indices[
                reference_target
            ]

            target_sims = similarities[
                reference_indices
            ]

            effective_m = min(
                local_top_m,
                len(target_sims),
            )

            if effective_m == len(
                target_sims
            ):
                top_values = target_sims
            else:
                top_values = np.partition(
                    target_sims,
                    len(target_sims)
                    - effective_m,
                )[
                    -effective_m:
                ]

            local_scores[
                reference_target
            ] = float(
                np.mean(
                    top_values
                )
            )

        requested_score = local_scores[
            requested_target
        ]

        competing_targets = [
            target
            for target in target_uniprots
            if target != requested_target
        ]

        strongest_competitor = max(
            competing_targets,
            key=lambda target: (
                local_scores[target],
                target,
            ),
        )

        strongest_competitor_score = (
            local_scores[
                strongest_competitor
            ]
        )

        margin = (
            requested_score
            - strongest_competitor_score
        )

        local_ranking = sorted(
            target_uniprots,
            key=lambda target: (
                -local_scores[target],
                target,
            ),
        )

        requested_rank = (
            local_ranking.index(
                requested_target
            )
            + 1
        )

        for k in k_values:
            purity = purity_by_k[k]
            purity_values[
                requested_target
            ][k].append(
                purity
            )
            pooled_purity_values[
                k
            ].append(
                purity
            )

        local_margins[
            requested_target
        ].append(
            float(margin)
        )
        requested_scores[
            requested_target
        ].append(
            float(requested_score)
        )
        strongest_competitor_scores[
            requested_target
        ].append(
            float(
                strongest_competitor_score
            )
        )
        local_top1[
            requested_target
        ].append(
            float(
                requested_rank == 1
            )
        )
        local_top5[
            requested_target
        ].append(
            float(
                requested_rank
                <= min(
                    5,
                    len(target_uniprots),
                )
            )
        )

        pooled_margins.append(
            float(margin)
        )
        pooled_top1.append(
            float(
                requested_rank == 1
            )
        )
        pooled_top5.append(
            float(
                requested_rank
                <= min(
                    5,
                    len(target_uniprots),
                )
            )
        )

        matrix_counts[
            requested_target
        ] += 1

        for reference_target in target_uniprots:
            matrix_sums[
                requested_target
            ][reference_target] += (
                local_scores[
                    reference_target
                ]
            )

    pooled: dict[str, Any] = {
        "num_queries": len(
            pooled_margins
        ),
        "local_margin_mean": (
            _safe_mean(
                pooled_margins
            )
        ),
        "local_margin_median": (
            _safe_median(
                pooled_margins
            )
        ),
        "local_fraction_positive": (
            _safe_mean(
                [
                    float(value > 0.0)
                    for value
                    in pooled_margins
                ]
            )
        ),
        "local_top1_accuracy": (
            _safe_mean(
                pooled_top1
            )
        ),
        "local_top5_accuracy": (
            _safe_mean(
                pooled_top5
            )
        ),
    }

    for k in k_values:
        pooled[
            f"knn_purity_k{k}"
        ] = _safe_mean(
            pooled_purity_values[k]
        )

    per_poi: dict[
        str,
        dict[str, Any],
    ] = {}

    for target_uniprot in target_uniprots:
        target_margins = local_margins[
            target_uniprot
        ]

        target_row: dict[
            str,
            Any,
        ] = {
            "num_queries": len(
                target_margins
            ),
            "local_margin_mean": (
                _safe_mean(
                    target_margins
                )
            ),
            "local_margin_median": (
                _safe_median(
                    target_margins
                )
            ),
            "local_fraction_positive": (
                _safe_mean(
                    [
                        float(value > 0.0)
                        for value
                        in target_margins
                    ]
                )
            ),
            "local_top1_accuracy": (
                _safe_mean(
                    local_top1[
                        target_uniprot
                    ]
                )
            ),
            "local_top5_accuracy": (
                _safe_mean(
                    local_top5[
                        target_uniprot
                    ]
                )
            ),
            "requested_score_mean": (
                _safe_mean(
                    requested_scores[
                        target_uniprot
                    ]
                )
            ),
            "strongest_competitor_score_mean": (
                _safe_mean(
                    strongest_competitor_scores[
                        target_uniprot
                    ]
                )
            ),
        }

        for k in k_values:
            target_row[
                f"knn_purity_k{k}"
            ] = _safe_mean(
                purity_values[
                    target_uniprot
                ][k]
            )

        per_poi[
            target_uniprot
        ] = target_row

    macro_metric_names = [
        *[
            f"knn_purity_k{k}"
            for k in k_values
        ],
        "local_margin_mean",
        "local_margin_median",
        "local_fraction_positive",
        "local_top1_accuracy",
        "local_top5_accuracy",
        "requested_score_mean",
        "strongest_competitor_score_mean",
    ]

    macro = {
        metric_name: _safe_mean(
            [
                float(
                    per_poi[
                        target
                    ][metric_name]
                )
                for target
                in target_uniprots
                if per_poi[
                    target
                ].get(
                    metric_name
                )
                is not None
            ]
        )
        for metric_name
        in macro_metric_names
    }

    matrix: dict[
        str,
        dict[str, float | None],
    ] = {}

    for requested_target in target_uniprots:
        count = matrix_counts[
            requested_target
        ]

        matrix[
            requested_target
        ] = {
            reference_target: (
                matrix_sums[
                    requested_target
                ][reference_target]
                / count
                if count > 0
                else None
            )
            for reference_target
            in target_uniprots
        }

    return {
        "generated_subset": (
            "unique_requested_poi_canonical_valid_and_novel_to_training"
        ),
        "k_values": k_values,
        "local_top_m": int(
            local_top_m
        ),
        "num_reference_molecules": len(
            reference_fingerprints
        ),
        "pooled": pooled,
        "macro": macro,
        "per_poi": per_poi,
        "matrix": matrix,
    }


#new
def _build_conditional_per_poi_rows(
    target_uniprots: list[str],
    poi_name_lookup: dict[str, str],
    nll_result: dict[str, Any],
    generated_fidelity: dict[str, Any],
    real_knn: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[
        dict[str, Any]
    ] = []

    matrix = generated_fidelity[
        "matrix"
    ]

    for target_uniprot in target_uniprots:
        row: dict[str, Any] = {
            "target_uniprot": target_uniprot,
            "poi_norm": poi_name_lookup.get(
                target_uniprot,
                target_uniprot,
            ),
        }

        nll_metrics = nll_result[
            "per_poi"
        ][target_uniprot]

        row.update(
            nll_metrics
        )

        for k_metric, value in real_knn[
            "per_poi"
        ][target_uniprot].items():
            row[
                f"real_test_{k_metric}"
            ] = value

        generated_metrics = generated_fidelity[
            "per_poi"
        ][target_uniprot]

        for metric_name, value in generated_metrics.items():
            row[
                f"generated_novel_{metric_name}"
            ] = value

        diagonal = matrix[
            target_uniprot
        ][target_uniprot]

        competitors = [
            (
                other_target,
                matrix[
                    target_uniprot
                ][other_target],
            )
            for other_target
            in target_uniprots
            if other_target
            != target_uniprot
            and matrix[
                target_uniprot
            ][other_target]
            is not None
        ]

        if (
            diagonal is not None
            and competitors
        ):
            strongest_target, strongest_score = max(
                competitors,
                key=lambda item: (
                    float(item[1]),
                    item[0],
                ),
            )

            competitor_scores = [
                float(score)
                for _, score
                in competitors
            ]

            row[
                "alignment_novel_diagonal"
            ] = diagonal
            row[
                "alignment_novel_strongest_competitor_target_uniprot"
            ] = strongest_target
            row[
                "alignment_novel_strongest_competitor_poi_norm"
            ] = poi_name_lookup.get(
                strongest_target,
                strongest_target,
            )
            row[
                "alignment_novel_dmax"
            ] = float(
                diagonal
                - float(strongest_score)
            )
            row[
                "alignment_novel_dmean"
            ] = float(
                diagonal
                - np.mean(
                    np.asarray(
                        competitor_scores,
                        dtype=float,
                    )
                )
            )
        else:
            row[
                "alignment_novel_diagonal"
            ] = diagonal
            row[
                "alignment_novel_strongest_competitor_target_uniprot"
            ] = None
            row[
                "alignment_novel_strongest_competitor_poi_norm"
            ] = None
            row[
                "alignment_novel_dmax"
            ] = None
            row[
                "alignment_novel_dmean"
            ] = None

        rows.append(
            row
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated PROTAC SMILES."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Optional key=value overrides.",
    )

    args = parser.parse_args()

    cfg = apply_cli_overrides(
        load_yaml(args.config),
        args.override,
    )

    eval_cfg = cfg["evaluation"]

    #new
    conditional_cfg = eval_cfg.get(
        "conditional_fidelity",
        {},
    )
    conditional_enabled = bool(
        conditional_cfg.get(
            "enabled",
            False,
        )
    )

    output_dir = ensure_dir(eval_cfg["output_dir"])

    save_yaml(
        cfg,
        output_dir / "evaluate.resolved.yaml",
    )

    seed = eval_cfg.get("seed", 42)
    set_seed(seed)

    if eval_cfg.get("suppress_rdkit_warnings", True):
        suppress_rdkit_warnings()

    generated_jsonl = eval_cfg["generated_jsonl"]

    generated_smiles_field = eval_cfg.get(
        "generated_smiles_field",
        "raw_smiles",
    )

    #new
    # TRAIN is the only reference for exact novelty, reference matching,
    # and scaffold novelty for both unconditional and conditional evaluation.
    #new
    reference_paths = get_reference_paths(eval_cfg)

    if len(reference_paths) == 0:
        #new
        raise ValueError(
            #new
            "No training reference provided. Add "
            #new
            "evaluation.reference_train_jsonl to the YAML."
            #new
        )

    reference_smiles_field = eval_cfg.get(
        "reference_smiles_field",
        "smiles",
    )

    fp_radius = eval_cfg.get("fp_radius", 2)
    fp_n_bits = eval_cfg.get("fp_n_bits", 2048)
    max_diversity_samples = eval_cfg.get(
        "max_diversity_samples",
        1000,
    )

    max_invalid_examples = eval_cfg.get(
        "max_invalid_examples",
        50,
    )

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

    (
        annotated_records,
        valid_canonical_all,
        unique_valid_mols_ordered,
    ) = annotate_generated_records(
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

    unique_valid_ordered = list(
        dict.fromkeys(valid_canonical_all)
    )

    unique_novel_valid_ordered = [
        smiles
        for smiles in unique_valid_ordered
        if smiles not in reference_set
    ]

    invalid_examples = [
        record
        for record in annotated_records
        if not record["is_valid"]
    ][:max_invalid_examples]

    write_json(
        metrics,
        output_dir / "metrics.json",
    )

    write_smi(
        output_dir / "valid_all.smi",
        valid_canonical_all,
    )

    write_smi(
        output_dir / "valid_unique.smi",
        unique_valid_ordered,
    )

    write_smi(
        output_dir / "valid_unique_novel.smi",
        unique_novel_valid_ordered,
    )

    write_jsonl(
        invalid_examples,
        output_dir / "invalid_examples.jsonl",
    )

    #new
    # ------------------------------------------------------------
    # Optional final standalone conditional-fidelity evaluation.
    # If absent/disabled, all original unconditional behavior above
    # remains unchanged.
    # ------------------------------------------------------------
    if conditional_enabled:
        test_jsonl = eval_cfg[
            "test_jsonl"
        ]
        test_smiles_field = eval_cfg.get(
            "test_smiles_field",
            "smiles",
        )
        target_uniprot_field = eval_cfg.get(
            "target_uniprot_field",
            "target_uniprot",
        )
        poi_name_field = eval_cfg.get(
            "poi_name_field",
            "poi_norm",
        )

        test_records = read_jsonl(
            test_jsonl
        )

        #new
        poi_name_lookup = {
            record[target_uniprot_field]:
            record[poi_name_field]
            for record in test_records
        }

        #new
        # The final conditional-evaluation panel is defined by the POIs
        # requested in the generated final experiment. TEST is then
        # restricted to chemistry for those same POIs.
        target_uniprots = list(
            dict.fromkeys(
                str(
                    record[
                        target_uniprot_field
                    ]
                ).strip()
                for record
                in generated_records
            )
        )

        if not target_uniprots:
            raise ValueError(
                "No target UniProt IDs were found in generated records."
            )

        target_uniprot_set = set(
            target_uniprots
        )

        conditional_test_records = [
            record
            for record
            in test_records
            if str(
                record[
                    target_uniprot_field
                ]
            ).strip()
            in target_uniprot_set
        ]

        conditional_test_target_set = {
            str(
                record[
                    target_uniprot_field
                ]
            ).strip()
            for record
            in conditional_test_records
        }

        missing_test_targets = sorted(
            target_uniprot_set
            - conditional_test_target_set
        )

        if missing_test_targets:
            raise ValueError(
                "Held-out TEST contains no records for generated POIs: "
                f"{missing_test_targets}."
            )

        print(
            "[INFO] Final conditional fidelity enabled."
        )
        print(
            f"[INFO] Held-out test JSONL: {test_jsonl}"
        )
        print(
            f"[INFO] Conditional POIs: {len(target_uniprots)}"
        )
        print(
            f"[INFO] Held-out test records for generated POIs: "
            f"{len(conditional_test_records)}"
        )

        #new
        # Build TEST chemistry once and reuse the same reference object for
        # NLL, real-test k-NN, and generated conditional-fidelity metrics.
        test_reference_pool = (
            _build_test_reference_pool(
                records=conditional_test_records,
                smiles_field=(
                    test_smiles_field
                ),
                target_uniprot_field=(
                    target_uniprot_field
                ),
                fp_radius=fp_radius,
                fp_n_bits=fp_n_bits,
            )
        )

        #new
        # Reuse the molecule objects created during global generated-sample
        # annotation. Per-POI metrics only recompute POI-local counts.
        generated_mol_by_canonical = dict(
            zip(
                unique_valid_ordered,
                unique_valid_mols_ordered,
            )
        )

        per_poi_generation_rows = (
            _compute_per_poi_generation_metrics(
                annotated_generated_records=(
                    annotated_records
                ),
                mol_by_canonical=(
                    generated_mol_by_canonical
                ),
                target_uniprot_field=(
                    target_uniprot_field
                ),
                poi_name_lookup=(
                    poi_name_lookup
                ),
                reference_set=reference_set,
                reference_scaffold_set=(
                    reference_scaffold_set
                ),
                fp_radius=fp_radius,
                fp_n_bits=fp_n_bits,
                max_diversity_samples=(
                    max_diversity_samples
                ),
                seed=seed,
            )
        )

        write_jsonl(
            per_poi_generation_rows,
            output_dir
            / "per_poi_metrics.jsonl",
        )

        per_poi_generation_summary = (
            _summarize_per_poi_generation_metrics(
                per_poi_generation_rows
            )
        )

        #new
        # Load the selected conditional checkpoint only for the final
        # multi-condition NLL evaluation with known valid alternative
        # POIs excluded from wrong-condition comparisons.
        from tokenizer import SmilesTokenizer  #new
        from transformerv2 import GPTForCausalLM  #new

        device = torch.device(
            conditional_cfg.get(
                "device",
                "cuda",
            )
        )

        tokenizer = SmilesTokenizer.load(
            cfg["tokenizer"]["path"]
        )

        model = GPTForCausalLM.from_pretrained(
            cfg["model"]["path"]
        )
        model.to(device)
        model.eval()

        poi_embedding_payload = torch.load(
            cfg["conditioning"]["poi_embeddings_path"],
            map_location="cpu",
        )
        poi_embeddings_by_uniprot = (
            poi_embedding_payload["embeddings_by_uniprot"]
        )

        nll_result = (
            compute_full_condition_nll_metrics(
                model=model,
                tokenizer=tokenizer,
                test_reference_pool=(
                    test_reference_pool
                ),
                poi_embeddings_by_uniprot=(
                    poi_embeddings_by_uniprot
                ),
                target_uniprots=(
                    target_uniprots
                ),
                batch_size=int(
                    conditional_cfg.get(
                        "nll_batch_size",
                        64,
                    )
                ),
                device=device,
                max_seq_len=int(
                    model.config.max_seq_len
                ),
            )
        )

        #new
        # Model is no longer needed after NLL scoring; free GPU memory
        # before the fingerprint-based evaluations.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        #new
        k_values = [
            int(k)
            for k in conditional_cfg.get(
                "knn_k_values",
                [5, 10, 20],
            )
        ]

        real_knn_result = (
            compute_real_test_knn_calibration(
                reference_pool=(
                    test_reference_pool
                ),
                target_uniprots=(
                    target_uniprots
                ),
                k_values=k_values,
            )
        )

        generated_fidelity = (
            compute_final_generated_fidelity(
                annotated_generated_records=(
                    annotated_records
                ),
                reference_pool=(
                    test_reference_pool
                ),
                target_uniprots=(
                    target_uniprots
                ),
                target_uniprot_field=(
                    target_uniprot_field
                ),
                k_values=k_values,
                local_top_m=int(
                    conditional_cfg.get(
                        "local_top_m",
                        5,
                    )
                ),
            )
        )

        conditional_output_dir = ensure_dir(
            output_dir
            / "conditional_fidelity"
        )

        #new
        alignment_column_labels = [
            f"{poi_name_lookup.get(target, target)}|{target}"
            for target in target_uniprots
        ]

        #new
        alignment_rows = [
            [
                requested_target,
                poi_name_lookup.get(
                    requested_target,
                    requested_target,
                ),
                *[
                    generated_fidelity[
                        "matrix"
                    ][requested_target][reference_target]
                    for reference_target
                    in target_uniprots
                ],
            ]
            for requested_target
            in target_uniprots
        ]

        #new
        alignment_df = pd.DataFrame(
            alignment_rows,
            columns=[
                "generated_target_uniprot",
                "generated_poi_norm",
                *alignment_column_labels,
            ],
        )

        #new
        alignment_df.to_csv(
            conditional_output_dir
            / "alignment_matrix_novel.csv",
            index=False,
        )

        conditional_per_poi_rows = (
            _build_conditional_per_poi_rows(
                target_uniprots=(
                    target_uniprots
                ),
                poi_name_lookup=(
                    poi_name_lookup
                ),
                nll_result=nll_result,
                generated_fidelity=(
                    generated_fidelity
                ),
                real_knn=(
                    real_knn_result
                ),
            )
        )

        write_jsonl(
            conditional_per_poi_rows,
            conditional_output_dir
            / "per_poi_metrics.jsonl",
        )

        conditional_summary = {
            "num_pois": len(
                target_uniprots
            ),
            "targets": [
                {
                    "target_uniprot": target,
                    "poi_norm": (
                        poi_name_lookup.get(
                            target,
                            target,
                        )
                    ),
                }
                for target
                in target_uniprots
            ],
            "data_roles": {
                "train_reference_jsonls": (
                    reference_paths
                ),
                "held_out_test_jsonl": (
                    test_jsonl
                ),
                "novelty_reference": (
                    "train"
                ),
                "conditional_fidelity_reference": (
                    "test"
                ),
            },
            "aggregation": {
                "level_1_pooled": (
                    "global_library_level; cross_poi_duplicates_are_collapsed"
                ),
                "level_2_macro": (
                    "equal_weight_per_poi"
                ),
                "level_2_weighted": (
                    "ratio_of_summed_per_poi_numerators_and_denominators; "
                    "cross_poi_duplicates_are_not_collapsed"
                ),
                "level_3_pooled": (
                    "query_or_association_weighted"
                ),
                "level_3_macro": (
                    "equal_weight_per_poi"
                ),
            },
            #new
            "per_poi_generation_summary": (
                per_poi_generation_summary
            ),
            "full_condition_nll": {
                "num_unique_test_molecules_scored": (
                    nll_result[
                        "num_unique_test_molecules_scored"
                    ]
                ),
                "num_test_associations_scored": (
                    nll_result[
                        "num_test_associations_scored"
                    ]
                ),
                "pooled": nll_result[
                    "pooled"
                ],
                "macro": nll_result[
                    "macro"
                ],
            },
            "real_test_knn_calibration": {
                "num_reference_molecules": (
                    real_knn_result[
                        "num_reference_molecules"
                    ]
                ),
                "num_query_associations": (
                    real_knn_result[
                        "num_query_associations"
                    ]
                ),
                "pooled": real_knn_result[
                    "pooled"
                ],
                "macro": real_knn_result[
                    "macro"
                ],
            },
            "generated_conditional_fidelity": {
                "primary_generated_subset": (
                    generated_fidelity[
                        "generated_subset"
                    ]
                ),
                "k_values": (
                    generated_fidelity[
                        "k_values"
                    ]
                ),
                "local_top_m": (
                    generated_fidelity[
                        "local_top_m"
                    ]
                ),
                "num_reference_molecules": (
                    generated_fidelity[
                        "num_reference_molecules"
                    ]
                ),
                "pooled": (
                    generated_fidelity[
                        "pooled"
                    ]
                ),
                "macro": (
                    generated_fidelity[
                        "macro"
                    ]
                ),
            },
        }

        write_json(
            conditional_summary,
            conditional_output_dir
            / "summary.json",
        )

        print(
            "[DONE] Final conditional-fidelity evaluation finished."
        )
        print(
            f"[DONE] Conditional outputs: {conditional_output_dir}"
        )
    

    write_jsonl(
        annotated_records,
        output_dir / "annotated_generated_f.jsonl",
    )
    print("[DONE] Evaluation finished.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()