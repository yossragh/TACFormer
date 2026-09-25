from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem

from chemistry import (
    canonicalize_from_mol,
    cap_mapped_dummy_with_hydrogen,
    internal_diversity_from_mols,
    mol_from_smiles,
    suppress_rdkit_warnings,
)
from utils import ensure_dir, read_jsonl, write_json, write_jsonl


def _resolve_path(
    project_root: Path,
    value: str | Path,
) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _extract_poi_warhead(
    prediction: Any,
) -> str | None:
    if not isinstance(prediction, str) or not prediction.strip():
        return None

    parts = prediction.strip().split(".")

    if len(parts) != 3:
        return None

    warhead = parts[2].strip()
    return warhead if warhead else None


def _prepare_warhead(
    raw_warhead: str,
    keep_attachment_atoms: bool,
    attachment_atom_map_num: int,
) -> tuple[str | None, Chem.Mol | None, str]:
    mol = mol_from_smiles(raw_warhead)

    if mol is None:
        return None, None, "rdkit_parse_failed"

    if keep_attachment_atoms:
        processed_mol = mol
        status = "attachment_aware"
    else:
        processed_mol = cap_mapped_dummy_with_hydrogen(
            mol,
            atom_map_num=attachment_atom_map_num,
        )

        if processed_mol is None:
            return None, None, "parent_conversion_failed"

        status = "parent_warhead"

    canonical = canonicalize_from_mol(processed_mol)

    if canonical is None:
        return None, None, "canonicalization_failed"

    return canonical, processed_mol, status


def _build_train_warhead_sets(
    train_records: list[dict[str, Any]],
    target_ids: list[str],
    target_field: str,
    split_field: str,
    keep_attachment_atoms: bool,
    attachment_atom_map_num: int,
) -> tuple[
    dict[str, set[str]],
    dict[str, dict[str, int | float | None]],
]:
    target_id_set = set(target_ids)

    train_warheads = {
        target_id: set()
        for target_id in target_ids
    }

    qc: dict[str, dict[str, int | float | None]] = {
        target_id: {
            "train_rows": 0,
            "train_splitter_predictions": 0,
            "train_processed_warheads": 0,
            "train_warhead_processing_coverage": None,
        }
        for target_id in target_ids
    }

    for record in train_records:
        target_id = record.get(target_field)

        if target_id not in target_id_set:
            continue

        qc[target_id]["train_rows"] += 1

        raw_warhead = _extract_poi_warhead(
            record.get(split_field)
        )

        if raw_warhead is None:
            continue

        qc[target_id]["train_splitter_predictions"] += 1

        canonical_warhead, _, _ = _prepare_warhead(
            raw_warhead=raw_warhead,
            keep_attachment_atoms=keep_attachment_atoms,
            attachment_atom_map_num=attachment_atom_map_num,
        )

        if canonical_warhead is None:
            continue

        qc[target_id]["train_processed_warheads"] += 1
        train_warheads[target_id].add(canonical_warhead)

    for target_id in target_ids:
        n_predictions = int(
            qc[target_id]["train_splitter_predictions"]
        )
        n_processed = int(
            qc[target_id]["train_processed_warheads"]
        )

        qc[target_id]["train_warhead_processing_coverage"] = (
            n_processed / n_predictions
            if n_predictions > 0
            else None
        )

    return train_warheads, qc


def _collect_unique_valid_generated(
    generated_records: list[dict[str, Any]],
    target_ids: list[str],
    target_field: str,
    smiles_field: str,
    valid_field: str,
) -> dict[str, dict[str, dict[str, Any]]]:

    target_id_set = set(target_ids)

    grouped = {
        target_id: {}
        for target_id in target_ids
    }

    for record in generated_records:
        target_id = record.get(target_field)

        if target_id not in target_id_set:
            continue

        if not bool(record.get(valid_field, False)):
            continue

        canonical_smiles = record.get(smiles_field)

        if not isinstance(canonical_smiles, str):
            continue

        canonical_smiles = canonical_smiles.strip()

        if not canonical_smiles:
            continue

        grouped[target_id].setdefault(
            canonical_smiles,
            record,
        )

    return grouped


def _compute_warhead_diversity(
    unique_warhead_mols: list[Chem.Mol],
    radius: int,
    n_bits: int,
) -> float | None:
    if len(unique_warhead_mols) < 2:
        return None

    return internal_diversity_from_mols(
        mols=unique_warhead_mols,
        radius=radius,
        n_bits=n_bits,
        max_samples=len(unique_warhead_mols),
        seed=42,
    )


def _analyze_one_poi(
    target_id: str,
    poi_name: str,
    generated_unique: dict[str, dict[str, Any]],
    train_warheads: set[str],
    split_field: str,
    keep_attachment_atoms: bool,
    attachment_atom_map_num: int,
    radius: int,
    n_bits: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Compute all warhead metrics for one POI.
    """
    n_unique_valid_protacs = len(generated_unique)

    generated_warheads: dict[str, Chem.Mol] = {}
    assignment_rows: list[dict[str, Any]] = []

    n_splitter_predictions = 0
    n_processed_warheads = 0

    for canonical_protac, record in generated_unique.items():
        raw_warhead = _extract_poi_warhead(
            record.get(split_field)
        )

        if raw_warhead is None:
            assignment_rows.append(
                {
                    "poi": poi_name,
                    "target_uniprot": target_id,
                    "canonical_protac": canonical_protac,
                    "raw_warhead": None,
                    "metric_warhead": None,
                    "warhead_processing_status": "missing_or_invalid_split_prediction",
                    "warhead_in_train_for_same_poi": None,
                    "splitter_model_name": record.get("splitter_model_name"),
                    "splitter_n_flags": record.get("splitter_n_flags"),
                    "splitter_review_reasons": record.get(
                        "splitter_review_reasons"
                    ),
                }
            )
            continue

        n_splitter_predictions += 1

        (
            metric_warhead,
            metric_mol,
            processing_status,
        ) = _prepare_warhead(
            raw_warhead=raw_warhead,
            keep_attachment_atoms=keep_attachment_atoms,
            attachment_atom_map_num=attachment_atom_map_num,
        )

        if metric_warhead is None or metric_mol is None:
            assignment_rows.append(
                {
                    "poi": poi_name,
                    "target_uniprot": target_id,
                    "canonical_protac": canonical_protac,
                    "raw_warhead": raw_warhead,
                    "metric_warhead": None,
                    "warhead_processing_status": processing_status,
                    "warhead_in_train_for_same_poi": None,
                    "splitter_model_name": record.get("splitter_model_name"),
                    "splitter_n_flags": record.get("splitter_n_flags"),
                    "splitter_review_reasons": record.get(
                        "splitter_review_reasons"
                    ),
                }
            )
            continue

        n_processed_warheads += 1

        generated_warheads.setdefault(
            metric_warhead,
            metric_mol,
        )

        assignment_rows.append(
            {
                "poi": poi_name,
                "target_uniprot": target_id,
                "canonical_protac": canonical_protac,
                "raw_warhead": raw_warhead,
                "metric_warhead": metric_warhead,
                "warhead_processing_status": processing_status,
                "warhead_in_train_for_same_poi": (
                    metric_warhead in train_warheads
                ),
                "splitter_model_name": record.get("splitter_model_name"),
                "splitter_n_flags": record.get("splitter_n_flags"),
                "splitter_review_reasons": record.get(
                    "splitter_review_reasons"
                ),
            }
        )

    splitter_prediction_coverage = (
        n_splitter_predictions / n_unique_valid_protacs
        if n_unique_valid_protacs > 0
        else None
    )

    warhead_processing_coverage = (
        n_processed_warheads / n_splitter_predictions
        if n_splitter_predictions > 0
        else None
    )

    n_unique_warheads = len(generated_warheads)

    warhead_uniqueness = (
        n_unique_warheads / n_processed_warheads
        if n_processed_warheads > 0
        else None
    )

    warhead_diversity = _compute_warhead_diversity(
        unique_warhead_mols=list(
            generated_warheads.values()
        ),
        radius=radius,
        n_bits=n_bits,
    )

    novel_warheads = (
        set(generated_warheads)
        - train_warheads
    )

    n_novel_warheads = len(novel_warheads)

    warhead_novelty = (
        n_novel_warheads / n_unique_warheads
        if n_unique_warheads > 0
        else None
    )

    metrics = {
        "poi": poi_name,
        "target_uniprot": target_id,
        "unique_valid_protacs": n_unique_valid_protacs,
        "splitter_predictions": n_splitter_predictions,
        "splitter_prediction_coverage": splitter_prediction_coverage,
        "processed_warheads": n_processed_warheads,
        "warhead_processing_coverage": warhead_processing_coverage,
        "unique_warheads": n_unique_warheads,
        "warhead_uniqueness": warhead_uniqueness,
        "warhead_diversity": warhead_diversity,
        "train_unique_warheads": len(train_warheads),
        "novel_unique_warheads": n_novel_warheads,
        "warhead_novelty": warhead_novelty,
    }

    return metrics, assignment_rows


def run_analysis(
    cfg: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    """
    Run the standalone per-POI warhead analysis.

    This script does not modify or call evaluate.py.
    """
    suppress_rdkit_warnings()

    data_cfg = cfg["data"]
    fields_cfg = cfg["fields"]
    warhead_cfg = cfg["warhead"]
    fingerprint_cfg = cfg["fingerprint"]
    output_cfg = cfg["output"]

    target_to_name = dict(cfg["targets"])
    target_ids = list(target_to_name)

    train_path = _resolve_path(
        project_root,
        data_cfg["train_jsonl"],
    )
    generated_path = _resolve_path(
        project_root,
        data_cfg["generated_jsonl"],
    )
    output_dir = ensure_dir(
        _resolve_path(
            project_root,
            output_cfg["dir"],
        )
    )

    train_records = read_jsonl(train_path)
    generated_records = read_jsonl(generated_path)

    keep_attachment_atoms = bool(
        warhead_cfg["keep_attachment_atoms"]
    )
    attachment_atom_map_num = int(
        warhead_cfg.get(
            "attachment_atom_map_num",
            1,
        )
    )
    radius = int(fingerprint_cfg["radius"])
    n_bits = int(fingerprint_cfg["n_bits"])

    train_warhead_sets, train_qc = _build_train_warhead_sets(
        train_records=train_records,
        target_ids=target_ids,
        target_field=fields_cfg["target_uniprot"],
        split_field=fields_cfg["split_prediction"],
        keep_attachment_atoms=keep_attachment_atoms,
        attachment_atom_map_num=attachment_atom_map_num,
    )

    generated_by_target = _collect_unique_valid_generated(
        generated_records=generated_records,
        target_ids=target_ids,
        target_field=fields_cfg["target_uniprot"],
        smiles_field=fields_cfg["generated_smiles"],
        valid_field=fields_cfg["generated_valid"],
    )

    metrics_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []

    for target_id in target_ids:
        metrics, assignments = _analyze_one_poi(
            target_id=target_id,
            poi_name=target_to_name[target_id],
            generated_unique=generated_by_target[target_id],
            train_warheads=train_warhead_sets[target_id],
            split_field=fields_cfg["split_prediction"],
            keep_attachment_atoms=keep_attachment_atoms,
            attachment_atom_map_num=attachment_atom_map_num,
            radius=radius,
            n_bits=n_bits,
        )

        metrics.update(train_qc[target_id])

        metrics_rows.append(metrics)
        assignment_rows.extend(assignments)

    metrics_df = pd.DataFrame(metrics_rows)

    per_poi_path = output_dir / output_cfg["per_poi_jsonl"]
    assignments_path = output_dir / output_cfg["details_jsonl"]
    summary_path = output_dir / output_cfg["summary_json"]

    write_jsonl(
        metrics_rows,
        per_poi_path,
    )

    write_jsonl(
        assignment_rows,
        assignments_path,
    )

    macro = {
        metric: float(metrics_df[metric].mean())
        for metric in [
            "splitter_prediction_coverage",
            "warhead_processing_coverage",
            "warhead_uniqueness",
            "warhead_diversity",
            "warhead_novelty",
            "train_warhead_processing_coverage",
        ]
    }

    representation = (
        "attachment_aware_splitter_warhead"
        if keep_attachment_atoms
        else "hydrogen_capped_parent_warhead"
    )

    summary = {
        "num_pois": len(target_ids),
        "targets": target_to_name,
        "warhead_representation": representation,
        "keep_attachment_atoms": keep_attachment_atoms,
        "attachment_atom_map_num": attachment_atom_map_num,
        "definitions": {
            "splitter_prediction_coverage": (
                "unique valid generated PROTACs with a valid 3-component "
                "PROTAC-Splitter prediction / unique valid generated PROTACs"
            ),
            "warhead_processing_coverage": (
                "generated POI warheads successfully converted into the metric "
                "representation / valid 3-component splitter predictions"
            ),
            "warhead_uniqueness": (
                "unique processed generated POI warheads / processed generated "
                "POI warheads, computed independently per POI"
            ),
            "warhead_diversity": (
                "mean pairwise Morgan-Tanimoto distance among unique processed "
                "generated warheads from the same POI"
            ),
            "warhead_novelty": (
                "unique processed generated warheads absent from processed TRAIN "
                "warheads for the same POI / unique processed generated warheads"
            ),
        },
        "fingerprint": {
            "type": "Morgan",
            "radius": radius,
            "n_bits": n_bits,
        },
        "macro": macro,
    }

    write_json(
        summary,
        summary_path,
    )

    display_df = metrics_df[
        [
            "poi",
            "target_uniprot",
            "unique_valid_protacs",
            "warhead_processing_coverage",
            "unique_warheads",
            "warhead_uniqueness",
            "warhead_diversity",
            "warhead_novelty",
        ]
    ].copy()

    for column in [
        "warhead_processing_coverage",
        "warhead_uniqueness",
        "warhead_novelty",
    ]:
        display_df[column] = display_df[column].map(
            lambda value: (
                f"{100.0 * value:.2f}%"
                if pd.notna(value)
                else "NA"
            )
        )

    display_df["warhead_diversity"] = (
        display_df["warhead_diversity"].map(
            lambda value: (
                f"{value:.3f}"
                if pd.notna(value)
                else "NA"
            )
        )
    )

    print(
        f"\n[INFO] Warhead representation: {representation}"
    )

    print("\n[RESULT] Per-POI warhead analysis")
    print(display_df.to_string(index=False))

    print("\n[RESULT] Macro")
    print(
        f"  Warhead processing coverage: "
        f"{100.0 * macro['warhead_processing_coverage']:.2f}%"
    )
    print(
        f"  Warhead uniqueness:          "
        f"{100.0 * macro['warhead_uniqueness']:.2f}%"
    )
    print(
        f"  Warhead diversity:           "
        f"{macro['warhead_diversity']:.3f}"
    )
    print(
        f"  Warhead novelty:             "
        f"{100.0 * macro['warhead_novelty']:.2f}%"
    )

    print(f"\n[DONE] Per-POI JSONL: {per_poi_path}")
    print(f"[DONE] Assignment JSONL: {assignments_path}")
    print(f"[DONE] Summary JSON: {summary_path}")

    return {
        "metrics": metrics_df,
        "assignments": assignment_rows,
        "summary": summary,
    }
