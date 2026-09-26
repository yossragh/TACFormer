#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("unseen_poi_selection"),
    )

    p.add_argument("--poi-col", default="poi_norm")
    p.add_argument("--uniprot-col", default="target_uniprot")
    p.add_argument("--smiles-col", default="smiles_std")

    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--min-molecules", type=int, default=50)
    p.add_argument("--max-total-removed", type=int, default=150)

    p.add_argument("--report-top-pairs", type=int, default=10)
    p.add_argument("--report-neighbors", type=int, default=5)

    return p.parse_args()


def load_embeddings(
    path: Path,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, str],
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {path}"
        )

    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        payload = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(payload, dict):
        raise TypeError(
            "Expected the .pt file to contain "
            "a dictionary payload."
        )

    if "embeddings_by_uniprot" not in payload:
        raise KeyError(
            "Missing 'embeddings_by_uniprot' "
            "in embedding payload."
        )

    raw = payload["embeddings_by_uniprot"]

    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "'embeddings_by_uniprot' is empty "
            "or not a dictionary."
        )

    embeddings: Dict[str, torch.Tensor] = {}

    for uniprot, vector in raw.items():
        u = str(uniprot).strip()

        v = torch.as_tensor(
            vector,
            dtype=torch.float32,
        ).flatten()

        if v.numel() == 0:
            raise ValueError(
                f"Empty embedding for {u}"
            )

        # Your saved ESM-2 embeddings are not
        # normalized. Normalize here for cosine
        # similarity calculations.
        embeddings[u] = F.normalize(
            v,
            p=2,
            dim=0,
        )

    poi_by_uniprot: Dict[str, str] = {}

    metadata = payload.get(
        "metadata_by_uniprot",
        {},
    )

    if isinstance(metadata, dict):
        for uniprot, info in metadata.items():

            if (
                isinstance(info, dict)
                and info.get("poi_norm")
            ):
                poi_by_uniprot[
                    str(uniprot).strip()
                ] = str(
                    info["poi_norm"]
                ).strip()

    return embeddings, poi_by_uniprot


def validate_dataset(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:

    required = [
        args.poi_col,
        args.uniprot_col,
        args.smiles_col,
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing required columns: {missing}"
        )

    work = df[required].copy()

    for col in required:
        work[col] = (
            work[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    bad = {
        col: int(
            (work[col] == "").sum()
        )
        for col in required
        if int(
            (work[col] == "").sum()
        ) > 0
    }

    if bad:
        raise ValueError(
            f"Missing/blank required values: {bad}"
        )

    # Dataset is already clean.
    # Do NOT canonicalize smiles_std again.
    #
    # Only remove completely identical rows
    # for counting purposes.
    return (
        work
        .drop_duplicates()
        .reset_index(drop=True)
    )


def nearest_training_neighbors(
    query_uniprot: str,
    training_uniprots: Iterable[str],
    embeddings: Dict[str, torch.Tensor],
    poi_by_uniprot: Dict[str, str],
    top_k: int,
) -> List[
    Tuple[str, str, float, float]
]:

    query = embeddings[query_uniprot]

    rows = []

    for uniprot in training_uniprots:

        similarity = float(
            torch.dot(
                query,
                embeddings[uniprot],
            ).item()
        )

        similarity = max(
            -1.0,
            min(1.0, similarity),
        )

        distance = 1.0 - similarity

        rows.append(
            (
                uniprot,
                poi_by_uniprot.get(
                    uniprot,
                    "",
                ),
                similarity,
                distance,
            )
        )

    rows.sort(
        key=lambda x: x[2],
        reverse=True,
    )

    return rows[:top_k]


def main() -> None:

    args = parse_args()

    if args.top_n < 1:
        raise ValueError(
            "--top-n must be >= 1"
        )

    if args.min_molecules < 1:
        raise ValueError(
            "--min-molecules must be >= 1"
        )

    if args.max_total_removed < 2:
        raise ValueError(
            "--max-total-removed must be >= 2"
        )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        args.input
    )

    df = validate_dataset(
        df,
        args,
    )

    # --------------------------------------------------------
    # Load ESM-2 embeddings
    # --------------------------------------------------------

    embeddings, pt_poi_by_uniprot = (
        load_embeddings(
            args.embeddings
        )
    )

    # UniProt -> POI lookup from dataset.
    dataset_poi_by_uniprot = (
        df[
            [
                args.uniprot_col,
                args.poi_col,
            ]
        ]
        .drop_duplicates()
        .groupby(
            args.uniprot_col
        )[args.poi_col]
        .agg(
            lambda x:
            "|".join(
                sorted(
                    set(x)
                )
            )
        )
        .to_dict()
    )

    # Prefer the POI metadata stored with
    # the ESM embeddings if available.
    poi_by_uniprot = dict(
        dataset_poi_by_uniprot
    )

    poi_by_uniprot.update(
        pt_poi_by_uniprot
    )

    # --------------------------------------------------------
    # Make sure every dataset target has an embedding
    # --------------------------------------------------------

    dataset_uniprots = sorted(
        set(
            df[
                args.uniprot_col
            ]
        )
    )

    missing_embeddings = [
        u
        for u in dataset_uniprots
        if u not in embeddings
    ]

    if missing_embeddings:

        preview = ", ".join(
            missing_embeddings[:20]
        )

        raise ValueError(
            "The embedding file does not cover "
            "every target_uniprot in the dataset. "
            f"Missing {len(missing_embeddings)} "
            f"accession(s): {preview}"
        )

    # --------------------------------------------------------
    # Count unique molecules per POI
    # --------------------------------------------------------

    poi_smiles = (
        df[
            [
                args.poi_col,
                args.smiles_col,
            ]
        ]
        .drop_duplicates()
    )

    molecule_counts = (
        poi_smiles
        .groupby(
            args.poi_col
        )[args.smiles_col]
        .nunique()
        .sort_values(
            ascending=False
        )
    )

    if len(molecule_counts) <= args.top_n:
        raise ValueError(
            f"Dataset has only "
            f"{len(molecule_counts)} POIs; "
            f"cannot reserve top "
            f"{args.top_n}."
        )

    # --------------------------------------------------------
    # Current top 20
    #
    # These are NOT candidates for unseen POI selection.
    # They remain exactly as they are in the existing
    # seen-POI experiment.
    # --------------------------------------------------------

    top_pois = list(
        molecule_counts
        .head(args.top_n)
        .index
    )

    top_poi_set = set(
        top_pois
    )

    # --------------------------------------------------------
    # Determine whether a molecular structure appears
    # under more than one POI.
    # --------------------------------------------------------

    smiles_n_pois = (
        poi_smiles
        .groupby(
            args.smiles_col
        )[args.poi_col]
        .nunique()
    )

    shared_smiles = set(
        smiles_n_pois[
            smiles_n_pois > 1
        ].index
    )

    # --------------------------------------------------------
    # POI -> UniProt mapping
    # --------------------------------------------------------

    poi_uniprots = (
        df
        .groupby(
            args.poi_col
        )[args.uniprot_col]
        .agg(
            lambda x:
            sorted(
                set(x)
            )
        )
        .to_dict()
    )

    candidate_rows = []

    eligible_pois = []

    eligible_uniprot = {}

    all_training_uniprots = sorted(
        set(
            dataset_uniprots
        )
    )

    # --------------------------------------------------------
    # Evaluate every POI as an individual candidate
    # --------------------------------------------------------

    for poi, n_molecules in molecule_counts.items():

        smiles = set(
            poi_smiles.loc[
                poi_smiles[
                    args.poi_col
                ] == poi,
                args.smiles_col,
            ]
        )

        n_shared = len(
            smiles
            & shared_smiles
        )

        accessions = (
            poi_uniprots
            .get(
                poi,
                [],
            )
        )

        single_uniprot = (
            len(accessions) == 1
        )

        has_embedding = (
            single_uniprot
            and accessions[0]
            in embeddings
        )

        eligible = (
            poi not in top_poi_set
            and int(
                n_molecules
            ) >= args.min_molecules
            and n_shared == 0
            and single_uniprot
            and has_embedding
        )

        nearest_poi = ""
        nearest_uniprot = ""
        nearest_similarity = float(
            "nan"
        )
        nearest_distance = float(
            "nan"
        )

        if eligible:

            uniprot = accessions[0]

            # For the individual-candidate summary,
            # remove only this candidate.
            training_uniprots = [
                u
                for u
                in all_training_uniprots
                if u != uniprot
            ]

            nearest = (
                nearest_training_neighbors(
                    query_uniprot=uniprot,
                    training_uniprots=(
                        training_uniprots
                    ),
                    embeddings=embeddings,
                    poi_by_uniprot=(
                        poi_by_uniprot
                    ),
                    top_k=1,
                )[0]
            )

            (
                nearest_uniprot,
                nearest_poi,
                nearest_similarity,
                nearest_distance,
            ) = nearest

            eligible_pois.append(
                poi
            )

            eligible_uniprot[
                poi
            ] = uniprot

        candidate_rows.append(
            {
                "poi":
                    poi,

                "target_uniprot":
                    (
                        accessions[0]
                        if single_uniprot
                        else "|".join(
                            accessions
                        )
                    ),

                "n_molecules":
                    int(
                        n_molecules
                    ),

                "is_top20":
                    poi
                    in top_poi_set,

                "n_smiles_shared_with_other_pois":
                    int(
                        n_shared
                    ),

                "all_structures_exclusive":
                    n_shared == 0,

                "single_target_uniprot":
                    single_uniprot,

                "eligible_candidate":
                    eligible,

                "nearest_training_poi_single_holdout":
                    nearest_poi,

                "nearest_training_uniprot_single_holdout":
                    nearest_uniprot,

                "max_cosine_similarity_single_holdout":
                    nearest_similarity,

                "cosine_distance_single_holdout":
                    nearest_distance,
            }
        )

    candidate_df = pd.DataFrame(
        candidate_rows
    )

    candidate_df = (
        candidate_df
        .sort_values(
            [
                "eligible_candidate",
                "cosine_distance_single_holdout",
                "n_molecules",
            ],
            ascending=[
                False,
                False,
                True,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    candidate_path = (
        args.out_dir
        / "poi_candidate_summary.csv"
    )

    candidate_df.to_csv(
        candidate_path,
        index=False,
    )

    if len(eligible_pois) < 2:

        raise RuntimeError(
            f"Only {len(eligible_pois)} "
            "eligible POIs remain. "
            "Lower --min-molecules or inspect "
            "poi_candidate_summary.csv."
        )

    # --------------------------------------------------------
    # Evaluate all pairs
    #
    # IMPORTANT:
    # For each A+B pair, BOTH A and B are removed
    # before measuring their nearest training target.
    # --------------------------------------------------------

    pair_rows = []

    for poi_a, poi_b in itertools.combinations(
        sorted(
            eligible_pois
        ),
        2,
    ):

        u_a = (
            eligible_uniprot[
                poi_a
            ]
        )

        u_b = (
            eligible_uniprot[
                poi_b
            ]
        )

        training_uniprots = [
            u
            for u
            in all_training_uniprots
            if u not in {
                u_a,
                u_b,
            }
        ]

        nn_a = (
            nearest_training_neighbors(
                query_uniprot=u_a,
                training_uniprots=(
                    training_uniprots
                ),
                embeddings=embeddings,
                poi_by_uniprot=(
                    poi_by_uniprot
                ),
                top_k=1,
            )[0]
        )

        nn_b = (
            nearest_training_neighbors(
                query_uniprot=u_b,
                training_uniprots=(
                    training_uniprots
                ),
                embeddings=embeddings,
                poi_by_uniprot=(
                    poi_by_uniprot
                ),
                top_k=1,
            )[0]
        )

        n_a = int(
            molecule_counts[
                poi_a
            ]
        )

        n_b = int(
            molecule_counts[
                poi_b
            ]
        )

        total_removed = (
            n_a
            + n_b
        )

        similarity_a = (
            nn_a[2]
        )

        similarity_b = (
            nn_b[2]
        )

        distance_a = (
            nn_a[3]
        )

        distance_b = (
            nn_b[3]
        )

        pair_rows.append(
            {
                "poi_a":
                    poi_a,

                "uniprot_a":
                    u_a,

                "n_molecules_a":
                    n_a,

                "nearest_training_poi_a":
                    nn_a[1],

                "nearest_training_uniprot_a":
                    nn_a[0],

                "max_cosine_similarity_a":
                    similarity_a,

                "cosine_distance_a":
                    distance_a,

                "poi_b":
                    poi_b,

                "uniprot_b":
                    u_b,

                "n_molecules_b":
                    n_b,

                "nearest_training_poi_b":
                    nn_b[1],

                "nearest_training_uniprot_b":
                    nn_b[0],

                "max_cosine_similarity_b":
                    similarity_b,

                "cosine_distance_b":
                    distance_b,

                "total_molecules_removed":
                    total_removed,

                "within_removal_budget":
                    (
                        total_removed
                        <=
                        args.max_total_removed
                    ),

                # Conservative pair score.
                #
                # We want BOTH unseen POIs
                # to be distant from training.
                "pair_min_cosine_distance":
                    min(
                        distance_a,
                        distance_b,
                    ),

                "pair_mean_cosine_distance":
                    (
                        distance_a
                        + distance_b
                    ) / 2.0,
            }
        )

    pair_df = pd.DataFrame(
        pair_rows
    )

    # --------------------------------------------------------
    # Ranking:
    #
    # 1. Must satisfy molecule-removal budget.
    # 2. Maximize the weaker of the two distances.
    # 3. Maximize mean distance.
    # 4. If still tied, remove fewer molecules.
    # --------------------------------------------------------

    pair_df = (
        pair_df
        .sort_values(
            [
                "within_removal_budget",
                "pair_min_cosine_distance",
                "pair_mean_cosine_distance",
                "total_molecules_removed",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    pair_df.insert(
        0,
        "rank",
        range(
            1,
            len(pair_df) + 1,
        ),
    )

    within_budget = (
        pair_df[
            pair_df[
                "within_removal_budget"
            ]
        ]
        .copy()
    )

    if within_budget.empty:

        raise RuntimeError(
            "No eligible pair satisfies "
            "--max-total-removed="
            f"{args.max_total_removed}. "
            "Increase the budget or lower "
            "--min-molecules."
        )

    selected = (
        within_budget
        .iloc[0]
    )

    pair_df[
        "selected"
    ] = False

    pair_df.loc[
        selected.name,
        "selected",
    ] = True

    pair_path = (
        args.out_dir
        / "candidate_pair_ranking.csv"
    )

    pair_df.to_csv(
        pair_path,
        index=False,
    )

    # --------------------------------------------------------
    # Build short report
    # --------------------------------------------------------

    selected_pois = [
        selected["poi_a"],
        selected["poi_b"],
    ]

    selected_uniprots = [
        selected["uniprot_a"],
        selected["uniprot_b"],
    ]

    selected_training = [
        u
        for u
        in all_training_uniprots
        if u not in set(
            selected_uniprots
        )
    ]

    report = []

    report.append(
        "UNSEEN-POI SELECTION REPORT"
    )

    report.append(
        "=" * 80
    )

    report.append(
        f"Input dataset: {args.input}"
    )

    report.append(
        f"Embeddings: {args.embeddings}"
    )

    report.append(
        f"Total POIs: "
        f"{len(molecule_counts)}"
    )

    report.append(
        f"Top-N excluded from candidates: "
        f"{args.top_n}"
    )

    report.append(
        f"Minimum molecules per candidate: "
        f"{args.min_molecules}"
    )

    report.append(
        f"Maximum total molecules removed: "
        f"{args.max_total_removed}"
    )

    report.append(
        f"Eligible candidate POIs: "
        f"{len(eligible_pois)}"
    )

    report.append("")

    # --------------------------------------------------------
    # Show current top 20
    # --------------------------------------------------------

    report.append(
        "TOP POIs KEPT OUT OF "
        "THE HOLDOUT CANDIDATE POOL"
    )

    for rank, poi in enumerate(
        top_pois,
        start=1,
    ):

        report.append(
            f"  {rank:2d}. "
            f"{poi}: "
            f"{int(molecule_counts[poi])} "
            "molecules"
        )

    report.append("")

    # --------------------------------------------------------
    # Selected pair
    # --------------------------------------------------------

    report.append(
        "SELECTED PAIR"
    )

    report.append(
        "-" * 80
    )

    report.append(
        f"{selected['poi_a']} "
        f"({selected['uniprot_a']}): "
        f"{int(selected['n_molecules_a'])} "
        "molecules"
    )

    report.append(
        "  nearest remaining training POI: "
        f"{selected['nearest_training_poi_a']} "
        f"({selected['nearest_training_uniprot_a']})"
    )

    report.append(
        "  cosine similarity: "
        f"{selected['max_cosine_similarity_a']:.6f}"
    )

    report.append(
        "  cosine distance:   "
        f"{selected['cosine_distance_a']:.6f}"
    )

    report.append("")

    report.append(
        f"{selected['poi_b']} "
        f"({selected['uniprot_b']}): "
        f"{int(selected['n_molecules_b'])} "
        "molecules"
    )

    report.append(
        "  nearest remaining training POI: "
        f"{selected['nearest_training_poi_b']} "
        f"({selected['nearest_training_uniprot_b']})"
    )

    report.append(
        "  cosine similarity: "
        f"{selected['max_cosine_similarity_b']:.6f}"
    )

    report.append(
        "  cosine distance:   "
        f"{selected['cosine_distance_b']:.6f}"
    )

    report.append("")

    report.append(
        "Total molecules removed: "
        f"{int(selected['total_molecules_removed'])}"
    )

    report.append(
        "Pair minimum distance: "
        f"{selected['pair_min_cosine_distance']:.6f}"
    )

    report.append(
        "Pair mean distance:    "
        f"{selected['pair_mean_cosine_distance']:.6f}"
    )

    report.append("")

    # --------------------------------------------------------
    # Top nearest training proteins for selected pair
    # --------------------------------------------------------

    report.append(
        f"TOP {args.report_neighbors} "
        "TRAINING NEIGHBORS FOR EACH "
        "SELECTED HOLDOUT"
    )

    report.append(
        "-" * 80
    )

    for poi, uniprot in zip(
        selected_pois,
        selected_uniprots,
    ):

        report.append(
            f"{poi} ({uniprot})"
        )

        neighbors = (
            nearest_training_neighbors(
                query_uniprot=uniprot,
                training_uniprots=(
                    selected_training
                ),
                embeddings=embeddings,
                poi_by_uniprot=(
                    poi_by_uniprot
                ),
                top_k=(
                    args.report_neighbors
                ),
            )
        )

        for i, (
            nn_uniprot,
            nn_poi,
            similarity,
            distance,
        ) in enumerate(
            neighbors,
            start=1,
        ):

            report.append(
                f"  {i}. "
                f"{nn_poi} "
                f"({nn_uniprot})  "
                f"similarity="
                f"{similarity:.6f}  "
                f"distance="
                f"{distance:.6f}"
            )

        report.append("")

    # --------------------------------------------------------
    # Best alternative pairs
    # --------------------------------------------------------

    report.append(
        f"TOP {args.report_top_pairs} "
        "PAIRS WITHIN THE REMOVAL BUDGET"
    )

    report.append(
        "-" * 80
    )

    for _, row in (
        within_budget
        .head(
            args.report_top_pairs
        )
        .iterrows()
    ):

        report.append(
            f"{int(row['rank']):2d}. "
            f"{row['poi_a']} + "
            f"{row['poi_b']} | "
            f"removed="
            f"{int(row['total_molecules_removed'])} | "
            f"min_distance="
            f"{row['pair_min_cosine_distance']:.6f} | "
            f"mean_distance="
            f"{row['pair_mean_cosine_distance']:.6f}"
        )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_path = (
        args.out_dir
        / "REPORT.txt"
    )

    report_text = (
        "\n".join(report)
        + "\n"
    )

    report_path.write_text(
        report_text,
        encoding="utf-8",
    )

    # Also print report into SLURM log.
    print(
        report_text
    )

    print(
        "Saved:"
    )

    print(
        f"  {candidate_path}"
    )

    print(
        f"  {pair_path}"
    )

    print(
        f"  {report_path}"
    )


if __name__ == "__main__":
    main()