#!/usr/bin/env python3
"""Analyze similarity across all POIs and optionally select two unseen POIs.

The script does two related but distinct analyses.

A) GLOBAL POI SIMILARITY STUDY
   - Computes cosine similarity for every unordered POI pair.
   - Summarizes, for every POI:
       * nearest POI globally
       * d_nearest_global = 1 - max cosine similarity to any other POI
       * most dissimilar POI
       * mean/median similarity to all other POIs

B) OPTIONAL UNSEEN-POI SELECTION (--select-unseen)
   Candidate requirements:
   - POI is outside the current top-N POIs.
   - ALL of its smiles_std structures are exclusive to that POI.
   - No molecule-count threshold is applied.

   For every eligible pair (A, B):
   - Remove BOTH A and B from the hypothetical training POI set.
   - Compute:

       d_A = 1 - max cosine(A, t), for t in remaining training POIs
       d_B = 1 - max cosine(B, t), for t in remaining training POIs

   - Score the pair using:

       pair_min_d_nearest = min(d_A, d_B)

   - Rank primarily by largest pair_min_d_nearest.
   - Tie-break by largest mean(d_A, d_B).
   - Molecule counts are REPORTED ONLY and never used for ranking.

This makes the unseen-POI selection answer:
"Which pair contains two POIs that are both as dissimilar as possible
from everything that remains in training?"

The input dataset is assumed to be already cleaned. `smiles_std` is used
exactly as supplied and is never re-standardized.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
        default=Path("poi_similarity_analysis"),
    )

    p.add_argument("--poi-col", default="poi_norm")
    p.add_argument("--uniprot-col", default="target_uniprot")
    p.add_argument("--smiles-col", default="smiles_std")

    p.add_argument(
        "--select-unseen",
        action="store_true",
        help="Also rank and select unseen-POI pairs.",
    )

    p.add_argument(
        "--exclude-top-n",
        type=int,
        default=20,
        help="Top-N most represented POIs excluded from unseen candidates.",
    )

    p.add_argument(
        "--top-k-pairs",
        type=int,
        default=5,
        help="Number of unseen candidate pairs to print in the report.",
    )

    return p.parse_args()


def load_data(
    path: Path,
    poi_col: str,
    uniprot_col: str,
    smiles_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = [poi_col, uniprot_col, smiles_col]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df[required].copy()

    for col in required:
        work[col] = (
            work[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    blanks = {
        col: int((work[col] == "").sum())
        for col in required
        if int((work[col] == "").sum()) > 0
    }

    if blanks:
        raise ValueError(f"Blank required values: {blanks}")

    # Dataset is already standardized/clean.
    # Do NOT canonicalize smiles_std again.
    return work.drop_duplicates().reset_index(drop=True)


def load_embeddings(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(payload, dict):
        raise TypeError("Expected a dictionary payload in the .pt file.")

    raw = payload.get("embeddings_by_uniprot")

    if not isinstance(raw, dict) or not raw:
        raise KeyError(
            "The .pt file must contain a non-empty "
            "'embeddings_by_uniprot' dictionary."
        )

    embeddings = {}

    for uniprot, vector in raw.items():
        key = str(uniprot).strip()

        vec = torch.as_tensor(
            vector,
            dtype=torch.float32,
        ).flatten()

        if vec.numel() == 0:
            raise ValueError(f"Empty embedding for {key}")

        # Normalize for cosine similarity.
        embeddings[key] = F.normalize(
            vec,
            p=2,
            dim=0,
        )

    dims = {int(v.numel()) for v in embeddings.values()}

    if len(dims) != 1:
        raise ValueError(
            f"Inconsistent embedding dimensions: {sorted(dims)}"
        )

    return embeddings


def build_poi_metadata(
    df: pd.DataFrame,
    poi_col: str,
    uniprot_col: str,
    smiles_col: str,
    exclude_top_n: int,
) -> pd.DataFrame:
    poi_smiles = (
        df[[poi_col, smiles_col]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    counts = (
        poi_smiles
        .groupby(poi_col)[smiles_col]
        .nunique()
        .sort_values(ascending=False)
    )

    n_pois_per_smiles = (
        poi_smiles
        .groupby(smiles_col)[poi_col]
        .nunique()
    )

    shared_smiles = set(
        n_pois_per_smiles[
            n_pois_per_smiles > 1
        ].index
    )

    poi_to_uniprots = (
        df
        .groupby(poi_col)[uniprot_col]
        .agg(lambda x: sorted(set(x)))
        .to_dict()
    )

    if exclude_top_n < 0:
        raise ValueError("--exclude-top-n must be >= 0")

    top_set = set(
        counts.head(exclude_top_n).index
        if exclude_top_n > 0
        else []
    )

    rows = []

    for poi, n_molecules in counts.items():
        smiles = set(
            poi_smiles.loc[
                poi_smiles[poi_col] == poi,
                smiles_col,
            ]
        )

        n_shared = len(smiles & shared_smiles)

        accessions = poi_to_uniprots.get(poi, [])

        if len(accessions) != 1:
            raise ValueError(
                f"POI {poi!r} maps to {len(accessions)} UniProt IDs: "
                f"{accessions}. One embedding per POI is required."
            )

        rows.append(
            {
                "poi": poi,
                "target_uniprot": accessions[0],
                "n_molecules": int(n_molecules),
                "n_smiles_shared_with_other_pois": int(n_shared),
                "all_structures_exclusive": bool(n_shared == 0),
                "is_top_n": bool(poi in top_set),
            }
        )

    metadata = pd.DataFrame(rows)

    # No molecule-count criterion.
    metadata["eligible_unseen_candidate"] = (
        (~metadata["is_top_n"])
        & metadata["all_structures_exclusive"]
    )

    return metadata


def build_similarity_matrix(
    metadata: pd.DataFrame,
    embeddings: dict[str, torch.Tensor],
) -> torch.Tensor:
    uniprots = metadata["target_uniprot"].tolist()

    missing = [
        u for u in uniprots
        if u not in embeddings
    ]

    if missing:
        raise ValueError(
            f"{len(missing)} POIs are missing embeddings. "
            f"First missing accessions: {missing[:20]}"
        )

    matrix = torch.stack(
        [embeddings[u] for u in uniprots],
        dim=0,
    )

    sim = matrix @ matrix.T
    return torch.clamp(sim, -1.0, 1.0)


def nearest_remaining_training(
    query_idx: int,
    held_out: set[int],
    sim: torch.Tensor,
    metadata: pd.DataFrame,
) -> dict:
    train_indices = [
        j
        for j in range(len(metadata))
        if j not in held_out
    ]

    if not train_indices:
        raise RuntimeError("No POIs remain in training.")

    idx_tensor = torch.tensor(
        train_indices,
        dtype=torch.long,
    )

    vals = sim[
        query_idx,
        idx_tensor,
    ]

    local_argmax = int(
        torch.argmax(vals).item()
    )

    nearest_idx = train_indices[
        local_argmax
    ]

    similarity = float(
        vals[local_argmax].item()
    )

    nearest = metadata.iloc[
        nearest_idx
    ]

    return {
        "nearest_training_poi":
            nearest["poi"],
        "nearest_training_uniprot":
            nearest["target_uniprot"],
        "max_training_cosine_similarity":
            similarity,
        "d_nearest_training":
            1.0 - similarity,
    }


def main() -> None:
    args = parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data(
        args.input,
        args.poi_col,
        args.uniprot_col,
        args.smiles_col,
    )

    embeddings = load_embeddings(
        args.embeddings
    )

    metadata = build_poi_metadata(
        df,
        args.poi_col,
        args.uniprot_col,
        args.smiles_col,
        args.exclude_top_n,
    )

    sim = build_similarity_matrix(
        metadata,
        embeddings,
    )

    n_pois = len(metadata)

    if n_pois < 2:
        raise RuntimeError("At least two POIs are required.")

    # ------------------------------------------------------------
    # 1. PER-POI GLOBAL SUMMARY
    # ------------------------------------------------------------

    summary_rows = []

    for i in range(n_pois):
        other_indices = [
            j
            for j in range(n_pois)
            if j != i
        ]

        values = (
            sim[
                i,
                torch.tensor(
                    other_indices,
                    dtype=torch.long,
                ),
            ]
            .cpu()
            .numpy()
        )

        nearest_local = int(
            np.argmax(values)
        )

        farthest_local = int(
            np.argmin(values)
        )

        nearest_idx = other_indices[
            nearest_local
        ]

        farthest_idx = other_indices[
            farthest_local
        ]

        nearest_similarity = float(
            values[nearest_local]
        )

        farthest_similarity = float(
            values[farthest_local]
        )

        base = metadata.iloc[i]

        summary_rows.append(
            {
                "poi":
                    base["poi"],

                "target_uniprot":
                    base["target_uniprot"],

                "n_molecules":
                    int(base["n_molecules"]),

                "n_smiles_shared_with_other_pois":
                    int(
                        base[
                            "n_smiles_shared_with_other_pois"
                        ]
                    ),

                "all_structures_exclusive":
                    bool(
                        base[
                            "all_structures_exclusive"
                        ]
                    ),

                "is_top_n":
                    bool(
                        base["is_top_n"]
                    ),

                "eligible_unseen_candidate":
                    bool(
                        base[
                            "eligible_unseen_candidate"
                        ]
                    ),

                "nearest_poi_global":
                    metadata.iloc[
                        nearest_idx
                    ]["poi"],

                "nearest_uniprot_global":
                    metadata.iloc[
                        nearest_idx
                    ]["target_uniprot"],

                "max_cosine_similarity_global":
                    nearest_similarity,

                "d_nearest_global":
                    1.0 - nearest_similarity,

                "most_dissimilar_poi_global":
                    metadata.iloc[
                        farthest_idx
                    ]["poi"],

                "most_dissimilar_uniprot_global":
                    metadata.iloc[
                        farthest_idx
                    ]["target_uniprot"],

                "min_cosine_similarity_global":
                    farthest_similarity,

                "max_cosine_distance_global":
                    1.0 - farthest_similarity,

                "mean_cosine_similarity_global":
                    float(np.mean(values)),

                "median_cosine_similarity_global":
                    float(np.median(values)),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    ).sort_values(
        "d_nearest_global",
        ascending=False,
    ).reset_index(
        drop=True
    )

    summary_path = (
        args.out_dir
        / "poi_similarity_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ------------------------------------------------------------
    # 2. ALL POI-POI SIMILARITIES
    #
    # For eligible unseen pairs, also compute the pair-specific
    # distance of EACH held-out POI to the remaining training POIs.
    # ------------------------------------------------------------

    pair_rows = []

    for i in range(n_pois):
        for j in range(
            i + 1,
            n_pois,
        ):
            a = metadata.iloc[i]
            b = metadata.iloc[j]

            mutual_similarity = float(
                sim[i, j].item()
            )

            mutual_distance = (
                1.0
                - mutual_similarity
            )

            eligible_pair = bool(
                a[
                    "eligible_unseen_candidate"
                ]
                and
                b[
                    "eligible_unseen_candidate"
                ]
            )

            row = {
                "poi_a":
                    a["poi"],

                "uniprot_a":
                    a["target_uniprot"],

                "n_molecules_a":
                    int(a["n_molecules"]),

                "exclusive_a":
                    bool(
                        a[
                            "all_structures_exclusive"
                        ]
                    ),

                "eligible_unseen_a":
                    bool(
                        a[
                            "eligible_unseen_candidate"
                        ]
                    ),

                "poi_b":
                    b["poi"],

                "uniprot_b":
                    b["target_uniprot"],

                "n_molecules_b":
                    int(b["n_molecules"]),

                "exclusive_b":
                    bool(
                        b[
                            "all_structures_exclusive"
                        ]
                    ),

                "eligible_unseen_b":
                    bool(
                        b[
                            "eligible_unseen_candidate"
                        ]
                    ),

                "cosine_similarity":
                    mutual_similarity,

                "cosine_distance":
                    mutual_distance,

                "eligible_unseen_pair":
                    eligible_pair,

                # Filled only for eligible unseen pairs:
                "nearest_training_poi_a":
                    "",

                "nearest_training_uniprot_a":
                    "",

                "d_nearest_training_a":
                    np.nan,

                "nearest_training_poi_b":
                    "",

                "nearest_training_uniprot_b":
                    "",

                "d_nearest_training_b":
                    np.nan,

                "pair_min_d_nearest":
                    np.nan,

                "pair_mean_d_nearest":
                    np.nan,

                "total_molecules_held_out":
                    int(
                        a["n_molecules"]
                        + b["n_molecules"]
                    ),
            }

            if eligible_pair:
                held_out = {
                    i,
                    j,
                }

                train_a = nearest_remaining_training(
                    i,
                    held_out,
                    sim,
                    metadata,
                )

                train_b = nearest_remaining_training(
                    j,
                    held_out,
                    sim,
                    metadata,
                )

                d_a = train_a[
                    "d_nearest_training"
                ]

                d_b = train_b[
                    "d_nearest_training"
                ]

                row.update(
                    {
                        "nearest_training_poi_a":
                            train_a[
                                "nearest_training_poi"
                            ],

                        "nearest_training_uniprot_a":
                            train_a[
                                "nearest_training_uniprot"
                            ],

                        "d_nearest_training_a":
                            d_a,

                        "nearest_training_poi_b":
                            train_b[
                                "nearest_training_poi"
                            ],

                        "nearest_training_uniprot_b":
                            train_b[
                                "nearest_training_uniprot"
                            ],

                        "d_nearest_training_b":
                            d_b,

                        # PRIMARY SELECTION SCORE:
                        # both POIs must be difficult.
                        "pair_min_d_nearest":
                            min(
                                d_a,
                                d_b,
                            ),

                        "pair_mean_d_nearest":
                            (
                                d_a
                                + d_b
                            ) / 2.0,
                    }
                )

            pair_rows.append(
                row
            )

    pair_df = pd.DataFrame(
        pair_rows
    )

    # All-pairs file remains sorted by direct pairwise similarity
    # because it is primarily the GLOBAL similarity-study output.
    all_pairs_path = (
        args.out_dir
        / "all_poi_pairwise_similarity.csv"
    )

    pair_df.sort_values(
        "cosine_similarity",
        ascending=False,
    ).to_csv(
        all_pairs_path,
        index=False,
    )

    # ------------------------------------------------------------
    # 3. REPORT
    # ------------------------------------------------------------

    all_pair_similarities = (
        pair_df[
            "cosine_similarity"
        ]
        .to_numpy()
    )

    report = []

    report.append(
        "ALL-POI ESM-2 SIMILARITY REPORT"
    )

    report.append(
        "=" * 80
    )

    report.append(
        f"Total POIs: {n_pois}"
    )

    report.append(
        "Unique unordered POI pairs: "
        f"{len(pair_df):,}"
    )

    report.append("")

    report.append(
        "GLOBAL PAIRWISE COSINE SIMILARITY"
    )

    report.append(
        "-" * 80
    )

    report.append(
        f"Minimum: "
        f"{all_pair_similarities.min():.6f}"
    )

    report.append(
        "5th percentile: "
        f"{np.quantile(all_pair_similarities, 0.05):.6f}"
    )

    report.append(
        f"Median: "
        f"{np.median(all_pair_similarities):.6f}"
    )

    report.append(
        f"Mean: "
        f"{all_pair_similarities.mean():.6f}"
    )

    report.append(
        "95th percentile: "
        f"{np.quantile(all_pair_similarities, 0.95):.6f}"
    )

    report.append(
        f"Maximum: "
        f"{all_pair_similarities.max():.6f}"
    )

    # Most globally isolated POIs
    report.append("")
    report.append(
        "MOST GLOBALLY ISOLATED POIs"
    )
    report.append(
        "(largest d_nearest_global)"
    )
    report.append(
        "-" * 80
    )

    for rank, (_, row) in enumerate(
        summary_df
        .head(10)
        .iterrows(),
        start=1,
    ):
        report.append(
            f"{rank:2d}. "
            f"{row['poi']} "
            f"({row['target_uniprot']}) | "
            f"molecules="
            f"{int(row['n_molecules'])} | "
            f"nearest="
            f"{row['nearest_poi_global']} | "
            f"similarity="
            f"{row['max_cosine_similarity_global']:.6f} | "
            f"d_nearest="
            f"{row['d_nearest_global']:.6f}"
        )

    # Most similar global pairs
    report.append("")
    report.append(
        "MOST SIMILAR POI PAIRS"
    )
    report.append(
        "-" * 80
    )

    most_similar = (
        pair_df
        .sort_values(
            "cosine_similarity",
            ascending=False,
        )
        .head(10)
    )

    for rank, (_, row) in enumerate(
        most_similar.iterrows(),
        start=1,
    ):
        report.append(
            f"{rank:2d}. "
            f"{row['poi_a']} + "
            f"{row['poi_b']} | "
            f"similarity="
            f"{row['cosine_similarity']:.6f} | "
            f"distance="
            f"{row['cosine_distance']:.6f}"
        )

    # Most dissimilar global pairs
    report.append("")
    report.append(
        "MOST DISSIMILAR POI PAIRS"
    )
    report.append(
        "-" * 80
    )

    most_dissimilar = (
        pair_df
        .sort_values(
            "cosine_similarity",
            ascending=True,
        )
        .head(10)
    )

    for rank, (_, row) in enumerate(
        most_dissimilar.iterrows(),
        start=1,
    ):
        report.append(
            f"{rank:2d}. "
            f"{row['poi_a']} + "
            f"{row['poi_b']} | "
            f"similarity="
            f"{row['cosine_similarity']:.6f} | "
            f"distance="
            f"{row['cosine_distance']:.6f}"
        )

    # ------------------------------------------------------------
    # 4. OPTIONAL UNSEEN SELECTION
    # ------------------------------------------------------------

    if args.select_unseen:
        eligible_pois = metadata[
            metadata[
                "eligible_unseen_candidate"
            ]
        ]

        eligible_pairs = pair_df[
            pair_df[
                "eligible_unseen_pair"
            ]
        ].copy()

        # CORRECT unseen ranking:
        #
        # Rank by how far BOTH held-out proteins are from
        # everything that remains in training.
        #
        # Molecule count is NOT part of the ranking.
        eligible_pairs = (
            eligible_pairs
            .sort_values(
                [
                    "pair_min_d_nearest",
                    "pair_mean_d_nearest",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        report.append("")
        report.append(
            "UNSEEN-POI CANDIDATE FILTER"
        )
        report.append(
            "-" * 80
        )

        report.append(
            f"Excluded top-{args.exclude_top_n} "
            "POIs: yes"
        )

        report.append(
            "Require all standardized SMILES "
            "to be exclusive: yes"
        )

        report.append(
            "Molecule count used as criterion: no"
        )

        report.append(
            "Eligible unseen POIs: "
            f"{len(eligible_pois)}"
        )

        report.append(
            "Eligible unseen POI pairs: "
            f"{len(eligible_pairs):,}"
        )

        report.append("")
        report.append(
            "UNSEEN SELECTION RULE"
        )
        report.append(
            "For every pair A+B, remove BOTH from training, "
            "then compute d_nearest for A and B against all "
            "remaining training POIs."
        )
        report.append(
            "Primary score = min(d_nearest_A, d_nearest_B)."
        )
        report.append(
            "Molecule counts are reported only and do not "
            "affect ranking."
        )

        report.append("")
        report.append(
            f"TOP {args.top_k_pairs} "
            "UNSEEN-POI CANDIDATE PAIRS"
        )
        report.append(
            "(ranked by distance to remaining training POIs)"
        )
        report.append(
            "-" * 80
        )

        for rank, (_, row) in enumerate(
            eligible_pairs
            .head(
                args.top_k_pairs
            )
            .iterrows(),
            start=1,
        ):
            report.append(
                f"{rank}. "
                f"{row['poi_a']} "
                f"({row['uniprot_a']}, "
                f"n={int(row['n_molecules_a'])}) "
                f"+ "
                f"{row['poi_b']} "
                f"({row['uniprot_b']}, "
                f"n={int(row['n_molecules_b'])})"
            )

            report.append(
                f"   {row['poi_a']}: "
                f"nearest training="
                f"{row['nearest_training_poi_a']} "
                f"({row['nearest_training_uniprot_a']}), "
                f"d_nearest="
                f"{row['d_nearest_training_a']:.6f}"
            )

            report.append(
                f"   {row['poi_b']}: "
                f"nearest training="
                f"{row['nearest_training_poi_b']} "
                f"({row['nearest_training_uniprot_b']}), "
                f"d_nearest="
                f"{row['d_nearest_training_b']:.6f}"
            )

            report.append(
                "   pair min d_nearest="
                f"{row['pair_min_d_nearest']:.6f}, "
                "mean d_nearest="
                f"{row['pair_mean_d_nearest']:.6f}"
            )

            report.append(
                "   mutual POI distance="
                f"{row['cosine_distance']:.6f}"
            )

            report.append(
                "   total held-out molecules="
                f"{int(row['total_molecules_held_out'])}"
            )

            report.append("")

        selected = eligible_pairs.iloc[0]

        report.append(
            "AUTOMATIC TOP-RANKED PAIR"
        )
        report.append(
            "-" * 80
        )

        report.append(
            f"{selected['poi_a']} + "
            f"{selected['poi_b']}"
        )

        report.append(
            "This is the top-ranked pair by protein-space "
            "separation from the remaining training POIs. "
            "Molecule count should still be considered manually "
            "before making the final experimental choice."
        )

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

    print(report_text)

    print("Saved:")
    print(f"  {summary_path}")
    print(f"  {all_pairs_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
