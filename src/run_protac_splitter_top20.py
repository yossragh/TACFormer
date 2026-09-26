#!/usr/bin/env python3
"""Run PROTAC-Splitter on all unique smiles_std from the top-N POIs.

Designed to mirror the notebook's PROTAC-Splitter configuration while removing
only the old 200-molecules-per-POI cap.

Important data rule
-------------------
The input dataset is already cleaned/standardized. This script DOES NOT
standardize or canonicalize SMILES. It uses the existing `smiles_std` column
exactly as supplied.

Repeated `smiles_std` values are deduplicated only for splitter inference so an
identical molecule is not split repeatedly. In the notebook, the resulting
one-row-per-SMILES lookup is merged back to the full POI--SMILES table, which
preserves molecules associated with multiple POIs.

Speed / sharding
----------------
For Arrhenius, independent shards can be run concurrently. Each shard uses the
same adaptive PROTAC-Splitter configuration as the notebook. Afterwards this
same script can merge the shard CSVs into one Option-B-compatible CSV.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p.add_argument("--smiles-col", default="smiles_std")
    p.add_argument("--poi-col", default="poi_norm")
    p.add_argument("--top-n", type=int, default=20)

    p.add_argument(
        "--num-proc",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
        help="PROTAC-Splitter heuristic worker processes.",
    )

    p.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Molecules per checkpointed split_protac call.",
    )

    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total independent shards.",
    )

    p.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Zero-based shard index.",
    )

    p.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge completed shard CSVs into final output.",
    )

    return p.parse_args()


def validate_args(a: argparse.Namespace) -> None:
    if a.top_n < 1:
        raise ValueError("--top-n must be >= 1")

    if a.num_proc < 1:
        raise ValueError("--num-proc must be >= 1")

    if a.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")

    if a.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")

    if not a.merge_only:

        if a.num_shards > 1 and a.shard_index is None:
            raise ValueError(
                "--shard-index is required when --num-shards > 1"
            )

        if (
            a.shard_index is not None
            and not (0 <= a.shard_index < a.num_shards)
        ):
            raise ValueError(
                "--shard-index must satisfy "
                "0 <= shard-index < num-shards"
            )


def build_full_split_input(
    input_path: Path,
    smiles_col: str,
    poi_col: str,
    top_n: int,
) -> tuple[pd.DataFrame, pd.Series, int]:

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    required_cols = [smiles_col, poi_col]

    missing_cols = [
        c for c in required_cols
        if c not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}"
        )

    # ------------------------------------------------------------
    # Dataset is already cleaned.
    # We DO NOT standardize/canonicalize smiles_std.
    # ------------------------------------------------------------

    n_missing_smiles = int(
        df[smiles_col].isna().sum()
    )

    n_missing_poi = int(
        df[poi_col].isna().sum()
    )

    if n_missing_smiles or n_missing_poi:
        raise ValueError(
            f"Missing required values: "
            f"{smiles_col}={n_missing_smiles}, "
            f"{poi_col}={n_missing_poi}"
        )

    # ------------------------------------------------------------
    # Same top-20 logic as notebook
    # ------------------------------------------------------------

    poi_counts = df[poi_col].value_counts()

    if len(poi_counts) < top_n:
        raise ValueError(
            f"Requested top {top_n} POIs, "
            f"but dataset contains only "
            f"{len(poi_counts)}."
        )

    top_pois = poi_counts.head(top_n)

    top_df = df[
        df[poi_col].isin(top_pois.index)
    ].copy()

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # Deduplicate ONLY for PROTAC-Splitter inference.
    #
    # If the same molecule belongs to multiple POIs,
    # it only needs to be split once.
    #
    # Later the result is merged back onto the full
    # POI--SMILES dataframe in the notebook.
    # ------------------------------------------------------------

    split_input = (
        top_df[[smiles_col]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return (
        split_input,
        top_pois,
        len(top_df),
    )


def shard_view(
    full: pd.DataFrame,
    num_shards: int,
    shard_index: int | None,
) -> pd.DataFrame:

    if num_shards == 1:
        return full.copy().reset_index(drop=True)

    assert shard_index is not None

    # Round-robin assignment keeps shard sizes
    # approximately identical.
    mask = (
        np.arange(len(full)) % num_shards
    ) == shard_index

    return (
        full.loc[mask]
        .reset_index(drop=True)
    )


def shard_path(
    output: Path,
    shard_index: int,
    num_shards: int,
) -> Path:

    width = max(
        2,
        len(str(num_shards - 1)),
    )

    return output.with_name(
        f"{output.stem}."
        f"part{shard_index:0{width}d}"
        f"-of-{num_shards:0{width}d}"
        f"{output.suffix}"
    )


def load_cached(
    path: Path,
    smiles_col: str,
) -> pd.DataFrame:

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if smiles_col not in df.columns:
        raise ValueError(
            f"Existing output {path} "
            f"has no '{smiles_col}' column"
        )

    return (
        df
        .drop_duplicates(
            smiles_col,
            keep="last",
        )
        .reset_index(drop=True)
    )


def atomic_save(
    df: pd.DataFrame,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
    )

    os.replace(
        tmp,
        path,
    )


def ordered_processed_rows(
    target: pd.DataFrame,
    cached: pd.DataFrame,
    smiles_col: str,
) -> pd.DataFrame:

    if cached.empty:
        return pd.DataFrame(
            columns=[smiles_col]
        )

    cached = (
        cached
        .drop_duplicates(
            smiles_col,
            keep="last",
        )
    )

    return target.merge(
        cached,
        on=smiles_col,
        how="inner",
        validate="one_to_one",
    )


def merge_shards(
    a: argparse.Namespace,
    full_input: pd.DataFrame,
) -> int:

    parts = []

    for i in range(a.num_shards):

        p = shard_path(
            a.output,
            i,
            a.num_shards,
        )

        if not p.exists():
            raise FileNotFoundError(
                f"Missing shard output: {p}"
            )

        part = pd.read_csv(p)

        if a.smiles_col not in part.columns:
            raise ValueError(
                f"Shard {p} lacks "
                f"'{a.smiles_col}'"
            )

        parts.append(part)

    merged = pd.concat(
        parts,
        ignore_index=True,
    )

    # There should be no overlap between shards.
    if merged[a.smiles_col].duplicated().any():

        dup_n = int(
            merged[
                a.smiles_col
            ].duplicated().sum()
        )

        raise ValueError(
            f"Shard outputs overlap: "
            f"{dup_n} duplicate SMILES"
        )

    # Reorder according to the original expected
    # unique-SMILES input.
    merged = full_input.merge(
        merged,
        on=a.smiles_col,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing = int(
        (
            merged["_merge"] != "both"
        ).sum()
    )

    merged = merged.drop(
        columns="_merge"
    )

    if missing:
        raise ValueError(
            f"Cannot create final CSV: "
            f"{missing:,} expected molecules "
            f"are missing."
        )

    required = [
        a.smiles_col,
        "default_pred_n0",
        "model_name",
    ]

    missing_output_cols = [
        c for c in required
        if c not in merged.columns
    ]

    if missing_output_cols:
        raise ValueError(
            "Merged output missing columns: "
            f"{missing_output_cols}"
        )

    atomic_save(
        merged,
        a.output,
    )

    print(
        f"Merged {a.num_shards} shards "
        f"-> {a.output}",
        flush=True,
    )

    print(
        f"Final rows: "
        f"{len(merged):,}",
        flush=True,
    )

    print(
        "Rows with default_pred_n0: "
        f"{int(merged['default_pred_n0'].notna().sum()):,}"
        f"/{len(merged):,}",
        flush=True,
    )

    return 0


def run_shard(
    a: argparse.Namespace,
    full_input: pd.DataFrame,
) -> int:

    try:
        from protac_splitter import split_protac

    except ImportError as exc:

        raise ImportError(
            "PROTAC-Splitter is not installed.\n"
            "Install using:\n"
            "pip install "
            "git+https://github.com/"
            "ribesstefano/PROTAC-Splitter.git"
        ) from exc

    target = shard_view(
        full_input,
        a.num_shards,
        a.shard_index,
    )

    if a.num_shards > 1:

        out = shard_path(
            a.output,
            a.shard_index,
            a.num_shards,
        )

    else:
        out = a.output

    cached = load_cached(
        out,
        a.smiles_col,
    )

    target_set = set(
        target[a.smiles_col].astype(str)
    )

    if not cached.empty:

        cached = cached[
            cached[
                a.smiles_col
            ]
            .astype(str)
            .isin(target_set)
        ].copy()

    cached_smiles = (
        set(
            cached[
                a.smiles_col
            ].astype(str)
        )
        if not cached.empty
        else set()
    )

    missing = (
        target[
            ~target[
                a.smiles_col
            ]
            .astype(str)
            .isin(cached_smiles)
        ]
        .reset_index(drop=True)
    )

    if a.num_shards > 1:

        label = (
            f"shard "
            f"{a.shard_index + 1}"
            f"/{a.num_shards}"
        )

    else:
        label = "single job"

    print(
        f"Running {label}",
        flush=True,
    )

    print(
        f"Target molecules: "
        f"{len(target):,}",
        flush=True,
    )

    print(
        f"Already processed: "
        f"{len(target) - len(missing):,}",
        flush=True,
    )

    print(
        f"Still to split: "
        f"{len(missing):,}",
        flush=True,
    )

    print(
        f"Heuristic workers: "
        f"{a.num_proc}",
        flush=True,
    )

    print(
        f"Checkpoint chunk size: "
        f"{a.chunk_size:,}",
        flush=True,
    )

    # ------------------------------------------------------------
    # Create/update resumable checkpoint
    # ------------------------------------------------------------

    atomic_save(
        ordered_processed_rows(
            target,
            cached,
            a.smiles_col,
        ),
        out,
    )

    # ------------------------------------------------------------
    # Run missing molecules
    # ------------------------------------------------------------

    if len(missing):

        n_chunks = (
            len(missing)
            + a.chunk_size
            - 1
        ) // a.chunk_size

        for chunk_no, start in enumerate(
            range(
                0,
                len(missing),
                a.chunk_size,
            ),
            start=1,
        ):

            chunk = missing.iloc[
                start:
                start + a.chunk_size
            ].copy()

            print(
                f"\n[{label}] "
                f"chunk "
                f"{chunk_no}/{n_chunks}: "
                f"{len(chunk):,} molecules",
                flush=True,
            )

            # ----------------------------------------------------
            # SAME splitting method used in the notebook.
            #
            # Keep adaptive.
            # Do NOT use Transformer.
            # ----------------------------------------------------

            new = split_protac(
                chunk,
                protac_smiles_col=a.smiles_col,
                model="adaptive",
                adaptive_use_transformer=False,
                num_proc=a.num_proc,
                verbose=1,
            )

            if not isinstance(
                new,
                pd.DataFrame,
            ):
                new = pd.DataFrame(new)

            if (
                a.smiles_col
                not in new.columns
            ):
                raise ValueError(
                    "PROTAC-Splitter returned "
                    f"no '{a.smiles_col}' "
                    f"column.\n"
                    f"Columns: "
                    f"{list(new.columns)}"
                )

            cached = pd.concat(
                [
                    cached,
                    new,
                ],
                ignore_index=True,
            )

            checkpoint = (
                ordered_processed_rows(
                    target,
                    cached,
                    a.smiles_col,
                )
            )

            atomic_save(
                checkpoint,
                out,
            )

            print(
                f"Checkpoint: "
                f"{len(checkpoint):,}/"
                f"{len(target):,} "
                f"molecules processed "
                f"-> {out}",
                flush=True,
            )

    # ------------------------------------------------------------
    # Final validation of this shard
    # ------------------------------------------------------------

    final = pd.read_csv(out)

    if len(final) != len(target):
        raise RuntimeError(
            f"Shard incomplete: "
            f"output has "
            f"{len(final):,} rows, "
            f"expected "
            f"{len(target):,}."
        )

    required = [
        a.smiles_col,
        "default_pred_n0",
        "model_name",
    ]

    missing_cols = [
        c for c in required
        if c not in final.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Output missing columns: "
            f"{missing_cols}"
        )

    print(
        f"\nDONE {label}: {out}",
        flush=True,
    )

    print(
        f"Rows: {len(final):,}",
        flush=True,
    )

    print(
        "Rows with default_pred_n0: "
        f"{int(final['default_pred_n0'].notna().sum()):,}"
        f"/{len(final):,}",
        flush=True,
    )

    return 0


def main() -> int:

    a = parse_args()

    validate_args(a)

    (
        full_input,
        top_pois,
        top_rows,
    ) = build_full_split_input(
        a.input,
        a.smiles_col,
        a.poi_col,
        a.top_n,
    )

    print(
        f"Input: {a.input}",
        flush=True,
    )

    print(
        f"Top-{a.top_n} "
        f"POI--molecule rows: "
        f"{top_rows:,}",
        flush=True,
    )

    print(
        "Unique smiles_std "
        "requiring splitting: "
        f"{len(full_input):,}",
        flush=True,
    )

    print(
        "Top POIs:",
        flush=True,
    )

    for poi, count in top_pois.items():

        print(
            f"  {poi}: "
            f"{int(count):,}",
            flush=True,
        )

    if a.merge_only:

        return merge_shards(
            a,
            full_input,
        )

    return run_shard(
        a,
        full_input,
    )


if __name__ == "__main__":

    try:
        raise SystemExit(
            main()
        )

    except Exception as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
            flush=True,
        )

        raise