#!/usr/bin/env python3
"""
Run PROTAC-Splitter on unique valid generated PROTACs and merge the
splitter annotations back onto the original annotated_generated.jsonl.

Input
-----
annotated_generated.jsonl

Expected fields:
    canonical_smiles
    is_valid

Workflow
--------
1. Read annotated_generated.jsonl.
2. Keep valid records with non-empty canonical_smiles.
3. Deduplicate canonical_smiles ONLY for PROTAC-Splitter inference.
4. Run PROTAC-Splitter once per unique canonical molecule.
5. Support multiple independent SLURM shards.
6. Merge shard outputs into one one-row-per-canonical-SMILES CSV.
7. Merge the splitter results back onto EVERY original JSONL record.
8. Write annotated_generated_with_split.jsonl.

Important
---------
The original generated JSONL is never overwritten.

All PROTAC-Splitter output fields are added with the prefix:

    splitter_

For example:

    splitter_default_pred_n0
    splitter_model_name

Two additional fields are written:

    splitter_lookup_found
    splitter_has_prediction
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="annotated_generated.jsonl",
    )

    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "One-row-per-unique-PROTAC splitter lookup CSV. "
            "Shard files are derived from this path."
        ),
    )

    p.add_argument(
        "--merged-jsonl",
        type=Path,
        default=None,
        help=(
            "Final annotated JSONL with splitter predictions merged back. "
            "Required for --merge-only."
        ),
    )

    p.add_argument(
        "--smiles-col",
        default="canonical_smiles",
    )

    p.add_argument(
        "--valid-col",
        default="is_valid",
    )

    p.add_argument(
        "--splitter-prefix",
        default="splitter_",
    )

    p.add_argument(
        "--num-proc",
        type=int,
        default=int(
            os.environ.get(
                "SLURM_CPUS_PER_TASK",
                "8",
            )
        ),
        help="PROTAC-Splitter worker processes.",
    )

    p.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help=(
            "Unique molecules per checkpointed "
            "split_protac call."
        ),
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
        help=(
            "Merge completed shard CSVs and then merge "
            "the lookup back onto the original JSONL."
        ),
    )

    return p.parse_args()


def validate_args(a: argparse.Namespace) -> None:
    if not a.input.exists():
        raise FileNotFoundError(
            f"Input JSONL not found: {a.input}"
        )

    if a.num_proc < 1:
        raise ValueError(
            "--num-proc must be >= 1"
        )

    if a.chunk_size < 1:
        raise ValueError(
            "--chunk-size must be >= 1"
        )

    if a.num_shards < 1:
        raise ValueError(
            "--num-shards must be >= 1"
        )

    if not a.merge_only:
        if (
            a.num_shards > 1
            and a.shard_index is None
        ):
            raise ValueError(
                "--shard-index is required "
                "when --num-shards > 1"
            )

        if (
            a.shard_index is not None
            and not (
                0
                <= a.shard_index
                < a.num_shards
            )
        ):
            raise ValueError(
                "--shard-index must satisfy "
                "0 <= shard-index < num-shards"
            )

    if (
        a.merge_only
        and a.merged_jsonl is None
    ):
        raise ValueError(
            "--merged-jsonl is required "
            "with --merge-only"
        )


# ============================================================
# JSONL helpers
# ============================================================

def iter_jsonl(
    path: Path,
) -> Iterator[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at "
                    f"{path}:{line_no}"
                ) from exc

            if not isinstance(
                record,
                dict,
            ):
                raise ValueError(
                    f"Expected JSON object at "
                    f"{path}:{line_no}"
                )

            yield record


def write_jsonl_record(
    handle,
    record: dict[str, Any],
) -> None:
    handle.write(
        json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def is_valid_record(
    record: dict[str, Any],
    valid_col: str,
) -> bool:
    value = record.get(
        valid_col
    )

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        (int, np.integer),
    ):
        return bool(value)

    if isinstance(
        value,
        str,
    ):
        return (
            value.strip().lower()
            in {
                "true",
                "1",
                "yes",
            }
        )

    return False


# ============================================================
# Build splitter input
# ============================================================

def build_full_split_input(
    input_path: Path,
    smiles_col: str,
    valid_col: str,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
]:
    """
    Build a one-row-per-unique-canonical-SMILES DataFrame.

    Deduplication here is ONLY for splitter inference.

    The full original JSONL is preserved and will later receive
    splitter annotations by exact canonical_smiles lookup.
    """

    unique_smiles: list[str] = []
    seen: set[str] = set()

    num_records = 0
    num_valid_records = 0
    num_valid_with_smiles = 0
    num_valid_missing_smiles = 0

    for record in iter_jsonl(
        input_path
    ):
        num_records += 1

        if valid_col not in record:
            raise KeyError(
                f"Input record is missing "
                f"'{valid_col}'."
            )

        if not is_valid_record(
            record,
            valid_col,
        ):
            continue

        num_valid_records += 1

        smiles_value = record.get(
            smiles_col
        )

        if (
            smiles_value is None
            or str(
                smiles_value
            ).strip() == ""
        ):
            num_valid_missing_smiles += 1
            continue

        smiles = str(
            smiles_value
        ).strip()

        num_valid_with_smiles += 1

        if smiles in seen:
            continue

        seen.add(
            smiles
        )
        unique_smiles.append(
            smiles
        )

    if (
        num_valid_missing_smiles
        > 0
    ):
        raise ValueError(
            f"{num_valid_missing_smiles:,} "
            f"valid records are missing "
            f"'{smiles_col}'."
        )

    split_input = pd.DataFrame(
        {
            smiles_col:
            unique_smiles
        }
    )

    stats = {
        "num_records":
            num_records,
        "num_valid_records":
            num_valid_records,
        "num_valid_with_smiles":
            num_valid_with_smiles,
        "num_unique_valid_smiles":
            len(unique_smiles),
    }

    return (
        split_input,
        stats,
    )


# ============================================================
# Sharding helpers
# ============================================================

def shard_view(
    full: pd.DataFrame,
    num_shards: int,
    shard_index: int | None,
) -> pd.DataFrame:
    if num_shards == 1:
        return (
            full.copy()
            .reset_index(
                drop=True
            )
        )

    assert (
        shard_index
        is not None
    )

    mask = (
        np.arange(
            len(full)
        )
        % num_shards
    ) == shard_index

    return (
        full.loc[
            mask
        ]
        .reset_index(
            drop=True
        )
    )


def shard_path(
    output: Path,
    shard_index: int,
    num_shards: int,
) -> Path:
    width = max(
        2,
        len(
            str(
                num_shards - 1
            )
        ),
    )

    return output.with_name(
        f"{output.stem}."
        f"part"
        f"{shard_index:0{width}d}"
        f"-of-"
        f"{num_shards:0{width}d}"
        f"{output.suffix}"
    )


# ============================================================
# Checkpoint helpers
# ============================================================

def load_cached(
    path: Path,
    smiles_col: str,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        path
    )

    if smiles_col not in df.columns:
        raise ValueError(
            f"Existing output "
            f"{path} has no "
            f"'{smiles_col}' column."
        )

    return (
        df
        .drop_duplicates(
            smiles_col,
            keep="last",
        )
        .reset_index(
            drop=True
        )
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
        path.suffix
        + ".tmp"
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
            columns=[
                smiles_col
            ]
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


# ============================================================
# JSON-safe conversion
# ============================================================

def json_safe_value(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        np.generic,
    ):
        value = value.item()

    if isinstance(
        value,
        float,
    ):
        if math.isnan(
            value
        ):
            return None

        if math.isinf(
            value
        ):
            return None

        return value

    if isinstance(
        value,
        (
            str,
            int,
            bool,
        ),
    ):
        return value

    # PROTAC-Splitter CSV output should normally
    # contain scalar values. This is a safe fallback.
    try:
        if pd.isna(
            value
        ):
            return None
    except Exception:
        pass

    return value


# ============================================================
# Merge splitter lookup back into original JSONL
# ============================================================

def merge_lookup_back_to_jsonl(
    input_jsonl: Path,
    lookup: pd.DataFrame,
    output_jsonl: Path,
    smiles_col: str,
    valid_col: str,
    prefix: str,
) -> None:
    if (
        lookup[
            smiles_col
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Splitter lookup contains "
            "duplicate canonical SMILES."
        )

    lookup_by_smiles: dict[
        str,
        dict[str, Any],
    ] = {}

    splitter_columns = [
        column
        for column
        in lookup.columns
        if column
        != smiles_col
    ]

    for _, row in (
        lookup.iterrows()
    ):
        smiles = str(
            row[
                smiles_col
            ]
        ).strip()

        lookup_by_smiles[
            smiles
        ] = {
            column:
            json_safe_value(
                row[column]
            )
            for column
            in splitter_columns
        }

    output_jsonl.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = output_jsonl.with_suffix(
        output_jsonl.suffix
        + ".tmp"
    )

    num_records = 0
    num_valid = 0
    num_lookup_found = 0
    num_predictions = 0

    prediction_column = (
        "default_pred_n0"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as out:
        for record in iter_jsonl(
            input_jsonl
        ):
            num_records += 1

            valid = is_valid_record(
                record,
                valid_col,
            )

            if valid:
                num_valid += 1

            smiles_value = record.get(
                smiles_col
            )

            smiles = (
                str(
                    smiles_value
                ).strip()
                if smiles_value
                is not None
                else ""
            )

            split_values = (
                lookup_by_smiles.get(
                    smiles
                )
                if (
                    valid
                    and smiles
                )
                else None
            )

            if split_values is None:
                record[
                    f"{prefix}lookup_found"
                ] = False

                record[
                    f"{prefix}has_prediction"
                ] = False

            else:
                num_lookup_found += 1

                record[
                    f"{prefix}lookup_found"
                ] = True

                for (
                    column,
                    value,
                ) in (
                    split_values.items()
                ):
                    output_field = (
                        f"{prefix}"
                        f"{column}"
                    )

                    if (
                        output_field
                        in record
                    ):
                        raise KeyError(
                            "Refusing to overwrite "
                            f"existing JSON field: "
                            f"{output_field}"
                        )

                    record[
                        output_field
                    ] = value

                prediction = (
                    split_values.get(
                        prediction_column
                    )
                )

                has_prediction = (
                    prediction
                    is not None
                    and str(
                        prediction
                    ).strip()
                    != ""
                )

                record[
                    f"{prefix}has_prediction"
                ] = bool(
                    has_prediction
                )

                if has_prediction:
                    num_predictions += 1

            write_jsonl_record(
                out,
                record,
            )

    os.replace(
        tmp,
        output_jsonl,
    )

    print(
        "[INFO] Merge-back summary:",
        flush=True,
    )
    print(
        f"  Original records: "
        f"{num_records:,}",
        flush=True,
    )
    print(
        f"  Valid records: "
        f"{num_valid:,}",
        flush=True,
    )
    print(
        f"  Valid records with "
        f"splitter lookup: "
        f"{num_lookup_found:,}",
        flush=True,
    )
    print(
        f"  Valid records with "
        f"default prediction: "
        f"{num_predictions:,}",
        flush=True,
    )
    print(
        f"  Final JSONL: "
        f"{output_jsonl}",
        flush=True,
    )


# ============================================================
# Merge completed shards
# ============================================================

def merge_shards(
    a: argparse.Namespace,
    full_input: pd.DataFrame,
) -> int:
    parts: list[
        pd.DataFrame
    ] = []

    for i in range(
        a.num_shards
    ):
        path = shard_path(
            a.output,
            i,
            a.num_shards,
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing shard "
                f"output: {path}"
            )

        part = pd.read_csv(
            path
        )

        if (
            a.smiles_col
            not in part.columns
        ):
            raise ValueError(
                f"Shard {path} lacks "
                f"'{a.smiles_col}'."
            )

        parts.append(
            part
        )

    merged = pd.concat(
        parts,
        ignore_index=True,
    )

    if (
        merged[
            a.smiles_col
        ]
        .duplicated()
        .any()
    ):
        dup_n = int(
            merged[
                a.smiles_col
            ]
            .duplicated()
            .sum()
        )

        raise ValueError(
            f"Shard outputs overlap: "
            f"{dup_n} duplicate SMILES."
        )

    # Restore the exact expected unique-SMILES order.
    merged = full_input.merge(
        merged,
        on=a.smiles_col,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing = int(
        (
            merged[
                "_merge"
            ]
            != "both"
        ).sum()
    )

    merged = merged.drop(
        columns=[
            "_merge"
        ]
    )

    if missing:
        raise ValueError(
            f"Cannot create final CSV: "
            f"{missing:,} expected "
            f"molecules are missing."
        )

    required = [
        a.smiles_col,
        "default_pred_n0",
        "model_name",
    ]

    missing_output_cols = [
        column
        for column
        in required
        if column
        not in merged.columns
    ]

    if missing_output_cols:
        raise ValueError(
            "Merged output missing "
            f"columns: "
            f"{missing_output_cols}"
        )

    atomic_save(
        merged,
        a.output,
    )

    print(
        f"[INFO] Merged "
        f"{a.num_shards} shards "
        f"-> {a.output}",
        flush=True,
    )

    print(
        f"[INFO] Unique valid "
        f"PROTACs in lookup: "
        f"{len(merged):,}",
        flush=True,
    )

    print(
        "[INFO] Rows with "
        "default_pred_n0: "
        f"{int(merged['default_pred_n0'].notna().sum()):,}"
        f"/{len(merged):,}",
        flush=True,
    )

    assert (
        a.merged_jsonl
        is not None
    )

    merge_lookup_back_to_jsonl(
        input_jsonl=a.input,
        lookup=merged,
        output_jsonl=(
            a.merged_jsonl
        ),
        smiles_col=(
            a.smiles_col
        ),
        valid_col=(
            a.valid_col
        ),
        prefix=(
            a.splitter_prefix
        ),
    )

    return 0


# ============================================================
# Run one shard
# ============================================================

def run_shard(
    a: argparse.Namespace,
    full_input: pd.DataFrame,
) -> int:
    try:
        from protac_splitter import (
            split_protac,
        )

    except ImportError as exc:
        raise ImportError(
            "PROTAC-Splitter is not installed."
        ) from exc

    target = shard_view(
        full_input,
        a.num_shards,
        a.shard_index,
    )

    if a.num_shards > 1:
        assert (
            a.shard_index
            is not None
        )

        out = shard_path(
            a.output,
            a.shard_index,
            a.num_shards,
        )

        label = (
            f"shard "
            f"{a.shard_index + 1}"
            f"/{a.num_shards}"
        )

    else:
        out = a.output
        label = "single job"

    cached = load_cached(
        out,
        a.smiles_col,
    )

    target_set = set(
        target[
            a.smiles_col
        ].astype(
            str
        )
    )

    if not cached.empty:
        cached = cached[
            cached[
                a.smiles_col
            ]
            .astype(
                str
            )
            .isin(
                target_set
            )
        ].copy()

    cached_smiles = (
        set(
            cached[
                a.smiles_col
            ].astype(
                str
            )
        )
        if not cached.empty
        else set()
    )

    missing = (
        target[
            ~target[
                a.smiles_col
            ]
            .astype(
                str
            )
            .isin(
                cached_smiles
            )
        ]
        .reset_index(
            drop=True
        )
    )

    print(
        f"[INFO] Running "
        f"{label}",
        flush=True,
    )

    print(
        f"[INFO] Target unique "
        f"molecules: "
        f"{len(target):,}",
        flush=True,
    )

    print(
        f"[INFO] Already processed: "
        f"{len(target) - len(missing):,}",
        flush=True,
    )

    print(
        f"[INFO] Still to split: "
        f"{len(missing):,}",
        flush=True,
    )

    print(
        f"[INFO] PROTAC-Splitter "
        f"workers: "
        f"{a.num_proc}",
        flush=True,
    )

    print(
        f"[INFO] Checkpoint "
        f"chunk size: "
        f"{a.chunk_size:,}",
        flush=True,
    )

    # Create/update resumable checkpoint.
    atomic_save(
        ordered_processed_rows(
            target,
            cached,
            a.smiles_col,
        ),
        out,
    )

    if len(
        missing
    ):
        n_chunks = (
            len(
                missing
            )
            + a.chunk_size
            - 1
        ) // a.chunk_size

        for (
            chunk_no,
            start,
        ) in enumerate(
            range(
                0,
                len(
                    missing
                ),
                a.chunk_size,
            ),
            start=1,
        ):
            chunk = missing.iloc[
                start:
                start
                + a.chunk_size
            ].copy()

            print(
                f"\n[{label}] "
                f"chunk "
                f"{chunk_no}/"
                f"{n_chunks}: "
                f"{len(chunk):,} "
                f"molecules",
                flush=True,
            )

            # ------------------------------------------------
            # EXACT SAME splitting configuration as the
            # existing full-data script:
            #
            #   model="adaptive"
            #   adaptive_use_transformer=False
            # ------------------------------------------------

            try:
                new = split_protac(
                    chunk,
                    protac_smiles_col=a.smiles_col,
                    model="adaptive",
                    adaptive_use_transformer=False,
                    num_proc=a.num_proc,
                    verbose=1,
                )

            except Exception as exc:
                print(
                    f"[WARNING] Chunk failed: {exc}. "
                    "Retrying molecule-by-molecule.",
                    flush=True,
                )

                rows = []

                for _, row in chunk.iterrows():
                    one = row.to_frame().T

                    try:
                        result = split_protac(
                            one,
                            protac_smiles_col=a.smiles_col,
                            model="adaptive",
                            adaptive_use_transformer=False,
                            num_proc=1,
                            verbose=0,
                        )

                        if not isinstance(result, pd.DataFrame):
                            result = pd.DataFrame(result)

                        if (
                            result.empty
                            or a.smiles_col not in result.columns
                        ):
                            raise RuntimeError(
                                "PROTAC-Splitter returned no result."
                            )

                        rows.append(result)

                    except Exception as mol_exc:
                        print(
                            f"[WARNING] Splitter failed for: "
                            f"{row[a.smiles_col]} | {mol_exc}",
                            flush=True,
                        )

                        rows.append(
                            pd.DataFrame(
                                [{
                                    a.smiles_col: row[a.smiles_col],
                                    "default_pred_n0": None,
                                    "model_name": None,
                                }]
                            )
                        )

                new = pd.concat(
                    rows,
                    ignore_index=True,
                )

            if not isinstance(
                new,
                pd.DataFrame,
            ):
                new = pd.DataFrame(
                    new
                )

            if (
                a.smiles_col
                not in new.columns
            ):
                raise ValueError(
                    "PROTAC-Splitter "
                    "returned no "
                    f"'{a.smiles_col}' "
                    "column.\n"
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
                f"[INFO] Checkpoint: "
                f"{len(checkpoint):,}/"
                f"{len(target):,} "
                f"molecules processed "
                f"-> {out}",
                flush=True,
            )

    final = pd.read_csv(
        out
    )

    if len(
        final
    ) != len(
        target
    ):
        raise RuntimeError(
            f"Shard incomplete: "
            f"{out} has "
            f"{len(final):,} rows; "
            f"expected "
            f"{len(target):,}."
        )

    required = [
        a.smiles_col,
        "default_pred_n0",
        "model_name",
    ]

    missing_cols = [
        column
        for column
        in required
        if column
        not in final.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Output missing "
            f"columns: "
            f"{missing_cols}"
        )

    print(
        f"\n[DONE] "
        f"{label}: "
        f"{out}",
        flush=True,
    )

    print(
        f"[DONE] Rows: "
        f"{len(final):,}",
        flush=True,
    )

    print(
        "[DONE] Rows with "
        "default_pred_n0: "
        f"{int(final['default_pred_n0'].notna().sum()):,}"
        f"/{len(final):,}",
        flush=True,
    )

    return 0


# ============================================================
# Main
# ============================================================

def main() -> int:
    a = parse_args()

    validate_args(
        a
    )

    (
        full_input,
        input_stats,
    ) = build_full_split_input(
        input_path=(
            a.input
        ),
        smiles_col=(
            a.smiles_col
        ),
        valid_col=(
            a.valid_col
        ),
    )

    print(
        f"[INFO] Input JSONL: "
        f"{a.input}",
        flush=True,
    )

    print(
        f"[INFO] Total generated "
        f"records: "
        f"{input_stats['num_records']:,}",
        flush=True,
    )

    print(
        f"[INFO] Valid generated "
        f"records: "
        f"{input_stats['num_valid_records']:,}",
        flush=True,
    )

    print(
        f"[INFO] Unique valid "
        f"canonical SMILES "
        f"requiring splitting: "
        f"{input_stats['num_unique_valid_smiles']:,}",
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