from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from chemistry import mol_from_smiles, murcko_scaffold_from_mol
from utils import ensure_dir, write_jsonl


def random_split_records(
    records: list[dict],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    indices = list(range(len(records)))

    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(len(records) * train_frac)
    n_val = int(len(records) * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    test = [records[i] for i in test_idx]

    return train, val, test


def scaffold_split_records(
    records: list[dict],
    smiles_key: str,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    scaffold_to_records: dict[str | None, list[dict]] = {}

    for record in records:
        mol = mol_from_smiles(record[smiles_key])
        scaffold = murcko_scaffold_from_mol(mol)

        scaffold_to_records.setdefault(
            scaffold,
            [],
        ).append(record)

    groups = list(scaffold_to_records.values())

    rng = random.Random(seed)
    rng.shuffle(groups)

    groups.sort(
        key=len,
        reverse=True,
    )

    n_total = len(records)
    train_target = int(n_total * train_frac)
    val_target = int(n_total * val_frac)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []

    for group in groups:
        if len(train) + len(group) <= train_target:
            train.extend(group)

        elif len(val) + len(group) <= val_target:
            val.extend(group)

        else:
            test.extend(group)

    return train, val, test


def conditional_top_poi_split_records(
    records: list[dict],
    poi_key: str,
    val_frac: float,
    test_frac: float,
    seed: int,
    top_n_eval_pois: int,
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
    dict,
]:
    """
    Seen-POI conditional split.

    The top-N most frequent POIs are selected as evaluation POIs.
    Each selected POI is split independently into train/validation/test.
    Every remaining POI is assigned entirely to training.
    """
    if top_n_eval_pois <= 0:
        raise ValueError(
            "top_n_eval_pois must be > 0"
        )

    poi_counts = Counter(
        str(record[poi_key]).strip()
        for record in records
    )

    if top_n_eval_pois > len(poi_counts):
        raise ValueError(
            f"top_n_eval_pois={top_n_eval_pois} exceeds "
            f"the number of unique POIs ({len(poi_counts)})."
        )

    # Highest molecule count first.
    # POI name is used as a deterministic tie-break.
    eval_pois = sorted(
        poi_counts,
        key=lambda poi: (
            -poi_counts[poi],
            poi,
        ),
    )[:top_n_eval_pois]

    eval_poi_set = set(eval_pois)

    poi_to_records: dict[
        str,
        list[dict],
    ] = defaultdict(list)

    for record in records:
        poi = str(
            record[poi_key]
        ).strip()

        poi_to_records[poi].append(
            record
        )

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []

    # All POIs outside the evaluation panel are train-only.
    for poi, group in poi_to_records.items():
        if poi not in eval_poi_set:
            train.extend(group)

    rng = random.Random(seed)

    # Split every evaluation POI independently.
    for poi in eval_pois:
        group = list(
            poi_to_records[poi]
        )

        rng.shuffle(group)

        n_total = len(group)
        n_val = int(
            round(
                n_total * val_frac
            )
        )
        n_test = int(
            round(
                n_total * test_frac
            )
        )
        n_train = (
            n_total
            - n_val
            - n_test
        )

        if min(n_train, n_val, n_test) < 1:
            raise ValueError(
                f"POI {poi!r} has too few records for the requested split: "
                f"train={n_train}, val={n_val}, test={n_test}."
            )

        train.extend(
            group[:n_train]
        )

        val.extend(
            group[
                n_train:
                n_train + n_val
            ]
        )

        test.extend(
            group[
                n_train + n_val:
            ]
        )

    train_counts = Counter(
        str(record[poi_key]).strip()
        for record in train
    )

    val_counts = Counter(
        str(record[poi_key]).strip()
        for record in val
    )

    test_counts = Counter(
        str(record[poi_key]).strip()
        for record in test
    )

    num_eval_poi_records = sum(
        poi_counts[poi]
        for poi in eval_pois
    )

    split_info = {
        "poi_key": poi_key,
        "top_n_eval_pois": (
            top_n_eval_pois
        ),
        "eval_pois": eval_pois,
        "eval_poi_split_counts": [
            {
                "poi": poi,
                "total": int(
                    poi_counts[poi]
                ),
                "train": int(
                    train_counts[poi]
                ),
                "val": int(
                    val_counts[poi]
                ),
                "test": int(
                    test_counts[poi]
                ),
            }
            for poi in eval_pois
        ],
        "num_unique_pois": int(
            len(poi_counts)
        ),
        "num_eval_poi_records": int(
            num_eval_poi_records
        ),
        "num_train_only_poi_records": int(
            len(records)
            - num_eval_poi_records
        ),
        "num_train_only_pois": int(
            len(poi_counts)
            - len(eval_pois)
        ),
    }

    return (
        train,
        val,
        test,
        split_info,
    )


def shuffle_split_records(
    train: list[dict],
    val: list[dict],
    test: list[dict],
    seed: int,
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
]:
    rng = random.Random(seed)

    train = list(train)
    val = list(val)
    test = list(test)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def prepare_dataset(
    input_csv: str | Path,
    output_dir: str | Path,
    smiles_column: str,
    keep_columns: list[str],
    split_method: str,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    poi_column: str | None = None,
    top_n_eval_pois: int | None = None,
) -> dict:
    if abs(
        (
            train_frac
            + val_frac
            + test_frac
        )
        - 1.0
    ) > 1e-8:
        raise ValueError(
            "train_frac + val_frac + test_frac "
            "must sum to 1.0"
        )

    df = pd.read_csv(
        input_csv
    )

    input_records = len(df)

    selected_columns = [
        smiles_column,
        *keep_columns,
    ]

    if split_method == "conditional_top_poi":
        if poi_column is None:
            raise ValueError(
                "poi_column is required for "
                "split_method='conditional_top_poi'"
            )

        if top_n_eval_pois is None:
            raise ValueError(
                "top_n_eval_pois is required for "
                "split_method='conditional_top_poi'"
            )

        if poi_column not in selected_columns:
            selected_columns.append(
                poi_column
            )

    selected_columns = list(
        dict.fromkeys(
            selected_columns
        )
    )

    # The canonical CSV schema is expected here.
    # Pandas raises KeyError automatically if a requested
    # column does not exist.
    df = df[
        selected_columns
    ].copy()

    df = df.rename(
        columns={
            smiles_column: "smiles",
        }
    )

    records = df.to_dict(
        orient="records"
    )

    split_info: dict | None = None

    if split_method == "random":
        train, val, test = (
            random_split_records(
                records=records,
                train_frac=train_frac,
                val_frac=val_frac,
                seed=seed,
            )
        )

    elif split_method == "scaffold":
        train, val, test = (
            scaffold_split_records(
                records=records,
                smiles_key="smiles",
                train_frac=train_frac,
                val_frac=val_frac,
                seed=seed,
            )
        )

    elif split_method == "conditional_top_poi":
        (
            train,
            val,
            test,
            split_info,
        ) = conditional_top_poi_split_records(
            records=records,
            poi_key=poi_column,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
            top_n_eval_pois=(
                top_n_eval_pois
            ),
        )

        train, val, test = (
            shuffle_split_records(
                train=train,
                val=val,
                test=test,
                seed=seed,
            )
        )

    else:
        raise ValueError(
            f"Unknown split method: "
            f"{split_method}"
        )

    output_dir = ensure_dir(
        output_dir
    )

    write_jsonl(
        train,
        output_dir / "train.jsonl",
    )

    write_jsonl(
        val,
        output_dir / "val.jsonl",
    )

    write_jsonl(
        test,
        output_dir / "test.jsonl",
    )

    stats = {
        "input_csv": str(
            input_csv
        ),
        "output_dir": str(
            output_dir
        ),
        "input_records": (
            input_records
        ),
        "train_records": len(
            train
        ),
        "val_records": len(
            val
        ),
        "test_records": len(
            test
        ),
        "split_method": (
            split_method
        ),
        "seed": seed,
        "train_frac": (
            train_frac
        ),
        "val_frac": (
            val_frac
        ),
        "test_frac": (
            test_frac
        ),
        "kept_columns": list(
            df.columns
        ),
    }

    if split_info is not None:
        stats[
            "conditional_split"
        ] = split_info

    return stats