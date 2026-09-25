from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from utils import ensure_dir, write_json


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    return torch.device(device)


def _load_poi_metadata(
    metadata_csv: str | Path,
    uniprot_field: str,
    poi_field: str,
    sequence_field: str,
) -> list[dict[str, str]]:
    metadata_csv = Path(metadata_csv)

    df = pd.read_csv(
        metadata_csv,
        keep_default_na=False,
    )

    required_columns = [
        uniprot_field,
        poi_field,
        sequence_field,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing columns in {metadata_csv}: "
            f"{missing_columns}"
        )

    records: list[dict[str, str]] = []
    seen_uniprots: set[str] = set()

    for row_number, row in enumerate(
        df.to_dict(orient="records"),
        start=2,
    ):
        target_uniprot = row[uniprot_field]
        poi_norm = row[poi_field]
        sequence = row[sequence_field]

        if (
            not isinstance(target_uniprot, str)
            or not target_uniprot
        ):
            raise ValueError(
                f"Missing {uniprot_field!r} in "
                f"{metadata_csv} row {row_number}."
            )

        if (
            not isinstance(poi_norm, str)
            or not poi_norm
        ):
            raise ValueError(
                f"Missing {poi_field!r} in "
                f"{metadata_csv} row {row_number}."
            )

        if (
            not isinstance(sequence, str)
            or not sequence
        ):
            raise ValueError(
                f"Missing {sequence_field!r} in "
                f"{metadata_csv} row {row_number}."
            )

        if target_uniprot in seen_uniprots:
            raise ValueError(
                "Duplicate target_uniprot in "
                f"{metadata_csv}: "
                f"{target_uniprot!r}."
            )

        seen_uniprots.add(
            target_uniprot
        )

        records.append(
            {
                "target_uniprot": target_uniprot,
                "poi_norm": poi_norm,
                "sequence": sequence,
            }
        )

    if not records:
        raise ValueError(
            f"No POI metadata rows found in "
            f"{metadata_csv}."
        )

    return sorted(
        records,
        key=lambda record: record["target_uniprot"],
    )


def _validate_window_config(
    max_amino_acids_per_chunk: int,
    chunk_overlap: int,
) -> None:
    if max_amino_acids_per_chunk <= 0:
        raise ValueError(
            "max_amino_acids_per_chunk "
            "must be positive."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap must be >= 0."
        )

    if chunk_overlap >= max_amino_acids_per_chunk:
        raise ValueError(
            "chunk_overlap must be smaller than "
            "max_amino_acids_per_chunk."
        )


def _check_long_sequences(
    records: list[dict[str, str]],
    max_amino_acids_per_chunk: int,
    chunk_long_sequences: bool,
) -> None:
    if chunk_long_sequences:
        return

    long_uniprots = [
        record["target_uniprot"]
        for record in records
        if (
            len(record["sequence"])
            > max_amino_acids_per_chunk
        )
    ]

    if long_uniprots:
        raise ValueError(
            "Found proteins longer than "
            "max_amino_acids_per_chunk while "
            "chunk_long_sequences=False: "
            f"{long_uniprots}"
        )


def _build_sequence_windows(
    sequence: str,
    max_amino_acids_per_chunk: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    sequence_length = len(sequence)

    if (
        sequence_length
        <= max_amino_acids_per_chunk
    ):
        return [
            {
                "start": 0,
                "end": sequence_length,
                "sequence": sequence,
            }
        ]

    stride = (
        max_amino_acids_per_chunk
        - chunk_overlap
    )

    windows: list[dict[str, Any]] = []

    start = 0

    while start < sequence_length:
        end = min(
            start
            + max_amino_acids_per_chunk,
            sequence_length,
        )

        windows.append(
            {
                "start": int(start),
                "end": int(end),
                "sequence": sequence[start:end],
            }
        )

        if end == sequence_length:
            break

        start += stride

    return windows


def _extract_residue_embeddings(
    last_hidden_state: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    expected_lengths: list[int],
    cls_token_id: int,
    eos_token_id: int,
) -> list[torch.Tensor]:
    residue_embeddings: list[
        torch.Tensor
    ] = []

    for (
        batch_index,
        expected_length,
    ) in enumerate(expected_lengths):
        non_padding_tokens = int(
            attention_mask[
                batch_index
            ].sum().item()
        )

        expected_tokens = (
            expected_length + 2
        )

        if (
            non_padding_tokens
            != expected_tokens
        ):
            raise ValueError(
                "ESM token/residue count mismatch. "
                f"Expected {expected_tokens} "
                "non-padding tokens for "
                f"{expected_length} residues, "
                f"found {non_padding_tokens}."
            )

        if (
            int(
                input_ids[
                    batch_index,
                    0,
                ].item()
            )
            != cls_token_id
        ):
            raise ValueError(
                "Expected the first ESM token "
                "to be <cls>."
            )

        if (
            int(
                input_ids[
                    batch_index,
                    expected_length + 1,
                ].item()
            )
            != eos_token_id
        ):
            raise ValueError(
                "Expected the token after the "
                "residues to be <eos>."
            )

        residue_hidden = (
            last_hidden_state[
                batch_index,
                1:1 + expected_length,
                :,
            ]
        )

        if (
            int(
                residue_hidden.shape[0]
            )
            != expected_length
        ):
            raise ValueError(
                "ESM residue representation "
                "count mismatch. "
                f"Expected {expected_length}, "
                f"found "
                f"{int(residue_hidden.shape[0])}."
            )

        residue_embeddings.append(
            residue_hidden
            .detach()
            .cpu()
            .float()
        )

    return residue_embeddings


def _stitch_window_embeddings(
    windows: list[dict[str, Any]],
    window_embeddings: list[torch.Tensor],
    sequence_length: int,
) -> torch.Tensor:
    """
    Keep exactly one ESM-2 representation
    per original residue.

    Adjacent overlapping windows are split
    at the midpoint of their overlap.
    """
    if (
        len(windows)
        != len(window_embeddings)
    ):
        raise ValueError(
            "Number of sequence windows and "
            "window embeddings does not match: "
            f"{len(windows)} vs "
            f"{len(window_embeddings)}."
        )

    stitched_parts: list[
        torch.Tensor
    ] = []

    for (
        index,
        (
            window,
            residue_embedding,
        ),
    ) in enumerate(
        zip(
            windows,
            window_embeddings,
        )
    ):
        window_start = int(
            window["start"]
        )

        window_end = int(
            window["end"]
        )

        if index == 0:
            keep_start_global = (
                window_start
            )
        else:
            previous_window_end = int(
                windows[
                    index - 1
                ]["end"]
            )

            keep_start_global = (
                window_start
                + previous_window_end
            ) // 2

        if (
            index
            == len(windows) - 1
        ):
            keep_end_global = (
                window_end
            )
        else:
            next_window_start = int(
                windows[
                    index + 1
                ]["start"]
            )

            keep_end_global = (
                window_end
                + next_window_start
            ) // 2

        local_start = (
            keep_start_global
            - window_start
        )

        local_end = (
            keep_end_global
            - window_start
        )

        stitched_parts.append(
            residue_embedding[
                local_start:local_end
            ]
        )

    stitched = torch.cat(
        stitched_parts,
        dim=0,
    ).float()

    if (
        int(stitched.shape[0])
        != sequence_length
    ):
        raise ValueError(
            "Stitched residue count does "
            "not match protein length. "
            f"Expected {sequence_length}, "
            f"found "
            f"{int(stitched.shape[0])}."
        )

    return stitched


@torch.inference_mode()
def _embed_protein_sequence(
    sequence: str,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_amino_acids_per_chunk: int,
    chunk_long_sequences: bool,
    chunk_overlap: int,
) -> tuple[
    torch.Tensor,
    dict[str, int],
]:
    effective_overlap = (
        chunk_overlap
        if (
            len(sequence)
            > max_amino_acids_per_chunk
            and chunk_long_sequences
        )
        else 0
    )

    windows = _build_sequence_windows(
        sequence=sequence,
        max_amino_acids_per_chunk=(
            max_amino_acids_per_chunk
        ),
        chunk_overlap=effective_overlap,
    )

    window_embeddings: list[
        torch.Tensor
    ] = []

    for start in range(
        0,
        len(windows),
        batch_size,
    ):
        batch_windows = windows[
            start:start + batch_size
        ]

        batch_sequences = [
            window["sequence"]
            for window in batch_windows
        ]

        expected_lengths = [
            int(window["end"])
            - int(window["start"])
            for window in batch_windows
        ]

        encoded = tokenizer(
            batch_sequences,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value
            in encoded.items()
        }

        outputs = model(
            **encoded
        )

        window_embeddings.extend(
            _extract_residue_embeddings(
                last_hidden_state=(
                    outputs.last_hidden_state
                ),
                input_ids=(
                    encoded["input_ids"]
                ),
                attention_mask=(
                    encoded["attention_mask"]
                ),
                expected_lengths=(
                    expected_lengths
                ),
                cls_token_id=int(
                    tokenizer.cls_token_id
                ),
                eos_token_id=int(
                    tokenizer.eos_token_id
                ),
            )
        )

    stitched_residue_embeddings = (
        _stitch_window_embeddings(
            windows=windows,
            window_embeddings=(
                window_embeddings
            ),
            sequence_length=len(
                sequence
            ),
        )
    )

    protein_embedding = (
        stitched_residue_embeddings
        .mean(dim=0)
        .float()
    )

    embedding_info = {
        "sequence_length": int(
            len(sequence)
        ),
        "num_chunks": int(
            len(windows)
        ),
    }

    return (
        protein_embedding,
        embedding_info,
    )


def build_poi_embeddings(
    metadata_csv: str | Path,
    output_pt: str | Path,
    stats_json: str | Path,
    model_name_or_path: str,
    uniprot_field: str,
    poi_field: str,
    sequence_field: str,
    device: str,
    dtype: str,
    batch_size: int,
    max_amino_acids_per_chunk: int,
    chunk_long_sequences: bool,
    chunk_overlap: int,
    local_files_only: bool,
    pooling: str,
) -> dict[str, Any]:
    if pooling != "mean":
        raise ValueError(
            "Only pooling='mean' "
            "is supported."
        )

    if dtype != "float32":
        raise ValueError(
            "This pipeline requires "
            "dtype='float32'."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    _validate_window_config(
        max_amino_acids_per_chunk=(
            max_amino_acids_per_chunk
        ),
        chunk_overlap=chunk_overlap,
    )

    records = _load_poi_metadata(
        metadata_csv=metadata_csv,
        uniprot_field=uniprot_field,
        poi_field=poi_field,
        sequence_field=sequence_field,
    )

    _check_long_sequences(
        records=records,
        max_amino_acids_per_chunk=(
            max_amino_acids_per_chunk
        ),
        chunk_long_sequences=(
            chunk_long_sequences
        ),
    )

    device_obj = _resolve_device(
        device
    )

    configured_stride = (
        max_amino_acids_per_chunk
        - chunk_overlap
    )

    print(
        f"[INFO] Loading tokenizer: "
        f"{model_name_or_path}"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=(
                local_files_only
            ),
        )
    )

    if (
        tokenizer.cls_token_id
        is None
        or tokenizer.eos_token_id
        is None
    ):
        raise ValueError(
            "ESM tokenizer must define "
            "both cls_token_id and "
            "eos_token_id."
        )

    print(
        f"[INFO] Loading model: "
        f"{model_name_or_path}"
    )

    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
        local_files_only=(
            local_files_only
        ),
    )

    model.to(
        device=device_obj,
        dtype=torch.float32,
    )

    model.eval()
    model.requires_grad_(
        False
    )

    print(
        "[INFO] ESM-2 representation: "
        f"device={device_obj}, "
        "dtype=float32, "
        "pooling=mean, "
        f"chunk_size="
        f"{max_amino_acids_per_chunk}, "
        f"overlap={chunk_overlap}, "
        f"stride={configured_stride}, "
        f"chunk_long_sequences="
        f"{chunk_long_sequences}"
    )

    target_uniprots: list[str] = []

    embeddings_by_uniprot: dict[
        str,
        torch.Tensor,
    ] = {}

    metadata_by_uniprot: dict[
        str,
        dict[str, Any],
    ] = {}

    embedding_infos: dict[
        str,
        dict[str, int],
    ] = {}

    for index, record in enumerate(
        records,
        start=1,
    ):
        target_uniprot = (
            record["target_uniprot"]
        )

        poi_norm = (
            record["poi_norm"]
        )

        sequence = (
            record["sequence"]
        )

        print(
            f"[INFO] Embedding "
            f"{index}/{len(records)}: "
            f"{target_uniprot} "
            f"({poi_norm}), "
            f"length={len(sequence)}"
        )

        (
            embedding,
            embedding_info,
        ) = _embed_protein_sequence(
            sequence=sequence,
            tokenizer=tokenizer,
            model=model,
            device=device_obj,
            batch_size=batch_size,
            max_amino_acids_per_chunk=(
                max_amino_acids_per_chunk
            ),
            chunk_long_sequences=(
                chunk_long_sequences
            ),
            chunk_overlap=(
                chunk_overlap
            ),
        )

        target_uniprots.append(
            target_uniprot
        )

        embeddings_by_uniprot[
            target_uniprot
        ] = embedding

        metadata_by_uniprot[
            target_uniprot
        ] = {
            "poi_norm": poi_norm,
            "sequence_length": int(
                len(sequence)
            ),
        }

        embedding_infos[
            target_uniprot
        ] = embedding_info

    embedding_matrix = torch.stack(
        [
            embeddings_by_uniprot[
                target
            ]
            for target
            in target_uniprots
        ],
        dim=0,
    ).float()

    output_pt = Path(
        output_pt
    )

    stats_json = Path(
        stats_json
    )

    ensure_dir(
        output_pt.parent
    )

    ensure_dir(
        stats_json.parent
    )

    payload = {
        "target_uniprots": (
            target_uniprots
        ),
        "embeddings": (
            embedding_matrix
        ),
        "embeddings_by_uniprot": (
            embeddings_by_uniprot
        ),
        "metadata_by_uniprot": (
            metadata_by_uniprot
        ),
        "embedding_infos": (
            embedding_infos
        ),
        "model_name_or_path": (
            model_name_or_path
        ),
        "pooling": "mean",
        "normalize_embeddings": False,
        "dtype": "float32",
        "embedding_dim": int(
            embedding_matrix.shape[1]
        ),
        "long_sequence_strategy": (
            "overlapping_windows_"
            "stitch_then_mean"
            if chunk_long_sequences
            else "error_if_over_limit"
        ),
        "max_amino_acids_per_chunk": int(
            max_amino_acids_per_chunk
        ),
        "chunk_overlap": int(
            chunk_overlap
        ),
        "chunk_stride": int(
            configured_stride
        ),
        "stitching": (
            "overlap_midpoint_trim"
        ),
    }

    torch.save(
        payload,
        output_pt,
    )

    sequence_lengths = [
        len(record["sequence"])
        for record in records
    ]

    num_chunked_sequences = sum(
        info["num_chunks"] > 1
        for info
        in embedding_infos.values()
    )

    stats = {
        "metadata_csv": str(
            metadata_csv
        ),
        "output_pt": str(
            output_pt
        ),
        "stats_json": str(
            stats_json
        ),
        "model_name_or_path": (
            model_name_or_path
        ),
        "uniprot_field": (
            uniprot_field
        ),
        "poi_field": (
            poi_field
        ),
        "sequence_field": (
            sequence_field
        ),
        "device": str(
            device_obj
        ),
        "dtype": "float32",
        "pooling": "mean",
        "normalize_embeddings": False,
        "batch_size": int(
            batch_size
        ),
        "max_amino_acids_per_chunk": int(
            max_amino_acids_per_chunk
        ),
        "chunk_long_sequences": bool(
            chunk_long_sequences
        ),
        "chunk_overlap": int(
            chunk_overlap
        ),
        "chunk_stride": int(
            configured_stride
        ),
        "stitching": (
            "overlap_midpoint_trim"
        ),
        "long_sequence_strategy": (
            "overlapping_windows_"
            "stitch_then_mean"
            if chunk_long_sequences
            else "error_if_over_limit"
        ),
        "local_files_only": bool(
            local_files_only
        ),
        "num_embeddings": int(
            len(target_uniprots)
        ),
        "embedding_dim": int(
            embedding_matrix.shape[1]
        ),
        "min_sequence_length": int(
            min(sequence_lengths)
        ),
        "median_sequence_length": float(
            pd.Series(
                sequence_lengths
            ).median()
        ),
        "max_sequence_length": int(
            max(sequence_lengths)
        ),
        "num_chunked_sequences": int(
            num_chunked_sequences
        ),
        "max_num_chunks": int(
            max(
                info["num_chunks"]
                for info
                in embedding_infos.values()
            )
        ),
        "target_uniprots": (
            target_uniprots
        ),
    }

    write_json(
        stats,
        stats_json,
    )

    return stats