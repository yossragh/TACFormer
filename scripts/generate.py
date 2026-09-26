#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    load_yaml,
    parse_optional_float,
    parse_optional_int,
    save_yaml,
)
from tokenizer import SmilesTokenizer  # noqa: E402
from transformer import GPTForCausalLM  # noqa: E402
from utils import ensure_dir, write_json  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Config requested CUDA, but no GPU is available."
        )

    return torch.device(device_name)


def load_model(
    model_path: str | Path,
    device: torch.device,
) -> GPTForCausalLM:
    model_path = Path(model_path)

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Model directory does not exist: {model_path}"
        )

    model = GPTForCausalLM.from_pretrained(
        str(model_path)
    )
    model.to(device)
    model.eval()

    return model


def _model_uses_poi_conditioning(
    model: GPTForCausalLM,
) -> bool:
    return bool(
        getattr(
            model.config,
            "poi_conditioning",
            False,
        )
    )


def _resolve_max_new_tokens(
    max_new_tokens: Any,
    model: GPTForCausalLM,
) -> int:
    if max_new_tokens is None:
        max_new_tokens = (
            int(model.config.max_seq_len) - 1
        )

    max_new_tokens = int(max_new_tokens)
    max_allowed = (
        int(model.config.max_seq_len) - 1
    )

    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must be > 0."
        )

    if max_new_tokens > max_allowed:
        raise ValueError(
            f"max_new_tokens={max_new_tokens} exceeds "
            f"the available generation length {max_allowed} "
            "when starting from one BOS token."
        )

    return max_new_tokens


def _validate_generation_plan(
    num_samples: int,
    batch_size: int,
    poi_embeddings: torch.Tensor | None,
    sample_metadata: list[dict[str, Any]] | None,
) -> None:
    """
    Validate only script-level alignment.

    Model-specific checks such as POI embedding dimension,
    finite values, conditioning mode, temperature, top-k,
    and top-p are handled by GPTForCausalLM.generate/forward.
    """
    if num_samples <= 0:
        raise ValueError(
            "num_samples must be > 0."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be > 0."
        )

    if poi_embeddings is not None:
        if not isinstance(
            poi_embeddings,
            torch.Tensor,
        ):
            raise TypeError(
                "poi_embeddings must be a torch.Tensor or None."
            )

        if poi_embeddings.ndim != 2:
            raise ValueError(
                "poi_embeddings must have shape [N, D]. "
                f"Got {tuple(poi_embeddings.shape)}."
            )

        if int(poi_embeddings.shape[0]) != num_samples:
            raise ValueError(
                "poi_embeddings must contain one row per "
                f"requested sample: rows={poi_embeddings.shape[0]}, "
                f"num_samples={num_samples}."
            )

    if sample_metadata is not None:
        if len(sample_metadata) != num_samples:
            raise ValueError(
                "sample_metadata must contain one dictionary "
                f"per sample: len={len(sample_metadata)}, "
                f"num_samples={num_samples}."
            )

        for index, metadata in enumerate(
            sample_metadata
        ):
            if not isinstance(metadata, dict):
                raise TypeError(
                    "Every sample_metadata entry must be a dictionary. "
                    f"Entry {index} has type {type(metadata)}."
                )

            reserved = {
                "sample_id",
                "raw_smiles",
            }.intersection(metadata)

            if reserved:
                raise ValueError(
                    "sample_metadata may not overwrite reserved "
                    f"generation fields: {sorted(reserved)}"
                )


def _get_bad_token_ids(
    tokenizer: SmilesTokenizer,
) -> list[int]:
    # EOS is intentionally allowed so generation can stop.
    return [
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.unk_id,
    ]


def _run_generation_loop(
    model: GPTForCausalLM,
    tokenizer: SmilesTokenizer,
    num_samples: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    device: torch.device,
    on_batch: Callable[
        [list[dict[str, Any]]],
        None,
    ],
    show_progress: bool = False,
    poi_embeddings: torch.Tensor | None = None,
    sample_metadata: list[dict[str, Any]] | None = None,
) -> int:

    _validate_generation_plan(
        num_samples=num_samples,
        batch_size=batch_size,
        poi_embeddings=poi_embeddings,
        sample_metadata=sample_metadata,
    )

    bad_token_ids = _get_bad_token_ids(
        tokenizer
    )

    generated = 0

    with tqdm(
        total=num_samples,
        desc="Generating",
        unit="mol",
        disable=not show_progress,
    ) as pbar:
        while generated < num_samples:
            current_batch_size = min(
                batch_size,
                num_samples - generated,
            )

            batch_start = generated
            batch_end = (
                batch_start
                + current_batch_size
            )

            start_ids = torch.full(
                (
                    current_batch_size,
                    1,
                ),
                tokenizer.bos_id,
                dtype=torch.long,
                device=device,
            )

            batch_poi_embeddings = None

            if poi_embeddings is not None:
                # Move each POI batch to the model device once.
                batch_poi_embeddings = (
                    poi_embeddings[
                        batch_start:batch_end
                    ].to(device=device)
                )

            output_ids = model.generate(
                input_ids=start_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                bad_token_ids=bad_token_ids,
                poi_embeddings=batch_poi_embeddings,
            )

            batch_records: list[
                dict[str, Any]
            ] = []

            for local_idx, token_ids in enumerate(
                output_ids.tolist()
            ):
                global_idx = (
                    batch_start + local_idx
                )

                raw_smiles = tokenizer.decode(
                    token_ids,
                    stop_at_eos=True,
                    skip_special_tokens=True,
                )

                record: dict[str, Any] = {
                    "sample_id": global_idx,
                    "raw_smiles": raw_smiles,
                }

                if sample_metadata is not None:
                    record.update(
                        sample_metadata[
                            global_idx
                        ]
                    )

                batch_records.append(
                    record
                )

            on_batch(
                batch_records
            )

            generated += (
                current_batch_size
            )

            pbar.update(
                current_batch_size
            )

    return generated


def generate_to_file(
    model: GPTForCausalLM,
    tokenizer: SmilesTokenizer,
    output_path: str | Path,
    num_samples: int,
    batch_size: int,
    max_new_tokens: int | None,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    device: torch.device,
    poi_embeddings: torch.Tensor | None = None,
    sample_metadata: list[dict[str, Any]] | None = None,
) -> tuple[int, int]:
    """
    Generate raw SMILES and write them incrementally to JSONL.
    """
    output_path = Path(
        output_path
    )

    ensure_dir(
        output_path.parent
    )

    resolved_max_new_tokens = (
        _resolve_max_new_tokens(
            max_new_tokens=max_new_tokens,
            model=model,
        )
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as out_f:

        def write_batch(
            records: list[dict[str, Any]],
        ) -> None:
            for record in records:
                out_f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            out_f.flush()

        n_written = _run_generation_loop(
            model=model,
            tokenizer=tokenizer,
            num_samples=num_samples,
            batch_size=batch_size,
            max_new_tokens=resolved_max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
            on_batch=write_batch,
            show_progress=True,
            poi_embeddings=poi_embeddings,
            sample_metadata=sample_metadata,
        )

    return (
        n_written,
        resolved_max_new_tokens,
    )


def generate_to_memory(
    model: GPTForCausalLM,
    tokenizer: SmilesTokenizer,
    num_samples: int,
    batch_size: int,
    max_new_tokens: int | None,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    device: torch.device,
    poi_embeddings: torch.Tensor | None = None,
    sample_metadata: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate raw SMILES in memory.

    Used by generation_monitor_callback.py.
    """
    resolved_max_new_tokens = (
        _resolve_max_new_tokens(
            max_new_tokens=max_new_tokens,
            model=model,
        )
    )

    records: list[
        dict[str, Any]
    ] = []

    def collect_batch(
        batch_records: list[dict[str, Any]],
    ) -> None:
        records.extend(
            batch_records
        )

    _run_generation_loop(
        model=model,
        tokenizer=tokenizer,
        num_samples=num_samples,
        batch_size=batch_size,
        max_new_tokens=resolved_max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
        on_batch=collect_batch,
        show_progress=False,
        poi_embeddings=poi_embeddings,
        sample_metadata=sample_metadata,
    )

    return records


def _build_balanced_target_plan(
    target_uniprots: list[str],
    num_samples: int,
) -> list[str]:
    """
    Assign generations as evenly as possible across requested POIs.
    """
    targets: list[str] = []
    seen: set[str] = set()

    for target in target_uniprots:
        target = str(
            target
        ).strip()

        if not target:
            raise ValueError(
                "target_uniprots contains an empty value."
            )

        if target not in seen:
            targets.append(
                target
            )
            seen.add(
                target
            )

    if not targets:
        raise ValueError(
            "At least one target UniProt ID is required."
        )

    base_count, remainder = divmod(
        num_samples,
        len(targets),
    )

    plan: list[str] = []

    for index, target in enumerate(
        targets
    ):
        count = (
            base_count
            + (
                1
                if index < remainder
                else 0
            )
        )

        plan.extend(
            [target] * count
        )

    return plan


def _resolve_cli_conditioning(
    cfg: dict[str, Any],
    model: GPTForCausalLM,
    num_samples: int,
) -> tuple[
    torch.Tensor | None,
    list[dict[str, Any]] | None,
]:
    """
    Build the per-sample POI conditioning for standalone generation.

    The POI embedding artifact is trusted as the canonical output of
    protein_embeddings.py. Model-level dimension/value checks remain in
    GPTForCausalLM.
    """
    generation_cfg = cfg["generation"]

    single_target = generation_cfg.get(
        "target_uniprot"
    )
    multiple_targets = generation_cfg.get(
        "target_uniprots"
    )

    if not _model_uses_poi_conditioning(
        model
    ):
        return None, None

    conditioning_cfg = cfg.get(
        "conditioning",
        {},
    )

    embeddings_path = (
        conditioning_cfg.get(
            "poi_embeddings_path"
        )
    )

    if not embeddings_path:
        raise ValueError(
            "Conditional generation requires "
            "conditioning.poi_embeddings_path."
        )

    if (
        single_target is not None
        and multiple_targets is not None
    ):
        raise ValueError(
            "Specify only one of "
            "generation.target_uniprot or "
            "generation.target_uniprots."
        )

    if single_target is not None:
        target_uniprots = [
            str(single_target)
        ]

    elif multiple_targets is not None:
        if not isinstance(
            multiple_targets,
            list,
        ):
            raise TypeError(
                "generation.target_uniprots "
                "must be a YAML list."
            )

        target_uniprots = [
            str(target)
            for target in multiple_targets
        ]

    else:
        raise ValueError(
            "Conditional generation requires "
            "generation.target_uniprot or "
            "generation.target_uniprots."
        )

    payload = torch.load(
        embeddings_path,
        map_location="cpu",
    )

    embeddings_by_uniprot = (
        payload[
            "embeddings_by_uniprot"
        ]
    )

    target_plan = (
        _build_balanced_target_plan(
            target_uniprots=(
                target_uniprots
            ),
            num_samples=num_samples,
        )
    )

    missing_targets = sorted(
        set(target_plan)
        - set(
            embeddings_by_uniprot
        )
    )

    if missing_targets:
        raise KeyError(
            "Requested target UniProt IDs are "
            "missing from poi_embeddings.pt: "
            f"{missing_targets}"
        )

    poi_embeddings = torch.stack(
        [
            embeddings_by_uniprot[
                target
            ]
            for target
            in target_plan
        ],
        dim=0,
    )

    sample_metadata = [
        {
            "target_uniprot": target,
        }
        for target in target_plan
    ]

    return (
        poi_embeddings,
        sample_metadata,
    )


def _write_generation_info(
    path: str | Path,
    model: GPTForCausalLM,
    model_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
    generation_cfg: dict[str, Any],
) -> None:
    write_json(
        {
            "model_path": str(
                model_path
            ),
            "tokenizer_path": str(
                tokenizer_path
            ),
            "output_path": str(
                output_path
            ),
            "model_config": (
                model.config.to_dict()
            ),
            "generation": (
                generation_cfg
            ),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate raw SMILES from an "
            "unconditional or POI-conditional "
            "TACFormer model."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )

    args = parser.parse_args()

    cfg = load_yaml(
        args.config
    )

    output_path = Path(
        cfg["generation"][
            "output_path"
        ]
    )

    ensure_dir(
        output_path.parent
    )

    save_yaml(
        cfg,
        output_path.parent
        / "generate.resolved.yaml",
    )

    _set_seed(
        int(
            cfg["generation"][
                "seed"
            ]
        )
    )

    device = resolve_device(
        cfg["generation"].get(
            "device",
            "cuda",
        )
    )

    tokenizer = (
        SmilesTokenizer.load(
            cfg["tokenizer"]["path"]
        )
    )

    model_path = (
        cfg["model"]["path"]
    )

    model = load_model(
        model_path=model_path,
        device=device,
    )

    if (
        tokenizer.vocab_size
        != model.config.vocab_size
    ):
        raise ValueError(
            f"Tokenizer vocab_size "
            f"({tokenizer.vocab_size}) does not match "
            f"model vocab_size ({model.config.vocab_size}). "
            "Use the tokenizer used during training."
        )

    generation_cfg = (
        cfg["generation"]
    )

    max_new_tokens_cfg = (
        generation_cfg.get(
            "max_new_tokens"
        )
    )

    top_k = parse_optional_int(
        generation_cfg.get(
            "top_k"
        )
    )

    top_p = parse_optional_float(
        generation_cfg.get(
            "top_p"
        )
    )

    num_samples = int(
        generation_cfg[
            "num_samples"
        ]
    )

    batch_size = int(
        generation_cfg[
            "batch_size"
        ]
    )

    temperature = float(
        generation_cfg[
            "temperature"
        ]
    )

    (
        poi_embeddings,
        sample_metadata,
    ) = _resolve_cli_conditioning(
        cfg=cfg,
        model=model,
        num_samples=num_samples,
    )

    print(
        f"[INFO] Model: {model_path}"
    )
    print(
        f"[INFO] Tokenizer: "
        f"{cfg['tokenizer']['path']}"
    )
    print(
        f"[INFO] Output: {output_path}"
    )
    print(
        f"[INFO] Device: {device}"
    )
    print(
        "[INFO] Mode: "
        + (
            "POI-conditional"
            if _model_uses_poi_conditioning(
                model
            )
            else "unconditional"
        )
    )

    if sample_metadata is not None:
        unique_targets = sorted(
            {
                row[
                    "target_uniprot"
                ]
                for row
                in sample_metadata
            }
        )

        print(
            "[INFO] Conditional targets: "
            f"{len(unique_targets)}"
        )
        print(
            "[INFO] Target UniProt IDs: "
            f"{unique_targets}"
        )

    (
        n_written,
        resolved_max_new_tokens,
    ) = generate_to_file(
        model=model,
        tokenizer=tokenizer,
        output_path=output_path,
        num_samples=num_samples,
        batch_size=batch_size,
        max_new_tokens=(
            max_new_tokens_cfg
        ),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
        poi_embeddings=(
            poi_embeddings
        ),
        sample_metadata=(
            sample_metadata
        ),
    )

    run_info = {
        "seed": int(
            generation_cfg["seed"]
        ),
        "device": str(
            device
        ),
        "num_samples": (
            num_samples
        ),
        "batch_size": (
            batch_size
        ),
        "max_new_tokens": (
            resolved_max_new_tokens
        ),
        "temperature": (
            temperature
        ),
        "top_k": (
            top_k
        ),
        "top_p": (
            top_p
        ),
        "poi_conditioning": (
            _model_uses_poi_conditioning(
                model
            )
        ),
    }

    if sample_metadata is not None:
        run_info[
            "target_uniprots"
        ] = sorted(
            {
                row[
                    "target_uniprot"
                ]
                for row
                in sample_metadata
            }
        )

    generation_info_path = (
        output_path.parent
        / "generation_info.json"
    )

    _write_generation_info(
        path=generation_info_path,
        model=model,
        model_path=model_path,
        tokenizer_path=(
            cfg["tokenizer"][
                "path"
            ]
        ),
        output_path=output_path,
        generation_cfg=run_info,
    )

    print(
        f"[DONE] Wrote {n_written} raw generated "
        f"SMILES to: {output_path}"
    )
    print(
        f"[INFO] Metadata: "
        f"{generation_info_path}"
    )


if __name__ == "__main__":
    main()
