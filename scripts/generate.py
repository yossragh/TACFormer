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
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but no GPU is available.")

    return torch.device(device_name)


def load_model(
    model_path: str | Path,
    device: torch.device,
) -> GPTForCausalLM:
    """
    Load a Hugging Face-style saved model.

    Expected directory structure:
        final_model/
            config.json
            model.safetensors

    or a Trainer checkpoint directory:
        checkpoint-XXXX/
            config.json
            model.safetensors
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if not model_path.is_dir():
        raise ValueError(
            f"Expected a Hugging Face model directory, but got a file: {model_path}"
        )

    config_path = model_path / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find config.json in {model_path}. "
            "Use a model directory saved by trainer.save_model(...), "
            "or a valid Trainer checkpoint directory."
        )

    model = GPTForCausalLM.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    return model


def _resolve_max_new_tokens(
    max_new_tokens: Any,
    model: GPTForCausalLM,
) -> int:
    """
    Resolve and validate max_new_tokens.

    Generation starts from one BOS token, so the maximum number of new tokens
    is model.config.max_seq_len - 1.
    """
    if max_new_tokens is None:
        max_new_tokens = model.config.max_seq_len - 1

    max_new_tokens = int(max_new_tokens)
    max_allowed = int(model.config.max_seq_len) - 1

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")

    if max_new_tokens > max_allowed:
        raise ValueError(
            f"max_new_tokens={max_new_tokens} exceeds model context limit "
            f"{max_allowed} when starting from BOS."
        )

    return max_new_tokens


def _validate_generation_args(
    num_samples: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> None:
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be > 0 or None")

    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1] or None")


def _get_bad_token_ids(tokenizer: SmilesTokenizer) -> list[int]:
    """
    Tokens blocked during generation.

    EOS is intentionally excluded because the model must be able to emit it
    to stop generation.
    """
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
    on_batch: Callable[[list[dict[str, Any]]], None],
    show_progress: bool = False,
) -> int:
    """
    Shared raw-generation loop.

    This function only generates raw decoded SMILES records.

    It does NOT:
        - canonicalize SMILES
        - compute validity
        - compute uniqueness
        - compute novelty
        - compute diversity

    Those steps belong in evaluate.py.

    Note:
        This function does not change model.train()/model.eval().
        The caller is responsible for setting the desired model mode.
    """
    _validate_generation_args(
        num_samples=num_samples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    bad_token_ids = _get_bad_token_ids(tokenizer)

    generated = 0

    with torch.inference_mode():
        with tqdm(
            total=num_samples,
            desc="Generating",
            unit="mol",
            disable=not show_progress,
        ) as pbar:
            while generated < num_samples:
                current_batch_size = min(batch_size, num_samples - generated)

                start_ids = torch.full(
                    (current_batch_size, 1),
                    tokenizer.bos_id,
                    dtype=torch.long,
                    device=device,
                )

                output_ids = model.generate(
                    input_ids=start_ids,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=tokenizer.eos_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    bad_token_ids=bad_token_ids,
                )

                batch_records: list[dict[str, Any]] = []

                for local_idx, token_ids in enumerate(output_ids.tolist()):
                    raw_smiles = tokenizer.decode(
                        token_ids,
                        stop_at_eos=True,
                        skip_special_tokens=True,
                    )

                    batch_records.append(
                        {
                            "sample_id": generated + local_idx,
                            "raw_smiles": raw_smiles,
                        }
                    )

                on_batch(batch_records)

                generated += current_batch_size
                pbar.update(current_batch_size)

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
) -> tuple[int, int]:
    """
    Generate raw SMILES records and write them incrementally to JSONL.

    This is the CLI/offline generation path.

    Returns:
        n_written:
            Number of generated records written.
        resolved_max_new_tokens:
            The validated max_new_tokens value actually used.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    resolved_max_new_tokens = _resolve_max_new_tokens(
        max_new_tokens=max_new_tokens,
        model=model,
    )

    with open(output_path, "w", encoding="utf-8") as out_f:

        def write_batch(records: list[dict[str, Any]]) -> None:
            for record in records:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

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
        )

    return n_written, resolved_max_new_tokens


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
) -> list[dict[str, Any]]:
    """
    Generate raw SMILES records in memory.

    This is the importable helper for training-time monitoring and W&B logging.

    Note:
        This function does not change model.train()/model.eval().
        The caller is responsible for setting the desired model mode.
    """
    resolved_max_new_tokens = _resolve_max_new_tokens(
        max_new_tokens=max_new_tokens,
        model=model,
    )

    records: list[dict[str, Any]] = []

    def collect_batch(batch_records: list[dict[str, Any]]) -> None:
        records.extend(batch_records)

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
    )

    return records


def _write_generation_info(
    path: str | Path,
    model: GPTForCausalLM,
    model_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
    generation_cfg: dict[str, Any],
) -> None:
    """
    Save generation metadata.

    This is not an evaluation summary.
    No validity, uniqueness, novelty, or diversity metrics are written here.
    """
    info = {
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "output_path": str(output_path),
        "model_config": model.config.to_dict(),
        "generation": generation_cfg,
    }

    write_json(info, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate raw SMILES from a trained unconditional GPT-style PROTAC model."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )

    args = parser.parse_args()

    cfg = load_yaml(args.config)

    output_path = Path(cfg["generation"]["output_path"])
    ensure_dir(output_path.parent)

    save_yaml(
        cfg,
        output_path.parent / "generate.resolved.yaml",
    )

    _set_seed(int(cfg["generation"]["seed"]))

    device = _resolve_device(
        cfg["generation"].get("device", "cuda")
    )

    tokenizer = SmilesTokenizer.load(cfg["tokenizer"]["path"])

    model_path = cfg["model"]["path"]

    model = load_model(
        model_path=model_path,
        device=device,
    )

    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size ({tokenizer.vocab_size}) does not match "
            f"model vocab_size ({model.config.vocab_size}). "
            "Use the tokenizer that was used when training this model."
        )

    max_new_tokens_cfg = cfg["generation"].get("max_new_tokens")
    top_k = parse_optional_int(cfg["generation"].get("top_k"))
    top_p = parse_optional_float(cfg["generation"].get("top_p"))

    num_samples = int(cfg["generation"]["num_samples"])
    batch_size = int(cfg["generation"]["batch_size"])
    temperature = float(cfg["generation"]["temperature"])

    print(f"[INFO] Model:                 {model_path}")
    print(f"[INFO] Tokenizer:             {cfg['tokenizer']['path']}")
    print(f"[INFO] Output:                {output_path}")
    print(f"[INFO] Device:                {device}")
    print(f"[INFO] num_samples:           {num_samples}")
    print(f"[INFO] batch_size:            {batch_size}")
    print(f"[INFO] max_new_tokens config: {max_new_tokens_cfg}")
    print(f"[INFO] temperature:           {temperature}")
    print(f"[INFO] top_k:                 {top_k}")
    print(f"[INFO] top_p:                 {top_p}")

    n_written, resolved_max_new_tokens = generate_to_file(
        model=model,
        tokenizer=tokenizer,
        output_path=output_path,
        num_samples=num_samples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens_cfg,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
    )

    print(f"[INFO] max_new_tokens used:   {resolved_max_new_tokens}")

    generation_cfg = {
        "seed": int(cfg["generation"]["seed"]),
        "device": cfg["generation"].get("device", "cuda"),
        "num_samples": num_samples,
        "batch_size": batch_size,
        "max_new_tokens": resolved_max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }

    generation_info_path = output_path.parent / "generation_info.json"

    _write_generation_info(
        path=generation_info_path,
        model=model,
        model_path=model_path,
        tokenizer_path=cfg["tokenizer"]["path"],
        output_path=output_path,
        generation_cfg=generation_cfg,
    )

    print(f"\n[DONE] Wrote {n_written} raw generated SMILES to: {output_path}")
    print(f"[INFO] Metadata: {generation_info_path}")
    print(
        "[INFO] Run evaluate.py later for validity, uniqueness, novelty, "
        "and diversity metrics."
    )


if __name__ == "__main__":
    main()