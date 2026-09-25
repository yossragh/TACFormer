#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import apply_cli_overrides, load_yaml
from datasetv2 import CausalLMCollator, SmilesDataset
from tokenizer import SmilesTokenizer
from transformerv2 import GPTConfig, GPTForCausalLM
from utils import ensure_dir, write_json


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total,
        "trainable_params": trainable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train unconditional or POI-conditional GPT-style SMILES model."
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

    output_dir = ensure_dir(cfg["train"]["output_dir"])

    seed = int(cfg["train"]["seed"])
    _set_seed(seed)

    # -------------------------
    # Conditioning
    # -------------------------
    # NEW: defaults preserve the original unconditional setup.
    conditioning_cfg = cfg.get("conditioning", {})
    conditioning_enabled = bool(
        conditioning_cfg.get("enabled", False)
    )

    poi_embeddings_by_uniprot = None

    target_uniprot_field = conditioning_cfg.get(
        "target_uniprot_field",
        "target_uniprot",
    )

    poi_embedding_dim = int(
        conditioning_cfg.get(
            "poi_embedding_dim",
            1280,
        )
    )

    # NEW: load the precomputed POI embedding lookup once.
    if conditioning_enabled:
        print(
            "[INFO] Loading POI embeddings from: "
            f"{conditioning_cfg['poi_embeddings_path']}"
        )

        poi_embedding_payload = torch.load(
            conditioning_cfg["poi_embeddings_path"],
            map_location="cpu",
        )

        poi_embeddings_by_uniprot = (
            poi_embedding_payload[
                "embeddings_by_uniprot"
            ]
        )

    # -------------------------
    # Tokenizer
    # -------------------------
    tokenizer = SmilesTokenizer.load(
        cfg["tokenizer"]["path"]
    )

    # -------------------------
    # Datasets
    # -------------------------
    train_dataset = SmilesDataset(
        path=cfg["data"]["train_jsonl"],
        tokenizer=tokenizer,
        smiles_field=cfg["data"]["smiles_field"],
        max_seq_len=int(
            cfg["data"]["max_seq_len"]
        ),
        randomize_smiles=bool(
            cfg["data"].get(
                "randomize_train_smiles",
                False,
            )
        ),
        randomize_probability=float(
            cfg["data"].get(
                "randomize_train_probability",
                1.0,
            )
        ),
        randomize_max_attempts=int(
            cfg["data"].get(
                "randomize_train_max_attempts",
                10,
            )
        ),

        # NEW: None preserves unconditional dataset behavior.
        poi_embeddings_by_uniprot=(
            poi_embeddings_by_uniprot
        ),
        target_uniprot_field=(
            target_uniprot_field
        ),
    )

    val_dataset = SmilesDataset(
        path=cfg["data"]["val_jsonl"],
        tokenizer=tokenizer,
        smiles_field=cfg["data"]["smiles_field"],
        max_seq_len=int(
            cfg["data"]["max_seq_len"]
        ),
        randomize_smiles=False,

        # NEW: None preserves unconditional dataset behavior.
        poi_embeddings_by_uniprot=(
            poi_embeddings_by_uniprot
        ),
        target_uniprot_field=(
            target_uniprot_field
        ),
    )

    data_collator = CausalLMCollator(
        pad_id=tokenizer.pad_id,
    )

    # -------------------------
    # Model
    # -------------------------
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=int(
            cfg["model"]["max_seq_len"]
        ),
        n_layers=int(
            cfg["model"]["n_layers"]
        ),
        d_model=int(
            cfg["model"]["d_model"]
        ),
        n_heads=int(
            cfg["model"]["n_heads"]
        ),
        mlp_ratio=int(
            cfg["model"]["mlp_ratio"]
        ),
        dropout=float(
            cfg["model"]["dropout"]
        ),
        bias=bool(
            cfg["model"]["bias"]
        ),
        pad_token_id=tokenizer.pad_id,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        tie_word_embeddings=True,

        # NEW
        poi_conditioning=(
            conditioning_enabled
        ),
        poi_embedding_dim=(
            poi_embedding_dim
        ),
    )

    #model = GPTForCausalLM(model_config)

    init_checkpoint = cfg["model"].get(
        "init_checkpoint"
    )

    if init_checkpoint:
        print(
            f"[INFO] Loading pretrained "
            f"model from: "
            f"{init_checkpoint}"
        )

        if conditioning_enabled:
            # NEW: load the same pretrained model configuration,
            # adding only the POI-conditioning fields.
            conditional_config = (
                GPTConfig.from_pretrained(
                    init_checkpoint,
                    poi_conditioning=True,
                    poi_embedding_dim=(
                        poi_embedding_dim
                    ),
                )
            )

            # NEW: pretrained chemical weights are loaded
            # into the expanded conditional architecture.
            model = (
                GPTForCausalLM.from_pretrained(
                    init_checkpoint,
                    config=conditional_config,
                )
            )

            # NEW: initialize only parameters that did not
            # exist in the unconditional checkpoint.
            model.poi_norm.reset_parameters()

            model._init_weights(
                model.poi_proj
            )

            # NEW: gates must start exactly at zero so the
            # initial conditional contribution is zero.
            with torch.no_grad():
                for block in model.blocks:
                    block.condition_gate.zero_()

        else:
            model = (
                GPTForCausalLM.from_pretrained(
                    init_checkpoint
                )
            )

    else:
        print(
            "[INFO] Initializing model "
            "from scratch"
        )

        model = GPTForCausalLM(
            model_config
        )

    # -------------------------
    # TrainingArguments
    # -------------------------
    training_args = TrainingArguments(
        output_dir=str(output_dir),

        seed=seed,

        num_train_epochs=(
            cfg["train"]["epochs"]
        ),

        per_device_train_batch_size=(
            cfg["train"][
                "per_device_train_batch_size"
            ]
        ),

        per_device_eval_batch_size=(
            cfg["train"][
                "per_device_eval_batch_size"
            ]
        ),

        learning_rate=(
            cfg["train"]["learning_rate"]
        ),

        weight_decay=(
            cfg["train"]["weight_decay"]
        ),

        gradient_accumulation_steps=(
            cfg["train"].get(
                "gradient_accumulation_steps",
                1,
            )
        ),

        max_grad_norm=(
            cfg["train"].get(
                "max_grad_norm",
                1.0,
            )
        ),

        warmup_steps=(
            cfg["train"].get(
                "warmup_ratio",
                0.05,
            )
        ),

        lr_scheduler_type=(
            cfg["train"].get(
                "lr_scheduler_type",
                "cosine",
            )
        ),

        logging_steps=(
            cfg["train"]["logging_steps"]
        ),

        save_strategy=(
            cfg["train"]["save_strategy"]
        ),

        eval_strategy=(
            cfg["train"]["eval_strategy"]
        ),

        save_total_limit=(
            cfg["train"]["save_total_limit"]
        ),

        fp16=cfg["train"]["fp16"],
        bf16=cfg["train"]["bf16"],

        report_to=(
            cfg["train"]["report_to"]
        ),

        run_name=(
            cfg["train"].get(
                "run_name"
            )
        ),

        remove_unused_columns=False,
        label_names=["labels"],
    )

    # -------------------------
    # Trainer
    # -------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # -------------------------
    # Optional generation monitoring
    # -------------------------
    generation_monitor_enabled = (
        cfg["train"].get(
            "generation_monitor",
            {},
        ).get(
            "enabled",
            False,
        )
    )

    if generation_monitor_enabled:
        from chemistry import (
            suppress_rdkit_warnings
        )

        from evaluate import (
            load_reference_sets
        )

        from generation_monitor_callbackv2 import (
            GenerationMonitorCallback
        )

        suppress_rdkit_warnings()

        gen_cfg = (
            cfg["train"][
                "generation_monitor"
            ]
        )

        reference_paths = gen_cfg.get(
            "reference_jsonls",
            [
                cfg["data"][
                    "train_jsonl"
                ]
            ],
        )

        reference_paths = [
            str(path)
            for path in reference_paths
            if (
                path is not None
                and str(path).strip()
            )
        ]

        (
            reference_set,
            reference_scaffold_set,
        ) = load_reference_sets(
            paths=reference_paths,
            smiles_field=(
                gen_cfg.get(
                    "reference_smiles_field",
                    cfg["data"][
                        "smiles_field"
                    ],
                )
            ),
        )

        # NEW: unconditional monitoring keeps the
        # original num_samples behavior.
        monitor_targets = None

        monitor_num_samples = (
            gen_cfg.get(
                "num_samples",
                100,
            )
        )

        if conditioning_enabled:
            # NEW: collect each validation POI once.
            monitor_targets_by_uniprot = {}

            for record in val_dataset.records:
                target_uniprot = str(
                    record[
                        target_uniprot_field
                    ]
                ).strip()

                if (
                    target_uniprot
                    not in
                    monitor_targets_by_uniprot
                ):
                    monitor_targets_by_uniprot[
                        target_uniprot
                    ] = {
                        "target_uniprot": (
                            target_uniprot
                        ),
                        "poi_norm": str(
                            record[
                                "poi_norm"
                            ]
                        ).strip(),
                    }

            monitor_targets = list(
                monitor_targets_by_uniprot.values()
            )

            # NEW: conditional monitoring is defined
            # by samples per POI.
            samples_per_poi = int(
                gen_cfg.get(
                    "samples_per_poi",
                    200,
                )
            )

            monitor_num_samples = (
                samples_per_poi
                * len(
                    monitor_targets
                )
            )

        gen_callback = (
            GenerationMonitorCallback(
                tokenizer=tokenizer,

                reference_set=(
                    reference_set
                ),

                reference_scaffold_set=(
                    reference_scaffold_set
                ),

                num_samples=(
                    monitor_num_samples
                ),

                batch_size=(
                    gen_cfg.get(
                        "batch_size",
                        32,
                    )
                ),

                max_new_tokens=(
                    gen_cfg.get(
                        "max_new_tokens"
                    )
                ),

                temperature=(
                    gen_cfg.get(
                        "temperature",
                        1.0,
                    )
                ),

                top_k=(
                    gen_cfg.get(
                        "top_k"
                    )
                ),

                top_p=(
                    gen_cfg.get(
                        "top_p",
                        0.9,
                    )
                ),

                every_n_epochs=(
                    gen_cfg.get(
                        "every_n_epochs",
                        1,
                    )
                ),

                fp_radius=(
                    gen_cfg.get(
                        "fp_radius",
                        2,
                    )
                ),

                fp_n_bits=(
                    gen_cfg.get(
                        "fp_n_bits",
                        2048,
                    )
                ),

                max_diversity_samples=(
                    gen_cfg.get(
                        "max_diversity_samples",
                        1000,
                    )
                ),

                seed=seed,

                log_molecule_images_to_wandb=(
                    gen_cfg.get(
                        "log_molecule_images_to_wandb",
                        True,
                    )
                ),

                num_valid_molecules_to_visualize=(
                    gen_cfg.get(
                        "num_valid_molecules_to_visualize",
                        10,
                    )
                ),

                mols_per_row=(
                    gen_cfg.get(
                        "mols_per_row",
                        5,
                    )
                ),

                sub_img_size=tuple(
                    gen_cfg.get(
                        "sub_img_size",
                        [
                            300,
                            300,
                        ],
                    )
                ),

                save_monitor_metrics_locally=(
                    gen_cfg.get(
                        "save_monitor_metrics_locally",
                        True,
                    )
                ),

                # NEW: both are None for unconditional monitoring.
                poi_embeddings_by_uniprot=(
                    poi_embeddings_by_uniprot
                    if conditioning_enabled
                    else None
                ),

                monitor_targets=(
                    monitor_targets
                    if conditioning_enabled
                    else None
                ),
            )
        )

        trainer.add_callback(
            gen_callback
        )

        gen_callback.add_trainer(
            trainer
        )

        print(
            "[INFO] Generation monitoring "
            "enabled."
        )

        print(
            "[INFO] Monitor samples "
            f"per epoch: "
            f"{monitor_num_samples}"
        )

        if conditioning_enabled:
            print(
                "[INFO] Monitor samples "
                "per POI: "
                f"{gen_cfg.get('samples_per_poi', 200)}"
            )

        print(
            "[INFO] Generation monitor "
            "batch size: "
            f"{gen_cfg.get('batch_size', 32)}"
        )

        print(
            f"[INFO] Reference molecules: "
            f"{len(reference_set)}"
        )

        print(
            f"[INFO] Reference scaffolds: "
            f"{len(reference_scaffold_set)}"
        )

        print(
            "[INFO] Molecule image logging: "
            f"{gen_cfg.get('log_molecule_images_to_wandb', True)}"
        )

        print(
            "[INFO] Valid molecules "
            "visualized per epoch: "
            f"{gen_cfg.get('num_valid_molecules_to_visualize', 10)}"
        )

    # -------------------------
    # Save run metadata
    # -------------------------
    run_info = {
        "seed": seed,

        "tokenizer_path": (
            cfg["tokenizer"]["path"]
        ),

        "train_size": len(
            train_dataset
        ),

        "val_size": len(
            val_dataset
        ),

        "init_checkpoint": (
            cfg["model"].get(
                "init_checkpoint"
            )
        ),

        "vocab_size": (
            tokenizer.vocab_size
        ),

        "randomize_smiles": (
            cfg["data"].get(
                "randomize_train_smiles",
                False,
            )
        ),

        "randomize_probability": (
            cfg["data"].get(
                "randomize_train_probability",
                1.0,
            )
        ),

        "randomize_max_attempts": (
            cfg["data"].get(
                "randomize_train_max_attempts",
                10,
            )
        ),

        "model_config": (
            model.config.to_dict()
        ),

        "generation_monitor_enabled": (
            generation_monitor_enabled
        ),

        "generation_monitor_config": (
            cfg["train"].get(
                "generation_monitor",
                {},
            )
        ),

        **_count_parameters(
            model
        ),
    }

    write_json(
        run_info,
        output_dir
        / "run_info.json",
    )

    print(
        f"[INFO] output_dir="
        f"{output_dir}"
    )

    print(
        f"[INFO] train_size="
        f"{len(train_dataset)} "
        f"val_size="
        f"{len(val_dataset)}"
    )

    print(
        f"[INFO] vocab_size="
        f"{tokenizer.vocab_size}"
    )

    print(
        f"[INFO] total_params="
        f"{run_info['total_params']:,}"
    )

    print(
        f"[INFO] trainable_params="
        f"{run_info['trainable_params']:,}"
    )

    print(
        f"[INFO] report_to="
        f"{cfg['train']['report_to']}"
    )

    print(
        f"[INFO] run_name="
        f"{cfg['train'].get('run_name')}"
    )

    # -------------------------
    # Train
    # -------------------------
    trainer.train()

    print(
        "[DONE] Training finished."
    )

    print(
        f"[DONE] Output dir: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()