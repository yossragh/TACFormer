from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import TrainerCallback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate import (
    annotate_generated_records,
    compute_condition_nll_metrics,
    compute_generated_poi_knn_purity,
    compute_metrics,
)
from generate import generate_to_memory
from metrics import extract_scalar_numeric_metrics
from utils import append_jsonl_record
from visualize import draw_highlighted_scaffold_grid, draw_molecule_grid


class GenerationMonitorCallback(TrainerCallback):

    def __init__(
        self,
        tokenizer,
        reference_set: set[str],
        reference_scaffold_set: set[str],
        num_samples: int = 100,
        batch_size: int = 32,
        max_new_tokens: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = 0.9,
        every_n_epochs: int = 1,
        fp_radius: int = 2,
        fp_n_bits: int = 2048,
        max_diversity_samples: int = 1000,
        seed: int = 42,
        log_molecule_images_to_wandb: bool = True,
        num_valid_molecules_to_visualize: int = 10,
        mols_per_row: int = 5,
        sub_img_size: tuple[int, int] = (300, 300),
        save_monitor_metrics_locally: bool = True,
        poi_embeddings_by_uniprot: dict[str, torch.Tensor] | None = None,
        monitor_targets: list[dict[str, str]] | None = None,
        condition_nll_examples: list[dict] | None = None,
        condition_nll_batch_size: int = 64,
        poi_knn_reference_pool: dict | None = None,
        poi_knn_k: int = 5,
        poi_knn_max_valid_per_poi: int = 25,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")

        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if every_n_epochs <= 0:
            raise ValueError("every_n_epochs must be > 0")

        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be > 0 or None")

        if top_p is not None and not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1] or None")

        self.tokenizer = tokenizer
        self.reference_set = reference_set
        self.reference_scaffold_set = reference_scaffold_set

        self.num_samples = num_samples
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.every_n_epochs = every_n_epochs

        self.fp_radius = fp_radius
        self.fp_n_bits = fp_n_bits
        self.max_diversity_samples = max_diversity_samples
        self.seed = seed

        self.log_molecule_images_to_wandb = log_molecule_images_to_wandb
        self.num_valid_molecules_to_visualize = num_valid_molecules_to_visualize
        self.mols_per_row = mols_per_row
        self.sub_img_size = sub_img_size
        self.save_monitor_metrics_locally = save_monitor_metrics_locally

        self.poi_embeddings_by_uniprot = poi_embeddings_by_uniprot
        self.monitor_targets = monitor_targets

        self.condition_nll_examples = condition_nll_examples
        self.condition_nll_batch_size = int(condition_nll_batch_size)

        self.poi_knn_reference_pool = poi_knn_reference_pool
        self.poi_knn_k = int(poi_knn_k)
        self.poi_knn_max_valid_per_poi = int(
            poi_knn_max_valid_per_poi
        )

        self.trainer = None

    def add_trainer(self, trainer) -> None:
        self.trainer = trainer

    def _build_conditional_generation_inputs(
        self,
    ) -> tuple[torch.Tensor, list[dict[str, str]]]:
        """
        Build an equal-sized generation panel for every monitored POI.
        """
        num_targets = len(self.monitor_targets)

        if self.num_samples % num_targets != 0:
            raise ValueError(
                "Conditional monitor num_samples must be divisible "
                "by the number of monitor targets."
            )

        samples_per_poi = self.num_samples // num_targets

        embeddings: list[torch.Tensor] = []
        metadata: list[dict[str, str]] = []

        for target in self.monitor_targets:
            target_uniprot = target["target_uniprot"]
            embedding = self.poi_embeddings_by_uniprot[target_uniprot]

            for _ in range(samples_per_poi):
                embeddings.append(embedding)
                metadata.append(dict(target))

        return (
            torch.stack(embeddings, dim=0).float(),
            metadata,
        )

    def _build_monitor_metrics_row(
        self,
        scalar_metrics: dict[str, int | float | None],
        epoch: int,
        global_step: int,
    ) -> dict:
        """
        Build one local JSONL row for reproducible plotting.
        """
        row: dict = {
            "epoch": int(epoch),
            "global_step": int(global_step),
        }

        row.update(scalar_metrics)

        return row

    def _append_monitor_metrics_locally(
        self,
        row: dict,
        output_dir: str | Path,
    ) -> None:
        """
        Append one pooled generation-monitor metrics row to JSONL.
        """
        if not self.save_monitor_metrics_locally:
            return

        path = (
            Path(output_dir)
            / "generation_monitor"
            / "metrics.jsonl"
        )

        append_jsonl_record(path, row)

        print(
            f"[GEN-MONITOR] Saved local metrics row to {path}"
        )

    def _compute_per_poi_metrics(
        self,
        generated_records: list[dict],
        epoch: int,
        global_step: int,
    ) -> list[dict]:
        """
        Compute one metrics row per monitored POI.
        """
        generated_by_poi: dict[str, list[dict]] = {}

        for record in generated_records:
            target_uniprot = record["target_uniprot"]

            generated_by_poi.setdefault(
                target_uniprot,
                [],
            ).append(
                dict(record)
            )

        rows: list[dict] = []

        for target in self.monitor_targets:
            target_uniprot = target["target_uniprot"]
            poi_records = generated_by_poi[target_uniprot]

            (
                annotated_records,
                valid_canonical_all,
                unique_valid_mols_ordered,
            ) = annotate_generated_records(
                generated_records=poi_records,
                generated_smiles_field="raw_smiles",
                reference_set=self.reference_set,
                reference_scaffold_set=self.reference_scaffold_set,
            )

            metrics = compute_metrics(
                annotated_records=annotated_records,
                valid_canonical_all=valid_canonical_all,
                unique_valid_mols_ordered=unique_valid_mols_ordered,
                reference_set=self.reference_set,
                reference_scaffold_set=self.reference_scaffold_set,
                fp_radius=self.fp_radius,
                fp_n_bits=self.fp_n_bits,
                max_diversity_samples=self.max_diversity_samples,
                seed=self.seed,
            )

            scalar_metrics = extract_scalar_numeric_metrics(metrics)

            row = {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "target_uniprot": target_uniprot,
                "poi_norm": target.get("poi_norm", ""),
                "num_samples": len(poi_records),
            }

            row.update(scalar_metrics)
            rows.append(row)

        return rows

    def _append_per_poi_metrics_locally(
        self,
        rows: list[dict],
        output_dir: str | Path,
    ) -> None:
        if not self.save_monitor_metrics_locally:
            return

        path = (
            Path(output_dir)
            / "generation_monitor"
            / "per_poi_metrics.jsonl"
        )

        for row in rows:
            append_jsonl_record(path, row)

        print(
            f"[GEN-MONITOR] Saved per-POI metrics to {path}"
        )

    def _add_per_poi_metrics_to_wandb_payload(
        self,
        payload: dict,
        rows: list[dict],
    ) -> None:
        metadata_keys = {
            "epoch",
            "global_step",
            "target_uniprot",
            "poi_norm",
            "num_samples",
        }

        for row in rows:
            poi_label = (
                row.get("poi_norm")
                or row["target_uniprot"]
            )

            for key, value in row.items():
                if key in metadata_keys:
                    continue

                if isinstance(value, (int, float)):
                    payload[
                        f"gen_per_poi/{poi_label}/{key}"
                    ] = float(value)

    def _select_valid_examples_for_visualization(
        self,
        annotated_records: list[dict],
    ) -> tuple[list[str], list[str]]:
        """
        Select valid generated molecules for RDKit visualization.
        """
        if self.num_valid_molecules_to_visualize <= 0:
            return [], []

        valid_records = [
            record
            for record in annotated_records
            if record.get("is_valid")
            and record.get("canonical_smiles")
        ]

        selected = valid_records[
            :self.num_valid_molecules_to_visualize
        ]

        smiles_list: list[str] = []
        legends: list[str] = []

        for index, record in enumerate(selected):
            smiles = record["canonical_smiles"]
            smiles_list.append(smiles)

            novelty_label = (
                "novel"
                if record.get("is_novel")
                else "seen"
            )

            ref_label = (
                "ref-match"
                if record.get("is_reference_match")
                else "not-ref"
            )

            dup_label = (
                "dup"
                if record.get("is_duplicate_generated")
                else "unique"
            )

            legends.append(
                f"#{index + 1} | "
                f"{novelty_label} | "
                f"{ref_label} | "
                f"{dup_label}"
            )

        return smiles_list, legends

    def _add_molecule_images_to_wandb_payload(
        self,
        payload: dict,
        annotated_records: list[dict],
        epoch: int,
        global_step: int,
    ) -> None:
        """
        Add RDKit molecule images to the W&B payload.
        """
        if not self.log_molecule_images_to_wandb:
            return

        try:
            import wandb
        except ImportError:
            print(
                "[GEN-MONITOR] wandb is not installed; "
                "skipping molecule images."
            )
            return

        smiles_list, legends = self._select_valid_examples_for_visualization(
            annotated_records
        )

        if len(smiles_list) == 0:
            print(
                "[GEN-MONITOR] No valid generated molecules "
                "available for visualization."
            )
            return

        molecule_grid = draw_molecule_grid(
            smiles_list=smiles_list,
            legends=legends,
            mols_per_row=self.mols_per_row,
            sub_img_size=self.sub_img_size,
        )

        scaffold_highlighted_grid = draw_highlighted_scaffold_grid(
            smiles_list=smiles_list,
            legends=legends,
            mols_per_row=self.mols_per_row,
            sub_img_size=self.sub_img_size,
        )

        if molecule_grid is not None:
            payload["gen/molecule_grid"] = wandb.Image(
                molecule_grid,
                caption=(
                    f"Epoch {epoch} | step {global_step} | "
                    f"{len(smiles_list)} valid generated molecules"
                ),
            )

        if scaffold_highlighted_grid is not None:
            payload["gen/scaffold_highlighted_grid"] = wandb.Image(
                scaffold_highlighted_grid,
                caption=(
                    f"Epoch {epoch} | step {global_step} | "
                    "Murcko scaffolds highlighted"
                ),
            )

    def _log_generation_payload_to_wandb(
        self,
        payload: dict,
        global_step: int,
    ) -> None:
        try:
            import wandb
        except ImportError:
            print(
                "[GEN-MONITOR] wandb is not installed; "
                "generation payload not logged."
            )
            return

        if wandb.run is None:
            print(
                "[GEN-MONITOR] No active W&B run; "
                "generation metrics and images will not be logged."
            )
            return

        wandb.log(
            payload,
            step=global_step,
        )

    def on_epoch_end(
        self,
        args,
        state,
        control,
        model=None,
        **kwargs,
    ):
        if model is None:
            if self.trainer is not None:
                model = self.trainer.model
            else:
                print(
                    "[GEN-MONITOR] Skipping: "
                    "no model and no trainer available."
                )
                return control

        if not getattr(
            state,
            "is_world_process_zero",
            True,
        ):
            return control

        if state.epoch is None:
            return control

        epoch = int(round(state.epoch))

        if epoch <= 0:
            return control

        if epoch % self.every_n_epochs != 0:
            return control

        device = next(model.parameters()).device

        poi_embeddings = None
        sample_metadata = None

        if self.monitor_targets is not None:
            (
                poi_embeddings,
                sample_metadata,
            ) = self._build_conditional_generation_inputs()

            samples_per_poi = (
                self.num_samples
                // len(self.monitor_targets)
            )

            print(
                f"[GEN-MONITOR] epoch={epoch}: "
                f"generating {samples_per_poi} molecules per POI "
                f"across {len(self.monitor_targets)} POIs "
                f"({self.num_samples} total)"
            )

            generated_records = generate_to_memory(
                model=model,
                tokenizer=self.tokenizer,
                num_samples=self.num_samples,
                batch_size=self.batch_size,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                device=device,
                poi_embeddings=poi_embeddings,
                sample_metadata=sample_metadata,
            )
        else:
            print(
                f"[GEN-MONITOR] epoch={epoch}: "
                f"generating {self.num_samples} SMILES "
                "for W&B monitoring"
            )

            generated_records = generate_to_memory(
                model=model,
                tokenizer=self.tokenizer,
                num_samples=self.num_samples,
                batch_size=self.batch_size,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                device=device,
            )

        per_poi_source_records = None

        if self.monitor_targets is not None:
            per_poi_source_records = [
                dict(record)
                for record in generated_records
            ]

        (
            annotated_records,
            valid_canonical_all,
            unique_valid_mols_ordered,
        ) = annotate_generated_records(
            generated_records=generated_records,
            generated_smiles_field="raw_smiles",
            reference_set=self.reference_set,
            reference_scaffold_set=self.reference_scaffold_set,
        )

        metrics = compute_metrics(
            annotated_records=annotated_records,
            valid_canonical_all=valid_canonical_all,
            unique_valid_mols_ordered=unique_valid_mols_ordered,
            reference_set=self.reference_set,
            reference_scaffold_set=self.reference_scaffold_set,
            fp_radius=self.fp_radius,
            fp_n_bits=self.fp_n_bits,
            max_diversity_samples=self.max_diversity_samples,
            seed=self.seed,
        )

        scalar_metrics = extract_scalar_numeric_metrics(metrics)

        condition_guardrail_metrics: dict[
            str,
            int | float | None,
        ] = {}

        condition_nll_per_poi: dict[str, dict] = {}
        poi_knn_per_poi: dict[str, dict] = {}

        if self.condition_nll_examples is not None:
            nll_metrics = compute_condition_nll_metrics(
                model=model,
                tokenizer=self.tokenizer,
                nll_examples=self.condition_nll_examples,
                poi_embeddings_by_uniprot=(
                    self.poi_embeddings_by_uniprot
                ),
                batch_size=self.condition_nll_batch_size,
                device=device,
                max_seq_len=int(model.config.max_seq_len),
                return_per_poi=True,
            )

            condition_nll_per_poi = nll_metrics.pop(
                "per_poi",
                {},
            )

            condition_guardrail_metrics.update(nll_metrics)

        if self.poi_knn_reference_pool is not None:
            poi_knn_metrics = compute_generated_poi_knn_purity(
                generated_records=generated_records,
                reference_pool=self.poi_knn_reference_pool,
                generated_smiles_field="raw_smiles",
                target_uniprot_field="target_uniprot",
                k=self.poi_knn_k,
                max_valid_per_poi=self.poi_knn_max_valid_per_poi,
                return_per_poi=True,
            )

            poi_knn_per_poi = poi_knn_metrics.pop(
                "per_poi",
                {},
            )

            condition_guardrail_metrics.update(poi_knn_metrics)

        monitor_scalar_metrics = dict(scalar_metrics)
        monitor_scalar_metrics.update(condition_guardrail_metrics)

        monitor_metrics_row = self._build_monitor_metrics_row(
            scalar_metrics=monitor_scalar_metrics,
            epoch=epoch,
            global_step=int(state.global_step),
        )

        self._append_monitor_metrics_locally(
            row=monitor_metrics_row,
            output_dir=args.output_dir,
        )

        per_poi_metrics_rows: list[dict] = []

        if per_poi_source_records is not None:
            per_poi_metrics_rows = self._compute_per_poi_metrics(
                generated_records=per_poi_source_records,
                epoch=epoch,
                global_step=int(state.global_step),
            )

            per_poi_metrics_rows_local = [
                dict(row)
                for row in per_poi_metrics_rows
            ]

            for row in per_poi_metrics_rows_local:
                target_uniprot = row["target_uniprot"]

                if self.condition_nll_examples is not None:
                    row["condition_nll_gap"] = (
                        condition_nll_per_poi.get(
                            target_uniprot,
                            {},
                        ).get(
                            "condition_nll_gap"
                        )
                    )

                if self.poi_knn_reference_pool is not None:
                    row["generated_poi_knn_purity_mean"] = (
                        poi_knn_per_poi.get(
                            target_uniprot,
                            {},
                        ).get(
                            "generated_poi_knn_purity_mean"
                        )
                    )

            self._append_per_poi_metrics_locally(
                rows=per_poi_metrics_rows_local,
                output_dir=args.output_dir,
            )

        wandb_payload: dict = {
            f"gen/{key}": float(value)
            for key, value in scalar_metrics.items()
        }

        for key, value in condition_guardrail_metrics.items():
            if isinstance(value, (int, float)):
                wandb_payload[f"condition/{key}"] = float(value)

        wandb_payload["gen/epoch"] = int(epoch)
        wandb_payload["gen/global_step"] = int(state.global_step)
        wandb_payload["gen/monitor_num_samples"] = int(self.num_samples)

        if per_poi_metrics_rows:
            wandb_payload["gen/monitor_num_pois"] = len(
                self.monitor_targets
            )

            self._add_per_poi_metrics_to_wandb_payload(
                payload=wandb_payload,
                rows=per_poi_metrics_rows,
            )

        self._add_molecule_images_to_wandb_payload(
            payload=wandb_payload,
            annotated_records=annotated_records,
            epoch=epoch,
            global_step=int(state.global_step),
        )

        self._log_generation_payload_to_wandb(
            payload=wandb_payload,
            global_step=int(state.global_step),
        )

        print(
            "[GEN-MONITOR] "
            f"epoch={epoch} "
            f"validity={metrics.get('validity')} "
            f"uniqueness={metrics.get('uniqueness_among_valid')} "
            f"novelty={metrics.get('novelty_among_unique_valid')} "
            f"dup_rate={metrics.get('duplicate_rate_among_valid')}"
        )

        return control