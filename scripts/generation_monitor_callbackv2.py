from __future__ import annotations

import sys
from pathlib import Path

import torch  # NEW
from transformers import TrainerCallback

# ---------------------------------------------------------------------
# Make project modules importable when this callback is imported
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate import annotate_generated_records, compute_metrics
from generatev2 import generate_to_memory
from metrics import extract_scalar_numeric_metrics
from utils import append_jsonl_record
from visualize import draw_highlighted_scaffold_grid, draw_molecule_grid


class GenerationMonitorCallback(TrainerCallback):
    """
    Training-time generation monitor.

    Unconditional:
        - preserves the original generation-monitor behavior
        - computes pooled generation metrics

    Conditional:
        - generates the same number of molecules for every monitored POI
        - computes pooled generation metrics across all generated molecules
        - computes the same generation metrics separately for every POI

    Raw generated SMILES are discarded after logging.
    """

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
        poi_embeddings_by_uniprot: dict[str, torch.Tensor] | None = None,  # NEW
        monitor_targets: list[dict[str, str]] | None = None,  # NEW
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

        # NEW: both remain None for unconditional monitoring.
        self.poi_embeddings_by_uniprot = poi_embeddings_by_uniprot
        self.monitor_targets = monitor_targets

        self.trainer = None

    def add_trainer(self, trainer) -> None:
        """
        Store the Trainer instance for fallback access to trainer.model.
        """
        self.trainer = trainer

    # NEW: build the conditional generation panel once per monitored epoch.
    def _build_conditional_generation_inputs(
        self,
    ) -> tuple[torch.Tensor, list[dict[str, str]]]:
        """
        Build an equal-sized generation panel for every monitored POI.

        train.py resolves:
            num_samples = samples_per_poi * number_of_POIs

        Example:
            20 POIs * 200 samples = 4000 total generations.
        """
        num_targets = len(self.monitor_targets)

        if self.num_samples % num_targets != 0:
            raise ValueError(
                "Conditional monitor num_samples must be divisible "
                "by the number of monitor targets."
            )

        samples_per_poi = (
            self.num_samples
            // num_targets
        )

        embeddings: list[torch.Tensor] = []
        metadata: list[dict[str, str]] = []

        for target in self.monitor_targets:
            target_uniprot = target[
                "target_uniprot"
            ]

            embedding = (
                self.poi_embeddings_by_uniprot[
                    target_uniprot
                ]
            )

            for _ in range(samples_per_poi):
                embeddings.append(
                    embedding
                )

                metadata.append(
                    dict(target)
                )

        return (
            torch.stack(
                embeddings,
                dim=0,
            ).float(),
            metadata,
        )

    def _build_monitor_metrics_row(
        self,
        scalar_metrics: dict[str, int | float],
        epoch: int,
        global_step: int,
    ) -> dict:
        """
        Build one local JSONL row for reproducible plotting.

        This stores only:
            - epoch
            - global_step
            - scalar generation metrics

        It does not store generated SMILES, W&B images, or repeated config values.
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

        Fixed output path:
            <output_dir>/generation_monitor/metrics.jsonl
        """
        if not self.save_monitor_metrics_locally:
            return

        path = (
            Path(output_dir)
            / "generation_monitor"
            / "metrics.jsonl"
        )

        append_jsonl_record(
            path,
            row,
        )

        print(
            f"[GEN-MONITOR] Saved local metrics row to {path}"
        )

    # NEW: compute the same generation metrics independently for each POI.
    def _compute_per_poi_metrics(
        self,
        generated_records: list[dict],
        epoch: int,
        global_step: int,
    ) -> list[dict]:
        """
        Compute one metrics row per monitored POI.

        The same global reference sets are used as for the pooled Stage-1
        generation metrics. Uniqueness and diversity are recomputed within
        each POI's generated set.
        """
        generated_by_poi: dict[
            str,
            list[dict],
        ] = {}

        for record in generated_records:
            target_uniprot = record[
                "target_uniprot"
            ]

            generated_by_poi.setdefault(
                target_uniprot,
                [],
            ).append(
                dict(record)
            )

        rows: list[dict] = []

        for target in self.monitor_targets:
            target_uniprot = target[
                "target_uniprot"
            ]

            poi_records = generated_by_poi[
                target_uniprot
            ]

            (
                annotated_records,
                valid_canonical_all,
                unique_valid_mols_ordered,
            ) = annotate_generated_records(
                generated_records=poi_records,
                generated_smiles_field="raw_smiles",
                reference_set=self.reference_set,
                reference_scaffold_set=(
                    self.reference_scaffold_set
                ),
            )

            metrics = compute_metrics(
                annotated_records=annotated_records,
                valid_canonical_all=(
                    valid_canonical_all
                ),
                unique_valid_mols_ordered=(
                    unique_valid_mols_ordered
                ),
                reference_set=self.reference_set,
                reference_scaffold_set=(
                    self.reference_scaffold_set
                ),
                fp_radius=self.fp_radius,
                fp_n_bits=self.fp_n_bits,
                max_diversity_samples=(
                    self.max_diversity_samples
                ),
                seed=self.seed,
            )

            scalar_metrics = (
                extract_scalar_numeric_metrics(
                    metrics
                )
            )

            row = {
                "epoch": int(epoch),
                "global_step": int(
                    global_step
                ),
                "target_uniprot": (
                    target_uniprot
                ),
                "poi_norm": target.get(
                    "poi_norm",
                    "",
                ),
                "num_samples": len(
                    poi_records
                ),
            }

            row.update(
                scalar_metrics
            )

            rows.append(
                row
            )

        return rows

    # NEW: save per-POI metrics in long format for later analysis.
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
            append_jsonl_record(
                path,
                row,
            )

        print(
            "[GEN-MONITOR] Saved per-POI metrics "
            f"to {path}"
        )

    # NEW: report per-POI scalar metrics to W&B.
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

                if isinstance(
                    value,
                    (int, float),
                ):
                    payload[
                        f"gen_per_poi/"
                        f"{poi_label}/"
                        f"{key}"
                    ] = float(value)

    def _select_valid_examples_for_visualization(
        self,
        annotated_records: list[dict],
    ) -> tuple[list[str], list[str]]:
        """
        Select valid generated molecules for RDKit visualization.

        We select only valid molecules because invalid SMILES cannot be drawn.
        Invalid generations are monitored quantitatively through:
            - gen/num_invalid
            - gen/validity
        """
        if self.num_valid_molecules_to_visualize <= 0:
            return [], []

        valid_records = [
            rec
            for rec in annotated_records
            if (
                rec.get("is_valid")
                and rec.get(
                    "canonical_smiles"
                )
            )
        ]

        selected = valid_records[
            : self.num_valid_molecules_to_visualize
        ]

        smiles_list: list[str] = []
        legends: list[str] = []

        for i, rec in enumerate(selected):
            smiles = rec[
                "canonical_smiles"
            ]

            smiles_list.append(
                smiles
            )

            novelty_label = (
                "novel"
                if rec.get("is_novel")
                else "seen"
            )

            ref_label = (
                "ref-match"
                if rec.get(
                    "is_reference_match"
                )
                else "not-ref"
            )

            dup_label = (
                "dup"
                if rec.get(
                    "is_duplicate_generated"
                )
                else "unique"
            )

            legends.append(
                f"#{i + 1} | "
                f"{novelty_label} | "
                f"{ref_label} | "
                f"{dup_label}"
            )

        return (
            smiles_list,
            legends,
        )

    def _add_molecule_images_to_wandb_payload(
        self,
        payload: dict,
        annotated_records: list[dict],
        epoch: int,
        global_step: int,
    ) -> None:
        """
        Add RDKit molecule images to the W&B payload.

        This function only modifies payload.
        It does not call wandb.log(...).

        This keeps generation-monitor logging as one single W&B call.
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

        (
            smiles_list,
            legends,
        ) = (
            self._select_valid_examples_for_visualization(
                annotated_records=(
                    annotated_records
                ),
            )
        )

        if len(smiles_list) == 0:
            print(
                "[GEN-MONITOR] No valid generated "
                "molecules available for visualization."
            )
            return

        molecule_grid = draw_molecule_grid(
            smiles_list=smiles_list,
            legends=legends,
            mols_per_row=self.mols_per_row,
            sub_img_size=self.sub_img_size,
        )

        scaffold_highlighted_grid = (
            draw_highlighted_scaffold_grid(
                smiles_list=smiles_list,
                legends=legends,
                mols_per_row=(
                    self.mols_per_row
                ),
                sub_img_size=(
                    self.sub_img_size
                ),
            )
        )

        if molecule_grid is not None:
            payload[
                "gen/molecule_grid"
            ] = wandb.Image(
                molecule_grid,
                caption=(
                    f"Epoch {epoch} | "
                    f"step {global_step} | "
                    f"{len(smiles_list)} "
                    "valid generated molecules"
                ),
            )

        if (
            scaffold_highlighted_grid
            is not None
        ):
            payload[
                "gen/scaffold_highlighted_grid"
            ] = wandb.Image(
                scaffold_highlighted_grid,
                caption=(
                    f"Epoch {epoch} | "
                    f"step {global_step} | "
                    "Murcko scaffolds highlighted"
                ),
            )

        print(
            f"[GEN-MONITOR] Prepared "
            f"{len(smiles_list)} valid "
            "molecule images and "
            "scaffold-highlighted images "
            "for W&B."
        )

    def _log_generation_payload_to_wandb(
        self,
        payload: dict,
        global_step: int,
    ) -> None:
        """
        Log generation-monitor scalars and images to W&B in one call.

        We do not use trainer.log(...) here because the payload may contain
        wandb.Image objects, which should not be stored in Hugging Face
        Trainer's log history.
        """
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
                "generation metrics and images "
                "will not be logged."
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
                model = (
                    self.trainer.model
                )
            else:
                print(
                    "[GEN-MONITOR] Skipping: "
                    "no model and no trainer available."
                )
                return control

        # In distributed training, only the main process should generate/log.
        if not getattr(
            state,
            "is_world_process_zero",
            True,
        ):
            return control

        if state.epoch is None:
            return control

        epoch = int(
            round(
                state.epoch
            )
        )

        if epoch <= 0:
            return control

        if (
            epoch
            % self.every_n_epochs
            != 0
        ):
            return control

        device = next(
            model.parameters()
        ).device

        # NEW: conditional monitoring builds one aligned
        # POI embedding and metadata row per generated molecule.
        poi_embeddings = None
        sample_metadata = None

        if self.monitor_targets is not None:
            (
                poi_embeddings,
                sample_metadata,
            ) = (
                self._build_conditional_generation_inputs()
            )

            samples_per_poi = (
                self.num_samples
                // len(
                    self.monitor_targets
                )
            )

            print(
                f"[GEN-MONITOR] epoch={epoch}: "
                f"generating {samples_per_poi} "
                "molecules per POI across "
                f"{len(self.monitor_targets)} POIs "
                f"({self.num_samples} total)"
            )

            # NEW: conditional generation.
            generated_records = (
                generate_to_memory(
                    model=model,
                    tokenizer=(
                        self.tokenizer
                    ),
                    num_samples=(
                        self.num_samples
                    ),
                    batch_size=(
                        self.batch_size
                    ),
                    max_new_tokens=(
                        self.max_new_tokens
                    ),
                    temperature=(
                        self.temperature
                    ),
                    top_k=self.top_k,
                    top_p=self.top_p,
                    device=device,
                    poi_embeddings=(
                        poi_embeddings
                    ),
                    sample_metadata=(
                        sample_metadata
                    ),
                )
            )

        else:
            # Preserve the original unconditional generation call.
            print(
                f"[GEN-MONITOR] epoch={epoch}: "
                f"generating {self.num_samples} "
                "SMILES for W&B monitoring"
            )

            generated_records = (
                generate_to_memory(
                    model=model,
                    tokenizer=(
                        self.tokenizer
                    ),
                    num_samples=(
                        self.num_samples
                    ),
                    batch_size=(
                        self.batch_size
                    ),
                    max_new_tokens=(
                        self.max_new_tokens
                    ),
                    temperature=(
                        self.temperature
                    ),
                    top_k=self.top_k,
                    top_p=self.top_p,
                    device=device,
                )
            )

        # NEW: preserve the raw conditional records for
        # independent per-POI annotation.
        per_poi_source_records = None

        if self.monitor_targets is not None:
            per_poi_source_records = [
                dict(record)
                for record
                in generated_records
            ]

        # Existing pooled metrics over all generated molecules.
        (
            annotated_records,
            valid_canonical_all,
            unique_valid_mols_ordered,
        ) = annotate_generated_records(
            generated_records=(
                generated_records
            ),
            generated_smiles_field=(
                "raw_smiles"
            ),
            reference_set=(
                self.reference_set
            ),
            reference_scaffold_set=(
                self.reference_scaffold_set
            ),
        )

        metrics = compute_metrics(
            annotated_records=(
                annotated_records
            ),
            valid_canonical_all=(
                valid_canonical_all
            ),
            unique_valid_mols_ordered=(
                unique_valid_mols_ordered
            ),
            reference_set=(
                self.reference_set
            ),
            reference_scaffold_set=(
                self.reference_scaffold_set
            ),
            fp_radius=(
                self.fp_radius
            ),
            fp_n_bits=(
                self.fp_n_bits
            ),
            max_diversity_samples=(
                self.max_diversity_samples
            ),
            seed=self.seed,
        )

        scalar_metrics = (
            extract_scalar_numeric_metrics(
                metrics
            )
        )

        monitor_metrics_row = (
            self._build_monitor_metrics_row(
                scalar_metrics=(
                    scalar_metrics
                ),
                epoch=epoch,
                global_step=int(
                    state.global_step
                ),
            )
        )

        # Existing pooled metrics file remains unchanged.
        self._append_monitor_metrics_locally(
            row=monitor_metrics_row,
            output_dir=args.output_dir,
        )

        # NEW: compute the same metrics independently for every POI.
        per_poi_metrics_rows: list[
            dict
        ] = []

        if (
            per_poi_source_records
            is not None
        ):
            per_poi_metrics_rows = (
                self._compute_per_poi_metrics(
                    generated_records=(
                        per_poi_source_records
                    ),
                    epoch=epoch,
                    global_step=int(
                        state.global_step
                    ),
                )
            )

            self._append_per_poi_metrics_locally(
                rows=(
                    per_poi_metrics_rows
                ),
                output_dir=(
                    args.output_dir
                ),
            )

        wandb_payload: dict = {
            f"gen/{key}": float(value)
            for key, value
            in scalar_metrics.items()
        }

        wandb_payload[
            "gen/epoch"
        ] = int(
            epoch
        )

        wandb_payload[
            "gen/global_step"
        ] = int(
            state.global_step
        )

        wandb_payload[
            "gen/monitor_num_samples"
        ] = int(
            self.num_samples
        )

        # NEW: add per-POI metrics to the same W&B logging call.
        if per_poi_metrics_rows:
            wandb_payload[
                "gen/monitor_num_pois"
            ] = len(
                self.monitor_targets
            )

            self._add_per_poi_metrics_to_wandb_payload(
                payload=(
                    wandb_payload
                ),
                rows=(
                    per_poi_metrics_rows
                ),
            )

        self._add_molecule_images_to_wandb_payload(
            payload=(
                wandb_payload
            ),
            annotated_records=(
                annotated_records
            ),
            epoch=epoch,
            global_step=int(
                state.global_step
            ),
        )

        self._log_generation_payload_to_wandb(
            payload=(
                wandb_payload
            ),
            global_step=int(
                state.global_step
            ),
        )

        print(
            "[GEN-MONITOR] "
            f"epoch={epoch} "
            f"validity="
            f"{metrics.get('validity')} "
            f"uniqueness="
            f"{metrics.get('uniqueness_among_valid')} "
            f"novelty="
            f"{metrics.get('novelty_among_unique_valid')} "
            f"dup_rate="
            f"{metrics.get('duplicate_rate_among_valid')}"
        )

        return control