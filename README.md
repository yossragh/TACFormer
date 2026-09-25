config.py = YAML config loading, saving, and CLI overrides for runtime changes such as SLURM multi-seed runs.

utils.py = generic path, JSON, JSONL, serialization, and safe filename helpers.

metrics.py = metric-table loading, cleaning, scalar extraction, selection-score calculation, and multi-seed mean/std aggregation.

plotting.py = reusable matplotlib plotting helpers for loss curves, metric curves, checkpoint comparison plots, and mean ± std shaded plots.

chemistry.py = low-level RDKit helpers for SMILES parsing, canonicalization from Mol objects, Murcko scaffolds, molecular properties, Morgan fingerprints, and chemical diversity.

prepare.py = dataset preparation workflow: read CSV, clean SMILES, validate/canonicalize using chemistry helpers, deduplicate, split, and write train/val/test JSONL.

tokenizer.py = atom-aware SMILES tokenization, vocabulary construction, encoding/decoding, and tokenizer save/load.

dataset.py = PyTorch dataset and collator for causal language modeling from JSONL SMILES records.

scripts/
generate.py = load trained model/tokenizer and generate raw SMILES records, either to JSONL or in memory for training-time monitoring.

evaluate.py = reusable evaluation logic for raw generated SMILES, supporting both CLI file-based evaluation and in-memory training/W&B monitoring.

generation_monitor_callback.py = training-time monitoring callback that periodically generates molecules, evaluates them using evaluate.py, logs generation metrics/images to W&B, and saves local per-epoch generation metrics for reproducible plotting.

train.py = orchestrates model training, optional generation monitoring, run metadata saving, and model saving.

plot_training_metrics.py = creates single-run and multi-seed training plots from local generation-monitor metrics and Trainer logs.
