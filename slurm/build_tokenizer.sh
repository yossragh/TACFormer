#!/bin/bash
#SBATCH -A NAISS2025-5-462
#SBATCH -p alvis
#SBATCH -t 00:30:00
#SBATCH -C NOGPU
#SBATCH --cpus-per-task=4
#SBATCH -J protac_tokenizer
#SBATCH -o logs/data/protac_tokenizer_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs/data artifacts/tokenizer_pretrain

APP_CONTAINER="${APP_CONTAINER:-./TACFormer.sif}"

apptainer exec \
  -B .:/workspace \
  --pwd /workspace \
  "$APP_CONTAINER" \
  python -u scripts/build_tokenizer.py --config configs/build_tokenizer_pretrain.yaml