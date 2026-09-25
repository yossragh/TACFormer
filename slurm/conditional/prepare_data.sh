#!/bin/bash
#SBATCH -A naiss2026-3-679-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 00:10:00
#SBATCH -J protac_prepare_conditional
#SBATCH -o logs/data/protac_prepare_conditional_%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs/data/
mkdir -p data/processed/conditional_top20

APP_CONTAINER="${APP_CONTAINER:-/nobackup/proj/disk/naiss2024-5-630/personal/yossra/containers/TACFormer.sif}"
CONFIG="${CONFIG:-configs/conditional/prepare_data.yaml}"

apptainer exec --nv \
  -B "$SLURM_SUBMIT_DIR":/workspace \
  --pwd /workspace \
  "$APP_CONTAINER" \
  python -u scripts/prepare_data.py \
    --config "$CONFIG"