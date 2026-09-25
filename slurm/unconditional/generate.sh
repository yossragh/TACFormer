#!/bin/bash
#SBATCH -A naiss2026-3-679-gpu
#SBATCH -p gpu
#SBATCH -t 08:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -J gen_uncond
#SBATCH --output=logs/unconditional/gen_%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export PROJECT_DIR="/nobackup/proj/disk/naiss2024-5-630/personal/yossra"
export MPLBACKEND=Agg

APP_CONTAINER="${APP_CONTAINER:-$PROJECT_DIR/containers/TACFormer.sif}"
CONFIG="${CONFIG:-configs/unconditional/generate.yaml}"

apptainer exec --nv \
  -B .:/workspace,"$PROJECT_DIR:$PROJECT_DIR" \
  --pwd /workspace \
  "$APP_CONTAINER" \
  python -u scripts/generatev2.py \
  --config "$CONFIG"