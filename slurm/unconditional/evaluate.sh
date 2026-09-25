#!/bin/bash
#SBATCH -A naiss2026-3-679-gpu
#SBATCH -p gpu
#SBATCH -t 4:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -J evaluate_unconditional
#SBATCH -o logs/unconditional/evaluate_%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export PROJECT_DIR="/nobackup/proj/disk/naiss2024-5-630/personal/yossra"

APP_CONTAINER="${APP_CONTAINER:-$PROJECT_DIR/containers/TACFormer.sif}"

CONFIG="${CONFIG:-configs/unconditional/evaluate.yaml}"

apptainer exec --nv \
  -B .:/workspace,"$PROJECT_DIR:$PROJECT_DIR" \
  --pwd /workspace \
  "$APP_CONTAINER" \
  python -u scripts/evaluatev3.py --config "$CONFIG"