#!/bin/bash
#SBATCH -A naiss2026-3-679-gpu
#SBATCH -p gpu
#SBATCH -t 4:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -J eval_prior
#SBATCH --output=logs/pretrain/eval_prior_%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export MIMER_DIR="/nobackup/proj/disk/naiss2024-5-630/personal/yossra"
export MPLBACKEND=Agg

APP_CONTAINER="${APP_CONTAINER:-/nobackup/proj/disk/naiss2024-5-630/personal/yossra/containers/TACFormer.sif}"
CONFIG="${CONFIG:-configs/pretraining/evaluate.yaml}"

apptainer exec \
  -B .:/workspace,"$MIMER_DIR:$MIMER_DIR" \
  --pwd /workspace \
  "$APP_CONTAINER" \
  python -u scripts/evaluatev3.py \
  --config "$CONFIG"