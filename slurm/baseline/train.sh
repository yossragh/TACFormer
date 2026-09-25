#!/bin/bash
#SBATCH -A naiss2026-3-679-gpu
#SBATCH -p gpu
#SBATCH -t 06:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -J train_baseline
#SBATCH -o logs/baseline/train_baseline_%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs/baseline

export PROJECT_DIR="/nobackup/proj/disk/naiss2024-5-630/personal/yossra"

export WANDB_API_KEY="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"
export WANDB_PROJECT="protac-gpt"
export WANDB_MODE="online"
export WANDB_DIR="$PROJECT_DIR/wandb"
export WANDB_LOG_MODEL="false"
export WANDB_WATCH="false"

APP_CONTAINER="${APP_CONTAINER:-$PROJECT_DIR/containers/TACFormer.sif}"
CONFIG="${CONFIG:-configs/baseline/train.yaml}"

echo "===================================================================="
echo "[INFO] Baseline unconditional training"
echo "[INFO] Config    : $CONFIG"
echo "[INFO] Container : $APP_CONTAINER"
echo "===================================================================="

apptainer exec --nv \
  -B .:/workspace,"$PROJECT_DIR:$PROJECT_DIR" \
  --pwd /workspace \
  "$APP_CONTAINER" \
  python -u scripts/trainv3.py \
    --config "$CONFIG"

echo "===================================================================="
echo "[DONE] Baseline training finished"
echo "===================================================================="