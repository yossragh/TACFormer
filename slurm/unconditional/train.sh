#!/bin/bash
#SBATCH -A naiss2026-3-679-gpu
#SBATCH -p gpu
#SBATCH -t 06:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -J finetune_uncond
#SBATCH -o logs/unconditional/finetune_uncond_%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export PROJECT_DIR="/nobackup/proj/disk/naiss2024-5-630/personal/yossra"

export WANDB_API_KEY="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"
export WANDB_PROJECT="protac-gpt"
export WANDB_MODE="online"
export WANDB_DIR="$PROJECT_DIR/wandb"
export WANDB_LOG_MODEL="false"
export WANDB_WATCH="false"

APP_CONTAINER="${APP_CONTAINER:-$PROJECT_DIR/containers/TACFormer.sif}"
CONFIG="${CONFIG:-configs/unconditional/train.yaml}"

echo "===================================================================="
echo "[INFO] Unconditional fine-tuning"
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
echo "[DONE] Unconditional fine-tuning finished"
echo "===================================================================="