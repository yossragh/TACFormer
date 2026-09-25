#!/bin/bash
#SBATCH -A naiss2026-3-679-gpu
#SBATCH -p gpu
#SBATCH -t 06:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -J pretrain
#SBATCH -o logs/pretrain/pretrain_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PROJECT_DIR="/nobackup/proj/disk/naiss2024-5-630/personal/yossra"
mkdir -p logs/pretrain

# -------------------------
# W&B settings
# -------------------------
export WANDB_API_KEY="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"
export WANDB_PROJECT="protac-pretrain"
export WANDB_NAME="pretrain_seed_11"
export WANDB_MODE="online"

# Store W&B local files on Mimer.
export WANDB_DIR="$PROJECT_DIR/wandb"
mkdir -p "$WANDB_DIR"

# Do not upload model checkpoints to W&B for now.
export WANDB_LOG_MODEL="false"

# Avoid extra parameter/gradient watching overhead.
export WANDB_WATCH="false"

APP_CONTAINER="${APP_CONTAINER:-/nobackup/proj/disk/naiss2024-5-630/personal/yossra/containers/TACFormer.sif}"

apptainer exec --nv \
  -B .:/workspace,"$PROJECT_DIR:$PROJECT_DIR" \
  --pwd /workspace \
  "$APP_CONTAINER" \
  python -u scripts/train.py \
  --config configs/pretraining/pretrain.yaml


