#!/bin/bash
#SBATCH -A naiss2026-3-679-gpu
#SBATCH -p gpu
#SBATCH -t 8:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -J conditional_seed_100
#SBATCH -o logs/conditional/train_seed_100_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PROJECT_DIR="/nobackup/proj/disk/naiss2024-5-630/personal/yossra"

SEED=100

OUT_ROOT="$PROJECT_DIR/TACFormer/conditional"
OUT_DIR="$OUT_ROOT/seed_${SEED}"
RUN_NAME="conditional_seed_${SEED}"

mkdir -p logs/conditional
mkdir -p "$OUT_DIR"

# -------------------------
# W&B settings
# -------------------------
export WANDB_API_KEY="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"

# Use a new W&B project/dashboard for this run.
export WANDB_PROJECT="protac-conditional-guardrails"
export WANDB_NAME="$RUN_NAME"
export WANDB_RUN_GROUP="conditional_single_seed"
export WANDB_MODE="online"

export WANDB_DIR="$PROJECT_DIR/wandb"
mkdir -p "$WANDB_DIR"

export WANDB_LOG_MODEL="false"
export WANDB_WATCH="false"

APP_CONTAINER="${APP_CONTAINER:-/nobackup/proj/disk/naiss2024-5-630/personal/yossra/containers/TACFormer.sif}"
CONFIG="${CONFIG:-configs/conditional/trainv3.yaml}"

echo "===================================================================="
echo "[INFO] Single-seed POI-conditional training"
echo "[INFO] Seed          : $SEED"
echo "[INFO] Output dir    : $OUT_DIR"
echo "[INFO] Run name      : $RUN_NAME"
echo "[INFO] W&B project   : $WANDB_PROJECT"
echo "[INFO] W&B group     : $WANDB_RUN_GROUP"
echo "[INFO] W&B name      : $WANDB_NAME"
echo "[INFO] Config        : $CONFIG"
echo "[INFO] Container     : $APP_CONTAINER"
echo "[INFO] PROJECT_DIR   : $PROJECT_DIR"
echo "===================================================================="

apptainer exec --nv \
  -B .:/workspace,"$PROJECT_DIR:$PROJECT_DIR" \
  --pwd /workspace \
  "$APP_CONTAINER" \
  python -u scripts/trainv3.py \
    --config "$CONFIG" \
    --override \
      train.seed="$SEED" \
      train.output_dir="$OUT_DIR" \
      train.run_name="$RUN_NAME"

echo "===================================================================="
echo "[DONE] Finished conditional seed $SEED"
echo "[DONE] Output dir: $OUT_DIR"
echo "===================================================================="