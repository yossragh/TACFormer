#!/bin/bash
#SBATCH -A naiss2026-3-679-gpu
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -J conditional_multiseed
#SBATCH --array=0-1
#SBATCH -o logs/conditional/conditional_multiseed_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PROJECT_DIR="/nobackup/proj/disk/naiss2024-5-630/personal/yossra"

# -------------------------
# Select seed for this array task
# -------------------------
SEEDS=(11 99)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

OUT_ROOT="$PROJECT_DIR/TACFormer/conditional"
OUT_DIR="$OUT_ROOT/seed_${SEED}"
RUN_NAME="conditional_multiseed_seed_${SEED}"

mkdir -p "$OUT_DIR"
# -------------------------
# W&B settings
# -------------------------
export WANDB_API_KEY="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"
export WANDB_PROJECT="protac-conditional"
export WANDB_NAME="$RUN_NAME"
export WANDB_RUN_GROUP="conditional_multiseed"
export WANDB_MODE="online"

# Store W&B local files on Mimer.
export WANDB_DIR="$PROJECT_DIR/wandb"
mkdir -p "$WANDB_DIR"

# Do not upload model checkpoints to W&B for now.
export WANDB_LOG_MODEL="false"

# Avoid extra parameter/gradient watching overhead.
export WANDB_WATCH="false"

APP_CONTAINER="${APP_CONTAINER:-/nobackup/proj/disk/naiss2024-5-630/personal/yossra/containers/TACFormer.sif}"
CONFIG="${CONFIG:-configs/conditional/train_conditional.yaml}"

echo "===================================================================="
echo "[INFO] Multi-seed POI-conditional training"
echo "[INFO] Array task ID : $SLURM_ARRAY_TASK_ID"
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
  python -u scripts/trainv2.py \
    --config "$CONFIG" \
    --override \
      train.seed="$SEED" \
      train.output_dir="$OUT_DIR" \
      train.run_name="$RUN_NAME"


echo "===================================================================="
echo "[DONE] Finished conditional seed $SEED"
echo "[DONE] Output dir: $OUT_DIR"
echo "===================================================================="