#!/bin/bash
#SBATCH -A naiss2026-3-679-gpu
#SBATCH -p gpu
#SBATCH -t 6:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -J generate_conditional_e40
#SBATCH -o logs/conditional/generate_e40_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PROJECT_DIR="/nobackup/proj/disk/naiss2024-5-630/personal/yossra"

APP_CONTAINER="${APP_CONTAINER:-/nobackup/proj/disk/naiss2024-5-630/personal/yossra/containers/TACFormer.sif}"
CONFIG="${CONFIG:-configs/conditional/generate.yaml}"

echo "===================================================================="
echo "[INFO] Final conditional generation"
echo "[INFO] Checkpoint     : checkpoint-32120 (epoch 40)"
echo "[INFO] Config         : $CONFIG"
echo "[INFO] Container      : $APP_CONTAINER"
echo "[INFO] PROJECT_DIR    : $PROJECT_DIR"
echo "===================================================================="

apptainer exec --nv \
  -B .:/workspace,"$PROJECT_DIR:$PROJECT_DIR" \
  --pwd /workspace \
  "$APP_CONTAINER" \
  python -u scripts/generatev2.py \
    --config "$CONFIG"

echo "===================================================================="
echo "[DONE] Final conditional generation finished"
echo "===================================================================="