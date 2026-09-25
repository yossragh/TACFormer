#!/bin/bash

#SBATCH -A naiss2026-3-679-gpu
#SBATCH -p gpu
#SBATCH -t 01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -J gen
#SBATCH --output=logs/pretrain/gen_%j.log

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs/pretrain

export PROJ_DIR="/nobackup/proj/disk/naiss2024-5-630/personal/yossra"
export MPLBACKEND=Agg

APP_CONTAINER="${APP_CONTAINER:-/nobackup/proj/disk/naiss2024-5-630/personal/yossra/containers/TACFormer.sif}"
CONFIG="${CONFIG:-configs/pretraining/generate.yaml}"

apptainer exec --nv \
  -B .:/workspace,"$PROJ_DIR:$PROJ_DIR" \
  --pwd /workspace \
  "$APP_CONTAINER" \
  python -u scripts/generatev2.py \
  --config "$CONFIG"