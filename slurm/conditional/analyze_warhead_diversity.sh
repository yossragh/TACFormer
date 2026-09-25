#!/bin/bash

#SBATCH -A naiss2026-3-679-gpu
#SBATCH -p gpu
#SBATCH -t 2:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -J warhead_diversity
#SBATCH -o logs/conditional/warhead_diversity_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PROJECT_DIR="/nobackup/proj/disk/naiss2024-5-630/personal/yossra"

APP_CONTAINER="${APP_CONTAINER:-/nobackup/proj/disk/naiss2024-5-630/personal/yossra/containers/TACFormer.sif}"
CONFIG="${CONFIG:-configs/conditional/analyze_warhead_diversity.yaml}"


apptainer exec --nv \
  -B .:/workspace,"$PROJECT_DIR:$PROJECT_DIR" \
  --pwd /workspace \
  "$APP_CONTAINER" \
  python -u scripts/analyze_warhead_diversity.py \
    --config "$CONFIG"