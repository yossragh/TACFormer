#!/bin/bash

#SBATCH -A naiss2026-3-679-gpu
#SBATCH -p gpu
#SBATCH -t 01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH -J build_poi_embeddings
#SBATCH -o logs/data/build_poi_embeddings_%j.out

set -euo pipefail

# The job should be submitted from the TACFormer project root.
cd "$SLURM_SUBMIT_DIR"

export PROJECT_DIR="/nobackup/proj/disk/naiss2024-5-630/personal/yossra"

mkdir -p logs/data
mkdir -p "$PROJECT_DIR/.cache/huggingface"

APP_CONTAINER="${APP_CONTAINER:-$PROJECT_DIR/containers/TACFormer.sif}"

CONFIG="${CONFIG:-configs/conditional/build_poi_embeddings.yaml}"

# Store the downloaded Hugging Face / ESM-2 model outside the container
# so it persists between jobs.
export HF_HOME="$PROJECT_DIR/.cache/huggingface"

echo "===================================================================="
echo "[INFO] Build POI ESM-2 embeddings"
echo "[INFO] Submit dir  : $SLURM_SUBMIT_DIR"
echo "[INFO] Project dir : $PROJECT_DIR"
echo "[INFO] Container   : $APP_CONTAINER"
echo "[INFO] Config      : $CONFIG"
echo "[INFO] HF_HOME     : $HF_HOME"
echo "===================================================================="

apptainer exec --nv \
  -B .:/workspace,"$PROJECT_DIR:$PROJECT_DIR" \
  --pwd /workspace \
  "$APP_CONTAINER" \
  python -u scripts/build_poi_embeddings.py \
  --config "$CONFIG"

echo "===================================================================="
echo "[DONE] POI ESM-2 embeddings finished"
echo "===================================================================="