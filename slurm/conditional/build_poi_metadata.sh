#!/bin/bash
#SBATCH -A naiss2026-3-679-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 00:20:00
#SBATCH -J build_poi_metadata
#SBATCH -o logs/data/build_poi_metadata_%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs/data/

APP_CONTAINER="${APP_CONTAINER:-/nobackup/proj/disk/naiss2024-5-630/personal/yossra/containers/TACFormer.sif}"
CONFIG="${CONFIG:-configs/conditional/build_poi_metadata.yaml}"

echo "===================================================================="
echo "[INFO] Build POI metadata"
echo "[INFO] Submit dir : $SLURM_SUBMIT_DIR"
echo "[INFO] Config     : $CONFIG"
echo "[INFO] Container  : $APP_CONTAINER"
echo "[INFO] Started at : $(date)"
echo "===================================================================="

apptainer exec --nv \
  -B "$SLURM_SUBMIT_DIR":/workspace \
  --pwd /workspace \
  "$APP_CONTAINER" \
  python -u scripts/build_poi_metadata.py \
    --config "$CONFIG"

echo "===================================================================="
echo "[DONE] POI metadata finished"
echo "[DONE] Finished at: $(date)"
echo "===================================================================="