#!/bin/bash
#SBATCH -A NAISS2025-5-462
#SBATCH -p alvis
#SBATCH -t 00:30:00
#SBATCH -C NOGPU
#SBATCH --cpus-per-task=2
#SBATCH -J check_tokenizer
#SBATCH -o logs/runs/check_tokenizer_%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
export MIMER_DIR="/mimer/NOBACKUP/groups/naiss2023-6-290/yossra"

mkdir -p logs/runs

APP_CONTAINER="${APP_CONTAINER:-./TACFormer.sif}"
CONFIG="${CONFIG:-configs/check_tokenizer_coverage.yaml}"

echo "===================================================================="
echo "[INFO] Tokenizer coverage check"
echo "[INFO] Submit dir : $SLURM_SUBMIT_DIR"
echo "[INFO] Container  : $APP_CONTAINER"
echo "[INFO] Config     : $CONFIG"
echo "[INFO] MIMER_DIR  : $MIMER_DIR"
echo "===================================================================="

# ------------------------------------------------------------
# Run tokenizer coverage script
# ------------------------------------------------------------
apptainer exec \
  -B .:/workspace,"$MIMER_DIR:$MIMER_DIR" \
  --pwd /workspace \
  "$APP_CONTAINER" \
  python -u scripts/check_tokenizer_coverage.py \
    --config "$CONFIG"

echo "===================================================================="
echo "[DONE] Tokenizer coverage check finished."
echo "===================================================================="