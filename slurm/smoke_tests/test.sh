#!/bin/bash
#SBATCH -A naiss2026-3-549-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 00:10:00
#SBATCH -J test_reload
#SBATCH -o logs/test_reload_%j.out
#SBATCH -e logs/test_reload_%j.err

mkdir -p logs

PROJECT_DIR="/home/yossra/projects/TACFormer"
CONTAINER="/nobackup/proj/disk/naiss2024-5-630/personal/yossra/containers/TACFormer.sif"

cd "$PROJECT_DIR"

echo "Running model reload verification on Grace Hopper GPU node..."

apptainer exec --nv "$CONTAINER" python3 - << 'EOF'
import sys
from pathlib import Path
import transformers

PROJECT_ROOT = Path("/home/yossra/projects/TACFormer")
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from transformer import GPTConfig, GPTForCausalLM

print(f"Transformers version: {transformers.__version__}")

cfg = GPTConfig(vocab_size=88, max_seq_len=64, n_layers=2, d_model=32, n_heads=4)
model = GPTForCausalLM(cfg)

with torch.no_grad():
    model.lm_head.weight.fill_(0.1234)

before = model.lm_head.weight.clone()
save_path = "/tmp/test_model"
model.save_pretrained(save_path)

reloaded = GPTForCausalLM.from_pretrained(save_path)
after = reloaded.lm_head.weight

print("----------------------------------------")
print("Weights survived reload:", torch.allclose(before, after))
print("Still tied:", reloaded.lm_head.weight.data_ptr() == reloaded.token_emb.weight.data_ptr())
print("----------------------------------------")
EOF