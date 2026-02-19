#!/bin/bash

set -euo pipefail
echo "[$(date +%F\ %T)] Job starting on $(hostname)"

# Set these environment variables or add them to your .env file
export MODEL_DIR=${MODEL_DIR:-./DeepSeek-R1-0528-UD-IQ3_XXS}
export MODEL=${MODEL:-/models/UD-IQ3_XXS/DeepSeek-R1-0528-UD-IQ3_XXS-00001-of-00007.gguf}
export LLM_API_KEY=$LLM_API_KEY
export LLM_BASE_URL=http://localhost:8080/v1/

export SIF=/scratch-scc/projects/ag_gipp/llama-server.sif

module load gcc cuda apptainer
mkdir -p logs

echo "Starting llama-server with model $MODEL"

apptainer run --nv \
  --bind "$MODEL_DIR:/models" \
  $SIF \
  -m "$MODEL" \
  --ctx-size 32768 \
  --n-gpu-layers 999 \
  --split-mode row \
  --flash-attn \
  --host 0.0.0.0 \
  --port 8080
