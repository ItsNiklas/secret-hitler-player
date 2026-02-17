#!/bin/bash

set -euo pipefail
echo "[$(date +%F\ %T)] Job starting on $HOSTNAME"

# Set these environment variables or add them to your .env file
export HF_HOME=/scratch-scc/users/$USER/hf
export MODEL=${MODEL:-openai/gpt-oss-120b}
export LLM_API_KEY=$LLM_API_KEY
export LLM_BASE_URL=http://localhost:8080/v1/

export SIF=/scratch-scc/projects/ag_gipp/vllm.sif

# export VLLM_USE_MODELSCOPE=true

module load gcc cuda apptainer
mkdir -p $HF_HOME logs

echo "Starting VLLM server with model $MODEL"
apptainer exec \
  --nv \
  --cleanenv \
  --env TIKTOKEN_RS_CACHE_DIR="$HF_HOME" \
  --env HF_HOME="$HF_HOME" \
  --env HF_HUB_OFFLINE=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -B "$HF_HOME:$HF_HOME:rw" \
  "$SIF" \
  vllm serve $MODEL \
    --port 8080 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.67 \
    --trust-remote-code \
    --download-dir "$HF_HOME" \
    --disable-custom-all-reduce \
    --limit-mm-per-prompt.image 0 \