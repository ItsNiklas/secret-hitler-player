#!/bin/bash
# Usage: ./vllmA-Kimi.sh <master_node_ip> <node_rank>
#   $1 = master node IP address (head node)
#   $2 = node rank (0 for head node, 1/2 for workers)

set -euo pipefail
echo "[$(date +%F\ %T)] Job starting on $(hostname)"

MASTER_ADDR="$1"
NODE_RANK="$2"

# Add --headless on worker nodes (any rank != 0)
if [ "$NODE_RANK" -ne 0 ]; then
  HEADLESS="--headless"
else
  HEADLESS=""
fi

# Set these environment variables or add them to your .env file
export HF_HOME=/scratch-scc/users/$USER/hf
export MODEL=deepseek-ai/DeepSeek-V3.1-Terminus
export LLM_API_KEY=$LLM_API_KEY
export LLM_BASE_URL=http://localhost:8080/v1/

export SIF=/scratch-scc/projects/ag_gipp/vllm-latest.sif

# export VLLM_USE_MODELSCOPE=true

module load gcc cuda apptainer nvhpc openmpi
mkdir -p $HF_HOME logs

# A100 (SM 8.0) does not support Marlin FP8 kernels (requires SM >= 8.9)
export VLLM_DISABLED_KERNELS=fp8_marlin

echo "Starting VLLM server with model $MODEL (master=$MASTER_ADDR, rank=$NODE_RANK)"
apptainer exec \
  --nv \
  --cleanenv \
  --env TIKTOKEN_RS_CACHE_DIR="$HF_HOME" \
  --env HF_HOME="$HF_HOME" \
  --env HF_HUB_OFFLINE=1 \
  --env VLLM_USE_DEEP_GEMM=0 \
  --env VLLM_DISABLED_KERNELS="$VLLM_DISABLED_KERNELS" \
  -B "$HF_HOME:$HF_HOME:rw" \
  "$SIF" \
  vllm serve $MODEL \
    --host 0.0.0.0 \
    --port 8080 \
    --tensor-parallel-size 4 \
    --dcp 4 \
    --pipeline-parallel-size 3 \
    --nnodes 3 \
    --node-rank $NODE_RANK \
    --master-addr $MASTER_ADDR \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --trust-remote-code \
    --download-dir "$HF_HOME" \
    --disable-custom-all-reduce \
    --async-scheduling \
    --quantization fp8 \
    --language-model-only \
    --reasoning-parser deepseek_v3 \
    ${HEADLESS}