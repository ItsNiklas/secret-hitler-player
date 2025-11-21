##!/bin/bash
#SBATCH --job-name=vllm-gpu
#SBATCH -t 48:00:00
#SBATCH -p scc-gpu
#SBATCH -c 16
#SBATCH -G A100:4
#SBATCH --constraint=80gb
#SBATCH -N 1

# Set these environment variables or add them to your .env file
export HF_TOKEN=${HF_TOKEN:-"your-huggingface-token-here"}
export HF_HOME=/scratch-scc/users/$USER/hf
export MODEL=Qwen/Qwen2.5-32B-Instruct
export LLM_API_KEY=${LLM_API_KEY:-"your-llm-api-key-here"}
export LLM_BASE_URL=http://localhost:8080/v1/

# export VLLM_USE_MODELSCOPE=true

module load gcc
module load cuda
source .venv/bin/activate

echo "Starting VLLM server with model $MODEL"

HF_HUB_OFFLINE=1 vllm serve $MODEL --port 8080 --api-key $LLM_API_KEY --tensor-parallel-size 4 --gpu-memory-utilization 0.8
