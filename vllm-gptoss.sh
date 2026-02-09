##!/bin/bash
#SBATCH --job-name=vllm-gpu
#SBATCH -t 48:00:00
#SBATCH -p scc-gpu
#SBATCH -c 16
#SBATCH -G A100:4
#SBATCH --constraint=80gb
#SBATCH -N 1

set -euo pipefail
echo "[$(date +%F\ %T)] Job starting on $HOSTNAME"

# Set these environment variables or add them to your .env file
export HF_HOME=/scratch-scc/users/$USER/hf
export MODEL=${MODEL:-meta-llama/Llama-3.3-70B-Instruct}
export LLM_API_KEY=$LLM_API_KEY
export LLM_BASE_URL=http://localhost:8080/v1/

export SIF = /scratch-scc/projects/ag_gipp/vllm-gptoss.sif

# export VLLM_USE_MODELSCOPE=true

module load gcc cuda apptainer
mkdir -p $HF_HOME logs

echo "Starting VLLM server with model $MODEL"
apptainer exec \
  --nv \
  --cleanenv \
  -B "$HF_HOME:$HF_HOME:rw" \
  "$SIF" \
  vllm serve $MODEL \
    --port 8080 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --download-dir "$HF_HOME" \
> "logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
