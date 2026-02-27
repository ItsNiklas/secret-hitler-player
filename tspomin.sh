#!/bin/bash
#SBATCH --job-name=tspo
#SBATCH -t 48:00:00              # walltime
#SBATCH -p scc-gpu               # partition/queue
#SBATCH -c 8                     # CPU cores
#SBATCH -G A100:1                # GPUs
#SBATCH --constraint=80gb        # GPU memory constraint
#SBATCH -N 1                     # number of nodes

# Print node information
echo "Job started on $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# Start a detached tmux session
tmux new-session -d "bash --login"

# Keep allocation alive as long as tmux is running
sleep infinity
