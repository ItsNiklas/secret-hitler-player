#!/bin/bash
#SBATCH --job-name=tspo4
#SBATCH -t 48:00:00              # walltime
#SBATCH -p scc-gpu               # partition/queue
#SBATCH -c 16                    # CPU cores per task
#SBATCH --gpus-per-node=A100:4   # 4 GPUs *per node* (16 total)
#SBATCH --constraint=80gb        # GPU memory constraint
#SBATCH -N 4                     # Request exactly 4 nodes

# 1. Extract the hostnames for all nodes
mapfile -t NODES < <(scontrol show hostnames $SLURM_JOB_NODELIST)
HEAD_NODE=${NODES[0]}
# Combine all other nodes into a single string for workers
WORKER_NODES="${NODES[@]:1}"

# 2. Print node information to your slurm-*.out file
echo "======================================================"
echo "Allocation Secured!"
echo "Head Node (SSH into this one): $HEAD_NODE"
echo "Worker Nodes (Remote LLM nodes): $WORKER_NODES"
echo "Allocated GPUs per node: $CUDA_VISIBLE_DEVICES"
echo "======================================================"

# 3. Start a detached tmux session named 'experiment'
tmux new-session -d -s experiment "bash --login"

# 4. Inject the hostnames into the tmux environment!
# This makes $HEAD_NODE and $WORKER_NODES available as variables
# inside your tmux session when you attach to it.
tmux set-environment -t experiment HEAD_NODE "$HEAD_NODE"
tmux set-environment -t experiment WORKER_NODES "$WORKER_NODES"

# 5. Keep allocation alive as long as the job is running
sleep infinity
