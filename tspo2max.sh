#!/bin/bash
#SBATCH --job-name=tspo2max
#SBATCH -t 48:00:00              # walltime
#SBATCH -p scc-gpu               # partition/queue
#SBATCH -c 16                    # CPU cores per task
#SBATCH --gpus-per-node=H100:4   # 4 H100 GPUs on first node
#SBATCH -N 1                     # Request 1 node for H100
#SBATCH hetjob
#SBATCH -p scc-gpu               # partition/queue for second component
#SBATCH -c 16                    # CPU cores per task for second node
#SBATCH --gpus-per-node=A100:4   # 4 A100 GPUs on second node
#SBATCH --constraint=80gb        # GPU memory constraint for A100
#SBATCH -N 1                     # Request 1 node for A100

# 1. Extract the hostnames for both nodes
mapfile -t NODES < <(scontrol show hostnames $SLURM_JOB_NODELIST)
HEAD_NODE=${NODES[0]}
WORKER_NODE=${NODES[1]}

# 2. Print node information to your slurm-*.out file
echo "======================================================"
echo "Allocation Secured!"
echo "Head Node (H100x4): $HEAD_NODE"
echo "Worker Node (A100x4): $WORKER_NODE"
echo "Allocated GPUs per node: $CUDA_VISIBLE_DEVICES"
echo "======================================================"

# 3. Start a detached tmux session named 'experiment'
tmux new-session -d -s experiment "bash --login"

# 4. Inject the hostnames into the tmux environment!
# This makes $HEAD_NODE and $WORKER_NODE available as variables
# inside your tmux session when you attach to it.
tmux set-environment -t experiment HEAD_NODE $HEAD_NODE
tmux set-environment -t experiment WORKER_NODE $WORKER_NODE

# 5. Keep allocation alive as long as the job is running
sleep infinity
