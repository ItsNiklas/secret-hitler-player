#!/bin/bash
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
export MODEL=google/gemma-3-12b-it
export LLM_API_KEY=${LLM_API_KEY:-"your-llm-api-key-here"}
export LLM_BASE_URL=http://localhost:8080/v1/

# export VLLM_USE_MODELSCOPE=true

module load gcc
module load cuda
source .venv/bin/activate

vllm serve $MODEL --port 8080 --api-key $LLM_API_KEY --tensor-parallel-size 4 --gpu-memory-utilization 0.8 &

sleep 2000

# Configure parallel processing
# Limit concurrent games to avoid overwhelming the VLLM server
MAX_PARALLEL_GAMES=4

# Function to wait for job slots to free up
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_GAMES ]; do
        sleep 5  # Check every 5 seconds
    done
}

echo "Running games with up to $MAX_PARALLEL_GAMES parallel processes"

# # Run simulator for each summary file in crawl/summaries with parallel processing
# for summary_file in crawl/summaries/*.json; do
#     # Wait for a free slot before starting new game
#     wait_for_slot
    
#     # Extract the base name without extension to match with xhr data
#     base_name=$(basename "$summary_file" _summary.json)
#     xhr_file="crawl/replay_data/${base_name}_xhr_data.json"
    
#     # Run the simulator in background and continue on errors
#     echo "Processing $base_name (parallel)"
#     (python simulator/HitlerGame.py -l DEBUG -g "$summary_file" -c "$xhr_file" -n 1 || echo "Failed on $summary_file, continuing...") &
# done

# Run simulator 100 times with error handling
for i in {1..200}; do
    # Wait for a free slot before starting new game
    wait_for_slot
    
    echo "Run $i of 100"
    (python simulator/HitlerGame.py --summary-path runsF1-G3-12B || echo "Run $i failed, continuing with next run")
done

# Wait for all background jobs to complete
echo "Waiting for all games to complete..."
wait
echo "All games completed!"

