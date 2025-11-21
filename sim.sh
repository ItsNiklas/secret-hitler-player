#!/bin/bash

# Set these environment variables or add them to your .env file
export HF_TOKEN=${HF_TOKEN:-"your-huggingface-token-here"}
export HF_HOME=/scratch-scc/users/$USER/hf
export LLM_API_KEY=${LLM_API_KEY:-"your-llm-api-key-here"}
export LLM_BASE_URL=http://localhost:8080/v1/
export ENABLE_PARALLEL_PROCESSING=true

source .venv/bin/activate

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

while true; do
    # Wait for a free slot before starting new game
    wait_for_slot
    
    echo "Run $i"
    (python simulator/HitlerGame.py --summary-path runsD1 || echo "Run $i failed, continuing with next run")
done

# Wait for all background jobs to complete
echo "Waiting for all games to complete..."
wait
echo "All games completed!"
