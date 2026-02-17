#!/bin/bash

source .venv/bin/activate

# Configure parallel processing
# Limit concurrent games to avoid overwhelming the VLLM server
MAX_PARALLEL_GAMES=${MAX_PARALLEL_GAMES:-4}

# Function to wait for job slots to free up
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_GAMES ]; do
        sleep 5  # Check every 5 seconds
    done
}

start_time=$(date)
echo "Running 100 games with up to $MAX_PARALLEL_GAMES parallel processes at $(date)..."

for i in {1..100}; do
    # Wait for a free slot before starting new game
    wait_for_slot
    
    echo "Run $i"
    # Use your config file
    (python simulator/HitlerGame.py --config config-local.yaml || echo "Run $i failed, continuing with next run") &
done

# Wait for all background jobs to complete
echo "Waiting for all games to complete..."
wait
echo "All games completed at $(date)!"