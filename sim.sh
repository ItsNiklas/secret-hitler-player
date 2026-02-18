#!/bin/bash

# Parse command-line arguments
CONFIG_FILE=${1:-config-local.yaml}

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
echo "Using config: $CONFIG_FILE"
echo "Alice (Player 1) role distribution: 60x Liberal, 20x Fascist, 20x Hitler"

for i in {1..100}; do
    # Wait for a free slot before starting new game
    wait_for_slot
    
    # Determine Alice's role based on game number
    # Games 1-60: Liberal
    # Games 61-80: Fascist
    # Games 81-100: Hitler
    if [ $i -le 60 ]; then
        ROLE="liberal"
    elif [ $i -le 80 ]; then
        ROLE="fascist"
    else
        ROLE="hitler"
    fi
    
    echo "Run $i (Alice: $ROLE)"
    # Use your config file and force Alice's role
    (python simulator/HitlerGame.py --config $CONFIG_FILE --role $ROLE || echo "Run $i failed, continuing with next run") &
done

# Wait for all background jobs to complete
echo "Waiting for all games to complete..."
wait
echo "All games completed at $(date)!"