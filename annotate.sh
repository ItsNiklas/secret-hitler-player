#!/bin/bash
# Set the folder containing JSON files
FOLDER="crawl/replay_data"  # Replace with your actual path
OUT_FOLDER="annotationQwen2532B"

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
current_jobs=0

# Function to wait for job slots to free up
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_GAMES ]; do
        sleep 5  # Check every 5 seconds
    done
}

echo "Running games with up to $MAX_PARALLEL_GAMES parallel processes"

# Process each JSON file in the folder
for filename in "$FOLDER"/*.json; do
    # Check if file exists and is readable
    if [ -r "$filename" ]; then
        echo "Processing $filename"
        wait_for_slot  # Wait for a free slot before starting new job
        
        # Run annotation script in background
        python annotation/persuasion_annotation.py "$filename" -f "$OUT_FOLDER" || {
            echo "Failed to process $filename, continuing with next file"
            continue
        } &
        
        # Brief pause to avoid overwhelming system with job starts
        sleep 1
    fi
done

# Wait for all background jobs to complete
echo "Waiting for all games to complete..."
wait
echo "All games completed!"

