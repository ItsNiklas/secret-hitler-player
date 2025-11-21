#!/bin/bash
#SBATCH --job-name=scc-cpu
#SBATCH -t 24:00:00
#SBATCH -p medium
#SBATCH -c 12
#SBATCH -N 1

export HF_HOME=/scratch-scc/users/$USER/hf

source .venv/bin/activate

# Run simulator for each summary file in crawl/summaries
for summary_file in crawl/summaries/*.json; do
    # # Wait for user input before processing each file
    # read -p "Press Enter to process $summary_file..."
    # Extract the base name without extension to match with xhr data
    base_name=$(basename "$summary_file" _summary.json)
    xhr_file="crawl/replay_data/${base_name}_xhr_data.json"
    
    # Run the simulator and continue on errors
    echo "Processing $base_name"
    python simulator/HitlerGame.py -l DEBUG -g "$summary_file" -c "$xhr_file" -n 1 || echo "Failed on $summary_file, continuing..."
done

