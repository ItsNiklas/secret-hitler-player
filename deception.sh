#!/bin/bash
# Set the folder containing JSON files
FOLDER=$1  # Replace with your actual path

# Set these environment variables or add them to your .env file
export HF_TOKEN=${HF_TOKEN:-"your-huggingface-token-here"}
export HF_HOME=/scratch-scc/users/$USER/hf
export LLM_API_KEY=${LLM_API_KEY:-"your-llm-api-key-here"}
export LLM_BASE_URL=http://localhost:8080/v1/
export ENABLE_PARALLEL_PROCESSING=true

source .venv/bin/activate

python deception-rate/deception.py $FOLDER
