#!/bin/bash
# Run the evaluation server with a local HuggingFace cache directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME="$SCRIPT_DIR/.hf/cache"
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$HF_HOME"

echo "HuggingFace cache: $HF_HOME"
echo "Starting evaluation server on port 8000..."

cd "$SCRIPT_DIR/server"
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
