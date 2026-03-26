#!/bin/bash

# Script to download AMD Llama-3.1-405B-Instruct FP4 quantized model from Hugging Face
# Usage: ./download_llama_405b_fp4.sh <HF_TOKEN>

set -e

MODEL_ID="amd/Llama-3.1-405B-Instruct-wmxfp4-amxfp4-kvfp8-scale-uint8-asq-mlperf"
DOWNLOAD_DIR="/data/inference/model/llama3.1-405b/fp4_quantized"

# Check if HF token is provided
if [ -z "$1" ]; then
    echo "Error: Hugging Face token is required"
    echo "Usage: $0 <HF_TOKEN>"
    exit 1
fi

HF_TOKEN="$1"

echo "=============================================="
echo "Downloading model: $MODEL_ID"
echo "Target directory: $DOWNLOAD_DIR"
echo "=============================================="

# Create target directory if it doesn't exist (using sudo)
# Set permissions on the entire path since HuggingFace creates temp files in parent dirs
BASE_DIR="/data/inference/model/llama3.1-405b"
if [ ! -d "$DOWNLOAD_DIR" ]; then
    echo "Creating directory: $DOWNLOAD_DIR"
    sudo mkdir -p "$DOWNLOAD_DIR"
fi
echo "Setting permissions on: $BASE_DIR"
sudo chown -R $(whoami):$(id -gn) "$BASE_DIR"

# Install/upgrade huggingface_hub and its dependencies
echo "Ensuring huggingface_hub and dependencies are up to date..."
pip install --upgrade huggingface_hub filelock

# Check if huggingface-cli is available
if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli to download model..."
    huggingface-cli download "$MODEL_ID" \
        --local-dir "$DOWNLOAD_DIR" \
        --token "$HF_TOKEN"
else
    echo "Using Python huggingface_hub to download model..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$MODEL_ID',
    local_dir='$DOWNLOAD_DIR',
    token='$HF_TOKEN'
)
"
fi

echo "=============================================="
echo "Download complete!"
echo "Model saved to: $DOWNLOAD_DIR"
echo "=============================================="