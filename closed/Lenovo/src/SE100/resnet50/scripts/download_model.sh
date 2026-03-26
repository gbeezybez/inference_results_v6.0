#!/bin/bash
# download_model.sh - Download ResNet-50 model and create fake ImageNet dataset
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_DIR="${BASE_DIR}/source-models/resnet-50"
DATASETS_DIR="${BASE_DIR}/datasets"

MODEL_URL="https://zenodo.org/record/4735647/files/resnet50_v1.onnx"

# Download ResNet-50 model
echo "[ Info ] Setting up model directory..."
mkdir -p "${MODEL_DIR}"

if [[ -f "${MODEL_DIR}/resnet50_v1.onnx" ]]; then
    echo "[ Info ] ResNet-50 model already exists, skipping download."
else
    echo "[ Info ] Downloading ResNet-50 ONNX model from Zenodo..."
    wget -q --show-progress -O "${MODEL_DIR}/resnet50_v1.onnx" "${MODEL_URL}" \
        || { echo "[ Error ] Failed to download ResNet-50 model."; exit 1; }
    echo "[ Done ] ResNet-50 model downloaded to ${MODEL_DIR}/"
fi

# Setup datasets directory
echo "[ Info ] Setting up datasets directory..."
mkdir -p "${DATASETS_DIR}"

# Create fake ImageNet dataset
if [[ -d "${DATASETS_DIR}/fake-imagenet" ]]; then
    echo "[ Info ] Fake ImageNet dataset already exists, skipping creation."
else
    echo "[ Info ] Creating fake ImageNet dataset for smoke testing..."
    
    if [[ ! -f "${BASE_DIR}/tools/make_fake_imagenet.sh" ]]; then
        echo "[ Error ] make_fake_imagenet.sh not found at ${BASE_DIR}/tools/"
        exit 1
    fi
    
    pushd "${BASE_DIR}" > /dev/null
    bash "./tools/make_fake_imagenet.sh" || { echo "[ Error ] Failed to create fake ImageNet."; popd > /dev/null; exit 1; }
    
    if [[ -d "fake_imagenet" ]]; then
        mv "fake_imagenet" "${DATASETS_DIR}/fake-imagenet"
    else
        echo "[ Error ] fake_imagenet directory was not created."
        popd > /dev/null
        exit 1
    fi
    popd > /dev/null
    
    echo "[ Done ] Fake ImageNet dataset created at ${DATASETS_DIR}/fake-imagenet/"
fi

echo ""
echo "[ Done ] All downloads completed successfully."
