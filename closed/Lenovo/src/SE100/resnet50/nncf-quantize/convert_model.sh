#!/bin/bash
# convert_model.sh - Convert ResNet-50 ONNX model to OpenVINO IR format
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOURCE_MODEL="${BASE_DIR}/source-models/resnet-50/resnet50_v1.onnx"
OUTPUT_DIR="${BASE_DIR}/converted-models/resnet-50"
VENV_DIR="${BASE_DIR}/mlperf_env"

# Check for source model
if [[ ! -f "${SOURCE_MODEL}" ]]; then
    echo "[ Error ] Source model not found at ${SOURCE_MODEL}"
    echo "[ Info ] Please run scripts/download_model.sh first."
    exit 1
fi

# Activate virtual environment
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "[ Error ] Virtual environment not found at ${VENV_DIR}"
    echo "[ Info ] Please run scripts/setup_env.sh first."
    exit 1
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# Create output directories
mkdir -p "${OUTPUT_DIR}/FP32"
mkdir -p "${OUTPUT_DIR}/FP16"

echo "[ Info ] Converting ONNX to OpenVINO IR (FP32)..."
ovc "${SOURCE_MODEL}" \
    --output_model "${OUTPUT_DIR}/FP32/resnet50_v1.xml" \
    --input "[1,3,224,224]" \
    --output "softmax_tensor" \
    --compress_to_fp16 False \
    || { echo "[ Error ] FP32 conversion failed."; exit 1; }
echo "[ Done ] FP32 model saved to ${OUTPUT_DIR}/FP32/"

echo "[ Info ] Converting ONNX to OpenVINO IR (FP16)..."
ovc "${SOURCE_MODEL}" \
    --output_model "${OUTPUT_DIR}/FP16/resnet50_v1.xml" \
    --input "[1,3,224,224]" \
    --output "softmax_tensor" \
    || { echo "[ Error ] FP16 conversion failed."; exit 1; }
echo "[ Done ] FP16 model saved to ${OUTPUT_DIR}/FP16/"

echo ""
echo "[ Done ] Model conversion completed successfully."
