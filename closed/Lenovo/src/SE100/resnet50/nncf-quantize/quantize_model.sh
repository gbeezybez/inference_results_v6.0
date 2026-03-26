#!/bin/bash
# quantize_model.sh - Quantize ResNet-50 to INT8 using NNCF
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Model paths
SOURCE_MODEL="${BASE_DIR}/converted-models/resnet-50/FP16/resnet50_v1.xml"
OUTPUT_DIR="${BASE_DIR}/converted-models/resnet-50/INT8"
VENV_DIR="${BASE_DIR}/mlperf_env"

# Dataset path (requires ImageNet tar files for torchvision.datasets.ImageNet)
IMAGENET_DIR="${BASE_DIR}/datasets/imagenet-packages"

# Quantization parameters
DEVICE="GPU"
CALIB_SUBSET=512
EVAL_SAMPLES=50000

# Check for source model
if [[ ! -f "${SOURCE_MODEL}" ]]; then
    echo "[ Error ] FP16 model not found at ${SOURCE_MODEL}"
    echo "[ Info ] Please run convert_model.sh first."
    exit 1
fi

# Check for ImageNet dataset
if [[ ! -d "${IMAGENET_DIR}" ]]; then
    echo "[ Error ] ImageNet packages directory not found at ${IMAGENET_DIR}"
    echo "[ Info ] Please place ILSVRC2012_img_val.tar and ILSVRC2012_devkit_t12.tar.gz in ${IMAGENET_DIR}/"
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

mkdir -p "${OUTPUT_DIR}"

echo "[ Info ] Starting INT8 quantization with NNCF..."
echo "[ Info ] Device: ${DEVICE}, Calibration samples: ${CALIB_SUBSET}, Eval samples: ${EVAL_SAMPLES}"

python3 "${SCRIPT_DIR}/quantize_resnet50.py" \
    --model "${SOURCE_MODEL}" \
    --imagenet-root "${IMAGENET_DIR}" \
    --out-dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --calib-subset "${CALIB_SUBSET}" \
    --eval-samples "${EVAL_SAMPLES}" \
    || { echo "[ Error ] Quantization failed."; exit 1; }

echo ""
echo "[ Done ] INT8 quantization completed successfully."
