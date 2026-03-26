#!/bin/bash
# prepare_imagenet_dataset.sh - Extract and prepare ImageNet validation dataset
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Accept IMAGENET_ROOT as argument or use default
PACKAGES_DIR="${1:-${BASE_DIR}/datasets/imagenet-packages}"
IMAGENET_DIR="${BASE_DIR}/datasets/imagenet"
VAL_TAR="${PACKAGES_DIR}/ILSVRC2012_img_val.tar"
VAL_MAP_URL="https://raw.githubusercontent.com/Abhishekghosh1998/MLPerf_ImageNet_val_vap_map_txt/refs/heads/main/val_map.txt"

# Check for validation tar
if [[ ! -f "${VAL_TAR}" ]]; then
    echo "[ Error ] ImageNet validation tar not found at ${VAL_TAR}"
    echo ""
    echo "Usage: $(basename "$0") [IMAGENET_ROOT]"
    echo ""
    echo "  IMAGENET_ROOT: Directory containing ILSVRC2012_img_val.tar"
    echo "                 (default: ${BASE_DIR}/datasets/imagenet-packages)"
    echo ""
    echo "Download the ImageNet 2012 validation dataset from https://image-net.org/download.php"
    echo "and place ILSVRC2012_img_val.tar in the specified directory."
    exit 1
fi

echo "[ Info ] Setting up ImageNet directory..."
mkdir -p "${IMAGENET_DIR}/val"

# Extract validation images
if [[ -z "$(ls -A "${IMAGENET_DIR}/val" 2>/dev/null)" ]]; then
    echo "[ Info ] Extracting ImageNet validation dataset..."
    tar -xf "${VAL_TAR}" -C "${IMAGENET_DIR}/val" \
        || { echo "[ Error ] Failed to extract validation tar."; exit 1; }
    echo "[ Done ] Extracted validation images to ${IMAGENET_DIR}/val/"
else
    echo "[ Info ] Validation images already extracted, skipping."
fi

# Download val_map.txt
if [[ -f "${IMAGENET_DIR}/val_map.txt" ]]; then
    echo "[ Info ] val_map.txt already exists, skipping download."
else
    echo "[ Info ] Downloading val_map.txt..."
    wget -q --show-progress -O "${IMAGENET_DIR}/val_map.txt" "${VAL_MAP_URL}" \
        || { echo "[ Error ] Failed to download val_map.txt."; exit 1; }
    
    echo "[ Info ] Updating val_map.txt paths..."
    sed -i 's/ILSV/val\/ILSV/g' "${IMAGENET_DIR}/val_map.txt"
    echo "[ Done ] val_map.txt downloaded and configured."
fi

echo ""
echo "[ Done ] ImageNet dataset ready at ${IMAGENET_DIR}/"
