#!/bin/bash
# =============================================================================
# install_visual_gen.sh - Install visual_gen for Wan-2.2 A14B
# =============================================================================
#
# This script follows the installation guide from:
#   code/wan22-a14b/tensorrt/visual_gen/README.md
#
# Installation steps:
#   1. Clear pip constraints (for NVIDIA container compatibility)
#   2. Check visual_gen source is available
#   3. Install visual_gen: pip install -e . --no-build-isolation --no-deps
#
# Note: This script should be run with root permission.
#       flashinfer should be installed with local user access separately.
#
# Usage:
#   sudo bash install_visual_gen.sh              # Run as root
#   pip install --user flashinfer-python   # Run as local user for flashinfer
# =============================================================================

set -Eeo pipefail
trap 'echo "[install_visual_gen.sh] Error on line $LINENO" >&2' ERR

# visual_gen source location (mounted at runtime via volume mount)
VISUAL_GEN_DIR="${VISUAL_GEN_DIR:-/work/3rdparty/trtllm/tensorrt_llm/visual_gen}"

echo "=========================================="
echo "Installing visual_gen for Wan-2.2 A14B"
echo "=========================================="
echo "VISUAL_GEN_DIR: ${VISUAL_GEN_DIR}"
echo "Running as user: $(whoami)"

# Check if running as root for system-wide installations
if [ "$(id -u)" -ne 0 ]; then
    echo "[WARNING] Not running as root. Some installations may fail."
    echo "[INFO] Run with: sudo bash install_visual_gen.sh"
fi

# =============================================================================
# 1. Clear pip constraints (for NVIDIA container compatibility)
# =============================================================================
echo "[1/4] Clearing pip constraints..."
{ echo -n > /etc/pip/constraint.txt; } 2>/dev/null || true


# =============================================================================
# 2. Check if visual_gen source is available
# =============================================================================
echo "[2/3] Check visual_gen path..."
if [ ! -d "${VISUAL_GEN_DIR}" ]; then
    echo "[INFO] visual_gen source not found at ${VISUAL_GEN_DIR}"
    echo "[INFO] visual_gen will be installed at runtime when source is mounted."
    echo "[INFO] Run this script again after launching the container:"
    echo "       VISUAL_GEN_DIR=/work/3rdparty/trtllm/tensorrt_llm/visual_gen bash docker/common/install_visual_gen.sh"
    exit 0
fi

cd "${VISUAL_GEN_DIR}"

# =============================================================================
# 3. Install visual_gen (as per README)
# =============================================================================
echo "[3/3] Installing visual_gen..."
# Try system-wide install first, fall back to user install if permission denied
pip install -e . --no-build-isolation --no-deps 2>/dev/null || \
pip install --user -e . --no-build-isolation --no-deps || {
    echo "[ERROR] visual_gen installation failed with exit code $?"
    exit 1
}

echo "=========================================="
echo "visual_gen installation completed!"
echo "=========================================="

# Verify installation
python -c "import visual_gen; from visual_gen.__version__ import __version__; print(f'visual_gen version: {__version__}')" || {
    echo "[WARNING] visual_gen import verification failed"
}
