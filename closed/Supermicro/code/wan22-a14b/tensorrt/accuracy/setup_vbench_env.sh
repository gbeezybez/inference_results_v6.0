#!/bin/bash
# =============================================================================
# setup_vbench_env.sh - Set up VBench venv for AccuracyOnly mode harness
#
# This script creates a Python venv with VBench dependencies for use by the
# MLPerf harness during AccuracyOnly mode. This is called automatically by
# Wan22AccuracyChecker in accuracy_checker.py.
#
# For standalone/manual setup with more options (conda support, etc.),
# use internal/setup_vbench_env.sh instead.
#
# Usage:
#   ./setup_vbench_env.sh [--env-path PATH]
#
# Options:
#   --env-path     Path for venv installation (default: /work/.vbench-venv)
#
# Example:
#   ./setup_vbench_env.sh --env-path /work/.vbench-venv
# =============================================================================

set -e

# Default values
VENV_PATH="/work/.vbench-venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-path)
            VENV_PATH="$2"
            shift 2
            ;;
        --help|-h)
            head -n 20 "$0" | tail -n 15
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "VBench Environment Setup for WAN22-A14B (Harness Mode)"
echo "============================================================"
echo "Setting up Python venv at: $VENV_PATH"
echo "------------------------------------------------------------"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is not installed or not in PATH"
    exit 1
fi

# Create venv if it doesn't exist
if [[ ! -d "$VENV_PATH" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
fi

# Activate venv
source "${VENV_PATH}/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install VBench requirements
echo "Installing VBench dependencies..."
pip install -r "${SCRIPT_DIR}/vbench_requirements.txt"

# Install VBench without dependencies
echo "Installing VBench..."
pip install vbench --no-deps

# Fix OpenCV for headless environments (common in containers)
echo "Fixing OpenCV for headless environments..."
pip uninstall -y opencv-python 2>/dev/null || true
pip install --force-reinstall opencv-python-headless>=4.8.0

echo ""
echo "============================================================"
echo "VBench venv created at '$VENV_PATH'"
echo "============================================================"
echo ""
echo "This venv is used automatically by the MLPerf harness during"
echo "AccuracyOnly mode via Wan22AccuracyChecker."
echo ""
echo "For manual activation:"
echo "  source ${VENV_PATH}/bin/activate"
echo ""

echo "Setup complete!"
