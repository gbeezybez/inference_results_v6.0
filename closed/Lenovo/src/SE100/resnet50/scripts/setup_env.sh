#!/bin/bash
# setup_env.sh - Create Python virtual environment and install dependencies
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOADGEN_DIR="$(cd "${BASE_DIR}/../../loadgen" && pwd)"

VENV_DIR="${BASE_DIR}/mlperf_env"
REQUIREMENTS_FILE="${BASE_DIR}/requirements.txt"

echo "[ Info ] Creating Python virtual environment at ${VENV_DIR}..."
python3 -m venv "${VENV_DIR}" || { echo "[ Error ] Failed to create virtual environment."; exit 1; }

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "[ Info ] Upgrading pip..."
python3 -m pip install --upgrade pip || { echo "[ Error ] Failed to upgrade pip."; exit 1; }

echo "[ Info ] Installing Python dependencies..."
python3 -m pip install -r "${REQUIREMENTS_FILE}" || { echo "[ Error ] Failed to install requirements."; exit 1; }

echo "[ Info ] Building and installing MLPerf loadgen..."
pushd "${LOADGEN_DIR}" > /dev/null
CFLAGS="-std=c++14" python3 setup.py install || { echo "[ Error ] Failed to install loadgen."; popd > /dev/null; exit 1; }
popd > /dev/null

echo "[ Info ] Installing onnxruntime-openvino..."
python3 -m pip uninstall -y onnxruntime-openvino 2>/dev/null || true
python3 -m pip install onnxruntime-openvino || { echo "[ Error ] Failed to install onnxruntime-openvino."; exit 1; }

echo ""
echo "[ Done ] MLPerf environment created successfully."
echo "[ Info ] To activate, run: source ${VENV_DIR}/bin/activate"
