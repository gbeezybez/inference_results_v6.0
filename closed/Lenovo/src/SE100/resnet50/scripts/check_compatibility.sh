#!/bin/bash

# SPDX-FileCopyrightText: (C) 2024 - 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# MLPerf Inference OpenVINO Pre-flight Compatibility Check
# Quick validation of system requirements before running benchmarks
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================"
echo "MLPerf Inference OpenVINO Compatibility Check"
echo "========================================================"
echo ""

# Color-coding
print_pass() { echo -e "\e[32m[ Pass ]\e[0m $1"; }
print_fail() { echo -e "\e[31m[ Fail ]\e[0m $1"; }
print_warn() { echo -e "\e[33m[ Warn ]\e[0m $1"; }
print_info() { echo -e "\e[34m[ Info ]\e[0m $1"; }

WARNINGS=0
ERRORS=0

# ==============================================================================
# Operating System
# ==============================================================================
echo "Checking Operating System..."
if [ -f /etc/os-release ]; then
    # shellcheck source=/dev/null
    . /etc/os-release
    if [[ "$ID" == "ubuntu" ]]; then
        if [[ "$VERSION_ID" == "22.04" || "$VERSION_ID" == "24.04" ]]; then
            print_pass "Ubuntu $VERSION_ID detected"
        else
            print_warn "Ubuntu $VERSION_ID detected. Recommended: 22.04 or 24.04"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        print_warn "Non-Ubuntu system detected: $NAME $VERSION_ID"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    print_fail "Cannot detect operating system"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# ==============================================================================
# Python Environment
# ==============================================================================
echo "Checking Python Environment..."

# Check for venv
VENV_PATH="${SCRIPT_DIR}/../mlperf_env"
if [[ -d "${VENV_PATH}" && -f "${VENV_PATH}/bin/activate" ]]; then
    print_pass "Virtual environment found: mlperf_env/"
    
    # Activate and check packages
    # shellcheck source=/dev/null
    source "${VENV_PATH}/bin/activate"
    
    # Python version
    PYTHON_VERSION=$(python3 --version 2>/dev/null | awk '{print $2}')
    if [[ -n "$PYTHON_VERSION" ]]; then
        print_pass "Python $PYTHON_VERSION"
    fi
    
    # OpenVINO
    if python3 -c "import openvino" 2>/dev/null; then
        OV_VERSION=$(python3 -c "from openvino import get_version; print(get_version())" 2>/dev/null)
        print_pass "OpenVINO installed: $OV_VERSION"
    else
        print_fail "OpenVINO not installed in venv"
        print_info "Run: ./scripts/setup_env.sh"
        ERRORS=$((ERRORS + 1))
    fi
    
    # MLPerf LoadGen
    if python3 -c "import mlperf_loadgen" 2>/dev/null; then
        print_pass "MLPerf LoadGen installed"
    else
        print_fail "MLPerf LoadGen not installed"
        print_info "Run: ./scripts/setup_env.sh"
        ERRORS=$((ERRORS + 1))
    fi
    
    deactivate
else
    print_fail "Virtual environment not found"
    print_info "Run: ./scripts/setup_env.sh"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# ==============================================================================
# Model Files
# ==============================================================================
echo "Checking Model Files..."

# FP32 ONNX model
ONNX_MODEL="${SCRIPT_DIR}/../source-models/resnet-50/resnet50_v1.onnx"
if [[ -f "${ONNX_MODEL}" ]]; then
    print_pass "FP32 ONNX model found"
else
    print_warn "FP32 ONNX model not found"
    print_info "Run: ./scripts/download_model.sh"
    WARNINGS=$((WARNINGS + 1))
fi

# FP32 OpenVINO IR
FP32_IR="${SCRIPT_DIR}/../converted-models/resnet-50/FP32/resnet50_v1.xml"
if [[ -f "${FP32_IR}" ]]; then
    print_pass "FP32 OpenVINO IR found"
else
    print_warn "FP32 OpenVINO IR not found"
    print_info "Run: ./nncf-quantize/convert_model.sh"
    WARNINGS=$((WARNINGS + 1))
fi

# INT8 OpenVINO IR
INT8_IR="${SCRIPT_DIR}/../converted-models/resnet-50/INT8/resnet50_v1.xml"
if [[ -f "${INT8_IR}" ]]; then
    print_pass "INT8 OpenVINO IR found"
else
    print_warn "INT8 OpenVINO IR not found"
    print_info "Run: ./nncf-quantize/quantize_model.sh"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# ==============================================================================
# Dataset
# ==============================================================================
echo "Checking Dataset..."

IMAGENET_DIR="${SCRIPT_DIR}/../datasets/imagenet/val"
if [[ -d "${IMAGENET_DIR}" ]]; then
    IMAGE_COUNT=$(find "${IMAGENET_DIR}" -type f \( -name "*.JPEG" -o -name "*.jpeg" -o -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
    if [[ ${IMAGE_COUNT} -ge 50000 ]]; then
        print_pass "ImageNet validation set: ${IMAGE_COUNT} images"
    elif [[ ${IMAGE_COUNT} -gt 0 ]]; then
        print_warn "ImageNet partial: ${IMAGE_COUNT} images (50000 expected)"
        WARNINGS=$((WARNINGS + 1))
    else
        print_fail "ImageNet directory exists but no images found"
        ERRORS=$((ERRORS + 1))
    fi
else
    print_fail "ImageNet dataset not found"
    print_info "Run: ./scripts/prepare_imagenet_dataset.sh"
    ERRORS=$((ERRORS + 1))
fi

# val_map.txt
VAL_MAP="${SCRIPT_DIR}/../datasets/imagenet/val_map.txt"
if [[ -f "${VAL_MAP}" ]]; then
    MAP_LINES=$(wc -l < "${VAL_MAP}")
    print_pass "val_map.txt found: ${MAP_LINES} entries"
else
    print_fail "val_map.txt not found"
    print_info "Run: ./scripts/prepare_imagenet_dataset.sh"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# ==============================================================================
# GPU Support (Optional)
# ==============================================================================
echo "Checking GPU Support (optional)..."

if command -v clinfo >/dev/null 2>&1; then
    GPU_INFO=$(clinfo 2>/dev/null | grep -m1 "Device Name" | sed 's/.*: *//' || echo "")
    if [[ -n "$GPU_INFO" ]]; then
        print_pass "OpenCL device: $GPU_INFO"
    else
        print_info "No OpenCL devices found"
    fi
else
    print_info "clinfo not installed (optional for GPU)"
fi

# Check for GPU render devices
if ls /dev/dri/render* >/dev/null 2>&1; then
    RENDER_DEVICES=$(find /dev/dri -name "render*" -type c 2>/dev/null | wc -l)
    print_pass "GPU render devices: $RENDER_DEVICES"
else
    print_info "No GPU render devices at /dev/dri/render*"
fi
echo ""

# ==============================================================================
# NPU Support (Optional)
# ==============================================================================
echo "Checking NPU Support (optional)..."

if ls /dev/accel/accel* >/dev/null 2>&1; then
    NPU_DEVICES=$(find /dev/accel -name "accel*" -type c 2>/dev/null | wc -l)
    print_pass "NPU device(s) detected: $NPU_DEVICES"
    
    if dpkg -l intel-driver-compiler-npu 2>/dev/null | grep -q "^ii"; then
        NPU_VERSION=$(dpkg -l intel-driver-compiler-npu 2>/dev/null | grep "^ii" | awk '{print $3}')
        print_pass "NPU driver installed: $NPU_VERSION"
    else
        print_warn "NPU hardware detected but driver not installed"
        print_info "Install NPU drivers with: ./scripts/drivers/install_npu_driver.sh"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    print_info "No NPU detected (optional - requires Core Ultra)"
fi
echo ""

# ==============================================================================
# System Resources
# ==============================================================================
echo "Checking System Resources..."

# RAM
TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
if [[ "$TOTAL_RAM_GB" -ge 16 ]]; then
    print_pass "RAM: ${TOTAL_RAM_GB}GB (16GB+ recommended)"
elif [[ "$TOTAL_RAM_GB" -ge 8 ]]; then
    print_warn "RAM: ${TOTAL_RAM_GB}GB (16GB+ recommended for full dataset)"
    WARNINGS=$((WARNINGS + 1))
else
    print_fail "RAM: ${TOTAL_RAM_GB}GB (minimum 8GB required)"
    ERRORS=$((ERRORS + 1))
fi

# Disk Space
AVAILABLE_SPACE=$(df -h "${SCRIPT_DIR}/.." | tail -n1 | awk '{print $4}')
AVAILABLE_SPACE_NUM=$(echo "$AVAILABLE_SPACE" | grep -oP '^\d+' || echo "0")
AVAILABLE_SPACE_UNIT=$(echo "$AVAILABLE_SPACE" | grep -oP '[A-Z]+$' || echo "")

if [[ "$AVAILABLE_SPACE_UNIT" == "G" && "$AVAILABLE_SPACE_NUM" -ge 20 ]]; then
    print_pass "Available disk space: ${AVAILABLE_SPACE} (20GB+ recommended)"
elif [[ "$AVAILABLE_SPACE_UNIT" == "T" ]]; then
    print_pass "Available disk space: ${AVAILABLE_SPACE} (20GB+ recommended)"
else
    print_warn "Available disk space: ${AVAILABLE_SPACE} (20GB+ recommended)"
    WARNINGS=$((WARNINGS + 1))
fi

# CPU cores
CPU_CORES=$(nproc)
print_pass "CPU cores: ${CPU_CORES}"
echo ""

# ==============================================================================
# Compliance Test Files
# ==============================================================================
echo "Checking Compliance Test Files..."

COMPLIANCE_DIR="${SCRIPT_DIR}/../../../compliance"
if [[ -d "${COMPLIANCE_DIR}/TEST01" ]]; then
    print_pass "TEST01 audit config found"
else
    print_warn "TEST01 audit config not found"
    WARNINGS=$((WARNINGS + 1))
fi

if [[ -d "${COMPLIANCE_DIR}/TEST04" ]]; then
    print_pass "TEST04 audit config found"
else
    print_warn "TEST04 audit config not found"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# ==============================================================================
# Summary
# ==============================================================================
echo "========================================================"
echo "Compatibility Check Summary"
echo "========================================================"

if [[ $ERRORS -eq 0 ]] && [[ $WARNINGS -eq 0 ]]; then
    print_pass "All checks passed! System is ready for MLPerf Inference benchmarks."
    echo ""
    echo "Run benchmarks with:"
    echo "  ./run_mlperf_ov_resnet.sh -t performance -d CPU -p INT8 -s SingleStream"
    echo "  ./run_mlperf_ov_resnet.sh -t accuracy -d CPU -p INT8 -s SingleStream"
    exit 0
elif [[ $ERRORS -eq 0 ]]; then
    print_warn "Completed with $WARNINGS warning(s)"
    echo ""
    echo "You can proceed, but some features may be limited."
    echo "Review the warnings above."
    exit 0
else
    print_fail "Failed with $ERRORS error(s) and $WARNINGS warning(s)"
    echo ""
    echo "Please resolve the errors before running benchmarks."
    echo ""
    echo "Quick setup:"
    echo "  1. ./scripts/install_prerequisites.sh"
    echo "  2. ./scripts/setup_env.sh"
    echo "  3. ./scripts/download_model.sh"
    echo "  4. ./nncf-quantize/convert_model.sh"
    echo "  5. ./nncf-quantize/quantize_model.sh"
    echo "  6. ./scripts/prepare_imagenet_dataset.sh"
    exit 1
fi
