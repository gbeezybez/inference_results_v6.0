#!/bin/bash
# run_mlperf_ov_resnet.sh - Run MLPerf Inference benchmark for ResNet-50 with OpenVINO
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
TEST_TYPE="performance"
SMOKE_TEST=false
DEVICE="CPU"
PRECISION="FP16"
SCENARIO="SingleStream"
AUDIT="none"

# Smoke test duration (seconds)
SMOKE_DURATION=10

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run MLPerf Inference benchmark for ResNet-50 using OpenVINO backend.

Options:
    -t, --test-type TYPE    Test type: 'performance' or 'accuracy' (default: performance)
    -d, --device DEVICE     OpenVINO device: CPU, GPU, or NPU (default: CPU)
    -p, --precision PREC    Model precision: FP32, FP16, or INT8 (default: FP16)
    -s, --scenario SCENARIO MLPerf scenario: SingleStream, Offline, or MultiStream (default: SingleStream)
    -a, --audit AUDIT       MLPerf compliance audit: none, TEST01, or TEST04 (default: none)
        --smoke-test        Enable smoke test mode (uses fake ImageNet, ${SMOKE_DURATION}s duration)
    -h, --help              Show this help message and exit

Examples:
    $(basename "$0") --smoke-test
        Run a quick smoke test with default settings

    $(basename "$0") -t accuracy -d GPU -p INT8
        Run accuracy test on GPU with INT8 model

    $(basename "$0") -t performance -d GPU -p FP16 -s Offline
        Run performance test in Offline scenario on GPU with FP16 model

    $(basename "$0") -t performance -d GPU -p INT8 --audit TEST01
        Run performance test with TEST01 compliance audit

EOF
    exit 0
}

die() {
    echo "[ Error ] $1" >&2
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--test-type)
            [[ -z "${2:-}" ]] && die "Missing argument for $1"
            TEST_TYPE="$2"
            shift 2
            ;;
        -d|--device)
            [[ -z "${2:-}" ]] && die "Missing argument for $1"
            DEVICE="$2"
            shift 2
            ;;
        -p|--precision)
            [[ -z "${2:-}" ]] && die "Missing argument for $1"
            PRECISION="$2"
            shift 2
            ;;
        -s|--scenario)
            [[ -z "${2:-}" ]] && die "Missing argument for $1"
            SCENARIO="$2"
            shift 2
            ;;
        -a|--audit)
            [[ -z "${2:-}" ]] && die "Missing argument for $1"
            AUDIT="$2"
            shift 2
            ;;
        --smoke-test)
            SMOKE_TEST=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            die "Unknown option: $1. Use --help for usage information."
            ;;
    esac
done

# Validate test type
case "${TEST_TYPE}" in
    performance|accuracy) ;;
    *) die "Invalid test type '${TEST_TYPE}'. Must be 'performance' or 'accuracy'." ;;
esac

# Validate device
case "${DEVICE}" in
    CPU|GPU|NPU) ;;
    *) die "Invalid device '${DEVICE}'. Must be CPU, GPU, or NPU." ;;
esac

# Validate precision
case "${PRECISION}" in
    FP32|FP16|INT8) ;;
    *) die "Invalid precision '${PRECISION}'. Must be FP32, FP16, or INT8." ;;
esac

# Validate scenario
case "${SCENARIO}" in
    SingleStream|Offline|MultiStream) ;;
    Server) die "Server scenario is not supported for this benchmark." ;;
    *) die "Invalid scenario '${SCENARIO}'. Must be SingleStream, Offline, or MultiStream." ;;
esac

# Validate audit
case "${AUDIT}" in
    none|TEST01|TEST04) ;;
    *) die "Invalid audit '${AUDIT}'. Must be none, TEST01, or TEST04." ;;
esac

# Set paths
MODEL_PATH="${SCRIPT_DIR}/converted-models/resnet-50/${PRECISION}/resnet50_v1.xml"
VENV_DIR="${SCRIPT_DIR}/mlperf_env"

# Build config identifier: {precision}-{device} in lowercase
CONFIG_ID="${PRECISION,,}-${DEVICE,,}"

if ${SMOKE_TEST}; then
    DATA_DIR="${SCRIPT_DIR}/datasets/fake-imagenet"
    OUTPUT_DIR="${SCRIPT_DIR}/output-logs/smoke-test-${CONFIG_ID}-${SCENARIO,,}"
    TIME_FLAG="--time=${SMOKE_DURATION}"
else
    DATA_DIR="${SCRIPT_DIR}/datasets/imagenet"
    if [[ "${AUDIT}" != "none" ]]; then
        OUTPUT_DIR="${SCRIPT_DIR}/audit-logs/resnet-${CONFIG_ID}-${SCENARIO,,}/${AUDIT}"
    else
        OUTPUT_DIR="${SCRIPT_DIR}/output-logs/resnet-${CONFIG_ID}-${TEST_TYPE}-${SCENARIO,,}"
    fi
    TIME_FLAG=""
fi

if [[ "${TEST_TYPE}" == "accuracy" ]]; then
    ACCURACY_FLAG="--accuracy"
else
    ACCURACY_FLAG=""
fi

# Pre-flight checks
if [[ ! -f "${MODEL_PATH}" ]]; then
    die "Model not found at ${MODEL_PATH}. Please run nncf-quantize/convert_model.sh first."
fi

if [[ ! -d "${DATA_DIR}" ]]; then
    if ${SMOKE_TEST}; then
        die "Fake ImageNet dataset not found at ${DATA_DIR}. Please run scripts/download_model.sh first."
    else
        die "ImageNet dataset not found at ${DATA_DIR}. Please run scripts/prepare_imagenet_dataset.sh first."
    fi
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    die "Virtual environment not found at ${VENV_DIR}. Please run scripts/setup_env.sh first."
fi

# Activate virtual environment
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Setup audit config if needed
AUDIT_CONFIG_COPIED=false
if [[ "${AUDIT}" != "none" ]]; then
    AUDIT_SRC="${SCRIPT_DIR}/audit-files/${AUDIT}/audit.config"
    AUDIT_DST="${SCRIPT_DIR}/audit.config"
    
    if [[ ! -f "${AUDIT_SRC}" ]]; then
        die "Audit config not found at ${AUDIT_SRC}"
    fi
    
    echo "[ Info ] Copying audit config for ${AUDIT}..."
    cp "${AUDIT_SRC}" "${AUDIT_DST}"
    AUDIT_CONFIG_COPIED=true
fi

# Cleanup function to remove audit config on exit
cleanup() {
    if ${AUDIT_CONFIG_COPIED} && [[ -f "${SCRIPT_DIR}/audit.config" ]]; then
        echo "[ Info ] Removing audit config..."
        rm -f "${SCRIPT_DIR}/audit.config"
    fi
}
trap cleanup EXIT

# Print configuration
echo ""
echo "[ Config ] MLPerf Inference - ResNet-50 OpenVINO"
echo "  Test Type:  ${TEST_TYPE}"
echo "  Device:     ${DEVICE}"
echo "  Precision:  ${PRECISION}"
echo "  Scenario:   ${SCENARIO}"
echo "  Audit:      ${AUDIT}"
echo "  Smoke Test: ${SMOKE_TEST}"
echo "  Model:      ${MODEL_PATH}"
echo "  Dataset:    ${DATA_DIR}"
echo "  Output:     ${OUTPUT_DIR}"
echo ""

echo "[ Info ] Starting MLPerf Inference benchmark..."

# Run benchmark
python3 "${SCRIPT_DIR}/python/main.py" \
    --profile resnet50-openvino \
    --model "${MODEL_PATH}" \
    --dataset-path "${DATA_DIR}" \
    --output "${OUTPUT_DIR}" \
    --scenario "${SCENARIO}" \
    --device "${DEVICE}" \
    ${TIME_FLAG} \
    ${ACCURACY_FLAG}

# Parse accuracy results if this was an accuracy test or TEST01 audit
# Skip accuracy generation for TEST04 (performance-only compliance test)
if [[ "${AUDIT}" != "TEST04" && ("${TEST_TYPE}" == "accuracy" || "${AUDIT}" == "TEST01") ]]; then
    ACCURACY_LOG="${OUTPUT_DIR}/mlperf_log_accuracy.json"
    VAL_MAP="${DATA_DIR}/val_map.txt"
    ACCURACY_OUTPUT="${OUTPUT_DIR}/accuracy.txt"
    
    if [[ -f "${ACCURACY_LOG}" && -f "${VAL_MAP}" ]]; then
        echo ""
        echo "[ Info ] Parsing accuracy results..."
        python3 "${SCRIPT_DIR}/tools/accuracy-imagenet.py" \
            --mlperf-accuracy-file="${ACCURACY_LOG}" \
            --imagenet-val-file="${VAL_MAP}" \
            > "${ACCURACY_OUTPUT}"
        echo "[ Done ] Accuracy results saved to ${ACCURACY_OUTPUT}"
        cat "${ACCURACY_OUTPUT}"
    else
        echo "[ Warning ] Could not parse accuracy: missing ${ACCURACY_LOG} or ${VAL_MAP}"
    fi
fi

echo ""
echo "[ Done ] Benchmark completed. Results saved to ${OUTPUT_DIR}/"
