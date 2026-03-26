#!/bin/bash
# run_verification.sh - Run MLPerf compliance verification for TEST01 and TEST04
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPLIANCE_DIR="${BASE_DIR}/../../compliance"
VENV_DIR="${BASE_DIR}/mlperf_env"

# Default values
DEVICE=""
PRECISION=""
SCENARIO=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run MLPerf compliance verification for TEST01 and TEST04.

Options:
    -d, --device DEVICE     Device: CPU, GPU, or NPU (required)
    -p, --precision PREC    Precision: FP32, FP16, or INT8 (required)
    -s, --scenario SCENARIO Scenario: SingleStream, Offline, or MultiStream (required)
    -h, --help              Show this help message and exit

Example:
    $(basename "$0") -d CPU -p INT8 -s SingleStream

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
        -h|--help)
            usage
            ;;
        *)
            die "Unknown option: $1. Use --help for usage information."
            ;;
    esac
done

# Validate required parameters
[[ -z "${DEVICE}" ]] && die "DEVICE is required"
[[ -z "${PRECISION}" ]] && die "PRECISION is required"
[[ -z "${SCENARIO}" ]] && die "SCENARIO is required"

# Build config identifier
CONFIG_ID="${PRECISION,,}-${DEVICE,,}"
SCENARIO_LC="${SCENARIO,,}"

# Convert scenario to MLPerf title case
case "${SCENARIO_LC}" in
    singlestream) SCENARIO_TITLE="SingleStream" ;;
    multistream)  SCENARIO_TITLE="MultiStream" ;;
    offline)      SCENARIO_TITLE="Offline" ;;
    *) die "Invalid scenario '${SCENARIO}'. Must be SingleStream, Offline, or MultiStream." ;;
esac

# Paths
RESULTS_DIR="${BASE_DIR}/output-logs/resnet-${CONFIG_ID}-performance-${SCENARIO_LC}"
AUDIT_DIR="${BASE_DIR}/audit-logs/resnet-${CONFIG_ID}-${SCENARIO_LC}"
OUTPUT_DIR="${BASE_DIR}/verification-output/resnet-${CONFIG_ID}-${SCENARIO_LC}"

# Check prerequisites
if [[ ! -d "${RESULTS_DIR}" ]]; then
    die "Performance results not found at ${RESULTS_DIR}. Run 'make benchmark' first."
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    die "Virtual environment not found. Run 'make setup' first."
fi

# Activate virtual environment
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "  MLPerf Compliance Verification"
echo "  Config:   ${PRECISION} / ${DEVICE} / ${SCENARIO_TITLE}"
echo "============================================================"
echo ""

VERIFICATION_PASSED=true

# TEST01 verification
TEST01_AUDIT="${AUDIT_DIR}/TEST01"
if [[ -d "${TEST01_AUDIT}" ]]; then
    echo "[ TEST01 ] Running verification..."
    if python3 "${COMPLIANCE_DIR}/TEST01/run_verification.py" \
        -r "${RESULTS_DIR}" \
        -c "${TEST01_AUDIT}" \
        -o "${OUTPUT_DIR}" 2>&1; then
        echo "[ TEST01 ] ✓ PASSED"
    else
        echo "[ TEST01 ] ✗ FAILED"
        VERIFICATION_PASSED=false
    fi
else
    echo "[ TEST01 ] - SKIPPED (no audit logs at ${TEST01_AUDIT})"
    echo "           Run: make benchmark DEVICE=${DEVICE} PRECISION=${PRECISION} SCENARIO=${SCENARIO} AUDIT=TEST01"
fi

echo ""

# TEST04 verification
TEST04_AUDIT="${AUDIT_DIR}/TEST04"
if [[ -d "${TEST04_AUDIT}" ]]; then
    echo "[ TEST04 ] Running verification..."
    if python3 "${COMPLIANCE_DIR}/TEST04/run_verification.py" \
        -r "${RESULTS_DIR}" \
        -c "${TEST04_AUDIT}" \
        -o "${OUTPUT_DIR}" 2>&1; then
        echo "[ TEST04 ] ✓ PASSED"
    else
        echo "[ TEST04 ] ✗ FAILED"
        VERIFICATION_PASSED=false
    fi
else
    echo "[ TEST04 ] - SKIPPED (no audit logs at ${TEST04_AUDIT})"
    echo "           Run: make benchmark DEVICE=${DEVICE} PRECISION=${PRECISION} SCENARIO=${SCENARIO} AUDIT=TEST04"
fi

echo ""
echo "============================================================"
if ${VERIFICATION_PASSED}; then
    echo "[ Result ] All verifications PASSED"
else
    echo "[ Result ] Some verifications FAILED"
fi
echo "  Output:  ${OUTPUT_DIR}/"
echo "============================================================"
echo ""

${VERIFICATION_PASSED}
