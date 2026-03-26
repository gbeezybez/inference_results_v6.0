#!/bin/bash
# show_status.sh - Display MLPerf benchmark coverage matrix for all configurations
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_LOGS="${BASE_DIR}/output-logs"
AUDIT_LOGS="${BASE_DIR}/audit-logs"
MODELS_DIR="${BASE_DIR}/converted-models/resnet-50"
VENV_DIR="${BASE_DIR}/mlperf_env"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Check marks
CHECK="${GREEN}✓${NC}"
CROSS="${RED}✗${NC}"
DASH="${YELLOW}-${NC}"

# ============================================================================
# Environment Status
# ============================================================================
show_environment_status() {
    echo ""
    echo "Environment Status"
    echo "──────────────────────────────────────────────────────────────────────"
    
    # Virtual environment
    if [[ -f "${VENV_DIR}/bin/activate" ]]; then
        echo -e "  ${CHECK} Python virtual environment (mlperf_env/)"
    else
        echo -e "  ${CROSS} Python virtual environment (run: make setup)"
    fi
    
    # Models by precision
    for precision in FP32 FP16 INT8; do
        if [[ -f "${MODELS_DIR}/${precision}/resnet50_v1.xml" ]]; then
            echo -e "  ${CHECK} ResNet-50 model (${precision})"
        else
            echo -e "  ${CROSS} ResNet-50 model (${precision})"
        fi
    done
    
    # Dataset
    if [[ -d "${BASE_DIR}/datasets/imagenet" && -f "${BASE_DIR}/datasets/imagenet/val_map.txt" ]]; then
        echo -e "  ${CHECK} ImageNet dataset"
    else
        echo -e "  ${CROSS} ImageNet dataset (required for full benchmarks)"
    fi
    
    if [[ -d "${BASE_DIR}/datasets/fake-imagenet" ]]; then
        echo -e "  ${CHECK} Fake ImageNet dataset (for smoke tests)"
    else
        echo -e "  ${CROSS} Fake ImageNet dataset"
    fi
}

# ============================================================================
# Discover all configurations
# ============================================================================
discover_configs() {
    local configs=()
    
    # Scan output-logs for patterns like resnet-{precision}-{device}-{test}-{scenario}
    if [[ -d "${OUTPUT_LOGS}" ]]; then
        for dir in "${OUTPUT_LOGS}"/resnet-*-*-*-*/; do
            if [[ -d "${dir}" ]]; then
                dir_name="$(basename "${dir}")"
                # Extract precision-device-scenario from resnet-{precision}-{device}-{test}-{scenario}
                if [[ "${dir_name}" =~ resnet-([a-z0-9]+)-([a-z]+)-(performance|accuracy)-([a-z]+) ]]; then
                    precision="${BASH_REMATCH[1]}"
                    device="${BASH_REMATCH[2]}"
                    scenario="${BASH_REMATCH[4]}"
                    config="${precision}-${device}-${scenario}"
                    if [[ ! " ${configs[*]:-} " =~ " ${config} " ]]; then
                        configs+=("${config}")
                    fi
                fi
            fi
        done
    fi
    
    # Also scan audit-logs
    if [[ -d "${AUDIT_LOGS}" ]]; then
        for dir in "${AUDIT_LOGS}"/resnet-*-*-*/; do
            if [[ -d "${dir}" ]]; then
                dir_name="$(basename "${dir}")"
                # Extract from resnet-{precision}-{device}-{scenario}
                if [[ "${dir_name}" =~ resnet-([a-z0-9]+)-([a-z]+)-([a-z]+) ]]; then
                    precision="${BASH_REMATCH[1]}"
                    device="${BASH_REMATCH[2]}"
                    scenario="${BASH_REMATCH[3]}"
                    config="${precision}-${device}-${scenario}"
                    if [[ ! " ${configs[*]:-} " =~ " ${config} " ]]; then
                        configs+=("${config}")
                    fi
                fi
            fi
        done
    fi
    
    # Sort and return
    printf '%s\n' "${configs[@]}" | sort -u
}

# ============================================================================
# Check status for a specific configuration
# ============================================================================
check_config_status() {
    local precision="$1"
    local device="$2"
    local scenario="$3"
    local config_id="${precision}-${device}"
    
    local perf_status="${CROSS}"
    local acc_status="${CROSS}"
    local test01_status="${CROSS}"
    local test04_status="${CROSS}"
    
    # Performance
    if [[ -f "${OUTPUT_LOGS}/resnet-${config_id}-performance-${scenario}/mlperf_log_summary.txt" ]]; then
        perf_status="${CHECK}"
    fi
    
    # Accuracy
    if [[ -f "${OUTPUT_LOGS}/resnet-${config_id}-accuracy-${scenario}/accuracy.txt" ]]; then
        acc_status="${CHECK}"
    fi
    
    # TEST01
    if [[ -f "${AUDIT_LOGS}/resnet-${config_id}-${scenario}/TEST01/mlperf_log_summary.txt" ]]; then
        test01_status="${CHECK}"
    fi
    
    # TEST04
    if [[ -f "${AUDIT_LOGS}/resnet-${config_id}-${scenario}/TEST04/mlperf_log_summary.txt" ]]; then
        test04_status="${CHECK}"
    fi
    
    echo -e "${perf_status}|${acc_status}|${test01_status}|${test04_status}"
}

# ============================================================================
# Show coverage matrix
# ============================================================================
show_coverage_matrix() {
    echo ""
    echo "Benchmark Coverage Matrix"
    echo "──────────────────────────────────────────────────────────────────────"
    
    # Discover all configs
    mapfile -t configs < <(discover_configs)
    
    if [[ ${#configs[@]} -eq 0 ]]; then
        echo "  No benchmark results found."
        echo ""
        echo "  Run benchmarks with:"
        echo "    make benchmark                      # Default: INT8, CPU, SingleStream"
        echo "    make benchmark DEVICE=GPU PRECISION=FP16"
        echo ""
        return
    fi
    
    # Print header
    printf "  %-12s %-8s %-14s │ %-4s %-4s %-6s %-6s │ %s\n" \
        "PRECISION" "DEVICE" "SCENARIO" "PERF" "ACC" "TEST01" "TEST04" "STATUS"
    echo "  ────────────────────────────────────┼─────────────────────────────┼────────"
    
    # Print each config
    for config in "${configs[@]}"; do
        # Skip empty configs
        [[ -z "${config}" ]] && continue
        
        # Parse config: precision-device-scenario
        IFS='-' read -r precision device scenario <<< "${config}"
        
        # Skip if parsing failed
        [[ -z "${precision}" || -z "${device}" || -z "${scenario}" ]] && continue
        
        # Get status
        status_str=$(check_config_status "${precision}" "${device}" "${scenario}")
        IFS='|' read -r perf acc test01 test04 <<< "${status_str}"
        
        # Determine overall status
        if [[ "${status_str}" == *"${CROSS}"* ]]; then
            overall="${YELLOW}incomplete${NC}"
        else
            overall="${GREEN}ready${NC}"
        fi
        
        printf "  %-12s %-8s %-14s │ %-4b %-4b %-6b %-6b │ %b\n" \
            "${precision^^}" "${device^^}" "${scenario}" \
            "${perf}" "${acc}" "${test01}" "${test04}" "${overall}"
    done
    
    echo ""
    echo "  Legend: ✓ = complete, ✗ = missing"
    echo "  A configuration is 'ready' when all 4 tests pass."
}

# ============================================================================
# Show submission status
# ============================================================================
show_submission_status() {
    echo ""
    echo "Submission Status"
    echo "──────────────────────────────────────────────────────────────────────"
    
    if [[ -d "${BASE_DIR}/generated-submission" ]]; then
        echo -e "  ${CHECK} generated-submission/ exists"
        # Count systems
        local system_count
        system_count=$(find "${BASE_DIR}/generated-submission" -name "*.json" -path "*/systems/*" 2>/dev/null | wc -l)
        echo "      Systems: ${system_count}"
    else
        echo -e "  ${CROSS} generated-submission/ (run: make configure-submission)"
    fi
    
    if [[ -d "${BASE_DIR}/submission-package/processed" ]]; then
        echo -e "  ${CHECK} submission-package/processed/ (validated)"
    else
        echo -e "  ${DASH} submission-package/processed/ (run: make finalize-submission)"
    fi
}

# ============================================================================
# Main
# ============================================================================
main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║              MLPerf Inference - ResNet-50 Status                     ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    
    show_environment_status
    show_coverage_matrix
    show_submission_status
    
    echo ""
}

main "$@"
