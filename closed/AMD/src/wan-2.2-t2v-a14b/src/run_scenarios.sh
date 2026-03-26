#!/usr/bin/env bash
#
# Run MLPerf wan-2.2-t2v-a14b scenarios (SingleStream, Offline).
# Intended to be run from: /app/mlperf/mlperf_inference/text_to_video/wan-2.2-t2v-a14b/
# (e.g. inside the Docker container). Set MLPERF_ROOT if running from elsewhere.
#

set -euo pipefail

# --- Configuration (override with env or pass args) ---
MLPERF_ROOT="${MLPERF_ROOT:-/app/mlperf}"
SCRIPT_DIR="${SCRIPT_DIR:-.}"
MODEL_PATH="${MODEL_PATH:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
DATASET="${DATASET:-./data/vbench_prompts.txt}"
CONFIG="${CONFIG:-./inference_config.yaml}"
FIXED_LATENT="${FIXED_LATENT:-./data/fixed_latent.pt}"
NPROC="${NPROC:-8}"
NUM_GPUS="${NUM_GPUS:-8}"
RESULTS_BASE="${RESULTS_BASE:-./outputs/results/wan-2.2-t2v-a14b}"
COMPLIANCE_BASE="${COMPLIANCE_BASE:-./outputs/compliance/wan-2.2-t2v-a14b}"
RUN_SUBDIR="${RUN_SUBDIR:-}"
SAGE_FRACTION="${SAGE_FRACTION:-0.9}"
# Compliance audit configs: from repo root (e.g. mlperf_inference), ../../compliance
COMPLIANCE_REPO="${COMPLIANCE_REPO:-${MLPERF_ROOT}/mlperf_inference}"
RUN_VBENCH="${RUN_VBENCH:-1}"
RUN_COMPLIANCE="${RUN_COMPLIANCE:-1}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [SCENARIO]

Run MLPerf wan-2.2-t2v-a14b scenarios.

SCENARIO:
  SingleStream   Run only SingleStream (accuracy, optional VBench, performance, compliance)
  Offline        Run only Offline (accuracy, optional VBench, performance, compliance)
  all            Run both SingleStream and Offline (default)

OPTIONS:
  --run-subdir PATH     Place PATH after ./outputs to distinguish runs (e.g. ./outputs/PATH/results/...)
  --sage-fraction FLOAT Fraction of steps using SAGE attention (default: 0.9)
  --skip-vbench        Do not run VBench after accuracy
  --skip-compliance  Do not run compliance tests (TEST01, TEST04)
  --dry-run          Print commands only, do not execute
  -h, --help         Show this help

ENVIRONMENT:
  MLPERF_ROOT      Root of mlperf repo (default: /app/mlperf)
  RUN_VBENCH        Set to 0 to skip VBench (same as --skip-vbench)
  RUN_COMPLIANCE    Set to 0 to skip compliance (same as --skip-compliance)
  DRY_RUN           Set to 1 to only print commands
  NPROC             torchrun processes (default: 8)
  NUM_GPUS          GPUs for VBench (default: 8)
  RUN_SUBDIR        Same as --run-subdir (default: unset)
  SAGE_FRACTION     Same as --sage-fraction (default: 0.9)
EOF
}

SCENARIO="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        SingleStream|Offline|all) SCENARIO="$1"; shift ;;
        --run-subdir)    RUN_SUBDIR="${2:?--run-subdir requires a path}"; shift 2 ;;
        --sage-fraction) SAGE_FRACTION="${2:?--sage-fraction requires a float}"; shift 2 ;;
        --skip-vbench)   RUN_VBENCH=0; shift ;;
        --skip-compliance) RUN_COMPLIANCE=0; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# When RUN_SUBDIR is set, place it after ./outputs to distinguish runs: ./outputs/<RUN_SUBDIR>/results/..., ./outputs/<RUN_SUBDIR>/videos/..., ./outputs/<RUN_SUBDIR>/compliance/...
if [[ -n "$RUN_SUBDIR" ]]; then
    RESULTS_BASE="./outputs/${RUN_SUBDIR}/results/wan-2.2-t2v-a14b"
    COMPLIANCE_BASE="./outputs/${RUN_SUBDIR}/compliance/wan-2.2-t2v-a14b"
    VIDEO_OUTPUT_BASE="./outputs/${RUN_SUBDIR}/videos/wan-2.2-t2v-a14b"
else
    VIDEO_OUTPUT_BASE="./outputs/videos/wan-2.2-t2v-a14b"
fi

run() {
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY-RUN] $*"
    else
        echo ">>> $*"
        "$@"
    fi
}

run_singlestream_accuracy() {
    run rm -f ./audit.config
    run torchrun --nproc-per-node="$NPROC" run_mlperf.py \
        --model-path "$MODEL_PATH" \
        --dataset "$DATASET" \
        --config "$CONFIG" \
        --fixed-latent "$FIXED_LATENT" \
        --sage-fraction "$SAGE_FRACTION" \
        --scenario SingleStream \
        --accuracy \
        --output-dir "${RESULTS_BASE}/SingleStream/accuracy" \
        --video_output_path "${VIDEO_OUTPUT_BASE}/SingleStream/accuracy" \
        --batch_size 1
}

run_singlestream_vbench() {
    local acc_dir="${RESULTS_BASE}/SingleStream/accuracy"
    local acc_videos="${VIDEO_OUTPUT_BASE}/SingleStream/accuracy"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY-RUN] with-vbench python run_evaluation.py ... > vbench.log; extract vbench_score into accuracy.txt"
    else
        echo ">>> with-vbench python run_evaluation.py (SingleStream VBench)"
        with-vbench python run_evaluation.py \
            --videos-path "${acc_videos}" \
            --output-path "${acc_dir}/vbench" \
            --num-gpus "$NUM_GPUS" \
            2>&1 | tee "${acc_dir}/vbench.log"
        # Extract "Overall Average : 0.7020" from vbench.log, write 'vbench_score': <value> to accuracy.txt
        local score
        score=$(grep "Overall Average" "${acc_dir}/vbench.log" | tail -1 | awk '{ print $NF*100 }')
        if [[ -n "$score" ]]; then
            echo " 'vbench_score': $score" > "${acc_dir}/accuracy.txt"
        fi
    fi
}

run_singlestream_performance() {
    run rm -f ./audit.config
    run torchrun --nproc-per-node="$NPROC" run_mlperf.py \
        --model-path "$MODEL_PATH" \
        --dataset "$DATASET" \
        --config "$CONFIG" \
        --fixed-latent "$FIXED_LATENT" \
        --sage-fraction "$SAGE_FRACTION" \
        --scenario SingleStream \
        --output-dir "${RESULTS_BASE}/SingleStream/performance/run_1" \
        --video_output_path "${VIDEO_OUTPUT_BASE}/SingleStream/performance/run_1" \
        --batch_size 1
}

run_singlestream_compliance_test01() {
    local audit_src="${COMPLIANCE_REPO}/compliance/TEST01/wan-2.2-t2v-a14b/audit.config"
    run cp "$audit_src" ./audit.config
    run torchrun --nproc-per-node="$NPROC" run_mlperf.py \
        --model-path "$MODEL_PATH" \
        --dataset "$DATASET" \
        --config "$CONFIG" \
        --fixed-latent "$FIXED_LATENT" \
        --sage-fraction "$SAGE_FRACTION" \
        --scenario SingleStream \
        --audit_conf ./audit.config \
        --output-dir "${RESULTS_BASE}/SingleStream/TEST01" \
        --video_output_path "${VIDEO_OUTPUT_BASE}/SingleStream/TEST01" \
        --batch_size 1
    run rm -f ./audit.config
    run python3 "${MLPERF_ROOT}/mlperf_inference/compliance/TEST01/run_verification.py" \
        --results_dir "${RESULTS_BASE}/SingleStream" \
        --compliance_dir "${RESULTS_BASE}/SingleStream/TEST01" \
        --output_dir "${COMPLIANCE_BASE}/SingleStream/"
}

run_singlestream_compliance_test04() {
    local audit_src="${COMPLIANCE_REPO}/compliance/TEST04/audit.config"
    run cp "$audit_src" ./audit.config
    run torchrun --nproc-per-node="$NPROC" run_mlperf.py \
        --model-path "$MODEL_PATH" \
        --dataset "$DATASET" \
        --config "$CONFIG" \
        --fixed-latent "$FIXED_LATENT" \
        --sage-fraction "$SAGE_FRACTION" \
        --scenario SingleStream \
        --audit_conf ./audit.config \
        --output-dir "${RESULTS_BASE}/SingleStream/TEST04" \
        --video_output_path "${VIDEO_OUTPUT_BASE}/SingleStream/TEST04" \
        --batch_size 1
    run rm -f ./audit.config
    run python3 "${MLPERF_ROOT}/mlperf_inference/compliance/TEST04/run_verification.py" \
        --results_dir "${RESULTS_BASE}/SingleStream" \
        --compliance_dir "${RESULTS_BASE}/SingleStream/TEST04" \
        --output_dir "${COMPLIANCE_BASE}/SingleStream/"
}

run_offline_accuracy() {
    run rm -f ./audit.config
    run torchrun --nproc-per-node="$NPROC" run_mlperf.py \
        --model-path "$MODEL_PATH" \
        --dataset "$DATASET" \
        --config "$CONFIG" \
        --fixed-latent "$FIXED_LATENT" \
        --sage-fraction "$SAGE_FRACTION" \
        --scenario Offline \
        --accuracy \
        --output-dir "${RESULTS_BASE}/Offline/accuracy" \
        --video_output_path "${VIDEO_OUTPUT_BASE}/Offline/accuracy" \
        --batch_size 16
}

run_offline_vbench() {
    local acc_dir="${RESULTS_BASE}/Offline/accuracy"
    local acc_videos="${VIDEO_OUTPUT_BASE}/Offline/accuracy"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY-RUN] with-vbench python run_evaluation.py ... > vbench.log; extract vbench_score into accuracy.txt"
    else
        echo ">>> with-vbench python run_evaluation.py (Offline VBench)"
        with-vbench python run_evaluation.py \
            --videos-path "${acc_videos}" \
            --output-path "${acc_dir}/vbench" \
            --num-gpus "$NUM_GPUS" \
            2>&1 | tee "${acc_dir}/vbench.log"
        # Extract "Overall Average : 0.7020" from vbench.log, write 'vbench_score': <value> to accuracy.txt
        local score
        score=$(grep "Overall Average" "${acc_dir}/vbench.log" | tail -1 | awk '{ print $NF*100 }')
        if [[ -n "$score" ]]; then
            echo " 'vbench_score': $score" > "${acc_dir}/accuracy.txt"
        fi
    fi
}

run_offline_performance() {
    run rm -f ./audit.config
    run torchrun --nproc-per-node="$NPROC" run_mlperf.py \
        --model-path "$MODEL_PATH" \
        --dataset "$DATASET" \
        --config "$CONFIG" \
        --fixed-latent "$FIXED_LATENT" \
        --sage-fraction "$SAGE_FRACTION" \
        --scenario Offline \
        --output-dir "${RESULTS_BASE}/Offline/performance/run_1" \
        --video_output_path "${VIDEO_OUTPUT_BASE}/Offline/performance/run_1" \
        --batch_size 16
}

run_offline_compliance_test01() {
    local audit_src="${COMPLIANCE_REPO}/compliance/TEST01/wan-2.2-t2v-a14b/audit.config"
    run cp "$audit_src" ./audit.config
    run torchrun --nproc-per-node="$NPROC" run_mlperf.py \
        --model-path "$MODEL_PATH" \
        --dataset "$DATASET" \
        --config "$CONFIG" \
        --fixed-latent "$FIXED_LATENT" \
        --sage-fraction "$SAGE_FRACTION" \
        --scenario Offline \
        --audit_conf ./audit.config \
        --output-dir "${RESULTS_BASE}/Offline/TEST01" \
        --video_output_path "${VIDEO_OUTPUT_BASE}/Offline/TEST01" \
        --batch_size 16
    run rm -f ./audit.config
    run python3 "${MLPERF_ROOT}/mlperf_inference/compliance/TEST01/run_verification.py" \
        --results_dir "${RESULTS_BASE}/Offline" \
        --compliance_dir "${RESULTS_BASE}/Offline/TEST01" \
        --output_dir "${COMPLIANCE_BASE}/Offline/"
}

run_offline_compliance_test04() {
    local audit_src="${COMPLIANCE_REPO}/compliance/TEST04/audit.config"
    run cp "$audit_src" ./audit.config
    run torchrun --nproc-per-node="$NPROC" run_mlperf.py \
        --model-path "$MODEL_PATH" \
        --dataset "$DATASET" \
        --config "$CONFIG" \
        --fixed-latent "$FIXED_LATENT" \
        --sage-fraction "$SAGE_FRACTION" \
        --scenario Offline \
        --audit_conf ./audit.config \
        --output-dir "${RESULTS_BASE}/Offline/TEST04" \
        --video_output_path "${VIDEO_OUTPUT_BASE}/Offline/TEST04" \
        --batch_size 16
    run rm -f ./audit.config
    run python3 "${MLPERF_ROOT}/mlperf_inference/compliance/TEST04/run_verification.py" \
        --results_dir "${RESULTS_BASE}/Offline" \
        --compliance_dir "${RESULTS_BASE}/Offline/TEST04" \
        --output_dir "${COMPLIANCE_BASE}/Offline/"
}

run_singlestream() {
    echo "========== SingleStream: Accuracy =========="
    run_singlestream_accuracy
    if [[ "$RUN_VBENCH" == "1" ]]; then
        echo "========== SingleStream: VBench =========="
        run_singlestream_vbench
    fi
    echo "========== SingleStream: Performance =========="
    run_singlestream_performance
    if [[ "$RUN_COMPLIANCE" == "1" ]]; then
        # echo "========== SingleStream: Compliance TEST01 =========="
        # run_singlestream_compliance_test01
        echo "========== SingleStream: Compliance TEST04 =========="
        run_singlestream_compliance_test04
    fi
}

run_offline() {
    echo "========== Offline: Accuracy =========="
    run_offline_accuracy
    if [[ "$RUN_VBENCH" == "1" ]]; then
        echo "========== Offline: VBench =========="
        run_offline_vbench
    fi
    echo "========== Offline: Performance =========="
    run_offline_performance
    if [[ "$RUN_COMPLIANCE" == "1" ]]; then
        # echo "========== Offline: Compliance TEST01 =========="
        # run_offline_compliance_test01
        echo "========== Offline: Compliance TEST04 =========="
        run_offline_compliance_test04
    fi
}

# --- Main ---
echo "Scenario: $SCENARIO | VBench: $RUN_VBENCH | Compliance: $RUN_COMPLIANCE | Dry-run: $DRY_RUN"
[[ -n "$RUN_SUBDIR" ]] && echo "Results base: $RESULTS_BASE | Compliance base: $COMPLIANCE_BASE"

case "$SCENARIO" in
    SingleStream) run_singlestream ;;
    Offline)      run_offline ;;
    all)          run_singlestream; run_offline ;;
    *)            echo "Invalid scenario: $SCENARIO"; exit 1 ;;
esac

echo "Done."
