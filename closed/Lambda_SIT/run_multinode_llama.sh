#!/bin/bash
# Multi-node Llama3.1-405B run script using SLURM scaleout

set -x

cd /sharedfs/mlperf_inference_v60/nv-mlpinf-partner/closed/NVIDIA

# Environment setup
export MLPERF_SCRATCH_PATH=/sharedfs/mlcommons/scratch/
export SYSTEM_NAME=GB300-NVL72_GB300-288GB_aarch64x72

# Run the scaleout script
./scaleout/run_scaleout.sh \
    --stage all \
    --atomic-system GB300-NVL72_GB300-288GB_aarch64x2 \
    --harness-system GB300-NVL72_GB300-288GB_aarch64x72 \
    --gpus-per-node 4 \
    --dp-multiplicity 36 \
    --run-args "--benchmarks=llama3.1-405b --scenarios=Offline" \
    --container-image build/sqsh_images/mlperf-inference-aarch64-release.sqsh \
    --mlperf-scratch-path /sharedfs/mlcommons/scratch/ \
    --nodelist $SLURM_JOB_NODELIST

