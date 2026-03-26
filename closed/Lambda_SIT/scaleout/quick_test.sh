#!/bin/bash
# Quick test to see what's mounted

srun \
    --export=ALL,MLPERF_SCRATCH_PATH=/home/mlperf_inference_storage \
    --container-image=/sharedfs/mlperf_inference_v60/nv-mlpinf-partner/closed/NVIDIA/build/sqsh_images/image_emailed_to_us.sqsh \
    --container-mounts="/sharedfs/chuan/nv-mlpinf-partner/closed/NVIDIA:/work,/sharedfs/mlcommons/scratch:/home/mlperf_inference_storage" \
    --container-workdir=/work \
    --container-remap-root \
    --nodes=1 \
    --ntasks=1 \
    bash -c 'echo "=== Mounted volumes ==="; df -h | grep -E "sharedfs|Filesystem"; echo ""; echo "=== /home/mlperf_inference_storage/models/gpt-oss ==="; ls -la /home/mlperf_inference_storage/models/gpt-oss 2>&1; echo ""; echo "=== Running make link_dirs ==="; cd /work && make link_dirs 2>&1; echo ""; echo "=== /work/build/models ==="; ls -la /work/build/models 2>&1; echo ""; echo "=== /work/build/models/gpt-oss ==="; ls -la /work/build/models/gpt-oss 2>&1'

