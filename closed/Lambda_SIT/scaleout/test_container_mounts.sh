#!/bin/bash
# Test script to verify container mounts and run link_dirs

set -x

echo "=== Testing container mounts and symlinks ==="
echo "Current directory: $(pwd)"
echo "SLURM_JOBID: ${SLURM_JOBID}"
echo ""

# Get the script directory and host volume
script_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
host_vol="$(readlink -f "$script_dir/..")"
container_vol="/work"
mlperf_scratch_path="/sharedfs/mlcommons/scratch"
container_image="/sharedfs/mlperf_inference_v60/nv-mlpinf-partner/closed/NVIDIA/build/sqsh_images/image_emailed_to_us.sqsh"

echo "host_vol: $host_vol"
echo "container_vol: $container_vol"
echo "mlperf_scratch_path: $mlperf_scratch_path"
echo ""

# Run debug inside container
srun \
    --export=ALL,MLPERF_SCRATCH_PATH=/home/mlperf_inference_storage,HF_HOME=/work/build/.cache/huggingface,FLASHINFER_WORKSPACE_BASE=/work/build/.cache,MPLCONFIGDIR=/work/build/.cache/matplotlib \
    --container-image="$container_image" \
    --container-mounts="${host_vol}:${container_vol},${mlperf_scratch_path}:/home/mlperf_inference_storage" \
    --container-workdir="$container_vol" \
    --container-remap-root \
    --nodes=1 \
    --ntasks=1 \
    bash -c '
        echo "=== Inside container ==="
        echo "MLPERF_SCRATCH_PATH: $MLPERF_SCRATCH_PATH"
        echo ""
        
        echo "=== Checking /home/mlperf_inference_storage ==="
        ls -la /home/mlperf_inference_storage/ || echo "ERROR: Directory does not exist"
        echo ""
        
        echo "=== Checking /home/mlperf_inference_storage/models ==="
        ls -la /home/mlperf_inference_storage/models/ || echo "ERROR: Directory does not exist"
        echo ""
        
        echo "=== Checking /home/mlperf_inference_storage/models/gpt-oss ==="
        ls -la /home/mlperf_inference_storage/models/gpt-oss/ || echo "ERROR: Directory does not exist"
        echo ""
        
        echo "=== Checking /home/mlperf_inference_storage/models/gpt-oss/gpt-oss-120b ==="
        ls -la /home/mlperf_inference_storage/models/gpt-oss/gpt-oss-120b/ | head -20 || echo "ERROR: Directory does not exist"
        echo ""
        
        echo "=== Checking /work/build ==="
        ls -la /work/build/ || echo "ERROR: Directory does not exist"
        echo ""
        
        echo "=== Checking /work/build/models ==="
        if [ -L /work/build/models ]; then
            echo "Symlink exists, points to: $(readlink /work/build/models)"
            ls -la /work/build/models || echo "ERROR: Symlink is broken"
        elif [ -d /work/build/models ]; then
            echo "Directory exists (not a symlink)"
            ls -la /work/build/models
        else
            echo "ERROR: /work/build/models does not exist"
        fi
        echo ""
        
        echo "=== Running make link_dirs ==="
        cd /work
        make link_dirs
        echo ""
        
        echo "=== After link_dirs: Checking /work/build/models ==="
        if [ -L /work/build/models ]; then
            echo "Symlink exists, points to: $(readlink /work/build/models)"
            ls -la /work/build/models || echo "ERROR: Symlink is broken"
        else
            echo "ERROR: /work/build/models is not a symlink"
        fi
        echo ""
        
        echo "=== Checking /work/build/models/gpt-oss/gpt-oss-120b ==="
        ls -la /work/build/models/gpt-oss/gpt-oss-120b/ | head -20 || echo "ERROR: Directory does not exist"
    '

