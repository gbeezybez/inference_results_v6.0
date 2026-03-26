#!/bin/bash
# Debug script to check mount points and symlinks

echo "=== Checking mount points ==="
echo "MLPERF_SCRATCH_PATH: ${MLPERF_SCRATCH_PATH}"
echo ""

echo "=== Checking /home/mlperf_inference_storage ==="
ls -la /home/mlperf_inference_storage/ || echo "Directory does not exist"
echo ""

echo "=== Checking /home/mlperf_inference_storage/models ==="
ls -la /home/mlperf_inference_storage/models/ || echo "Directory does not exist"
echo ""

echo "=== Checking /home/mlperf_inference_storage/models/gpt-oss ==="
ls -la /home/mlperf_inference_storage/models/gpt-oss/ || echo "Directory does not exist"
echo ""

echo "=== Checking /work/build ==="
ls -la /work/build/ || echo "Directory does not exist"
echo ""

echo "=== Checking /work/build/models symlink ==="
ls -la /work/build/models || echo "Symlink does not exist"
echo ""

echo "=== Checking if /work/build/models is a symlink ==="
if [ -L /work/build/models ]; then
    echo "Yes, it's a symlink pointing to: $(readlink /work/build/models)"
else
    echo "No, it's not a symlink"
fi
echo ""

echo "=== Checking /work/build/models/gpt-oss ==="
ls -la /work/build/models/gpt-oss || echo "Directory does not exist"
echo ""

echo "=== Checking /work/build/models/gpt-oss/gpt-oss-120b ==="
ls -la /work/build/models/gpt-oss/gpt-oss-120b || echo "Directory does not exist"
echo ""

echo "=== Environment variables ==="
env | grep -E "MLPERF|MODEL|BUILD" | sort

