#!/bin/bash
mkdir -p /workspace/dist

# triton
pip uninstall pytorch-triton-xpu triton -y
mkdir -p /workspace/third_party
cd /workspace/third_party
git clone https://github.com/intel/intel-xpu-backend-for-triton.git triton-internal
cd triton-internal && git checkout bd88137b
# Apply patch
cp /workspace/initialize/enable_small_dot.patch . && git apply enable_small_dot.patch
cd python
python setup.py install
