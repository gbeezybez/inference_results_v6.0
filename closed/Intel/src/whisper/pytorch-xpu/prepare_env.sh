#!/bin/bash

echo "Preparing wheels for XPU MLPerf workflow container"

mkdir -p /workspace/dist

# Clone internal repos
mkdir -p third_party
pushd third_party
# rm -rf ipex-gpu-internal vllm-internal

# ipex
git clone https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-gpu -b dev/vllm/2.9.10 ipex-gpu
# vllm
git clone https://github.com/intel-innersource/applications.ai.gpu.vllm-xpu.git -b mlperf/dev/6.0 vllm-xpu
popd

# vllm
pip uninstall -y vllm
pushd /workspace/third_party/vllm-xpu
pip install -r requirements/xpu.txt 
pip uninstall triton pytorch-triton-xpu -y
pip install pytorch-triton-xpu --extra-index-url=https://download.pytorch.org/whl/xpu
VLLM_TARGET_DEVICE=xpu pip install -e . --no-build-isolation
pip uninstall triton pytorch-triton-xpu -y
pip install pytorch-triton-xpu --extra-index-url=https://download.pytorch.org/whl/xpu
popd

# ipex internal
export BUILD_WITH_CPU=OFF
export BUILD_SEPARATE_OPS=ON
export USE_AOT_DEVLIST='bmg'
export TORCH_XPU_ARCH_LIST='bmg'
export MAX_JOBS=$(awk -v threads="$(nproc)" '/^Mem:/{mem=int($7/2); print(threads < mem ? threads : mem)}' <(free -g))
export USE_XETLA=ON

pushd /workspace/third_party/ipex-gpu
# # Inside ipex-gpu source directory
pip uninstall -y intel-extension-for-pytorch
pip install -r requirements.txt
git submodule sync
git submodule update --init --recursive
python setup.py bdist_wheel
cp dist/*.whl /workspace/dist/
pip install dist/*.whl
popd

pushd third_party
git clone https://github.com/intel/auto-round.git
cd autoround
# git checkout mlperf-awq
python setup.py install
popd

# Remove internal repos. Already built whls
pushd third_party
# rm -rf ipex-gpu-internal vllm-internal
popd

echo "Wheels prepared in dist/ directory"
