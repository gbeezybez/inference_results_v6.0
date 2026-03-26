#!/bin/bash

if [ -z "${MAX_JOBS:-}" ]; then
    export MAX_JOBS=$(awk -v threads="$(nproc)" '/^Mem:/{mem=int($7/2); print(threads < mem ? threads : mem)}' <(free -g))
fi


# Install compute runtime for vllm-xpu-kernels build
rm -rf neo && mkdir -p neo && cd neo
wget -nv https://github.com/intel/intel-graphics-compiler/releases/download/v2.20.5/intel-igc-core-2_2.20.5+19972_amd64.deb
wget -nv https://github.com/intel/intel-graphics-compiler/releases/download/v2.20.5/intel-igc-opencl-2_2.20.5+19972_amd64.deb
wget -nv https://github.com/intel/compute-runtime/releases/download/25.40.35563.10/intel-ocloc-dbgsym_25.40.35563.10-0_amd64.ddeb
wget -nv https://github.com/intel/compute-runtime/releases/download/25.40.35563.10/intel-ocloc_25.40.35563.10-0_amd64.deb
wget -nv https://github.com/intel/compute-runtime/releases/download/25.40.35563.10/intel-opencl-icd-dbgsym_25.40.35563.10-0_amd64.ddeb
wget -nv https://github.com/intel/compute-runtime/releases/download/25.40.35563.10/intel-opencl-icd_25.40.35563.10-0_amd64.deb
wget -nv https://github.com/intel/compute-runtime/releases/download/25.40.35563.10/libigdgmm12_22.8.2_amd64.deb
wget -nv https://github.com/intel/compute-runtime/releases/download/25.40.35563.10/libze-intel-gpu1-dbgsym_25.40.35563.10-0_amd64.ddeb
wget -nv https://github.com/intel/compute-runtime/releases/download/25.40.35563.10/libze-intel-gpu1_25.40.35563.10-0_amd64.deb
dpkg -i *.deb
cd .. && rm -rf neo

# Build and install vllm-xpu-kernels
rm -rf vllm-xpu-kernels-local
git clone https://github.com/vllm-project/vllm-xpu-kernels.git vllm-xpu-kernels-local
cd vllm-xpu-kernels-local && git checkout c771759
# Add g31
git apply /workspace/add_g31.patch
# End of add g31
# oneDNN gemm strategy
git submodule sync && git submodule update --init --recursive
cd third_party/oneDNN && git checkout 86b5567 && cd ../../
# end of oneDNN gemm strategy
pip wheel --extra-index-url=https://download.pytorch.org/whl/xpu  . -v
cp *.whl /workspace/dist/ && pip install vllm_xpu_kernels-*.whl
cd .. && rm -rf vllm-xpu-kernels-local