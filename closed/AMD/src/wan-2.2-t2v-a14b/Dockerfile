FROM rocm/pytorch-xdit:v25.13 AS build_torch_stack
ARG THEROCK_COMMIT="1728a81e5cd97ff366dc04e2517cf53774d53501"
ARG TRITON_COMMIT="ba5c1517e6f5906761cf5783036efb587026208d"
ARG PYTORCH_COMMIT="d9556b05edc2762cb07f7715dfeedb3cf26b6cdd"
ARG PYTORCH_VISION_COMMIT="82df5f599578b383987510836bb05ea97dcc9669"

WORKDIR /app
RUN git clone https://github.com/ROCm/TheRock.git && \
    cd TheRock && \
    git reset --hard ${THEROCK_COMMIT}
RUN git config --global user.name "TheRock Builder" && \
    git config --global user.email "builder@rocm.ai"
# update torch stack
WORKDIR /app/TheRock/external-builds/pytorch
RUN pip uninstall -y torch torchaudio torchvision torchcodec triton torchao && \
    pip install -r /app/TheRock/requirements.txt && \
    python pytorch_torch_repo.py checkout --gitrepo-origin https://github.com/ROCm/pytorch.git --repo-hashtag ${PYTORCH_COMMIT} && \
    python pytorch_triton_repo.py checkout --gitrepo-origin https://github.com/ROCm/triton.git --repo-hashtag ${TRITON_COMMIT} && \
    python pytorch_vision_repo.py checkout --gitrepo-origin https://github.com/pytorch/vision.git --repo-hashtag ${PYTORCH_VISION_COMMIT}
# install triton
RUN cd triton && \
    pip install -r python/requirements.txt && \
    pip wheel . --wheel-dir dist/ && \
    pip install dist/*.whl && \
    cp dist/*.whl /install
# install pytorch
RUN cd pytorch && \
    pip install -r requirements.txt && \
    USE_KINETO=ON USE_GLOO=ON USE_ROCM=ON USE_CUDA=OFF USE_MPI=OFF \
    USE_MEM_EFF_ATTENTION=ON USE_FLASH_ATTENTION=ON USE_ROCM_CK_SDPA=ON \
    PYTORCH_ROCM_ARCH="gfx942;gfx950" \
    CXXFLAGS="-I${ROCM_HOME}/include/roctracer -I${ROCM_HOME}/lib/rocm_sysdeps/include" \
    LDFLAGS="-L${ROCM_HOME}/lib/rocm_sysdeps/lib" \
    PKG_CONFIG_PATH="${ROCM_HOME}/lib/rocm_sysdeps/lib/pkgconfig" \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    cp dist/*.whl /install
# install vision
RUN cd pytorch_vision && \
    FORCE_CUDA=1 TORCHVISION_USE_NVJPEG=0 TORCHVISION_USE_VIDEO_CODEC=0 \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    cp dist/*.whl /install

FROM rocm/pytorch-xdit:v25.13 AS final
ARG MLPERF_INFERENCE_COMMIT="a67138434d5b677c5e1cb11ce2320e365dc30c8c"
ARG XDIT_COMMIT="52973faf89f71a423bea063e6997a9bd92a22a2a"
RUN pip uninstall -y torch torchaudio torchvision torchcodec triton torchao && \
    rm -rf /install/*
COPY --from=build_torch_stack /install/torch-2.10.0*.whl  /install/
COPY --from=build_torch_stack /install/triton-3.6.0*.whl  /install/
COPY --from=build_torch_stack /install/torchvision-0.25.0*.whl  /install/
RUN python -m pip install \
    --no-cache-dir \
    /install/*.whl

# reinstall ao
RUN cd /app/external/ao && \
    pip install -U "setuptools==78.1.1" && \
    pip install -e . --no-build-isolation

# update xDiT
RUN cd /app/external/xDiT && \
git stash && \
git pull && \
git reset --hard ${XDIT_COMMIT}

WORKDIR /app/mlperf
# For submissions, please use the master branch and any commit since the 6.0 seed release although it is best to use the latest commit in the master branch.
RUN git clone https://github.com/mlcommons/inference.git mlperf_inference && \
cd mlperf_inference && \
git reset --hard ${MLPERF_INFERENCE_COMMIT} && \
git submodule sync && \
git submodule update --init --recursive && \
cd loadgen && \
pip install .

# VBench deps - systemwide
RUN apt-get update && apt-get install -y --no-install-recommends \
unzip \
&& rm -rf /var/lib/apt/lists/*
# Create VBench venv that does not clash with the main environment, allowing the two to coexist.
RUN python3 -m venv /opt/vbench_env && \
/opt/vbench_env/bin/pip install --no-cache-dir \
--index-url https://rocm.nightlies.amd.com/v2-staging/gfx950-dcgpu/ "rocm[libraries,devel]==7.12.0a20260128" \
"torch==2.9.1+rocm7.12.0a20260128" \
"torchaudio==2.9.0+rocm7.12.0a20260128" \
"torchvision==0.24.0+rocm7.12.0a20260128" && \
/opt/vbench_env/bin/pip install --no-cache-dir \
"transformers==4.46.0" \
decord \
easydict \
"fairscale>=0.4.4" \
imageio \
omegaconf \
openai-clip \
opencv-python \
scipy \
"timm>=0.9,<=1.0.12"

# Add helper script to activate vbench env - any VBench evaluations should use "with-vbench" to activate the vbench environment.
RUN echo '#!/bin/bash\nsource /opt/vbench_env/bin/activate\nexec "$@"' > /usr/local/bin/with-vbench && \
chmod +x /usr/local/bin/with-vbench

# update aiter
ARG AITER_COMMIT="03c0c863d5b0cbbde0b60f8c35f1a5e5afb3bc25"
RUN cd /app/external/aiter && \
    git pull && \
    git reset --hard ${AITER_COMMIT} && \
    git submodule sync && \
    git submodule update --init --recursive && \
    pip uninstall -y aiter && \
    pip install -e .

# update xDiT
ARG XDIT_COMMIT="6ffc551265033abe6d45a7265d2f39fd5320b46c"
RUN cd /app/external/xDiT && \
    git pull && \
    git reset --hard ${XDIT_COMMIT}

COPY ./src/* /app/mlperf/mlperf_inference/text_to_video/wan-2.2-t2v-a14b/


# Setup MIOpen user dbs
COPY miopen/* $MIOPEN_USER_DB_PATH/
# Disable direct solution methods slow for large convs
ENV MIOPEN_DEBUG_CONV_DIRECT=0
# Enable incremental tuning for unknown shapes
ENV MIOPEN_FIND_MODE=3
ENV MIOPEN_FIND_ENFORCE=3
# OpenMP threads
ENV OMP_NUM_THREADS=16

# hotfixes for bugs in mlperf
COPY ./patches/ /patches/
RUN git -C /app/mlperf/mlperf_inference \
    apply \
    /patches/test01_print_bug.patch \
    /patches/test04_min_samples_bug.patch