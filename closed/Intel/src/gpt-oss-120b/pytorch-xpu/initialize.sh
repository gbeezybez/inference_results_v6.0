#!/bin/bash

TRITON_DIR=/workspace/third_party/triton-internal
if [ -d "${TRITON_DIR}" ]; then
    rm -rf "${TRITON_DIR}"
fi

export MAX_JOBS=32
cd /workspace/initialize
bash prepare_whl.sh
