#!/bin/bash

# set -x
set -e

# Set proxy for network operations
export http_proxy=${http_proxy:-"http://173.36.224.108:80/"}
export https_proxy=${https_proxy:-"http://173.36.224.108:80/"}
export HTTP_PROXY=${HTTP_PROXY:-"http://173.36.224.108:80/"}
export HTTPS_PROXY=${HTTPS_PROXY:-"http://173.36.224.108:80/"}

echo ">>> Starting accuracy check..."

VENV_BIN_PATH=/lab-mlperf-inference/code/llama2_accuracy_venv/bin
DATASET_FILE_PATH=/data/processed-openorca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl
MODEL_PATH=/model/llama2-70b-chat-hf/fp8_quantized
EVAL_SCRIPT_PATH=/lab-mlperf-inference/mlperf_inference/language/llama2-70b/evaluate-accuracy.py
MODEL_OVERRIDE_PATH=${MODEL_OVERRIDE_PATH:-""}

# Check if venv exists, otherwise use system python
USE_VENV=0
if [ -f "${VENV_BIN_PATH}/python3" ]; then
    PYTHON3_PATH=${VENV_BIN_PATH}/python3
    ACTIVATE_PATH=${VENV_BIN_PATH}/activate
    USE_VENV=1
else
    PYTHON3_PATH=$(which python3)
    echo "Using system python3: ${PYTHON3_PATH}"
fi

echo ">>> Detecting GPU architecture..."
ARCH=${ARCH:-$(rocminfo | grep "Name:" | grep "gfx" | awk 'NR==1' | awk '{print $2}')}
echo ">>> Detected ARCH: ${ARCH}"

if [[ "$ARCH" == "gfx950" ]]; then
    MODEL_PATH=/model/llama2-70b-chat-hf/fp4_quantized_gptq
fi

if [ -n "$MODEL_OVERRIDE_PATH" ]; then
    echo ">>> Overriding model path with ${MODEL_OVERRIDE_PATH}"
    MODEL_PATH=${MODEL_OVERRIDE_PATH}
fi

echo ">>> Using MODEL_PATH: ${MODEL_PATH}"

echo ">>> Checking dataset file..."
if [ ! -f ${DATASET_FILE_PATH} ]; then
    echo "dataset not found, check the README.md how to download it"
    exit 1
fi
echo ">>> Dataset found: ${DATASET_FILE_PATH}"

echo ">>> Checking model directory..."
if [ ! -d ${MODEL_PATH} ]; then
    echo "model not found, check the README.md how to download it"
    exit 1
elif [ -z "$(ls -A ${MODEL_PATH})" ]; then
    echo "model dir is empty, check the README.md how to download it"
    exit 1
fi
echo ">>> Model found"

echo ">>> Checking eval script..."
if [ ! -f ${EVAL_SCRIPT_PATH} ]; then
    echo "tools/evaluate-accuracy.py not found"
    exit 1
fi
echo ">>> Eval script found"

# Activate venv if using it
if [ ${USE_VENV} -eq 1 ]; then
    source ${ACTIVATE_PATH}
fi

ACCURACY_JSON=${1}

if [ -z ${ACCURACY_JSON} ]; then
    echo "incorrect accuracy path, set it with ${0} <path>"
    [ ${USE_VENV} -eq 1 ] && deactivate
    exit 1
fi

if [ ! -f ${ACCURACY_JSON} ]; then
    echo "incorrect accuracy path, set it with ${0} <path>"
    [ ${USE_VENV} -eq 1 ] && deactivate
    exit 1
fi

OUTPUT_DIR=$(dirname ${ACCURACY_JSON})
RESULT_TXT=${OUTPUT_DIR}/accuracy.txt

echo ">>> Pre-downloading nltk/evaluate datasets..."
"${PYTHON3_PATH}" -c 'import evaluate; evaluate.load("rouge"); import nltk; nltk.download("punkt"); nltk.download("punkt_tab")'

echo ">>> Running accuracy evaluation (this may take a while)..."
echo ">>> Python: ${PYTHON3_PATH}"
echo ">>> Eval script: ${EVAL_SCRIPT_PATH}"
"${PYTHON3_PATH}" -u "${EVAL_SCRIPT_PATH}" --checkpoint-path "${MODEL_PATH}" \
                                           --mlperf-accuracy-file "${ACCURACY_JSON}" \
                                           --dataset-file "${DATASET_FILE_PATH}" \
                                           --dtype int32 2>&1 | tee "${RESULT_TXT}"

# Deactivate venv if using it
[ ${USE_VENV} -eq 1 ] && deactivate

echo ">>> Done! Check $RESULT_TXT for the accuracy scores"
