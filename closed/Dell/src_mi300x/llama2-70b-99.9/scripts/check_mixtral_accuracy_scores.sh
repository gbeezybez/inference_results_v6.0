#!/bin/bash

# set -x
set -e

PYTHON3_BIN_PATH=/lab-mlperf-inference/code/moe_accuracy_venv/bin
PYTHON3_PATH=${PYTHON3_BIN_PATH}/python3
ACTIVATE_PATH=${PYTHON3_BIN_PATH}/activate
DATASET_FILE_PATH=/data/mixtral-8x7b/mlperf_mixtral8x7b_dataset_15k.pkl
MODEL_PATH=/model/mixtral-8x7b/fp8_quantized/
EVAL_SCRIPT_PATH=/app/mlperf_inference/language/mixtral-8x7b/evaluate-accuracy.py

if [ ! -f ${PYTHON3_PATH} ]; then
    echo "venv not found, run bash ./scripts/setup_mixtral_accuracy_env.sh"
    exit 1
fi

if [ ! -f ${DATASET_FILE_PATH} ]; then
    echo "dataset not found, check the README.md how to download it"
    exit 1
fi

if [ ! -f ${EVAL_SCRIPT_PATH} ]; then
    echo "tools/evaluate-accuracy.py not found"
    exit 1
fi

source ${ACTIVATE_PATH}

if [ ${PYTHON3_PATH} != `which python3` ]; then
    echo "incorrect python3 is used"
    exit 1
fi

ACCURACY_JSON=${1}

if [ -z ${ACCURACY_JSON} ]; then
    echo "incorrect accuracy path, set it with ${0} <path>"
    deactivate
    exit 1
fi

if [ ! -f ${ACCURACY_JSON} ]; then
    echo "incorrect accuracy path, set it with ${0} <path>"
    deactivate
    exit 1
fi

# Pre-download datasets to avoid multithreading issues
python -c 'import evaluate; evaluate.load("rouge"); import nltk; nltk.download("punkt"); nltk.download("punkt_tab")'

OUTPUT_DIR=$(dirname ${ACCURACY_JSON})
RESULT_TXT=${OUTPUT_DIR}/accuracy.txt

python -u ${EVAL_SCRIPT_PATH} --checkpoint-path ${MODEL_PATH} \
                              --mlperf-accuracy-file ${ACCURACY_JSON} \
                              --dataset-file ${DATASET_FILE_PATH} \
                              --dtype int32 \
                              --n_workers 8 > ${RESULT_TXT}
mv evaluated_test.json ${OUTPUT_DIR}

deactivate

echo "Check $RESULT_TXT for the accuracy scores"
