#!/bin/bash

QUANT_OUTPUT_DIR=${QUANT_OUTPUT_DIR:-"/model"}
mkdir -p ${QUANT_OUTPUT_DIR}
LOG_FILE=${LOG_FILE:-"${QUANT_OUTPUT_DIR}/quantization.log"}
MODEL_PATH=${MODEL_PATH:-/model/Llama-3.1-8B-Instruct}
CALIBRATION_DATASET_PATH=${CALIBRATION_DATASET_PATH:-"/data/cnn_dailymail_calibration.json"}

node0_cores=$(lscpu | grep "NUMA node0 CPU(s):" | awk '{print $4}')
taskset -c $node0_cores \
python3 run_quantization.py --model_name ${MODEL_PATH} \
    --nsamples 512 \
    --iters 128 \
    --bits 4 \
    --group_size 128 \
    --dataset-path ${CALIBRATION_DATASET_PATH} \
    --output_dir ${QUANT_OUTPUT_DIR} \
    2>&1 | tee -a ${LOG_FILE}
