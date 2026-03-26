#!/bin/bash

cd amd_quark-0.9/examples/torch/language_modeling/llm_ptq

DATASET="/data/mixtral-8x7b/mlperf_mixtral8x7b_calibration_dataset_1k.pkl"
MODEL="/model/mixtral-8x7b/orig"
OUTPUT_DIR="/model/mixtral-8x7b/fp8_quantized"

python3 quantize_quark.py --model_dir "${MODEL}" \
                          --output_dir "${OUTPUT_DIR}" \
                          --dataset "${DATASET}" \
                          --data_type float16 \
                          --multi_gpu \
                          --quant_scheme w_fp8_a_fp8 \
                          --kv_cache_dtype fp8 \
                          --num_calib_data 1024 \
                          --seq_len 1024 \
                          --min_kv_scale 1.0 \
                          --model_export hf_format \
                          --custom_mode fp8 \
                          --quant_algo autosmoothquant \
                          --exclude_layers "lm_head" "*.gate"
