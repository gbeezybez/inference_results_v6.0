#!/bin/bash

cd amd_quark-0.9/examples/torch/language_modeling/llm_ptq

ARCH=$(rocminfo | grep "Name:" | grep "gfx" | awk 'NR==1' | awk '{print $2}')
DATASET="/data/processed-openorca/open_orca_gpt4_tokenized_llama.calibration_1000.pkl"
MODEL="/model/llama2-70b-chat-hf/orig"
OUTPUT_DIR="/model/llama2-70b-chat-hf/fp8_quantized"

if [[ "$ARCH" == "gfx950" ]]; then
    OUTPUT_DIR="/model/llama2-70b-chat-hf/fp4_quantized"
    python3 quantize_quark.py --model_dir "${MODEL}" \
                          --output_dir "${OUTPUT_DIR}" \
                          --dataset "${DATASET}" \
                          --quant_scheme w_mxfp4_a_mxfp4 \
                          --group_size 32 \
                          --kv_cache_dtype fp8 \
                          --num_calib_data 1000 \
                          --multi_gpu \
                          --seq_len 1024 \
                          --exclude_layers "lm_head" \
                          --quant_algo gptq \
                          --model_export hf_format
else
    python3 quantize_quark.py --model_dir "${MODEL}" \
                          --output_dir "${OUTPUT_DIR}" \
                          --dataset "${DATASET}" \
                          --data_type float16 \
                          --multi_gpu \
                          --quant_scheme w_fp8_a_fp8 \
                          --kv_cache_dtype fp8 \
                          --num_calib_data 1000 \
                          --seq_len 1024 \
                          --model_export hf_format \
                          --custom_mode fp8 \
                          --exclude_layers "lm_head"
fi
