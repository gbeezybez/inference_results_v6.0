#!/bin/bash

HF_TOKEN=${1:-dummy}
QUANT_FORMAT=${2:-FP4}
QUANT_ALGO=${3:-gptq}
SKIP_DOWNLOAD=${4:-false}

MODEL=meta-llama/Llama-2-70b-chat-hf
MODEL_PATH="/model/llama2-70b-chat-hf/orig"
if [[ "${SKIP_DOWNLOAD}" != "true" ]]; then
    hf download $MODEL --token $HF_TOKEN --local-dir $MODEL_PATH
fi

DATASET="/data/processed-openorca/open_orca_gpt4_tokenized_llama.calibration_1000.pkl"

pushd "amd_quark-0.10/examples/torch/language_modeling/llm_ptq" > /dev/null
if [[ "$QUANT_FORMAT" == "FP8" ]]; then

    OUTPUT_DIR="/model/llama2-70b-chat-hf/fp8_quantized"
    python3 quantize_quark.py --model_dir "${MODEL_PATH}" \
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

elif [[ "$QUANT_FORMAT" == "FP4" ]]; then

    OUTPUT_DIR="/model/llama2-70b-chat-hf/fp4_quantized"
    OUTPUT_ALGO_DIR="/model/llama2-70b-chat-hf/fp4_quantized_${QUANT_ALGO}"
    python3 quantize_quark.py --model_dir "${MODEL_PATH}" \
                          --output_dir "${OUTPUT_ALGO_DIR}" \
                          --dataset "${DATASET}" \
                          --quant_scheme w_mxfp4_a_mxfp4 \
                          --data_type float16 \
                          --kv_cache_dtype fp8 \
                          --num_calib_data 1000 \
                          --multi_gpu \
                          --seq_len 1024 \
                          --exclude_layers "lm_head" \
                          --quant_algo "${QUANT_ALGO}" \
                          --model_export hf_format
    if [ ! -e $OUTPUT_DIR ]; then
        ln -s $OUTPUT_ALGO_DIR $OUTPUT_DIR
    fi
fi
popd > /dev/null
