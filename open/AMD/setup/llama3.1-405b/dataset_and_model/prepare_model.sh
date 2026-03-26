#!/bin/bash

HF_TOKEN=${1:-dummy}
QUANT_FORMAT=${2:-FP4}
QUANT_ALGO=${3:-autosmoothquant}
SKIP_DOWNLOAD=${4:-false}
DOWNLOAD_PREQUANTIZED=${5:-false}

if [[ "${DOWNLOAD_PREQUANTIZED}" == "true" ]]; then
    if [[ "$QUANT_FORMAT" == "FP8" ]]; then
        MODEL=amd/TODO-PLACEHOLDER
        OUTPUT_DIR="/model/llama3.1-405b/fp8_quantized"
    elif [[ "$QUANT_FORMAT" == "FP4" ]]; then
        MODEL=amd/TODO-PLACEHOLDER
        OUTPUT_DIR="/model/llama3.1-405b/fp4_quantized_autosmoothquant"
    else
        echo "Unsupported QUANT_FORMAT=$QUANT_FORMAT"
        exit 1
    fi
    hf download $MODEL --token $HF_TOKEN --local-dir $OUTPUT_DIR
    exit 0
fi

MODEL="meta-llama/Llama-3.1-405B-Instruct"
MODEL_PATH="/model/llama3.1-405b/orig"
if [[ "${SKIP_DOWNLOAD}" != "true" ]]; then
    hf download $MODEL --token $HF_TOKEN --local-dir $MODEL_PATH
fi

DATASET="/data/llama3.1-405b/mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval.pkl"

pushd "amd_quark-0.10/examples/torch/language_modeling/llm_ptq" > /dev/null
if [[ "$QUANT_FORMAT" == "FP8" ]]; then

    OUTPUT_DIR="/model/llama3.1-405b/fp8_quantized"
    python3 quantize_quark.py --model_dir "${MODEL_PATH}" \
                            --output_dir "${OUTPUT_DIR}" \
                            --dataset "${DATASET}" \
                            --multi_gpu \
                            --data_type auto \
                            --model_attn_implementation "sdpa" \
                            --quant_algo autosmoothquant \
                            --quant_scheme w_fp8_a_fp8 \
                            --kv_cache_dtype fp8 \
                            --min_kv_scale 1.0 \
                            --num_calib_data 512 \
                            --seq_len 8192 \
                            --model_export hf_format \
                            --custom_mode fp8 \
                            --exclude_layers "lm_head"

elif [[ "$QUANT_FORMAT" == "FP4" ]]; then

    OUTPUT_DIR="/model/llama3.1-405b/fp4_quantized"
    OUTPUT_ALGO_DIR="/model/llama3.1-405b/fp4_quantized_${QUANT_ALGO}"
    python3 quantize_quark.py --model_dir "${MODEL_PATH}" \
                          --output_dir "${OUTPUT_ALGO_DIR}" \
                          --dataset "${DATASET}" \
                          --model_attn_implementation "sdpa" \
                          --quant_algo "${QUANT_ALGO}" \
                          --quant_scheme w_mxfp4_a_mxfp4 \
                          --group_size 32 \
                          --data_type bfloat16 \
                          --kv_cache_dtype fp8 \
                          --min_kv_scale 1.0 \
                          --exclude_layers "lm_head" \
                          --model_export hf_format \
                          --multi_gpu
    if [ ! -e $OUTPUT_DIR ]; then
        ln -s $OUTPUT_ALGO_DIR $OUTPUT_DIR
    fi
fi
popd > /dev/null
