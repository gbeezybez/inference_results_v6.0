#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

HF_TOKEN=$1
QUANT_FORMAT=$2
QUANT_ALGO=${3:-autosmoothquant}

# Treat "NONE" placeholder as empty (used when no token provided)
if [ "$HF_TOKEN" == "NONE" ]; then
    HF_TOKEN=""
fi

MODEL="meta-llama/Llama-3.1-405B-Instruct"
MODEL_PATH="/model/llama3.1-405b/orig"

# Check if model directory exists and is not empty
if [ -d "$MODEL_PATH" ] && [ "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]; then
    echo -e "${GREEN}Found content under ${YELLOW}${MODEL_PATH}${GREEN}, skipping HF download${NC}"
else
    # HF token is only required when download is needed
    if [ -z "$HF_TOKEN" ]; then
        echo -e "${RED}Error: Model not found at ${MODEL_PATH} and no HF token provided${NC}"
        echo -e "${RED}Please provide HF token as the first parameter to download the model${NC}"
        exit 1
    fi

    echo -e "${YELLOW}Directory ${MODEL_PATH} is empty or doesn't exist, downloading from HF...${NC}"
    huggingface-cli download $MODEL --token $HF_TOKEN --local-dir $MODEL_PATH

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to download model from HF${NC}"
        exit 1
    fi
    echo -e "${GREEN}Model downloaded successfully${NC}"
fi

echo -e "${GREEN}Quantizing the model with format: ${YELLOW}${QUANT_FORMAT}${NC}"

DATASET="/data/llama3.1-405b/mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval.pkl"

pushd "amd_quark-0.10/examples/torch/language_modeling/llm_ptq" > /dev/null
if [[ "$QUANT_FORMAT" == "FP8" ]]; then
    echo -e "${GREEN}Running FP8 quantization...${NC}"

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
    echo -e "${GREEN}Running FP4 quantization with algorithm: ${YELLOW}${QUANT_ALGO}${NC}"

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
    if [ ! -e "$OUTPUT_DIR" ]; then
        ln -s "$OUTPUT_ALGO_DIR" "$OUTPUT_DIR"
    fi
else
    echo -e "${RED}Error: Invalid QUANT_FORMAT '${QUANT_FORMAT}'. Must be 'FP8' or 'FP4'${NC}"
    popd > /dev/null
    exit 1
fi
popd > /dev/null
echo -e "${GREEN}Quantization completed successfully${NC}"
