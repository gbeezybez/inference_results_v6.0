#!/bin/bash

set -eux
set -o pipefail

DEFAULT_MODE="performance_only"
mode=${DEFAULT_MODE}

DEFAULT_OUTPUT_DIR="./output/"
output_dir=${DEFAULT_OUTPUT_DIR}

hf_token_flag=""

DEFAULT_WANDB_ENTITY="nvidia"
wandb_entity=${DEFAULT_WANDB_ENTITY}

DEFAULT_WANDB_PROJECT="mlperf-inf-mm-q3vl-nv-v6.0"
wandb_project=${DEFAULT_WANDB_PROJECT}

DEFAULT_WANDB_NAME=""
wandb_name=${DEFAULT_WANDB_NAME}

DEFAULT_WANDB_API_KEY=""
wandb_api_key=${DEFAULT_WANDB_API_KEY}

function _exit_with_help_msg() {
  cat <<EOF
Run the benchmark in the offline scenario and performance-only mode.

Usage: ${BASH_SOURCE[0]}
  [-h | --help]     Print this help message.
  [--mode <mode>]   The mode to run the benchmark in (choices: "performance_only", "accuracy_only"). Default: ${DEFAULT_MODE}
  [--output-dir <output_dir>]   The directory to save the output. Default: ${DEFAULT_OUTPUT_DIR}
  [--hf-token <hf_token>]   The HuggingFace access token to access the model checkpoint.
  [--wandb-entity <wandb_entity>]   The Weights and Biases entity to use. Default: ${DEFAULT_WANDB_ENTITY}
  [--wandb-project <wandb_project>]   The Weights and Biases project to use. Default: ${DEFAULT_WANDB_PROJECT}
  [--wandb-name <wandb_name>]   The Weights and Biases name to use. Default: ${DEFAULT_WANDB_NAME}
  [--wandb-api-key <wandb_api_key>]   The Weights and Biases API key to use. Default: ${DEFAULT_WANDB_API_KEY}
EOF
  if [ -n "$1" ]; then
    echo "$(tput bold setab 1)$1$(tput sgr0)"
  fi
  exit "$2"
}

while [[ $# -gt 0 ]]; do
    case $1 in
    -h | --help)
        _exit_with_help_msg "" 0
        ;;
    --mode)
        mode=$2
        shift
        shift
        ;;
    --mode=*)
        mode=${1#*=}
        shift
        ;;
    --output-dir)
        output_dir=$2
        shift
        shift
        ;;
    --output-dir=*)
        output_dir=${1#*=}
        shift
        ;;
    --hf-token)
        hf_token_flag="--dynamo.model.token=$2"
        shift
        shift
        ;;
    --hf-token=*)
        hf_token_flag="--dynamo.model.token=${1#*=}"
        shift
        ;;
    --wandb-entity)
        wandb_entity=$2
        shift
        shift
        ;;
    --wandb-entity=*)
        wandb_entity=${1#*=}
        shift
        ;;
    --wandb-project)
        wandb_project=$2
        shift
        shift
        ;;
    --wandb-project=*)
        wandb_project=${1#*=}
        shift
        ;;
    --wandb-name)
        wandb_name=$2
        shift
        shift
        ;;
    --wandb-name=*)
        wandb_name=${1#*=}
        shift
        ;;
    --wandb-api-key)
        wandb_api_key=$2
        shift
        shift
        ;;
    --wandb-api-key=*)
        wandb_api_key=${1#*=}
        shift
        ;;
    *)
        _exit_with_help_msg "Unknown argument: $1" 1
        ;;
    esac
done

if [ -n "${wandb_entity}" ] && [ -n "${wandb_project}" ] && [ -n "${wandb_api_key}" ]; then
    wandb_flags="--wandb_config.entity=${wandb_entity} --wandb_config.project=${wandb_project} --wandb_config.api_key=${wandb_api_key}"
    if [ -n "${wandb_name}" ]; then
        wandb_flags="${wandb_flags} --wandb_config.name=${wandb_name}"
    fi
else
    wandb_flags=""
fi

mpirun -np 2 \
    --bind-to numa \
    --allow-run-as-root \
    -x TOKIO_WORKER_THREADS=32 \
    -x OMP_NUM_THREADS=64 \
    mlperf-inf-mm-q3vl benchmark nv mpi-dynamo-vllm \
    --dynamo.model.repo_id=RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic \
    --dynamo.model.revision=main \
    ${hf_token_flag} \
    ${wandb_flags} \
    --settings.test.scenario offline \
    --settings.test.mode ${mode} \
    --settings.test.qsl_rng_seed 2465351861681999779 \
    --settings.test.sample_index_rng_seed 14276810075590677512 \
    --settings.test.schedule_rng_seed 3936089224930324775 \
    --settings.logging.log_output.outdir ${output_dir} \
    --dynamo.vllm.cli=--tensor-parallel-size=4 \
    --dynamo.vllm.cli=--enable-expert-parallel \
    --dynamo.vllm.cli=--pipeline-parallel-size=1 \
    --dynamo.vllm.cli=--data-parallel-size=1 \
    --dynamo.vllm.cli=--async-scheduling \
    --dynamo.vllm.cli=--max-model-len=32768 \
    --dynamo.vllm.cli=--max-num-seqs=1024 \
    --dynamo.vllm.cli=--max-num-batched-tokens=8192 \
    --dynamo.vllm.cli=--compilation-config='{
        "max_cudagraph_capture_size": 8192,
        "cudagraph_capture_sizes": [
            1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
            136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248,
            256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480,
            496, 512, 1024, 1536, 8192
        ]
    }' \
    --dynamo.vllm.cli=--override-generation-config='{"max_new_tokens": 150}' \
    --dynamo.vllm.cli=--limit-mm-per-prompt.video=0 \
    --dynamo.vllm.cli=--no-enable-prefix-caching \
    --dynamo.vllm.cli=--enable-multimodal \
    --dynamo.vllm.cli=--connector=none \
    --dynamo.vllm.cli=--kv-events-config='{"publisher":"null"}' \
    --dynamo.vllm.cli=--distributed-executor-backend=mp

if [ "${mode}" == "accuracy_only" ]; then
    mlperf-inf-mm-q3vl evaluate --filename=${output_dir}/mlperf_log_accuracy.json
    mv accuracy.txt ${output_dir}/accuracy.txt
fi