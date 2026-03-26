#!/bin/bash
#SBATCH -A coreai_mlperf_inference
#SBATCH --output=outputs/%j-stdout.txt
#SBATCH --error=outputs/%j-stderr.txt
#SBATCH --partition=b200-a01r
#SBATCH --time=4:00:00
#SBATCH -N1
#SBATCH --ntasks-per-node=8

CONTAINER_IMAGE=${CONTAINER_IMAGE:-"gitlab-master.nvidia.com:5005/mlpinf/mlperf-inference/mlperf-inf-mm-q3vl-nv:amd64_cuda13.0.1_CentML_dynamo-mlperf-inf-mm-q3vl-v6.0_CentML_vllm-mlperf-inf-mm-q3vl-v6.0"}
HF_CACHE_HOST_DIR=${CACHE_HOST_DIR:-"/lustre/fsw/coreai_mlperf_inference/${USER}/.cache/huggingface"}
OUTPUT_HOST_DIR=${OUTPUT_HOST_DIR:-"$(pwd -P)/outputs/${SLURM_JOB_ID}/"}
HF_TOKEN=${HF_TOKEN:-""}
WANDB_ENTITY=${WANDB_ENTITY:-"nvidia"}
WANDB_PROJECT=${WANDB_PROJECT:-"mlperf-inf-mm-q3vl-nv-v6.0"}
WANDB_NAME=${WANDB_NAME:-"prenyx-${SLURM_JOB_ID}"}
WANDB_API_KEY=${WANDB_API_KEY:-""}

mkdir -p "${OUTPUT_HOST_DIR}"

mounts="${HF_CACHE_HOST_DIR}:/root/.cache/huggingface,${OUTPUT_HOST_DIR}:/output/"

if [ -n "${HF_TOKEN}" ]; then
    hf_token_flag="--dynamo.model.token=${HF_TOKEN}"
else
    hf_token_flag=""
fi

if [ -n "${WANDB_ENTITY}" ] && [ -n "${WANDB_PROJECT}" ] && [ -n "${WANDB_API_KEY}" ]; then
    wandb_flags="--wandb_config.entity=${WANDB_ENTITY} --wandb_config.project=${WANDB_PROJECT} --wandb_config.api_key=${WANDB_API_KEY}"
    if [ -n "${WANDB_NAME}" ]; then
        wandb_flags="${wandb_flags} --wandb_config.name=${WANDB_NAME}"
    fi
else
    wandb_flags=""
fi

export DYN_LOG=debug  # Change from 'info' to 'debug'
export VLLM_LOGGING_LEVEL=DEBUG

export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_FLASHINFER_MOE_BACKEND=latency
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=$((6 * 256 * 1024 * 1024)) 
export TOKIO_WORKER_THREADS=32
export OMP_NUM_THREADS=32

echo "Starting job at: $(date)"
echo $SLURM_GPUS_ON_NODE

# NUMA binding: GPU 0-1 use NUMA node 0, GPU 2-3 use NUMA node 1, GPU 4-5 use NUMA node 2, GPU 6-7 use NUMA node 3
srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${mounts}" \
    --no-container-mount-home \
    --mpi=pmix \
    bash -c "NUMA_NODE=\$((SLURM_LOCALID / 4)); exec numactl --cpunodebind=\$NUMA_NODE --membind=\$NUMA_NODE mlperf-inf-mm-q3vl benchmark nv mpi-dynamo-vllm \
    --dynamo.model.repo_id=nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4 \
    --dynamo.model.revision=main \
    ${hf_token_flag} \
    ${wandb_flags} \
    --settings.test.scenario offline \
    --settings.test.mode performance_only \
    --settings.test.qsl_rng_seed 2465351861681999779 \
    --settings.test.sample_index_rng_seed 14276810075590677512 \
    --settings.test.schedule_rng_seed 3936089224930324775 \
    --settings.logging.log_output.outdir /output/ \
    --dynamo.vllm.cli=--tensor-parallel-size=1 \
    --dynamo.vllm.cli=--pipeline-parallel-size=1 \
    --dynamo.vllm.cli=--data-parallel-size=1 \
    --dynamo.vllm.cli=--async-scheduling \
    --dynamo.vllm.cli=--max-model-len=32768 \
    --dynamo.vllm.cli=--max-num-seqs=1024 \
    --dynamo.vllm.cli=--mm-encoder-attn-backend=FLASH_ATTN_CUTE \
    --dynamo.vllm.cli=--max-num-batched-tokens=9216 \
    --dynamo.vllm.cli=--compilation-config='{
        \"max_cudagraph_capture_size\": 2048,
        \"cudagraph_capture_sizes\": [
            1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
            136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248,
            256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480,
            496, 512, 1024, 1536, 2048
        ]
    }' \
    --dynamo.vllm.cli=--override-generation-config='{\"max_new_tokens\": 150}' \
    --dynamo.vllm.cli=--limit-mm-per-prompt.video=0 \
    --dynamo.vllm.cli=--no-enable-prefix-caching \
    --dynamo.vllm.cli=--enable-multimodal \
    --dynamo.vllm.cli=--connector=none \
    --dynamo.vllm.cli=--kv-events-config='{\"publisher\":\"null\"}' \
    "