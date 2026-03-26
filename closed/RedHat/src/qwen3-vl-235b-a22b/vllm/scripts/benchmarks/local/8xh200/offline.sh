#!/bin/bash
set -ux
set -o pipefail

# -------- Edit these --------
NP=2
TP=4
MODE="performance_only"   # performance_only | accuracy_only
OUTDIR="${CONTAINER_OUTPUT_DIR:-output}/offline_np${NP}_tp${TP}_${MODE}_$(date +%Y%m%d_%H%M%S)"
# ----------------------------

HF_TOKEN="${HF_TOKEN:-}"  # optional: export HF_TOKEN=... before running

COMPILATION_JSON='{"max_cudagraph_capture_size":16384,"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,1024,1536,2048,3072,4096,6144,8192,12288,16384]}'
OVERRIDE_GEN_JSON='{"max_new_tokens":150}'

mkdir -p "${OUTDIR}"
export CONTAINER_OUTPUT_DIR="${OUTDIR}"

export TOKIO_WORKER_THREADS=32
export OMP_NUM_THREADS=32

args=(
  mpirun -np "${NP}" --bind-to none --allow-run-as-root
  -x VLLM_USE_FLASHINFER_SAMPLER=1
  -x VLLM_USE_FLASHINFER_MOE_FP8=0
  -x VLLM_USE_FLASHINFER_MOE_FP4=1
  -x VLLM_FLASHINFER_MOE_BACKEND=latency
  -x VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=$((6 * 256 * 1024 * 1024))
  -x TOKIO_WORKER_THREADS=32
  -x OMP_NUM_THREADS=32
  -x VLLM_USE_TRITON_POS_EMBED=1
  -x VLLM_MM_ENCODER_FP8_ATTN=0
  mlperf-inf-mm-q3vl benchmark nv mpi-dynamo-vllm --use-http-client
  --max-concurrency=2048
  --dynamo.vllm.enable_numa_binding=true
  --dynamo.frontend.enable_numa_binding=true
  --dynamo.model.repo_id=RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic
  --dynamo.model.revision=main
  --settings.logging.log_output.outdir "${CONTAINER_OUTPUT_DIR}"
  --settings.test.scenario offline
  --settings.test.mode "${MODE}"
  --settings.test.qsl_rng_seed=2465351861681999779
  --settings.test.sample_index_rng_seed=14276810075590677512
  --settings.test.schedule_rng_seed=3936089224930324775
  --dynamo.vllm.cli=--max-model-len=32768
  --dynamo.vllm.cli=--max-num-seqs=1024
  --dynamo.vllm.cli=--max-num-batched-tokens=16384
  --dynamo.vllm.cli=--tensor-parallel-size="${TP}"
  --dynamo.vllm.cli=--enable-expert-parallel
  --dynamo.vllm.cli=--pipeline-parallel-size=1
  --dynamo.vllm.cli=--data-parallel-size=1
  --dynamo.vllm.cli=--async-scheduling
  --dynamo.vllm.cli=--compilation-config="${COMPILATION_JSON}"
  --dynamo.vllm.cli=--no-enable-prefix-caching
  --dynamo.vllm.cli=--limit-mm-per-prompt.video=0
  --dynamo.vllm.cli=--enable-multimodal
  --dynamo.vllm.cli=--connector=none
  --dynamo.vllm.cli=--kv-events-config='{"publisher":"null"}'
  --dynamo.vllm.cli=--distributed-executor-backend=mp
  --dynamo.vllm.cli=--mm-encoder-attn-backend=FLASHINFER
  --dynamo.vllm.cli=--scheduling-policy=sjf
  --dynamo.vllm.cli=--kv-cache-dtype=fp8
  --dynamo.vllm.cli=--override-generation-config="${OVERRIDE_GEN_JSON}"
)

if [[ -n "${HF_TOKEN}" ]]; then
  args+=( "--dynamo.model.token=${HF_TOKEN}" )
fi

echo "OUTDIR=${OUTDIR}"
"${args[@]}" 2>&1 | tee "${OUTDIR}/run.log"
