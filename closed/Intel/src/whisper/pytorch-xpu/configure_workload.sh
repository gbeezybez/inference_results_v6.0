#### THIS SCRIPT IS NOT INTENDED FOR INDEPENDENT RUN. IT CONTROLS RUN CONFIGURATION FOR run_mlperf.sh ####

# Common workload parameters used by the run_mlperf.sh harness.
export WORKLOAD="whisper"
export MODEL="whisper"
export IMPL="pytorch-xpu"
export COMPLIANCE_TESTS="TEST01"
export COMPLIANCE_SUITE_DIR=${WORKSPACE_DIR}/third_party/mlperf-inference/compliance
# export MAX_LATENCY=10000000000

# This function should handle each combination of the following parameters:
# - SCENARIO: Offline or Server
# - MODE: Performance, Accuracy, and Compliance
workload_specific_run () {
  export SCENARIO=${SCENARIO}
  export MODE=${MODE}
  # Standard ENV settings (potentially redundant)
  export MODEL_DIR=${MODEL_DIR}/whisper-large-v3-w4a8g-1
  export DATA_DIR=${DATA_DIR}
  export MANIFEST_FILE=${DATA_DIR}/dev-all-repack.json
  export USER_CONF=${USER_CONF}
  export RUN_LOGS=${RUN_LOGS}
  export XPU_COUNT=$(python -c "import torch; count = 0; count = torch.xpu.device_count() if hasattr(torch, 'xpu') and torch.xpu.is_available() else 0; print(count)")
  export NUM_CORES=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
  export VLLM_USE_V1=1
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=2
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  #export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7
  export ONEDNN_VERBOSE=0
  export USE_PRIMITIVE_CACHE=ON
  export VLLM_XPU_USE_W4A8=1
  export VLLM_FUSE_QUANT=1
  export VLLM_USE_SPLIT_XPU_ATTN=1
  export VLLM_USE_BATCHED_ENCODE=1
  if [ "${SYSTEM}" == "1-node-${XPU_COUNT}x-BMG-B70" ]; then
      export BATCH_SIZE=192
      export MAX_NUM_BATCHED_TOKENS=175488
      export ENCODER_BATCH_SIZE=16
  elif [ "${SYSTEM}" == "1-node-${XPU_COUNT}x-BMG-B60" ]; then
      export BATCH_SIZE=144
      export MAX_NUM_BATCHED_TOKENS=128960
      export ENCODER_BATCH_SIZE=12
  elif [ "${SYSTEM}" == "1-node-${XPU_COUNT}x-BMG-B50" ]; then
      export BATCH_SIZE=92
      export MAX_NUM_BATCHED_TOKENS=83520
      export ENCODER_BATCH_SIZE=8
  else
      export BATCH_SIZE=104
      export MAX_NUM_BATCHED_TOKENS=94768
      export ENCODER_BATCH_SIZE=8
  fi
  # export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
  # export CCL_ZE_IPC_EXCHANGE=drmfd
  if [ "${MODE}" == "Accuracy" ]; then
      export EXTRA_ARGS="--accuracy"
  else
      export EXTRA_ARGS=""
  fi
  export PBAR=1
  python main.py \
      --dataset_dir ${DATA_DIR} \
      --model_path ${MODEL_DIR} \
      --manifest ${MANIFEST_FILE} \
      --scenario Offline \
      --log_dir ${RUN_LOGS} \
      --num_workers ${XPU_COUNT} \
      ${EXTRA_ARGS}
}
