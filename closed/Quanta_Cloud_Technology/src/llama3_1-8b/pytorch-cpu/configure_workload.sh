#### THIS SCRIPT IS NOT INTENDED FOR INDEPENDENT RUN. IT CONTROLS RUN CONFIGURATION FOR run_mlperf.sh ####

# Common workload parameters used by the run_mlperf.sh harness.
export WORKLOAD="llama3_1-8b"
export MODEL="llama3_1-8b"
export IMPL="pytorch-cpu"
export COMPLIANCE_TESTS="TEST06"
export COMPLIANCE_SUITE_DIR=${WORKSPACE_DIR}/third_party/mlperf-inference/compliance
#export MAX_LATENCY=10000000000

# This function should handle each combination of the following parameters:
# - SCENARIO: Offline or Server
# - MODE: Performance, Accuracy, and Compliance
workload_specific_run () {
  export MAX_REQUESTS=64
  export SGLANG_USE_CPU_W4A8=1
  export START_NODE=${START_NODE:-0} # Default to 0
  export START_PORT=${START_PORT:-30000} # Starting port for first server
  export NUM_NUMA_NODES=$(lscpu | grep "NUMA node(s)" | awk '{print $NF}')
  export NUM_CORES=$(($(lscpu | grep "Socket(s):" | awk '{print $2}') * $(lscpu | grep "Core(s) per socket:" | awk '{print $4}')))
  export CORES_PER_NODE=$(($NUM_CORES / $NUM_NUMA_NODES))
  export MODEL_PATH=/model/Llama-3.1-8B-Instruct-autoround-w4g128-iters128-cpu
  export DATASET_PATH=/data/cnn_eval.json
  export MEM_FRACTION_STATIC=0.9 #TODO: Need to calculate based on memory
  export MAX_CONCURRENT_REQUESTS=128
  export MAX_TOTAL_TOKENS=131072

  export NUM_CORES=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
  if [ "$SCENARIO" == "Server" ]; then
	  export MAX_REQUESTS=32
	  export MAX_CONCURRENT_REQUESTS=32
	  export MAX_TOTAL_TOKENS=65536
  fi

  if [ "${NUM_CORES}" == "192" ]; then 
      if [ "$SCENARIO" == "Server" ]; then
          export CORES_PER_INST=32
	  export BATCH_SIZE=1
      else
          export CORES_PER_INST=8
	  export BATCH_SIZE=64
      fi
  elif [ "${NUM_CORES}" == "172" ]; then
      if [ "$SCENARIO" == "Server" ]; then
          export CORES_PER_INST=42
	  export BATCH_SIZE=8
      else
          export CORES_PER_INST=7
	  export BATCH_SIZE=64
      fi
  else
      if [ "$SCENARIO" == "Server" ]; then
          export CORES_PER_INST=42
          export BATCH_SIZE=8
      else
          export CORES_PER_INST=7
          export BATCH_SIZE=64
      fi
  fi

  if [ "${MODE}" == "Compliance" ]; then
    export MODE="Performance"
  fi

  export TOTAL_INSTANCES=${TOTAL_INSTANCES:=$(($NUM_CORES / $CORES_PER_INST))}

  if [ "$SCENARIO" == "Server" ]; then
      if [ "${MODE}" == "Accuracy" ]; then
          echo "Run ${MODEL} (${SCENARIO} Accuracy)."
          bash run_accuracy_server.sh
      else
          echo "Run ${MODEL} (${SCENARIO} Performance)."
          bash run_server.sh
      fi
  else
      if [ "${MODE}" == "Accuracy" ]; then
          echo "Run ${MODEL} (${SCENARIO} Accuracy)."
          bash run_accuracy_offline.sh
      else
          echo "Run ${MODEL} (${SCENARIO} Performance)."
          bash run_offline.sh
      fi
  fi
}
