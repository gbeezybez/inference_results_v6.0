#### THIS SCRIPT IS NOT INTENDED FOR INDEPENDENT RUN. IT CONTROLS RUN CONFIGURATION FOR run_mlperf.sh ####

# Common workload parameters used by the run_mlperf.sh harness.
#export WORKLOAD="llama3_1-8b"
#export MODEL="llama3_1-8b"
export IMPL="pytorch-xpu"
export COMPLIANCE_TESTS="TEST06"
export COMPLIANCE_SUITE_DIR=${WORKSPACE_DIR}/third_party/mlperf-inference/compliance
#export MAX_LATENCY=10000000000

# Dynamically detecting model and workload based on contents of /model
if (( "$(find /model -name Llama-3.1-8B* | wc -l)" > 0 )); then 
  export WORKLOAD="llama3_1-8b"
  export MODEL="llama3_1-8b"
  echo "AUTODETECTED: MODEL=llama3_1-8b"
  sleep 2
elif (( "$(find /model -name Llama-2-70b* | wc -l)" > 0 )); then
  export WORKLOAD="llama2-70b-99.9"
  export MODEL="llama2-70b"
  echo "AUTODETECTED: MODEL=llama2-70b"
  sleep 2
else
  echo "ERROR: Model file not detected in /model. Exiting."
  exit
fi

# This function should handle each combination of the following parameters:
# - SCENARIO: Offline or Server
# - MODE: Performance, Accuracy, and Compliance
workload_specific_run () {
    unset GPU_MEMORY_UTILIZATION

    export WORKLOAD_NAME=${MODEL}
    export SCENARIO=${SCENARIO}
    export ROOT_DIR=/workspace
    export EVALUATION_DIR=/workspace/evaluation_scripts
    export MODEL_INIT_DIR=/workspace/model_init/${MODEL}
    export OUTPUT_DIR=${RUN_LOGS}

    source init_env_base
    source ${MODEL_INIT_DIR}/init_env_shared
    if [ "$SCENARIO" == "Server" ]; then
        echo "SCENARIO: Server"
	source ${MODEL_INIT_DIR}/init_env_server
    else
        echo "SCENARIO: Offline"
        source ${MODEL_INIT_DIR}/init_env_offline
    fi

    if [ "${MODE}" == "Accuracy" ]; then
        echo "MODE: Accuracy"
        ./run_local.sh accuracy
    else
        echo "MODE: Performance"
	./run_local.sh
    fi
}
