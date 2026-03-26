#!/bin/bash

##########     MLPERF COMMON FUNCTIONS     ##########
# This file contains shared functions used across multiple MLPerf scripts
# Source this file to use these common functions in your MLPerf scripts

# Setting standard environmental paths
export WORKSPACE_DIR=/workspace
export DATA_DIR=/data
export MODEL_DIR=/model
export LOG_DIR=/logs
export DOCUMENTATION_DIR=${LOG_DIR}/documentation
export CODE_DIR=${LOG_DIR}/code
export COMPLIANCE_DIR=${LOG_DIR}/compliance
export MEASUREMENTS_DIR=${LOG_DIR}/measurements
export RESULTS_DIR=${LOG_DIR}/results
export SYSTEMS_DIR=${LOG_DIR}/systems

##########     SUPPORT FUNCTIONS BEGIN HERE     ##########

# Creates the default user.conf file, either auto-selected, modified, or newly generated.
configure_userconf () {
  cd ${WORKSPACE_DIR}

  # Use pre-configured SYSTEM file if available.
  if [ -f "systems/user.conf.${SYSTEM}" ]; then
    if [ -f "${USER_CONF}" ]; then rm "${USER_CONF}"; fi
    cp systems/user.conf.${SYSTEM} ${USER_CONF}
  fi

  # If an Offline QPS is manually specified, modify the existing user.conf or add to a new one.
  if [ "${OFFLINE_QPS}" != "0" ]; then
      if [ "$(grep "Offline.target_qps" user.conf | wc -l)" == "0" ]; then
          echo "*.Offline.target_qps = ${OFFLINE_QPS}" >> ${USER_CONF}
      else
          sed -i 's/.*Offline.target_qps.*/\*\.Offline\.target_qps = '"${OFFLINE_QPS}"'/g' ${USER_CONF}
      fi
  fi

  # If a Server QPS is manually specified, modify the existing user.conf or add to a new one.
  if [ "${SERVER_QPS}" != "0" ]; then
      if [ "$(grep "Server.target_qps" user.conf | wc -l)" == "0" ]; then
          echo "*.Server.target_qps = ${SERVER_QPS}" >> ${USER_CONF}
      else
          sed -i 's/.*Server.target_qps.*/\*\.Server\.target_qps = '"${SERVER_QPS}"'/g' ${USER_CONF}
      fi
  fi
}

# Base function for preparing supplements - can be overridden in specific scripts
# This handles the common parts: system files and documentation
prepare_supplements_base () {
  cd ${WORKSPACE_DIR}
  # Ensure /logs/systems is populated or abort process.
  if [ -f "systems/${SYSTEM}.json" ]; then
    cp systems/${SYSTEM}.json ${SYSTEMS_DIR}/
  else
    echo '{ "submitter": "OEM", "system_name": "DEFAULT" }' > ${SYSTEMS_DIR}/${SYSTEM}.json
  fi

  # Populate /logs/documentation directory
  cp calibration.md ${DOCUMENTATION_DIR}/
}

# Initializes the system for an MLPerf run, then launches the run.
run_workload () {
  cd ${WORKSPACE_DIR}
  bash run_clean.sh
  if [ -f "${RUN_LOGS}" ]; then rm -r ${RUN_LOGS}; fi
  mkdir -p ${RUN_LOGS}
  workload_specific_run
}

# Places the standard MLPerf run log outputs to the specified final dir.
stage_logs () {
  OUTPUT_PATH=$1
  cd ${RUN_LOGS}
  mkdir -p ${OUTPUT_PATH}
  mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt ${OUTPUT_PATH}/
  if [ -f accuracy.txt ]; then mv accuracy.txt ${OUTPUT_PATH}/; fi
}

# Common setup for directory structure
setup_directories () {
  mkdir -p ${SYSTEMS_DIR}
  mkdir -p ${CODE_PATH}
  mkdir -p ${MEASUREMENTS_PATH}
  mkdir -p ${DOCUMENTATION_DIR}
}

# Common user.conf handling
handle_user_conf () {
  export USER_CONF=user.conf
  if [ "${AUTO_USER_CONF}" == "True" ]; then configure_userconf; fi
  if [ -f "${USER_CONF}" ]; then
    echo "LOG:::: Contents of user.conf:"
    cat ${USER_CONF}
  else
    echo "ERROR::: No user.conf file found."
  fi
}

# Common compliance test execution
run_compliance_tests () {
  for TEST in ${COMPLIANCE_TESTS}; do
      echo "Running compliance ${TEST} ..."
      
      if [ -f ${WORKSPACE_DIR}/audit.config ]; then rm ${WORKSPACE_DIR}/audit.config; fi
      if [ "$TEST" == "TEST01" ]; then
          cp ${COMPLIANCE_SUITE_DIR}/${TEST}/${MODEL}/audit.config .
          if ! [ -d ${RESULTS_PATH} ]; then
              echo "[ERROR] Compliance run could not be verified due to unspecified or non-existent RESULTS_PATH: ${RESULTS_PATH}"
              exit
          fi
          OUTPUT_PATH=${RUN_LOGS}
          run_workload
          python ${COMPLIANCE_SUITE_DIR}/${TEST}/run_verification.py -r ${RESULTS_PATH} -c ${OUTPUT_PATH} -o ${COMPLIANCE_PATH}
      elif [ "$TEST" == "TEST06" ]; then
          cp ${COMPLIANCE_SUITE_DIR}/${TEST}/audit.config .
          OUTPUT_PATH=${RUN_LOGS}
          run_workload
          python ${COMPLIANCE_SUITE_DIR}/${TEST}/run_verification.py -c ${OUTPUT_PATH} -o ${COMPLIANCE_PATH} -s ${SCENARIO}
      fi
  done
}

# Main execution logic - common across all scripts
execute_mlperf_run () {
  export RUN_LOGS=${WORKSPACE_DIR}/run_output
  if [ "${MODE}" == "Performance" ]; then
      run_workload
      stage_logs "${RESULTS_PATH}/performance/run_1"
  elif [ "${MODE}" == "Accuracy" ]; then
      run_workload
      stage_logs "${RESULTS_PATH}/accuracy"
  elif [ "${MODE}" == "Compliance" ]; then
      run_compliance_tests
  else
      echo "[ERROR] Missing value for MODE. Options: Performance, Accuracy, Compliance"
  fi
}

##########     WORKLOAD CONFIGURATION FUNCTIONS     ##########

# Common workload configuration - sets standard variables
configure_workload_base () {
  export IMPL="pytorch-xpu"
  export COMPLIANCE_TESTS="TEST06"
  export COMPLIANCE_SUITE_DIR=${WORKSPACE_DIR}/mlperf-inference/compliance/nvidia
  #export MAX_LATENCY=10000000000
}

# Base workload execution function - handles common setup and cleanup
workload_specific_run_base () {
  local model_name=$1
  local scenario_param=$2
  local primary_scenario=$3
  local secondary_scenario=$4
  
  rm -rf /workspace/log
  rm -rf /workspace/run_output
  mkdir -p /workspace/run_output
  
  if [ "$SCENARIO" == "$primary_scenario" ]; then
      source init_env.sh ${model_name} ${scenario_param}
      if [ "${MODE}" == "Accuracy" ]; then
          echo "Run ${MODEL} (${SCENARIO} Accuracy)."
          bash run_local.sh accuracy
      else
          echo "Run ${MODEL} (${SCENARIO} Performance)."
          bash run_local.sh
      fi
  else
      # Use secondary scenario (usually offline)
      source init_env.sh ${model_name} ${secondary_scenario}
      if [ "${MODE}" == "Accuracy" ]; then
          echo "Run ${MODEL} (${SCENARIO} Accuracy)."
          bash run_local.sh accuracy
      else
          echo "Run ${MODEL} (${SCENARIO} Performance)."
          bash run_local.sh
      fi
  fi

  mv log/llama/* /workspace/run_output/
}