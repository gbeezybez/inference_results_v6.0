#!/bin/bash

# Controls workload mode
export SCENARIO="${SCENARIO:-Offline}"
export MODE="${MODE:-Performance}"
export OFFLINE_QPS="${OFFLINE_QPS:-0}"
export SERVER_QPS="${SERVER_QPS:-0}"
export AUTO_USER_CONF="${AUTO_USER_CONF:-True}"
export SYSTEM="${SYSTEM:-AUTO}"

# Source common MLPerf functions
source "$(dirname "${BASH_SOURCE[0]}")/mlperf_common_functions.sh"

##########     WORKLOAD-SPECIFIC FUNCTIONS     ##########

# Set HW specific qps settings from a select list of SKUs, or default.
configure_system () {
  if [ "${SYSTEM}" != "AUTO" ] ; then
      export SYSTEM="${SYSTEM}"
  else
      export SYSTEM="1-node-4x-BMG-Pro-B60-Dual"
  fi

  echo ${SYSTEM}
}

# Creates the workload-specific supplement files (extends the base function).
prepare_supplements () {
  # Call the base function for common supplements
  prepare_supplements_base

  cd ${WORKSPACE_DIR}
  # Populate /logs/code directory
  cp -r README.llama2-70b.md ${CODE_PATH}/

  # Populate /logs/measurements directory
  cp measurements.json ${MEASUREMENTS_PATH}/${SYSTEM}.json
  cp README.llama2-70b.md user.conf calibration/quantize_70b.sh ${MEASUREMENTS_PATH}/
}

##########     RUN BEGINS HERE     ##########

# Using workload-specific parameters from 'configure_workload.sh', create the submission dir structure.
source configure_workload_llama2-70b.sh
export SYSTEM="$(configure_system)"
export CODE_PATH=${CODE_DIR}/${WORKLOAD}/${IMPL}
export MEASUREMENTS_PATH=${MEASUREMENTS_DIR}/${SYSTEM}/${WORKLOAD}/${SCENARIO}
export COMPLIANCE_PATH=${COMPLIANCE_DIR}/${SYSTEM}/${WORKLOAD}/${SCENARIO}
export RESULTS_PATH=${RESULTS_DIR}/${SYSTEM}/${WORKLOAD}/${SCENARIO}

# Setup directories and handle user.conf
setup_directories
handle_user_conf

# Creates the non-runtime submission content (code, systems, measurements)
prepare_supplements

# Execute the MLPerf run using common logic
execute_mlperf_run
