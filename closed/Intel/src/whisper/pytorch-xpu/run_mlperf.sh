#!/bin/bash

# Controls workload mode
export SCENARIO="${SCENARIO:-Offline}"
export MODE="${MODE:-Performance}"
export OFFLINE_QPS="${OFFLINE_QPS:-0}"
export SERVER_QPS="${SERVER_QPS:-0}"
export AUTO_USER_CONF="${AUTO_USER_CONF:-True}"
export SYSTEM="${SYSTEM:-AUTO}"
export DEBUG="${DEBUG:-False}"
export XPU_COUNT=${XPU_COUNT:-1}

# Setting standard environmental paths
export WORKSPACE_DIR=/workspace
export DATA_DIR=/data
export MODEL_DIR=/model
export LOG_DIR=/logs
export DOCUMENTATION_DIR=${LOG_DIR}/documentation
export SRC_DIR=${LOG_DIR}/src
export RESULTS_DIR=${LOG_DIR}/results
export SYSTEMS_DIR=${LOG_DIR}/systems

##########     SUPPORT FUNCTIONS BEGIN HERE     ##########

# Set HW specific qps settings from a select list of SKUs, or default.
configure_system () {
  export XPU_COUNT=$(python -c "import torch; count = 0; count = torch.xpu.device_count() if hasattr(torch, 'xpu') and torch.xpu.is_available() else 0; print(count)")
  export XPU_DEVICE_ID=$(python -c "import torch; print(torch.xpu.get_device_properties(0).device_id)")

  if (( XPU_DEVICE_ID == 57891 )); then
    export SYSTEM="1-node-${XPU_COUNT}x-BMG-B70"
  elif (( XPU_DEVICE_ID == 57873 )); then
    export SYSTEM="1-node-${XPU_COUNT}x-BMG-B60"
  elif (( XPU_DEVICE_ID == 57874 )); then
    export SYSTEM="1-node-${XPU_COUNT}x-BMG-B50"
  else
    export SYSTEM="DEFAULT"
  fi

  echo "${SYSTEM}"
}

# Creates the default user.conf file, either auto-selected, modified, or newly generated.
configure_userconf () {
  cd ${WORKSPACE_DIR}

  # Ensure no left-over user.conf files from previous runs, and use pre-configured SYSTEM file if available.
  if [ -f "${USER_CONF}" ]; then rm ${USER_CONF}; fi
  if [ -f "systems/user.conf.${SYSTEM}" ]; then cp systems/user.conf.${SYSTEM} ${USER_CONF}; fi

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

# Creates the non-run-specific submission files (necessary for final submission).
prepare_suplements () {
  cd ${WORKSPACE_DIR}
  # Ensure /logs/systems is populated or abort process.
  if [ -f "systems/${SYSTEM}.json" ]; then
    cp systems/${SYSTEM}.json ${SYSTEMS_DIR}/
  else
    echo '{ "submitter": "OEM", "system_name": "DEFAULT" }' > ${SYSTEMS_DIR}/${SYSTEM}.json
  fi

  # Populate /logs/src directory
  cp -r README.md ${SRC_PATH}/

  # Populate /logs/results scenario directory
  cp measurements.json user.conf mlperf.conf README.md ${RESULTS_PATH}/

  # Populate /logs/documentation directory
  cp calibration.md ${DOCUMENTATION_DIR}/
}

# Initializes the system for an MLPerf run, then launches the run.
run_workload () {
  cd ${WORKSPACE_DIR}
  if [ "${DEBUG}" == "False" ] ; then bash run_clean.sh; fi
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

##########     RUN BEGINS HERE     ##########

# Using workload-specific parameters from 'configure_workload.sh', create the submission dir structure.
source configure_workload.sh
export SYSTEM="$(configure_system)"
export SRC_PATH=${SRC_DIR}/${WORKLOAD}/${IMPL}
export RESULTS_PATH=${RESULTS_DIR}/${SYSTEM}/${WORKLOAD}/${SCENARIO}
mkdir -p ${DOCUMENTATION_DIR}
mkdir -p ${SYSTEMS_DIR}
mkdir -p ${SRC_PATH}
mkdir -p ${RESULTS_PATH}
echo "AUTODETECTED: SYSTEM=${SYSTEM}"
sleep 2

# Ensuring the user.conf file is created if auto is enabled. If disabled, checks for existing one.
export USER_CONF=user.conf
if [ "${AUTO_USER_CONF}" == "True" ]; then configure_userconf; fi
if [ -f "${USER_CONF}" ]; then
  echo "LOG:::: Contents of user.conf:"
  cat ${USER_CONF}
else
  echo "ERROR::: No user.conf file found."
fi

# Creates the non-runtime submission content (src, systems, documents)
if [ "${DEBUG}" == "False" ] ; then prepare_suplements; fi

# Begining workload runs, with Mode of: Performance, Accuracy, OR Compliance
export RUN_LOGS=${WORKSPACE_DIR}/run_output
if [ "${MODE}" == "Performance" ]; then
    run_workload
    stage_logs "${RESULTS_PATH}/performance/run_1"
elif [ "${MODE}" == "Accuracy" ]; then
    run_workload
    stage_logs "${RESULTS_PATH}/accuracy"
elif [ "${MODE}" == "Compliance" ]; then
    for TEST in ${COMPLIANCE_TESTS}; do
        echo "Running compliance ${TEST} ..."
        
        if [ -f ${WORKSPACE_DIR}/audit.config ]; then rm ${WORKSPACE_DIR}/audit.config; fi
        if [ "$TEST" == "TEST01" ]; then
            cp ${COMPLIANCE_SUITE_DIR}/${TEST}/${MODEL}/audit.config .
            COMPLIANCE_ARGS="-r ${RESULTS_PATH} -c ${RUN_LOGS} -o ${RESULTS_PATH}"
        else
            cp ${COMPLIANCE_SUITE_DIR}/${TEST}/audit.config .
            COMPLIANCE_ARGS="-c ${RUN_LOGS} -o ${RESULTS_PATH} -s ${SCENARIO}"
        fi

        if ! [ -d ${RESULTS_PATH} ]; then
            echo "[ERROR] Compliance run could not be verified due to unspecified or non-existant RESULTS_PATH: ${RESULTS_PATH}"
            exit
        fi
        

	run_workload
        if [ -f ${WORKSPACE_DIR}/audit.config ]; then rm ${WORKSPACE_DIR}/audit.config; fi

        python ${COMPLIANCE_SUITE_DIR}/${TEST}/run_verification.py ${COMPLIANCE_ARGS}
    done
else
    echo "[ERROR] Missing value for MODE. Options: Performance, Accuracy, Compliance"
fi
