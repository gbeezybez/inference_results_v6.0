#!/bin/bash

# Settings expected for run
export SCENARIO="${SCENARIO:-Offline}"
export OFFLINE_QPS="${OFFLINE_QPS:-0}"
export SERVER_QPS="${SERVER_QPS:-0}"

# Default settings for this workload (ResNet-50)
export OFFLINE_DURATION=120000
export SERVER_DURATION=120000
export INITIAL_LATENCY=1000000000000

# Setting standard environmental paths
export WORKSPACE_DIR=/workspace
export DATA_DIR=/data
export MODEL_DIR=/model
export LOG_DIR=/logs
export MODE=Performance

export USER_CONF=${WORKSPACE_DIR}/user.conf
export RUN_LOGS=${WORKSPACE_DIR}/run_output
export SUMMARY_FILE=${RUN_LOGS}/mlperf_log_summary.txt
if [ -f "${RUN_LOGS}" ]; then rm -r ${RUN_LOGS}; fi
  mkdir -p ${RUN_LOGS}

# Uses same configure_workload script as run_mlperf.sh
source configure_workload.sh

parse_logs () {
    if [[ "${WORKLOAD}" =~ "gptj" ]]; then
        THROUGHPUT=$(cat ${SUMMARY_FILE} | grep "okens per second" | head -1 | rev | cut -d' ' -f1 | rev)
    else
        THROUGHPUT=$(cat ${SUMMARY_FILE} | grep "amples per second" | head -1 | rev | cut -d' ' -f1 | rev)
    fi
    LATENCY=$(cat ${SUMMARY_FILE} | grep "99.00 percentile latency" | head -1 | rev | cut -d' ' -f1 | rev)
    echo "${THROUGHPUT},${LATENCY}"
}

create_user_config () {
    echo "*.Offline.target_qps = ${OFFLINE_QPS}" > ${USER_CONF}
    echo "*.Server.target_qps = ${SERVER_QPS}" >> ${USER_CONF}
    echo "*.Offline.min_duration = ${OFFLINE_DURATION}" >> ${USER_CONF}
    echo "*.Server.min_duration = ${SERVER_DURATION}" >> ${USER_CONF}
}

# For Server run, starts with initial setting, then narrows in on valid Server performance using 3 phases (reduce 5%->increase 1%->increase 0.2%)
if [ "$SCENARIO" == "Server" ]; then
    LATENCY=${INITIAL_LATENCY}
    # Starting (assumed target above actual passing Server qps), and reducing in 5% increments
    while [[ $(echo "${LATENCY} > ${MAX_LATENCY}" | bc -l) == "1" ]]; do
        if [ "${LATENCY}" != "${INITIAL_LATENCY}" ]; then export SERVER_QPS=$(echo "${SERVER_QPS} * 0.95" | bc -l); fi
        create_user_config
        workload_specific_run
        RESULTS="$(parse_logs)"
        THROUGHPUT=$(echo ${RESULTS} | cut -d',' -f1)
        LATENCY=$(echo ${RESULTS} | cut -d',' -f2)
        echo "SERVER_INTERMEDIATE(SERVER_QPS_THROUGHPUT_LATENCY): ${SERVER_QPS} ${THROUGHPUT} ${LATENCY}" | tee -a ${LOG_DIR}/FIND_QPS-Server.log
    done
    # Now below actual Server, and increasing in 1% increments
    while [[ $(echo "${LATENCY} < ${MAX_LATENCY}" | bc -l) == "1" ]]; do
        PASSING_SERVER_QPS=${SERVER_QPS}
        PASSING_THROUGHPUT=${THROUGHPUT}
        PASSING_LATENCY=${LATENCY}
        export SERVER_QPS=$(echo "${PASSING_SERVER_QPS} * 1.01" | bc -l)
        create_user_config
        workload_specific_run
        RESULTS="$(parse_logs)"
        THROUGHPUT=$(echo ${RESULTS} | cut -d',' -f1)
        LATENCY=$(echo ${RESULTS} | cut -d',' -f2)
        echo "SERVER_INTERMEDIATE(SERVER_QPS_THROUGHPUT_LATENCY): ${SERVER_QPS} ${THROUGHPUT} ${LATENCY}" | tee -a ${LOG_DIR}/FIND_QPS-Server.log
    done
    # Now believe within 1%, and increasing in 0.2% increments with double duration.
    export SERVER_DURATION=$(( ${SERVER_DURATION} * 2 ))
    SERVER_QPS=${PASSING_SERVER_QPS}
    THROUGHPUT=${PASSING_THROUGHPUT}
    LATENCY=${PASSING_LATENCY}
    while [[ $(echo "${LATENCY} < ${MAX_LATENCY}" | bc -l) == "1" ]]; do
        PASSING_SERVER_QPS=${SERVER_QPS}
        PASSING_THROUGHPUT=${THROUGHPUT}
        PASSING_LATENCY=${LATENCY}
        export SERVER_QPS=$(echo "${PASSING_SERVER_QPS} * 1.002" | bc -l)
        create_user_config
        workload_specific_run
        RESULTS="$(parse_logs)"
        THROUGHPUT=$(echo ${RESULTS} | cut -d',' -f1)
        LATENCY=$(echo ${RESULTS} | cut -d',' -f2)
        echo "SERVER_INTERMEDIATE(SERVER_QPS_THROUGHPUT_LATENCY): ${SERVER_QPS} ${THROUGHPUT} ${LATENCY}" | tee -a ${LOG_DIR}/FIND_QPS-Server.log
    done
    echo "SERVER_FINAL(THROUGHPUT_LATENCY): ${PASSING_THROUGHPUT} ${PASSING_LATENCY}" | tee -a ${LOG_DIR}/FIND_QPS-Server.log
else
    create_user_config
    workload_specific_run
    RESULTS="$(parse_logs)"
    THROUGHPUT=$(echo ${RESULTS} | cut -d',' -f1)
    LATENCY=$(echo ${RESULTS} | cut -d',' -f2)
    echo "OFFLINE_FINAL(THROUGHPUT_LATENCY): ${THROUGHPUT} ${LATENCY}" | tee -a ${LOG_DIR}/FIND_QPS-Offline.log
fi
