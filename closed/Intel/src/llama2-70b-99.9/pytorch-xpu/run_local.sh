#!/bin/bash
set -x

ACCURACY=$1

OUTPUT_DIR="${OUTPUT_DIR:-offline_accuracy_loadgen_logs}"

mkdir -p $OUTPUT_DIR

run_cmd="python -u main.py 
         --mlperf-conf mlperf.conf 
	 --model-path ${MODEL_NAME}
	 --workload-name ${WORKLOAD_NAME}
	 --dataset-path ${DATASET_PATH}
	 --total-sample-count ${TOTAL_SAMPLE_COUNT}
	 --batch-size ${BATCH_SIZE}
	 --num-workers ${NUM_INSTS}
	 --tensor-parallel ${TP}
	 --pipeline-parallel ${PP}
	 --output-log-dir ${OUTPUT_DIR}
	 --warmup
	 --user-conf user.conf "

if [[ "$ACCURACY" == "accuracy" ]]; then
	run_cmd+="--accuracy "
fi

run_cmd+="--scenario ${SCENARIO} "
#if [[ "$SCENARIO" == "offline" ]]; then
#    run_cmd+="--scenario Offline"
#elif [[ "$SCENARIO" == "server" ]]; then
#    run_cmd+="--scenario Server"
#elif [[ "$SCENARIO" == "singlestream" ]]; then
#    run_cmd+="--scenario SingleStream"
#fi

echo "RUN_COMMAND: ${run_cmd}"
sleep 5

$run_cmd 2>&1 | tee ${OUTPUT_DIR}/run.log

if [[ "$ACCURACY" == "accuracy" ]]; then
	python3 $EVAL_SCRIPT --checkpoint-path ${MODEL_NAME} \
        --mlperf-accuracy-file ${OUTPUT_DIR}/mlperf_log_accuracy.json \
        --dataset-file ${DATASET_PATH} \
        --dtype int64 2>&1 | tee ${OUTPUT_DIR}/accuracy.txt
fi
