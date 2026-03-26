#!/bin/bash
set -x

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

if [[ "$MODE" == "acc" ]]; then
	run_cmd+="--accuracy "
fi

if [[ "$SCENARIO" == "offline" ]]; then
    run_cmd+="--scenario Offline"
elif [[ "$SCENARIO" == "server" ]]; then
    run_cmd+="--scenario Server"
elif [[ "$SCENARIO" == "singlestream" ]]; then
    run_cmd+="--scenario SingleStream"
fi

echo $run_cmd

$run_cmd 2>&1 | tee ${OUTPUT_DIR}/run.log

if [[ "$MODE" == "acc" ]]; then
	python3 $EVAL_SCRIPT \
        --mlperf-log ${OUTPUT_DIR}/mlperf_log_accuracy.json \
        --reference-data ${DATASET_PATH} \
        --tokenizer openai/gpt-oss-120b 2>&1 | tee ${OUTPUT_DIR}/accuracy.txt
 elif [[ "$MODE" == test* ]]; then
        mkdir -p "${OUTPUT_DIR}/${MODE}"
	python3 $COMPLIANCE_SCRIPT \
        -c ${OUTPUT_DIR} \
        -o ${OUTPUT_DIR}/${MODE} \
        --audit-config ${AUDIT_CONFIG} \
        --accuracy-script "python3 ${EVAL_SCRIPT} \
            --mlperf-log ${OUTPUT_DIR}/mlperf_log_accuracy.json \
            --reference-data ${DATASET_PATH} \
            --tokenizer openai/gpt-oss-120b" 2>&1 | tee ${OUTPUT_DIR}/${MODE}.txt
        mv verify_*.txt ${OUTPUT_DIR}
fi
