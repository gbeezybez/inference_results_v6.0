#!/bin/bash

# Launch servers
export OUTPUT_DIR=${RUN_LOGS:-run_output}
mkdir -p ${OUTPUT_DIR}
source launch_servers.sh

# Start workers with parameters
export TOTAL_SAMPLE_COUNT=13368 # Total number of samples in the dataset
export TOTAL_INSTANCES=$NUM_LAUNCHED_SERVERS # Total number of instances launched
export UPDATED_URL_LIST=$(printf "%s " "${URL_LIST[@]}") # Convert array to space-separated string
export UPDATED_URL_LIST=(${UPDATED_URL_LIST}) # Convert string back to array
BATCH_SIZE=${BATCH_SIZE:-64} # Batch size for each worker

cmd="python3 -u main.py --dataset-path ${DATASET_PATH} \
    --scenario Offline \
    --mode Performance \
    --workload-name llama3_1-8b \
    --model-path ${MODEL_PATH} \
    --total-sample-count ${TOTAL_SAMPLE_COUNT} \
    --batch-size ${BATCH_SIZE} \
    --device cpu \
    --warmup \
    --num-workers ${TOTAL_INSTANCES} \
    --user-conf user.conf \
    --output-log-dir ${OUTPUT_DIR} \
    --url "${UPDATED_URL_LIST[@]}" 2>&1 | tee ${OUTPUT_DIR}/run.log"

# Run the command
echo "Running command: $cmd"
eval $cmd

pkill -9 python3
