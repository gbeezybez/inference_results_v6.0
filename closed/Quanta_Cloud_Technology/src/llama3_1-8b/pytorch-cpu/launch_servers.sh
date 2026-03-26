#!/bin/bash

MAX_REQUESTS=${MAX_CONCURRENT_REQUESTS:-64}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.1}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-65536}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-32768}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}

# Function to launch servers with specific port and CPU affinity list
function launch_server() {
    local port=$1
    local cpu_affinity=$2
    local model_path=${MODEL_PATH}

    echo "Launching server on port $port with CPU affinity $cpu_affinity"
    export SGLANG_USE_CPU_W4A8=1
    taskset -c $cpu_affinity \
    python3 -m sglang.launch_server \
        --model-path $model_path \
        --device cpu \
        --mem-fraction-static ${MEM_FRACTION_STATIC} \
        --host 127.0.0.1 \
        --port $port \
        --max-total-tokens ${MAX_TOTAL_TOKENS} \
        --max-prefill-tokens ${MAX_PREFILL_TOKENS} \
        --max-running-requests ${MAX_REQUESTS} \
        --chunked-prefill-size ${CHUNKED_PREFILL_SIZE} \
        --skip-tokenizer-init &
}

# Check for launched servers
HEALTHY_URL_LIST=()

function health_check_servers() {
    echo "Waiting for servers to start and checking health..."
    for url in "${URL_LIST[@]}"; do
        echo "Checking server at: $url"
        
        # Wait up to 60 seconds for server to become ready
        max_attempts=12
        attempt=0
        server_ready=false
        
        while [[ $attempt -lt $max_attempts ]]; do
            # Check if server responds with HTTP 200
            http_code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 5 --max-time 10 "$url/health" 2>/dev/null || echo "000")
            
            if [[ "$http_code" == "200" ]]; then
                server_ready=true
                break
            elif [[ "$http_code" == "000" ]]; then
                # Connection failed, try the root endpoint as fallback
                http_code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 5 --max-time 10 "$url" 2>/dev/null || echo "000")
                if [[ "$http_code" == "200" ]] || [[ "$http_code" == "404" ]] || [[ "$http_code" == "405" ]]; then
                    # 404 or 405 means server is running but endpoint doesn't exist, which is acceptable
                    server_ready=true
                    break
                fi
            fi
            
            echo "  Attempt $((attempt + 1))/$max_attempts: Server not ready (HTTP $http_code), waiting 5 seconds..."
            sleep 5
            ((attempt++))
        done
        
        if [[ "$server_ready" == true ]]; then
            echo "  ✓ Server at $url is reachable and healthy."
            HEALTHY_URL_LIST+=("$url")
        else
            echo "  ✗ Server at $url is not reachable after $max_attempts attempts."
        fi
    done

    echo "Health check completed. ${#HEALTHY_URL_LIST[@]} out of ${#URL_LIST[@]} servers are healthy."

}


TIC=$(date +%s) # Start timer

# List of launched servers
URL_LIST=()
# Launch servers on numa nodes with different CPU affinities
NUM_LAUNCHED_SERVERS=0
# Loop through each NUMA node starting from START_NODE
for ((i=START_NODE; i<$NUM_NUMA_NODES; i++)); do
    # Get the start and end cores for this NUMA node. Get the physical cores only.
    # lscpu output format: "NUMA node0 CPU(s): 0-29,120-149"
    # start_core=0, end_core=29 for node 0, start_core=30, end_core=59 for node 1, etc.
    numa_info=$(lscpu | grep "NUMA node$i CPU(s):")
    numa_cores_list=$(echo $numa_info | awk '{print $4}' | tr ',' ' ')
    start_core=$(echo $numa_cores_list | awk '{print $1}' | cut -d'-' -f1)
    end_core=$(echo $numa_cores_list | awk '{print $1}' | cut -d'-' -f2)

    # Create instances on this numa node
    while [[ $start_core -le $end_core ]] && [[ $NUM_LAUNCHED_SERVERS -lt $TOTAL_INSTANCES ]]; do
        # Calculate the CPU affinity for this instance
        cpu_affinity="$start_core-$((start_core + CORES_PER_INST - 1))"
        
        # Launch server with the calculated CPU affinity
        echo "Launching server on NUMA node $i with CPU affinity $cpu_affinity"
        port=$((START_PORT + NUM_LAUNCHED_SERVERS))
        launch_server $port "$cpu_affinity" > ${OUTPUT_DIR}/server_${i}_${start_core}.log 2>&1 &
        # Add the URL to the list
        URL_LIST+=("http://127.0.0.1:$port")
        # Wait for a short time to allow the server to start
        sleep 5
        # Increment start_core by CORES_PER_INST
        start_core=$((start_core + CORES_PER_INST))
        next_affinity_end=$((start_core + CORES_PER_INST - 1))
        
        NUM_LAUNCHED_SERVERS=$((NUM_LAUNCHED_SERVERS + 1))
        echo "Number of launched servers: $NUM_LAUNCHED_SERVERS"
        # Check if the next server would exceed the end_core
        if [[ $next_affinity_end -gt $end_core ]]; then
            break
        fi
    done
    echo -e "Finished launching servers on NUMA node $i\n"
done
# Check if servers were launched
health_check_servers
# Reassign URL_LIST to HEALTHY_URL_LIST if any servers are healthy
URL_LIST=("${HEALTHY_URL_LIST[@]}")
# Update the number of launched servers based on healthy servers
NUM_LAUNCHED_SERVERS=${#HEALTHY_URL_LIST[@]}

if [[ ${#HEALTHY_URL_LIST[@]} -eq 0 ]]; then
    echo "No reachable servers found. Exiting."
    # Set exit code to 3 to indicate failure
    exit 3
else
    # Export the updated URL list as a space-separated string for the workers
    export HEALTHY_URL_LIST_STR="${HEALTHY_URL_LIST[*]}"
    echo "Request producer started with URLs: ${HEALTHY_URL_LIST[@]}"
    echo "Logs can be found in ${OUTPUT_DIR}/run.log"
    echo "To stop the request producer, use: killall -9 python3"
    echo "To stop the servers, use: killall -9 python3"
    echo "To check the status of the servers, use: ps aux | grep sglang.launch_server"
    echo "To check the logs of the servers, use: tail -f server_*.log"
    echo "To check the logs of the request producer, use: tail -f request_producer.log"
    echo "To check the status of the launched servers, use: curl -s -o /dev/null -w '%{http_code}' ${HEALTHY_URL_LIST[@]}"
fi

# Calculate the elapsed time
TOC=$(date +%s)
ELAPSED_TIME=$((TOC - TIC))
echo "Total time taken to launch servers: $ELAPSED_TIME seconds"
