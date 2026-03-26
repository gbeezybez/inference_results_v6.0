#!/bin/bash
set -x

if [ -z "$1" ]; then
    echo "Error: Missing the first argument which should be workload name"
    return 1
fi

if [ -z "$2" ]; then
    echo "Error: Missing the second argument which should be scenario name"
    return 1
fi

supported_workloads=("llama2-70b" "llama3_1-8b" "llama3_1-8b-edge")
supported_scenarios=("offline" "server" "singlestream")

is_valid() {
    local arg="$1"
    local supported_values_name=$2[@]
    local supported_values=("${!supported_values_name}")
    for value in "${supported_values[@]}"; do
    if [ "$arg" == "$value" ]; then
        return 0
        break
    fi
    done
    return 1
}

if ! is_valid "$1" supported_workloads; then
    echo "Error: Specified workload $1 is not in the allowed set (${supported_workloads[*]})."
    return 1
fi

if ! is_valid "$2" supported_scenarios; then
    echo "Error: Specified scenario $2 is not in the allowed set (${supported_scenarios[*]})."
    return 1
fi

# uninit variables that could have been initialized before (e.g. during development)
unset GPU_MEMORY_UTILIZATION

export WORKLOAD_NAME="$1"
export SCENARIO="$2"
export ROOT_DIR=$PWD
export EVALUATION_DIR=$ROOT_DIR/evaluation_scripts
export MODEL_INIT_DIR=$ROOT_DIR/model_init/$WORKLOAD_NAME
source $MODEL_INIT_DIR/init_env_${SCENARIO,,}
