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

if [ -z "$3" ]; then
    echo "Error: Missing the third argument which should be perf/acc"
    return 1
fi

supported_workloads=("gpt-oss-120b")
supported_scenarios=("offline" "server")
supported_mode=("perf" "acc" "test07" "test09")

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
source uninit_env

export WORKLOAD_NAME="$1"
export SCENARIO="$2"
export MODE="$3"
export ROOT_DIR=$PWD
export MODEL_INIT_DIR=$ROOT_DIR/model_init/$WORKLOAD_NAME
source $MODEL_INIT_DIR/init_env_${SCENARIO,,}
