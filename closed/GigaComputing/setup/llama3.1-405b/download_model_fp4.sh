#!/bin/bash

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/../build_scripts/build_docker_general.sh"

export CONFIG_FILE="../build_scripts/dataset_and_model.config"
source "$SCRIPT_DIR/$CONFIG_FILE"

source "$SCRIPT_DIR/../build_scripts/bash_util/argparse.sh"

# Support both positional args (legacy) and named args
if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
    # Legacy positional argument mode
    HF_TOKEN=${1:-"NONE"}
else
    # Named argument mode
    declare -A ARG_SPECS=(
        ["--token"]="value optional HF_TOKEN default:NONE"
    )
    parse_args "$@"
    HF_TOKEN="${ARGS[--token]}"
fi

build_dataset_and_model

bash "$SCRIPT_DIR/../start_scripts/run_model_dataset.sh" $SCRIPT_DIR dataset_and_model/prepare_model.sh "$HF_TOKEN" "FP4" "autosmoothquant"
