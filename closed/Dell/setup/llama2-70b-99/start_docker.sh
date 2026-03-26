#!/bin/bash


export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export CONFIG_FILE="vllm_gfx9xx.config"

if [[ -n "$1" ]]; then
  CONFIG_FILE="$1"
fi

source "$SCRIPT_DIR/$CONFIG_FILE"

bash "$SCRIPT_DIR/../start_scripts/start_general.sh"
