#!/bin/bash

# set -x

# Set proxy for network operations
export http_proxy=${http_proxy:-"http://173.36.224.108:80/"}
export https_proxy=${https_proxy:-"http://173.36.224.108:80/"}
export HTTP_PROXY=${HTTP_PROXY:-"http://173.36.224.108:80/"}
export HTTPS_PROXY=${HTTPS_PROXY:-"http://173.36.224.108:80/"}

if [ -e /lab-mlperf-inference/code/scripts/setup_llama2_accuracy_env.sh ]
then
    # Try to create venv, but if it fails due to permissions, install directly
    VENV_DIR=/lab-mlperf-inference/code/llama2_accuracy_venv
    
    if python3 -m venv "$VENV_DIR" 2>/dev/null; then
        source "$VENV_DIR/bin/activate"
        pip install transformers google nltk==3.8.1 evaluate==0.4.0 absl-py==1.4.0 rouge-score==0.1.2 sentencepiece>=0.2.0 accelerate==0.21.0 numpy==1.26.4 protobuf==3.20.0
        deactivate
    else
        echo "Warning: Could not create venv, installing packages directly..."
        pip install --user transformers google nltk==3.8.1 evaluate==0.4.0 absl-py==1.4.0 rouge-score==0.1.2 sentencepiece>=0.2.0 accelerate==0.21.0 numpy==1.26.4 protobuf==3.20.0 || \
        pip install transformers google nltk==3.8.1 evaluate==0.4.0 absl-py==1.4.0 rouge-score==0.1.2 sentencepiece>=0.2.0 accelerate==0.21.0 numpy==1.26.4 protobuf==3.20.0
    fi
else
    echo "ERROR: Please enter the MLPerf container before running this script"
    exit 1
fi
