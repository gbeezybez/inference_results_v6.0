#!/bin/bash

cd /data

# Inference dataset
bash /workspace/mlc-r2-downloader.sh https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri

# Calibration dataset
bash /workspace/mlc-r2-downloader.sh https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-dailymail-calibration.uri
