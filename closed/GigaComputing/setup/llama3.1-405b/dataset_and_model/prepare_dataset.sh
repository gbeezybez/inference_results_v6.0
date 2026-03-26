#!/bin/bash

DOWNLOAD_DIR="/data/llama3.1-405b"

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
-d ${DOWNLOAD_DIR} https://inference.mlcommons-storage.org/metadata/llama3-1-405b-dataset-8313.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
-d ${DOWNLOAD_DIR} https://inference.mlcommons-storage.org/metadata/llama3-1-405b-calibration-dataset-512.uri
