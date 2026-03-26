#!/bin/bash

# Pre-requisite: Using a web browser, navigate to the following links to apply for the appropriate Llama model access: (1)[Meta](https://www.llama.com/llama-downloads/) and (2)[Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). During the 'download_model' script initialization, an authorized Hugging Face account and token must be provided to continue the download.

# Prepare workload resources [Llama3.1-8B only, one-time operations, un-comment below as desired]
#echo "Please ensure the appropriate Llama and Hugging Face access requests have ben applied for and approved before proceeding. These are required to download the Llama model. See 'README.llama3.1-8B.md' for more details." 
#bash scripts/download_model.llama3.1-8B.sh
#bash scripts/download_dataset.llama3.1-8B.sh
#bash scripts/run_calibration.sh

# Run Benchmark (all scenarios)
SCENARIO=Offline MODE=Performance bash run_mlperf.sh
SCENARIO=Server  MODE=Performance bash run_mlperf.sh
SCENARIO=Offline MODE=Accuracy    bash run_mlperf.sh
SCENARIO=Server  MODE=Accuracy    bash run_mlperf.sh

# Run Compliance (all tests)
SCENARIO=Offline MODE=Compliance  bash run_mlperf.sh
SCENARIO=Server  MODE=Compliance  bash run_mlperf.sh

# Build submission
VENDOR=Intel SYSTEM=1-node-8x-BMG-B60 bash scripts/prepare_submission.sh
