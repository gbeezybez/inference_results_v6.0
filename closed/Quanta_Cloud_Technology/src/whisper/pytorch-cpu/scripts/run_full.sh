#!/bin/bash

# Prepare workload resources [one-time operations]
bash scripts/download_model.sh
bash scripts/download_dataset.sh
bash scripts/run_calibration.sh

# Run Benchmark (Whisper supports only Offline scenario)
SCENARIO=Offline MODE=Performance bash run_mlperf.sh
SCENARIO=Offline MODE=Accuracy    bash run_mlperf.sh

# Run Compliance
SCENARIO=Offline MODE=Compliance  bash run_mlperf.sh

# Build submission repository (modify VENDOR and SYSTEM to reflect the system under test)
VENDOR=OEM SYSTEM=1-node-2S-GNR_86C bash scripts/prepare_submission.sh
