#!/bin/bash

# Prepare workload resources [one-time operations]
echo "NOTE: The standard scripts/download_model and scripts/download_dataset"
echo "are not available due to access restrictions applied by MLCommons scripts."
echo "See README for guidance on accessing model and dataset before proceeding"
echo "to calibration step." 
bash scripts/run_calibration.sh 

# Run Benchmark (all scenarios)
SCENARIO=Offline MODE=Performance bash run_mlperf.sh
SCENARIO=Server  MODE=Performance bash run_mlperf.sh
SCENARIO=Offline MODE=Accuracy    bash run_mlperf.sh
SCENARIO=Server  MODE=Accuracy    bash run_mlperf.sh

# Run Compliance (all tests)
SCENARIO=Offline MODE=Compliance  bash run_mlperf.sh
SCENARIO=Server  MODE=Compliance  bash run_mlperf.sh

# Build submission repository (modify VENDOR and SYSTEM to reflect the system under test)
VENDOR=OEM SYSTEM=1-node-2S-GNR_86C bash scripts/prepare_submission.sh
