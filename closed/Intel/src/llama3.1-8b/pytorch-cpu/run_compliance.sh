#!/bin/bash

cp /workspace/third_party/mlperf-inference/compliance/nvidia/TEST06/audit.config audit.conf 
cp /workspace/third_party/mlperf-inference/compliance/nvidia/TEST06/run_verification.py .
./run_offline_gptj.sh
./run_server_gptj.sh
python3 run_verification.py -c /logs/server_performance_loadgen_logs -o /logs/server_performance_loadgen_logs -s Server
python3 run_verification.py -c /logs/offline_performance_loadgen_logs -o /logs/offline_performance_loadgen_logs -s Offline

rm audit.conf
