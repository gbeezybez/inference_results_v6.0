#!/bin/bash
set -xeu

SUBMISSION_DIR=/workspace/apps/mlperf/submission
TEST06_DIR=/workspace/apps/mlperf/tools/compliance/nvidia/TEST06
SYSTEM_NAME=${SYSTEM_NAME:-"32xMI300X_2xEPYC_9534"} #CPU name: lscpu | grep name; count: lscpu | grep 'node(s)'
GPU_NAME=${GPU_NAME:-'mi300x'}
COMPANY=${COMPANY:-'Dell_MangoBoost'}


## Package the results

# 8xMI355X
SYSTEM_NAME=8xMI355X_2xEPYC_9965
USER_CONF=conf/user_llama2-70b_8x_mi355x.conf
python $SUBMISSION_DIR/package_submission.py --base-package-dir inference_results_6.0 --input-dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME --scenarios Server Offline --system-name $SYSTEM_NAME --benchmark llama2-70b-99   --user-conf ${USER_CONF} --company ${COMPANY} --system-description systems/${SYSTEM_NAME}.json
python $SUBMISSION_DIR/package_submission.py --base-package-dir inference_results_6.0 --input-dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME --scenarios Server Offline --system-name $SYSTEM_NAME --benchmark llama2-70b-99.9 --user-conf ${USER_CONF} --company ${COMPANY} --system-description systems/${SYSTEM_NAME}.json

# 8xMI355X Power Cap
SYSTEM_NAME=8xMI355X_2xEPYC_9965_Power_Cap_1000W
USER_CONF=conf/user_llama2-70b_8x_mi355x_power_1000w.conf
python $SUBMISSION_DIR/package_submission.py --base-package-dir inference_results_6.0 --input-dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME --scenarios Server Offline --system-name $SYSTEM_NAME --benchmark llama2-70b-99   --user-conf ${USER_CONF} --company ${COMPANY} --system-description systems/${SYSTEM_NAME}.json
python $SUBMISSION_DIR/package_submission.py --base-package-dir inference_results_6.0 --input-dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME --scenarios Server Offline --system-name $SYSTEM_NAME --benchmark llama2-70b-99.9 --user-conf ${USER_CONF} --company ${COMPANY} --system-description systems/${SYSTEM_NAME}.json


# 8xMI355X
SYSTEM_NAME=8xMI300X_2xEPYC_9534_8xMI325X_2xEPYC_9655_8xMI355X_2xEPYC_9965
USER_CONF=conf/user_llama2-70b_8x_mi300x_8x_mi325x_8x_mi355x.conf
python $SUBMISSION_DIR/package_submission.py --base-package-dir inference_results_6.0 --input-dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME --scenarios Server Offline --system-name $SYSTEM_NAME --benchmark llama2-70b-99   --user-conf ${USER_CONF} --company ${COMPANY} --system-description systems/${SYSTEM_NAME}.json
python $SUBMISSION_DIR/package_submission.py --base-package-dir inference_results_6.0 --input-dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME --scenarios Server Offline --system-name $SYSTEM_NAME --benchmark llama2-70b-99.9 --user-conf ${USER_CONF} --company ${COMPANY} --system-description systems/${SYSTEM_NAME}.json
