#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DATA_DIR=${DATA_DIR:-/data/mixtral-8x7b}
DATASET_MD5="ded6c711288c9bbca02929855557b8c1"
CALIBRATION_DATASET_MD5="75067c9fe5cb5baef216a4b124c61df1"


if [ -e /lab-mlperf-inference/setup/download_mixtral_8x7b.sh ]
then
    echo "Inside container, start downloading..."
    mkdir -p ${DATA_DIR}

    if [ -e "${DATA_DIR}/mlperf_mixtral8x7b_dataset_15k.pkl" ]
    then
        echo "Dataset for Mixtral-8x7b is already exist"
    else
        wget https://inference.mlcommons-storage.org/mixtral_8x7b/09292024_mixtral_15k_mintoken2_v1.pkl -O ${DATA_DIR}/mlperf_mixtral8x7b_dataset_15k.pkl
    fi

    if [ -e "${DATA_DIR}/mlperf_mixtral8x7b_calibration_dataset_1k.pkl" ]
    then
        echo "Calibration dataset for Mixtral-8x7b is already exist"
    else
        wget https://inference.mlcommons-storage.org/mixtral_8x7b%2F2024.06.06_mixtral_15k_calibration_v4.pkl -O ${DATA_DIR}/mlperf_mixtral8x7b_calibration_dataset_1k.pkl
    fi

    md5sum ${DATA_DIR}/mlperf_mixtral8x7b_dataset_15k.pkl | grep ${DATASET_MD5}
    if [ $? -ne 0 ]; then
        echo "md5sum of the data file mismatch. Should be ${DATASET_MD5}"
        exit -1
    fi

    md5sum ${DATA_DIR}/mlperf_mixtral8x7b_calibration_dataset_1k.pkl | grep ${CALIBRATION_DATASET_MD5}
    if [ $? -ne 0 ]; then
        echo "md5sum of the data file mismatch. Should be ${CALIBRATION_DATASET_MD5}"
        exit -1
    fi
else
    echo "WARNING: Please enter the MLPerf container before downloading dataset"
    echo "WARNING: Mixtral dataset is NOT downloaded! Exiting..."
    exit 0
fi
