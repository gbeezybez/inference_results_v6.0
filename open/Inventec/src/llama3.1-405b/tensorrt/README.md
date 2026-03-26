# Llama3.1 405B

## Getting started

### Prepare and enter the mlperf container

Note that the mlperf inference needs to be run by a non-root user, say, "franklin".  You first need to make sure the user have access to docker.  This could be done by running the following command as root (replace "franklin" with your own account name).

   ```bash
   usermod -aG docker franklin
   ```

Then logout and re-login as "franklin", and set the scratch path environment variable.  All downloaded models/data and preprocessed data would be stored at this scratch space.

   ```bash
   export MLPERF_SCRATCH_PATH=/hps/franklin/mlperf_scratch
   ```

Run the prebuild command to build and enter the mlperf container.

   ```bash
   mkdir -p ${MLPERF_SCRATCH_PATH}/models ${MLPERF_SCRATCH_PATH}/data ${MLPERF_SCRATCH_PATH}/preprocessed_data
   cd ${HOME}/inference_results_v5.1/closed/Inventec
   make prebuild
   ```

This takes a long time (~2 hours) for the first time...  You should be inside the mlperf container when it finishes.  Then do the following within the mlperf container, which links the build/ directory to the scratch space.

   ```bash
   make link_dirs
   ls -al build/
   ```

You should see an output similar to the following.  (There might be other files in the build/ directory if you have previously built some code within the container.)

   ```
   total 4
   drwxrwxr-x  2 user group 4096 Jun 24 18:49 .
   drwxrwxr-x 15 user group 4096 Jun 24 18:49 ..
   lrwxrwxrwx  1 user group   35 Jun 24 18:49 data -> $MLPERF_SCRATCH_PATH/data
   lrwxrwxrwx  1 user group   37 Jun 24 18:49 models -> $MLPERF_SCRATCH_PATH/models
   lrwxrwxrwx  1 user group   48 Jun 24 18:49 preprocessed_data -> $MLPERF_SCRATCH_PATH/preprocessed_data
   ```

### Download Model

The instruction for downloading the llama3.1 405b model could be found at [MLCommons Members Download (Recommended for official submission)](https://github.com/mlcommons/inference/tree/master/language/llama3.1-405b#mlcommons-members-download-recommended-for-official-submission).  Note the downloading would require a MLCommons Member associated account.

For Inventec AI Lab, the model has been downloaded and stored at `/hps/data/mlperf_inference/llama31/Llama-3.1-405B-Instruct`.  To copy the model for benchmarking, _exit the container_ and do the following.  This model is 2.4TB large, so this would take a while.

   ```bash
   mkdir -p ${MLPERF_SCRATCH_PATH}/models/Llama3.1-405B
   cp -r /hps/data/mlperf_inference/llama31/Llama-3.1-405B-Instruct ${MLPERF_SCRATCH_PATH}/models/Llama3.1-405B/Meta-Llama-3.1-405B-Instruct
   ```

_Enter the mlperf container_ again for the rest of the steps.

   ```bash
   cd ${HOME}/inference_results_v5.1/closed/Inventec
   make prebuild
   ```

### Download and Prepare Data

Do the following within the container to download and preprocess the data.

   ```bash
   mkdir -p build/data/llama3.1-405b/
   bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
     -d build/data/llama3.1-405b/ \
     https://inference.mlcommons-storage.org/metadata/llama3-1-405b-dataset-8313.uri
   bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
     -d build/data/llama3.1-405b/ \
     https://inference.mlcommons-storage.org/metadata/llama3-1-405b-calibration-dataset-512.uri

   python3 code/llama3_1-405b/tensorrt/preprocess_data.py --data_dir build/data/ --preprocessed_data_dir build/preprocessed_data

   mv build/preprocessed_data/llama3.1-405b/mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval build/preprocessed_data/llama3.1-405b/mlperf_llama3.1_405b_dataset_512_processed_fp16_calibration
   cp build/data/llama3.1-405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl build/preprocessed_data/llama3.1-405b/
   ```

Make sure after the steps above, you have:

1. model downloaded at: `build/models/Llama3.1-405B/Meta-Llama-3.1-405B-Instruct/`
2. preprocessed data at `build/preprocessed_data/llama3.1-405b/`:
   - `build/preprocessed_data/llama3.1-405b/input_lens.npy`
   - `build/preprocessed_data/llama3.1-405b/input_ids_padded.npy`
   - `build/preprocessed_data/llama3.1-405b/mlperf_llama3.1_405b_dataset_512_processed_fp16_calibration/data.parquet`

## Build and run the benchmarks

Please follow the steps below in the mlperf container.  The build step would build code for all mlperf inference tasks, including code for other benchmarks.  If you are not running this benchmark for the first time and you did not make any modifications to the code since last `make build`, it suffices to just run `make build_loadgen` to save time.

   ```bash
   make build
   ```

Next, do `generate_engines` which would take care of quantization/calibration of the model.

   ```bash
   make generate_engines RUN_ARGS="--benchmarks=llama3.1-405b --scenarios=Offline"
   ```

Then, do `run_harness` to validate the optimized model could achieve the required accuracy threshold.

   ```bash
   make run_harness RUN_ARGS="--benchmarks=llama3.1-405b --scenarios=Offline --test_mode=AccuracyOnly"
   ```

Finally, run the performance benchmark.

   ```bash
   make run_harness RUN_ARGS="--benchmarks=llama3.1-405b --scenarios=Offline"
   ```

TO-DO: How to run Interactive Scenario?  (Refer to run_disagg_405B/README.md?)
