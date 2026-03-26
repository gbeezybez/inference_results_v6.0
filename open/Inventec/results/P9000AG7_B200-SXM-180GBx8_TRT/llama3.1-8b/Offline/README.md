# Llama3.1 8B

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

The instruction for downloading the llama3.1 8b model could be found at [MLCommons Members Download (Recommended for official submission)](https://github.com/mlcommons/inference/tree/master/language/llama3.1-8b#mlcommons-members-download-recommended-for-official-submission).  Note the downloading would require a MLCommons Member associated account.

For Inventec AI Lab, the model has been downloaded and stored at `/hps/data/mlperf_inference/llama31/llama3-1-8b-instruct.uri/`.  To copy the model for benchmarking, _exit the container_ and do the following:

   ```bash
   mkdir -p ${MLPERF_SCRATCH_PATH}/models/Llama3.1-8B
   cp -r /hps/data/mlperf_inference/llama31/llama3-1-8b-instruct.uri ${MLPERF_SCRATCH_PATH}/models/Llama3.1-8B/Meta-Llama-3.1-8B-Instruct
   ```

_Enter the mlperf container_ again for the rest of the steps.

   ```bash
   cd ${HOME}/inference_results_v5.1/closed/Inventec
   make prebuild
   ```

Untar the quantized checkpoint packaged within the container:

   ```bash
   mkdir -p build/models/Llama3.1-8B/fp4-quantized-modelopt
   tar -xzf /opt/fp4-quantized-modelopt/llama3_1-8b-instruct-hf-torch-fp4.tar.gz -C build/models/Llama3.1-8B/fp4-quantized-modelopt/
   ```

### Download and Prepare Data

The instruction for downloading the dataset could be found at [Full dataset (datacenter)](https://github.com/mlcommons/inference/tree/master/language/llama3.1-8b#full-dataset-datacenter) and [Calibration](https://github.com/mlcommons/inference/tree/master/language/llama3.1-8b#calibration).

Do the following within the container to download and preprocess the data.

   ```bash
   mkdir -p build/data/llama3.1-8b
   bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
     -d build/data/llama3.1-8b https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri
   bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
     -d build/data/llama3.1-8b https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-dailymail-calibration.uri
   
   python3 code/llama3.1-8b/tensorrt/preprocess_data.py --data_dir build/data/ --preprocessed_data_dir build/preprocessed_data
   ```

Make sure after the steps above, you have:

1. model downloaded at: `build/models/Llama3.1-8B/Meta-Llama-3.1-8B-Instruct/`
2. preprocessed data at `build/preprocessed_data/llama3.1-8b/`:
   - `build/preprocessed_data/llama3.1-8b/input_lens.npy`
   - `build/preprocessed_data/llama3.1-8b/input_ids_padded.npy`
   - `build/preprocessed_data/llama3.1-8b/mlperf_llama3.1-8b_calibration_1k/data.parquet`

## Login Hugging Face

The benchmark code requires you to be logged in (authed) in Hugging Face and granted access to "meta-llama/Meta-Llama-3-8B-Instruct".  So, go to [https://huggingface.co/meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) and request access with your registered account.  Once access is granted, install Hugging Face CLI and log in with your own token (represented as $HF_TOKEN below).

   ```bash
   curl -LsSf https://hf.co/cli/install.sh | bash
   hf auth login --token $HF_TOKEN
   ```

## Build and run the benchmarks

Please follow the steps below in the mlperf container.  The build step would build code for all mlperf inference tasks, including code for other benchmarks.  If you are not running this benchmark for the first time and you did not make any modifications to the code since last `make build`, it suffices to just run `make build_loadgen` to save time.

   ```bash
   make build
   ```

Start the llm server.  Wait for a few seconds for the server to spin up (see logs).

   ```bash
   make run_llm_server RUN_ARGS="--core_type=trtllm_endpoint --benchmarks=llama3.1-8b --scenarios=Offline"
   ```

Validate accuracy of the optimized llama3.1-8b model.

   ```bash
   make run_harness RUN_ARGS="--core_type=trtllm_endpoint --benchmarks=llama3.1-8b --scenarios=Offline --test_mode=AccuracyOnly"
   ```

Run the benchmark.

   ```bash
   make run_harness RUN_ARGS="--core_type=trtllm_endpoint --benchmarks=llama3.1-8b --scenarios=Offline"
   ```

Based on [NVIDIA's recommendation](https://github.com/jkjung-avt/inference_results_v5.1/tree/main/closed/Inventec#do-i-need-to-restart-llm-server-each-time-before-running-benchmarks), it's better to exit and re-enter the mlperf container, and then restart the server for each performance run.  This ensures GPU is in a clean state with no residual memory from past runs.  It is okay to re-use server for accuracy/audit runs after performance runs.
