# Llama2 70b

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

The instruction for downloading the llama2 70b model could be found at [Download model through MLCFlow Automation](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/README.md#download-model-through-mlcflow-automation).  Note the downloading would require a MLCommons Member associated account.

For Inventec AI Lab, the model has been downloaded and stored at `/hps/data/mlperf_inference/llama2/llama-2-70b-chat-hf.uri/`.  To copy the model for benchmarking, _exit the container_ and do the following:


   ```bash
   mkdir -p ${MLPERF_SCRATCH_PATH}/models/Llama2
   cp -r /hps/data/mlperf_inference/llama2/llama-2-70b-chat-hf.uri ${MLPERF_SCRATCH_PATH}/models/Llama2/Llama-2-70b-chat-hf
   ```

### Download and Prepare Data

The instruction for downloading the dataset could be found at [Preprocessed](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/README.md#preprocessed).

   ```bash
   cd ${MLPERF_SCRATCH_PATH}/data
   bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://inference.mlcommons-storage.org/metadata/llama-2-70b-open-orca-dataset.uri
   mv open_orca llama2-70b
   cd llama2-70b
   gzip -dk open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz
   gzip -dk open_orca_gpt4_tokenized_llama.calibration_1000.pkl.gz
   ```

Enter the mlperf container for the rest of the steps.

   ```bash
   cd ${HOME}/inference_results_v5.1/closed/Inventec
   make prebuild
   ```

Run the required data pre-processing:

   ```bash
   mkdir -p build/preprocessed_data/open_orca/
   cp build/data/llama2-70b/open_orca_gpt4_tokenized_llama.sampled_24576.pkl build/preprocessed_data/open_orca/
   python3 code/llama2-70b/tensorrt/preprocess_data.py --data_dir build/data/ --preprocessed_data_dir build/preprocessed_data
   ```

Make sure after the steps above, you have:

1. model downloaded at: `build/models/Llama2/Llama-2-70b-chat-hf/`,
2. preprocessed data at `build/preprocessed_data/llama2-70b/`:
   - `build/preprocessed_data/llama2-70b/input_lens.npy`
   - `build/preprocessed_data/llama2-70b/input_ids_padded.npy`
   - `build/preprocessed_data/llama2-70b/mlperf_llama2_openorca_calibration_1k/data.parquet`

## Build and run the benchmarks

Please follow the steps below in the mlperf container.  The build step would build code for all mlperf inference tasks, including code for other benchmarks.  If you are not running this benchmark for the first time and you did not make any modifications to the code since last `make build`, it suffices to just run `make build_loadgen` to save time.

   ```bash
   make build
   ```

Next, do `generate_engines` which would take care of quantization/calibration of the model.

   ```bash
   make generate_engines SYSTEM_NAME=P9000AG7_B200-SXM-180GBx8 RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --config_ver=high_accuracy"
   ```

Then, do `run_harness` to validate the optimized model could achieve the required accuracy threshold.

   ```bash
   make run_harness RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --config_ver=high_accuracy --test_mode=AccuracyOnly"
   ```

You should expect to get the following results (the detailed number might be different):

   ```
   Results
   
   {'rouge1': 44.7482, 'rouge2': 22.3577, 'rougeL': 29.1381, 'rougeLsum': 42.2582, 'gen_len': 26547811, 'gen_num': 24576, 'gen_tok_len': 6717768, 'tokens_per_sample': 273.3}
   
   ======================== Result summaries: ========================
   
   Offline Scenario:
   +-------------------------------+---------------+-----------+------------------+-------------------+------------------+----------------------+
   | System Name                   | Benchmark     | Setting   | All Acc. Pass?   | Metric Name       |   Measured Value | Threshold            |
   +===============================+===============+===========+==================+===================+==================+======================+
   | P9000AG7_B200-SXM-180GBx8_TRT | llama2-70b-99 | cp990     | Yes              | ROUGE1            |            44.75 | >=43.98688799999999  |
   |                               |               |           |                  | ROUGE2            |            22.36 | >=21.814847999999998 |
   |                               |               |           |                  | ROUGEL            |            29.14 | >=28.330038          |
   |                               |               |           |                  | TOKENS_PER_SAMPLE |           273.30 | >=265.005            |
   +-------------------------------+---------------+-----------+------------------+-------------------+------------------+----------------------+
   ```

Finally, run the following command to get the benchmark number.

   ```bash
   make run_harness RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --config_ver=high_accuracy" 
   ```
