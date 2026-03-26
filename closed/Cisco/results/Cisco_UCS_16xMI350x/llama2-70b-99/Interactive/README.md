# MLPerf Inference 6.0

## Setup

### Model and Dataset

Download the dataset for the benchmark by running the below command

```bash
bash setup/llama2-70b-99/download_dataset.sh
```

Download the model for the benchmark by running the below command

```bash
HUGGINGFACE_ACCESS_TOKEN="<your HF token goes here>"
bash setup/llama2-70b-99/download_model_fp4.sh --token $HUGGINGFACE_ACCESS_TOKEN
```

## Inference

### Runtime tunables

To boost the machine's performance further, execute the following script before any performance test (should be set once after a reboot):

```bash
bash setup/runtime_tunables.sh
```

### Docker

Build the docker image for the benchmark by running the below command

```bash
bash setup/llama2-70b-99/build_docker.sh
```

Start the docker container for the benchmark by running the below commands

```bash
export EXTRA_ARGS="--rm --workdir /lab-mlperf-inference/submission"
bash setup/llama2-70b-99/start_docker.sh
```


### Running the benchmark

Run the following commands inside the docker container

``` bash
## Performance
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/llama2-70b-99/ \
   --config-name interactive_mi350x \
   test_mode=performance \
   harness_config.device_count=8 \
   harness_config.user_conf_path=/lab-mlperf-inference/code/llama2-70b-99/user_mi350x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b-99/Interactive/performance/run_1

## Accuracy
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/llama2-70b-99/ \
   --config-name interactive_mi350x \
   test_mode=accuracy \
   harness_config.device_count=8 \
   harness_config.user_conf_path=/lab-mlperf-inference/code/llama2-70b-99/user_mi350x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b-99/Interactive/accuracy

### Evaluate accuracy
bash /lab-mlperf-inference/code/scripts/setup_llama2_accuracy_env.sh
bash /lab-mlperf-inference/code/scripts/check_llama2_accuracy_scores.sh \
   /lab-mlperf-inference/results/llama2-70b-99/Interactive/accuracy/mlperf_log_accuracy.json
```
