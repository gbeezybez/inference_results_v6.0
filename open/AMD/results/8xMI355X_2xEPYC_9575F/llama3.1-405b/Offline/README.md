# MLPerf Inference 6.0

## Setup

### Model and Dataset

Download the dataset for the benchmark by running the below command

```bash
bash setup/llama3.1-405b/download_dataset.sh
```

Download the model for the benchmark by running the below command

```bash
HUGGINGFACE_ACCESS_TOKEN="<your HF token goes here>"
bash setup/llama3.1-405b/download_model_fp4.sh --token $HUGGINGFACE_ACCESS_TOKEN --download-prequantized
```

## Inference

### Runtime tunables

To boost the machine's performance further, execute the following script before any performance test (should be set once after a reboot):

```bash
bash setup/runtime_tunables.sh
```

### Docker

Build the docker image for the benchmark by running the below command

For the offline scenario, use
```bash
bash setup/llama3.1-405b/build_docker.sh vllm_offline_pruned_gfx9xx.config
```

For the interactive scenario, use
```bash
bash setup/llama3.1-405b/build_docker.sh vllm_interactive_gfx9xx.config
```

Start the docker container for the benchmark by running the below commands

For the offline scenario, use

```bash
export EXTRA_ARGS="--rm --workdir /lab-mlperf-inference/submission"
bash setup/llama3.1-405b/start_docker.sh vllm_offline_pruned_gfx9xx.config
```

For the interactive scenario, use

```bash
export EXTRA_ARGS="--rm --workdir /lab-mlperf-inference/submission"
bash setup/llama3.1-405b/start_docker.sh vllm_interactive_gfx9xx.config
```

### Pruning the model

Llama3.1-405B has 126 layers, where each layer is architecturally the same. However, we have identified that some of the layers are less important to the output in comparison to the other layers. Using the calibration dataset, we measured the sum of the magnitude of the outputs of each of the transformer layers just before the finaly rmsnorm layer. 
$L_i = sum(abs(output_i))$
 where, $L_i$ is the importance of the $i^{th}$ layer and $output_i$ is the tensor of outputs of the transformer layers feed-forward network (just before the rmsnorm layer)

Based on $L_i$ of a small number of samples in the calibration dataset (we used 170 samples in our experiments), we drop several contiguous layers. This allows us to prune the model with higher number of layers while having minimum impact to the accuracy of the model output.


```
export $GIT_ROOT=<path_to_the_code_folder>
export $MODEL_PATH=<path_to_the_model>
python3 $GIT_ROOT/scripts/drop_layers.py --model $MODEL_PATH --initial 59 --final 89
```

The pruned model is saved in the folder  ```$MODEL_PATH/pruned_59_89/```

Note: what this script does in the above example is to prune layers of the model from layer 59 to layer 89 and save the remaining layers of the model into a folder named pruned_59_89. Note that we need to link the model path to this pruned model in vLLM configurations, i.e., in the offline_mi355x_pruned.yaml file in this case.

### Running the benchmark

Run the following commands inside the docker container

``` bash
## Performance
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/llama3_1-405b/ \
   --config-name offline_mi355x_pruned \
   test_mode=performance \
   harness_config.device_count=8 \
   harness_config.user_conf_path=/lab-mlperf-inference/code/llama3_1-405b/user_mi355x_pruned.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/llama3_1-405b/Offline/performance/run_1

## Accuracy
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/llama3_1-405b/ \
   --config-name offline_mi355x_pruned \
   test_mode=accuracy \
   harness_config.device_count=8 \
   harness_config.user_conf_path=/lab-mlperf-inference/code/llama3_1-405b/user_mi355x_pruned.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/llama3_1-405b/Offline/accuracy

### Evaluate accuracy
bash /lab-mlperf-inference/code/scripts/setup_llama3_accuracy_env.sh
bash /lab-mlperf-inference/code/scripts/check_llama3_accuracy_scores.sh \
   /lab-mlperf-inference/results/llama3_1-405b/Offline/accuracy/mlperf_log_accuracy.json
```
