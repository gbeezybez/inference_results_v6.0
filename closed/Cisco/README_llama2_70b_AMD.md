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
# gfx942
bash setup/llama2-70b-99/download_model_fp8.sh --token $HUGGINGFACE_ACCESS_TOKEN
# gfx950
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
# gfx942
bash setup/llama2-70b-99/build_docker.sh vllm_gfx942.config
# gfx950
bash setup/llama2-70b-99/build_docker.sh vllm_gfx950.config
```

Start the docker container for the benchmark by running the below commands

```bash
export EXTRA_ARGS="--rm --workdir /lab-mlperf-inference/submission"
# gfx942
bash setup/llama2-70b-99/start_docker.sh vllm_gfx942.config
# gfx950
bash setup/llama2-70b-99/start_docker.sh vllm_gfx950.config
```


### Running the benchmark and submission packaging

We provide helper scripts to run the benchmark and create submission packages for llama2-70b ([llama2_70b.sh](./submission/llama2_70b.sh)).

The package will be generated in the `submission/inference_results_6.0` folder. This folder will contain all the results and information to recreate the results.

Run the below command in the container

```bash
# llama2_70b GPU_NAME can be mi300x/mi325x/mi355x.
# Set CPU_NAME based on your hardware you use. You can use `lscpu | grep name`.
# GPU_COUNT can be 1 or 8. 1 is only applicable for llama2, llama2-interactive.
# RESULTS can be set to the ouput for the results. By default it is set to the results directory of the CWD (/lab-mlperf-inference/submission/results).
# DIVISION can be closed or open. By default it is set to closed.
# ENABLE_POWER_SETUP is used to set GPU frequency and power state to a predetermined value for best performace. By default it is set to `1` set it to `0` to disable it.
COMPANY="<your company name>" CPU_NAME="EPYC_9575F" GPU_NAME="mi355x" GPU_COUNT=8 DIVISION=closed RESULTS="<output directory name>" ENABLE_POWER_SETUP=1 bash /lab-mlperf-inference/submission/llama2_70b.sh
```

To run the packaging script only, run the below command in the container
```bash
# The parameters are the same as above described
COMPANY="<your company name>" CPU_NAME="EPYC_9575F" GPU_NAME="mi355x" GPU_COUNT=8 RESULTS="<output directory name>" bash /lab-mlperf-inference/submission/package_submission.sh
```

### Distributed inference

Check the ZMQ's backend [README.md](code/harness_llm/backends/vllm/zmq/README.md) for details.

For packaging, it is recommended to follow [llama2_70b.sh](submission/llama2_70b.sh)'s paths (e.g. harness_config.output_log_dir=results/llama2-70b/Offline/performance/run_1).

Make sure to update [README_cmds.md](submission/README_cmds.md) to have the correct commands or update it manually in the generated package before uploading it.
