# Qwen3-VL-235B-A22B

This directory contains the source code for NVIDIA's submission towards the
[vision-language model (VLM) benchmark](https://github.com/mlcommons/inference/tree/master/multimodal/qwen3-vl)
in the MLPerf Inference Benchmark Suite, starting from the v6.0 round.

## Building the container image

You can leverage [scripts/build_image.sh](scripts/build_image.sh) to build a container
image end-to-end for running this benchmark. At the
[closed/NVIDIA/code/qwen3-vl-235b-a22b/vllm](closed/NVIDIA/code/qwen3-vl-235b-a22b/vllm)
directory (i.e., where this `README.md` is), run the following command:

```bash
bash scripts/build_image.sh
```

If you would like to run the benchmark on a `amd64` (i.e., x86) system, you would need
to build the image on an `amd64` (i.e., x86) system. Conversely, you would need to build
the image on an `arm64` (i.e., `aarch64`) system for running the benchmark on a `arm64`
(i.e., `aarch64`) system.

Building the vLLM base image can be intensive on CPU and host memory resources. We
recommend to build the image on a machine with at least 72 CPU threads and 574 GB of
host memory.

## NVFP4 Quantization with ModelOpt

1. Install ModelOpt from [this branch](https://github.com/NVIDIA/Model-Optimizer.git) in developer mode ```pip install -e .[dev]```
2. Navigate to post-training-quantization script at ```examples/llm_ptq``` and run the following example command
```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-235B-A22B-Instruct

python hf_ptq.py \
    --pyt_ckpt_path=$MODEL_NAME_OR_PATH \
    --export_path=/Path/to/Output \
    --qformat=nvfp4 \
    --kv_cache_qformat=fp8 \
    --calib_size=256 \
    --batch_size=24 \
    --trust_remote_code \
    --export_fmt=hf
```

> [!NOTE]
> 1. This script is device memory demanding, which requires at least 8*B200 to run.

## Launch a benchmarking run

### Local 4xGB300 node

Start the container in the interactive mode via:

```bash
docker run --gpus all --rm -it --ipc=host -v $(pwd -P):/mlperf-inf-mm-q3vl-nv -w /mlperf-inf-mm-q3vl-nv \
  gitlab-master.nvidia.com:5005/mlpinf/mlperf-inference/mlperf-inf-mm-q3vl-nv:arm64_cuda13.0.1_CentML_dynamo-mlperf-inf-mm-q3vl-v6.0_CentML_vllm-mlperf-inf-mm-q3vl-v6.0
```

Runn the offline scenario, performance only mode:

```bash
bash scripts/benchmarks/local/gb300-nvl4/offline-perf.sh
```

### Local 8xB300 node

Start the container in the interactive mode via:

```bash
docker run --gpus all --rm -it --ipc=host -v $(pwd -P):/mlperf-inf-mm-q3vl-nv -w /mlperf-inf-mm-q3vl-nv \
  gitlab-master.nvidia.com:5005/mlpinf/mlperf-inference/mlperf-inf-mm-q3vl-nv:amd64_cuda13.0.1_CentML_dynamo-mlperf-inf-mm-q3vl-v6.0_CentML_vllm-mlperf-inf-mm-q3vl-v6.0
```

Offline scenario, performance only mode:

```bash
bash scripts/benchmarks/local/8xb300/offline-perf.sh
```

### Local 8xH200 node

Start the container in the interactive mode via:

```bash
docker run --gpus all --rm -it --ipc=host -v $(pwd -P):/mlperf-inf-mm-q3vl-nv -w /mlperf-inf-mm-q3vl-nv \
  gitlab-master.nvidia.com:5005/mlpinf/mlperf-inference/mlperf-inf-mm-q3vl-nv:amd64_cuda13.0.1_CentML_dynamo-mlperf-inf-mm-q3vl-v6.0_CentML_vllm-mlperf-inf-mm-q3vl-v6.0
```

Offline scenario, performance only mode:

```bash
bash scripts/benchmarks/local/8xh200/offline-perf.sh
```

## Profile with Nsight Systems

To collect Nsight Systems traces on vLLM for this benchmark, you can leverage the
`mlperf-inf-mm-q3vl benchmark nv vllm-profiler` command:

1. Wrap the command with `nsys profile --options`.
2. Set `--vllm.profile True` as a CLI flag for `mlperf-inf-mm-q3vl benchmark nv vllm-profiler`
   to ask vLLM to start/stop the profiler.
3. Pass vLLM supported profiling flags to control the capture range. Check
   [vLLM Profile Documentation](https://docs.vllm.ai/en/stable/contributing/profiling/#openai-server) for more information.

### Example

If you want to profile a benchmark run where:
- It runs the performance only mode in the server scenario where the target QPS is 10
  requests per second.
- The maximum number of batched tokens is 32768 (therefore, you want to capture a CUDA
  graph that can support up to 32768 tokens).
- All 4 GPUs on a single node are configured in a fully tensor parallel fashion.
- Prefix caching across requests are disabled (this is required by the
  [MLPerf Inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#94-llm-benchmarks)).
- The `nsys` trace is captured starting from the 1000-th to the 1100-th `EngineCore`
  iterations.
The commands (inside the container) would look like the following:

```
# Set `HF_TOKEN` to the HuggingFace access token that you would like to use to access
# your model checkpoint (in this case, `nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.0`).
export HF_TOKEN=...

export VLLM_NVTX_SCOPES_FOR_PROFILING=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_FLASHINFER_MOE_BACKEND=latency
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=$((6 * 256 * 1024 * 1024))

nsys profile \
    --wait primary \
    -f true \
    -o test_output \
    --gpu-metrics-devices=all \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end repeat \
    --trace cuda,nvtx \
    mlperf-inf-mm-q3vl benchmark nv vllm-profiler \
    --settings.test.scenario server \
    --settings.test.mode performance_only \
    --settings.test.server_target_qps 10 \
    --vllm.profile True \
    --vllm.model.repo_id=nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.0 \
    --vllm.model.revision=main \
    --vllm.model.token ${HF_TOKEN} \
    --vllm.cli=--async-scheduling \
    --vllm.cli=--max-model-len=32768 \
    --vllm.cli=--max-num-seqs=1024 \
    --vllm.cli=--max-num-batched-tokens=32768 \
    --vllm.cli=--compilation-config='{
        "max_cudagraph_capture_size": 32768,
        "cudagraph_capture_sizes": [
            1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
            136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248,
            256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480,
            496, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 16384, 32768
        ]
    }' \
    --vllm.cli=--limit-mm-per-prompt.video=0 \
    --vllm.cli=--tensor-parallel-size=4 \
    --vllm.cli=--no-enable-prefix-caching \
    --vllm.cli=--enable-layerwise-nvtx-tracing \
    --vllm.cli=--enable-logging-iteration-details \
    --vllm.cli=--profiler-config.profiler=cuda \
    --vllm.cli="--profiler-config.delay_iterations=1000" \
    --vllm.cli="--profiler-config.max_iterations=100"
```

## Run through Harness

This repo integrate a highly automated workflow to get benchmarking results through harness.

### Docker building
*** Via docker only ***

```bash
cd mlperf-inference/closed/NVIDIA \
    && make -f Makefile.docker prebuild_q3vl
```
This command builds a vllm based container image and all mlperf-inference related tools and launches the result image with correct mounting for you.

When running on a slurm based cluster where docker is not avaibable, follow the steps below:
1. Clone this repo locally and make sure the submodules `closed/NVIDIA/3rdparty/mitten` and `closed/NVIDIA/3rdparty/mlc-inference` are initialized. (Example command `git submodule update --init 3rdparty/mitten`)
2. Pull our pre-built image from  `registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/v6.0-jan29-q3vl-aarch64:latest`
3. Mount the `mlperf-inference/closed/NVIDIA` folder to `/work` when launching the container

### Example Command

####Offline Scnario
```angular2html
srun -N1 --mpi=pmix  --ntasks-per-node=4  --container-image=registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/v6.0-jan27-q3vl-aarch64:latest \
--container-mounts mlperf-inference/closed/NVIDIA:/work \
--pty bash -c 'make run_harness RUN_ARGS="--benchmark=q3vl --benchmarks=qwen3-vl-235b-a22b \
            --scenarios=Offline  \
            --test_mode=PerformanceOnly"'
```

####Server Scnario

```angular2html
srun -N1 --mpi=pmix  --ntasks-per-node=1  --container-image=registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/v6.0-jan27-q3vl-aarch64:latest \
--container-mounts mlperf-inference/closed/NVIDIA:/work \
--pty bash -c 'make run_harness RUN_ARGS="--benchmark=q3vl --benchmarks=qwen3-vl-235b-a22b \
            --scenarios=Server  \
            --test_mode=PerformanceOnly"'
```
***Note***: Set `--ntasks-per-node` to be number of GPUs you want to use for Offline scenario science the most performant config here is DataParallel via multiple vllm server intances, which are managed by [Dynamo](https://github.com/ai-dynamo/dynamo) and launched with mpirun.
