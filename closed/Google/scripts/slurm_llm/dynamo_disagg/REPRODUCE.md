# MLPerf Inference Dynamo Disaggregated Serving - Reproduction Commands

This document contains commands to reproduce MLPerf Inference results using the Dynamo disaggregated serving framework.

## Prerequisites

1. Allocate nodes via SLURM:
   ```bash
   salloc --nodes=18 --partition=<partition_name>
   ```

2. Activate the disagg virtual environment (on login node):
   ```bash
   source /path/to/disagg_venv/bin/activate
   ```

3. Run commands from the workspace directory:
   ```bash
   cd /path/to/mlperf-inference/closed/NVIDIA
   ```

## DeepSeek-R1

### GB200 x72
Configuration: 2 CTX (DEP4) + 4 GEN (DEP16) = 72 GPUs

```bash
salloc --nodes=18 --partition=36x2-a01r # salloc --nodes=18 --partition=gb200
python3 -u scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \
    --system GB200-NVL72_GB200-186GB_aarch64x72 \
    --benchmark deepseek-r1 --scenario Interactive \
    --container-image /lustre/share/coreai_mlperf_inference/krish/images/dynamo-v0.8.0-mlperf-trtllm-1.2-mlpinf-b5ddff4-jan28-mlperf.sqsh \
    --storage-path /home/mlperf_inference_storage \
    --server-init-delay 1800 \
    --run-harness-args "--benchmarks=deepseek-r1 --scenarios=Interactive --test_mode=PerformanceOnly --server_target_qps=64 --min_query_count=144000" \
    --verbose
```

### GB300 x72
Configuration: 4 CTX (DEP2) + 4 GEN (DEP16) = 72 GPUs

```bash
salloc --nodes=18 --partition=gb300
python3 -u scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \
    --system GB300-NVL72_GB300-288GB_aarch64x72 \
    --benchmark deepseek-r1 --scenario Interactive \
    --container-image /lustre/share/coreai_mlperf_inference/krish/images/dynamo-v0.8.0-mlperf-trtllm-1.2-mlpinf-b5ddff4-jan28-mlperf.sqsh \
    --run-harness-args "--benchmarks=deepseek-r1 --scenarios=Interactive --test_mode=PerformanceOnly --server_target_qps=68 --min_query_count=244800" \
    --server-init-delay 1800 \
    --verbose
```
