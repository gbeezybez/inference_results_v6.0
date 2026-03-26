# MLPerf Python LLM harness.

## Targeted workloads

This supports all language models in mlperf-inference and is the default choice for running workloads.
This includes:

- gptj
- llama2-70b
- mixtral-8x7b
- llama2-70b-interactive
- llama3.1-405b

## Steps to run

0. Enter the container from `closed/NVIDIA`:

```bash
make prebuild
```

1. Build and Install dependencies including TensorRT-LLM:

```bash
make build
```

1. Generate the engines:

```bash
make generate_engines RUN_ARGS="--benchmarks=gptj --scenarios=Offline[,Server] ..."
make generate_engines RUN_ARGS="--benchmarks=llama2,llama3 --scenarios=Offline[,Server] ..."
make generate_engines RUN_ARGS="--benchmarks=llama2-70b-interactive --scenarios=Offline[,Server] ..."
```

3. Run the harness.

```bash
make run_harness RUN_ARGS="--benchmarks=llama2 --scenarios=Offline[,Server] ..."
```

### Configs

Configuration is EITHER:

- configuration class in `<config>/<model>/<scenario>/__init_.py`
- `RUN_ARGS` cli flags `--trtllm_build_flags --trtllm_runtime_flags --trtllm_checkpoint_flags`

Environment variables `TRTLLM_CHECKPOINT_FLAGS`, `TRTLLM_BUILD_FLAGS`, `TRTLLM_RUNTIME_FLAGS` can then be used to override on a per-flag basis. EG:

```bash
# eg: these CHECKPOINT,BUILD,RUNTIME flags will merge/override the trtllm flags from config
TRTLLM_CHECKPOINT_FLAGS=kv_cache_dtype:fp8 TRTLLM_BUILD_FLAGS=use_fused_mlp:enabled,max_batch_size:1024 TRTLLM_RUNTIME_FLAGS=max_num_tokens:32 make run RUN_ARGS="--benchmarks=llama2-70b --scenarios=offline"

# eg: enable verbose engine-build logs with: (dumps to <engine_dir>/rank0.stdout, <engine_dir>/rank0.stderr)
TRTLLM_BUILD_FLAGS=log_level:verbose make generate_engines RUN_ARGS="--benchmarks=llama2-70b --scenarios=offline"
```

### Notes:

1. to disable a flag which is inherited from parent class, you can set value to `None`. EG:

```python

class HopperOfflineGPUBaseConfig(OfflineGPUBaseConfig):
  precision = "fp8"
  vboost_slider = 1

  trtllm_checkpoint_flags = {'kv_cache_dtype': 'fp8'}
  trtllm_runtime_flags = {'kvcache_free_gpu_mem_frac': 0.90, 'enable_chunked_context': True}
  trtllm_build_flags = {
    'tokens_per_block': 32,
    'tensor_parallelism': 1,
    'pipeline_parallelism': 1,

    'use_paged_context_fmha': 'enable',
  }

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx1(HopperOfflineGPUBaseConfig):
  system = KnownSystem.H200_SXM_141GBx1

  offline_expected_qps = 13.4
  gpu_batch_size = {'llama2-70b': 850}
  trtllm_runtime_flags = {'max_num_tokens': 1024}
  trtllm_build_flags = {
    'max_num_tokens': 1024,

    # value==None disables flags, useful for disabling inherted flags
    # ie. H200_SXM_141GBx1 will not have flag 'use_paged_context_fmha' since value==None
    'use_paged_context_fmha': None
  }
```
