import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields

import os
os.environ['TRTLLM_SERVER_DISABLE_GC'] = '1'
os.environ['OMPI_MCA_coll_ucc_enable'] = '0'  # Hitting "Deserialization failed: invalid load key"
os.environ['TLLM_NUMA_AWARE_WORKER_AFFINITY'] = '1'  # perf improved from 180->187tps/gpu
os.environ['CUDA_SCALE_LAUNCH_QUEUES'] = '4x'  # perf improved from 152->177 along with cuda graph and mnt1024 changes
os.environ['TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED'] = '1'
os.environ['TRTLLM_LLAMA_EAGER_FUSION_DISABLED'] = '1'
os.environ['TRTLLM_DISABLE_NVFP4_LAYERNORM_FUSION'] = '0'

harness_config = {
    model_fields.input_dtype: 'int32',
    llm_fields.llm_gen_config_path: 'code/llama3_1-405b/tensorrt/generation_config.json',
    loadgen_fields.min_duration: 600000,
    model_fields.precision: 'fp4',
    loadgen_fields.server_target_qps: 19.8,
    loadgen_fields.min_query_count: 74808,
    harness_fields.tensor_path: 'build/preprocessed_data/llama3.1-405b/',
    llm_fields.traffic_distribution_policy: 'isl_load_balancing',
    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    llm_fields.trtllm_runtime_flags: {
        'max_concurrency': 256,
    },
    llm_fields.use_token_latencies: True,
    llm_fields.readiness_timeout: 600,
}
disagg_config = {
    model_fields.input_dtype: 'int32',
    llm_fields.llm_gen_config_path: 'code/llama3_1-405b/tensorrt/generation_config.json',
    loadgen_fields.min_duration: 600000,
    model_fields.precision: 'fp4',
    loadgen_fields.server_target_qps: 20.5,
    loadgen_fields.min_query_count: 74808,
    harness_fields.tensor_path: 'build/preprocessed_data/llama3.1-405b/',
    llm_fields.traffic_distribution_policy: 'isl_load_balancing',
    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    llm_fields.trtllm_runtime_flags: {
        'max_concurrency': 256,
    },
    llm_fields.use_token_latencies: True,
    llm_fields.readiness_timeout: 900,
}

# x72 scale configuration (14 CTX DEP4 + 2 GEN DEP8 = 56 + 16 = 72 GPUs)
dynamo_cluster = {
    # Harness settings
    model_fields.input_dtype: 'int32',
    llm_fields.llm_gen_config_path: 'code/llama3_1-405b/tensorrt/generation_config.json',
    loadgen_fields.min_duration: 600000,
    loadgen_fields.min_query_count: 49878,
    loadgen_fields.server_target_qps: 22.4,
    llm_fields.warmup_iterations: 0,
    model_fields.precision: 'fp4',
    harness_fields.tensor_path: 'build/preprocessed_data/llama3.1-405b/',
    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    llm_fields.trtllm_runtime_flags: {
        'max_concurrency': 2048,
    },
    llm_fields.use_token_latencies: True,
    llm_fields.readiness_timeout: 600,

    # Disaggregated cluster topology
    llm_fields.dynamo_cluster: {
        'num_prefill_workers': 14,
        'num_decode_workers': 2,
        'num_frontends': 4,
        'gpus_per_node': 4,
        'frontend': {
            'router_mode': 'kv',
            'kv_overlap_weight': 1.1,
        },
        'prefill': {
            'system': 'GB200-NVL72_GB200-186GB_aarch64x4',
            'trtllm_yml_override': '/work/configs/GB200-NVL72_GB200-186GB_aarch64x4/Server/disagg_prefill/llama3_1-405b.yaml',
            'env_vars': 'configs/GB200-NVL72_GB200-186GB_aarch64x4/Server/disagg_prefill/llama3_1-405b-env.yaml',
        },
        'decode': {
            'system': 'GB200-NVL72_GB200-186GB_aarch64x8',
            'trtllm_yml_override': '/work/configs/GB200-NVL72_GB200-186GB_aarch64x8/Server/disagg_decode/llama3_1-405b.yaml',
            'env_vars': 'configs/GB200-NVL72_GB200-186GB_aarch64x8/Server/disagg_decode/llama3_1-405b-env.yaml',
        },
    },

}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): harness_config,
}

# Atomic configs for leader mode with multiple variants
# Usage: --mpi_mode=leader --config_id=<variant_name>
ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": harness_config,
        "disagg": disagg_config,
        "dynamo_cluster": dynamo_cluster,
    },
}
