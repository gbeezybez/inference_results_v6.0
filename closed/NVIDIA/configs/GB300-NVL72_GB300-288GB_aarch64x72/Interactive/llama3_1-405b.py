import os

import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields
import code.fields.triton as triton_fields

os.environ['TLLM_NUMA_AWARE_WORKER_AFFINITY'] = '1'
os.environ['CUDA_SCALE_LAUNCH_QUEUES'] = '4x'
os.environ['OMPI_MCA_hwloc_base_binding_policy'] = 'none'
os.environ['OMPI_MCA_rmaps_base_inherit'] = '1'
os.environ['TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED'] = '1'
os.environ['TRTLLM_LLAMA_EAGER_FUSION_DISABLED'] = '1'
os.environ['TRTLLM_DISABLE_NVFP4_LAYERNORM_FUSION'] = '0'


harness_config = {
    model_fields.input_dtype: 'int32',
    llm_fields.llm_gen_config_path: 'code/llama3_1-405b/tensorrt/generation_config.json',
    loadgen_fields.min_duration: 600000,
    loadgen_fields.min_query_count: 33252,
    loadgen_fields.server_target_qps: 26,
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
}

# x72 scale configuration (26 CTX DEP2 + 5 GEN DEP4 = 52 + 20 = 72 GPUs)
dynamo_cluster = {
    # Harness settings
    model_fields.input_dtype: 'int32',
    llm_fields.llm_gen_config_path: 'code/llama3_1-405b/tensorrt/generation_config.json',
    loadgen_fields.min_duration: 600000,
    loadgen_fields.min_query_count: 33252,
    loadgen_fields.server_target_qps: 29.3,
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
        'num_prefill_workers': 26,
        'num_decode_workers': 5,
        'num_frontends': 8,
        'gpus_per_node': 4,
        'frontend': {
            'router_mode': 'kv',
            'kv_overlap_weight': 1.1,
        },
        'prefill': {
            'system': 'GB300-NVL72_GB300-288GB_aarch64x2',
            'trtllm_yml_override': '/work/configs/GB300-NVL72_GB300-288GB_aarch64x2/Interactive/disagg_prefill/llama3_1-405b.yaml',
            'env_vars': 'configs/GB300-NVL72_GB300-288GB_aarch64x2/Interactive/disagg_prefill/llama3_1-405b-env.yaml',
        },
        'decode': {
            'system': 'GB300-NVL72_GB300-288GB_aarch64x4',
            'trtllm_yml_override': '/work/configs/GB300-NVL72_GB300-288GB_aarch64x4/Interactive/disagg_decode/llama3_1-405b.yaml',
            'env_vars': 'configs/GB300-NVL72_GB300-288GB_aarch64x4/Interactive/disagg_decode/llama3_1-405b-env.yaml',
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
        "dynamo_cluster": dynamo_cluster,
    },
}
