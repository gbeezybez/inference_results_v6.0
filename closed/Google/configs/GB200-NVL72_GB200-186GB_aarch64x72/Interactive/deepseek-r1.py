# DeepSeek-R1 GB200 x72 Configuration for Disaggregated Serving
# Full 72-GPU system: 2 CTX DEP4 + 4 GEN DEP16 = 8 + 64 = 72 GPUs
# WIDEEP DS-R1 best config: 2ctxDEP4_4genDEP16_mbs64, qps=64
#
# Worker TRT-LLM configs are in worker system directories:
# - configs/GB200-NVL72_GB200-186GB_aarch64x4/Interactive/disagg_prefill/deepseek-r1.yml (CTX workers, DEP4)
# - configs/GB200-NVL72_GB200-186GB_aarch64x16/Interactive/disagg_decode/deepseek-r1.yml (GEN workers, DEP16)

import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields

import os
os.environ['TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER'] = '1'
os.environ['TRTLLM_SERVER_DISABLE_GC'] = '1'
os.environ['MLPINF_HTTP_USE_COMPLETIONS'] = '1'

# x72 scale configuration (2 CTX DEP4 + 4 GEN DEP16 = 8 + 64 = 72 GPUs)
dynamo_endpoint = {
    # Harness settings
    llm_fields.llm_gen_config_path: 'code/deepseek-r1/tensorrt/generation_config.json',
    harness_fields.tensor_path: 'build/preprocessed_data/deepseek-r1/',
    loadgen_fields.min_duration: 600000,
    loadgen_fields.min_query_count: 26328,
    loadgen_fields.server_target_qps: 64,
    llm_fields.warmup_iterations: 0,
    llm_fields.use_token_latencies: True,
    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32',
    llm_fields.trtllm_runtime_flags: {
        'exclude_input_from_output': True,
        'use_inflight_batching': True,
        'max_concurrency': 5120,
    },
    harness_fields.vboost_slider: 1,

    # Disaggregated cluster topology
    llm_fields.dynamo_cluster: {
        'num_prefill_workers': 2,
        'num_decode_workers': 4,
        'num_frontends': 4,
        'gpus_per_node': 4,
        'frontend': {
            'router_mode': 'kv',
            'kv_overlap_weight': 1.0,
        },
        'prefill': {
            'system': 'GB200-NVL72_GB200-186GB_aarch64x4',
            'trtllm_yml_override': '/work/configs/GB200-NVL72_GB200-186GB_aarch64x4/Interactive/disagg_prefill/deepseek-r1.yml',
            'env_vars': 'configs/GB200-NVL72_GB200-186GB_aarch64x4/Interactive/disagg_prefill/dsr1_env.yaml',
        },
        'decode': {
            'system': 'GB200-NVL72_GB200-186GB_aarch64x16',
            'trtllm_yml_override': '/work/configs/GB200-NVL72_GB200-186GB_aarch64x16/Interactive/disagg_decode/deepseek-r1.yml',
            'env_vars': 'configs/GB200-NVL72_GB200-186GB_aarch64x16/Interactive/disagg_decode/dsr1_env.yaml',
        },
    },
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): dynamo_endpoint,
}

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": dynamo_endpoint,
        "dynamo_endpoint": dynamo_endpoint,
    },
}
