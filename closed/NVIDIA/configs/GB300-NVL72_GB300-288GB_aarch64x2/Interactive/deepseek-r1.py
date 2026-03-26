# DeepSeek-R1 GB300 DEP2 Worker Configuration
# Used by disaggregated prefill workers (TP=2)

import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields

import os
os.environ['TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER'] = '1'
os.environ['TRTLLM_SERVER_DISABLE_GC'] = '1'

# DEP2 prefill worker config
base = {
    llm_fields.llm_gen_config_path: 'code/deepseek-r1/tensorrt/generation_config.json',
    harness_fields.tensor_path: 'build/preprocessed_data/deepseek-r1/',
    loadgen_fields.min_duration: 600000,
    loadgen_fields.min_query_count: 26328,
    llm_fields.warmup_iterations: 0,
    llm_fields.use_token_latencies: True,
    llm_fields.trtllm_build_flags: {
        'max_beam_width': 1,
        'kv_cache_type': 'paged',
        'remove_input_padding': 'enable',
        'multiple_profiles': 'enable',
        'use_fused_mlp': 'enable',
        'context_fmha': 'enable',
        'max_num_tokens': 12288,
        'max_input_len': 3140,
        'max_seq_len': 3200,
        'use_fp8_context_fmha': 'enable',
        'use_paged_context_fmha': 'enable',
        'enable_attention_dp': True,
    },
    llm_fields.trtllm_runtime_flags: {
        'exclude_input_from_output': True,
        'use_inflight_batching': True,
        'max_num_tokens': 12288,
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
        'kvcache_free_gpu_mem_frac': 0.85,
        'enable_chunked_context': False,
        'max_concurrency': 5120,
        'moe_backend': 'CUTEDSL',
        'cache_transceiver_max_tokens': 4608,
        'cache_transceiver_backend': 'UCX',
    },
    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    harness_fields.use_graphs: False,  # No CUDA graphs for prefill
    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32',
    model_fields.gpu_batch_size: {
        'deepseek-r1': 128,
    },
    llm_fields.tensor_parallelism: 2,
    llm_fields.pipeline_parallelism: 1,
    llm_fields.moe_expert_parallelism: 2,
    harness_fields.vboost_slider: 1,
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): base,
}

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": base,
    },
}
