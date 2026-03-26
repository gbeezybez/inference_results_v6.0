import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields

import os
os.environ['TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER'] = '1'
os.environ['TRTLLM_SERVER_DISABLE_GC'] = '1'
os.environ['MLPINF_HTTP_USE_COMPLETIONS'] = '1'

# =============================================================================
# Dynamo Disaggregated Serving - Generation (Decode) Worker Configuration
# =============================================================================
# This config is for Dynamo generation workers with TP=16 (DEP16).
# Launch with: srun --ntasks=16 --nodes=4 make run_llm_server RUN_ARGS="...
#              --core_type=disagg_decode --config_id=dynamo_generation ..."

dynamo_generation = {
    # Data paths
    llm_fields.llm_gen_config_path: 'code/deepseek-r1/tensorrt/generation_config.json',
    harness_fields.tensor_path: 'build/preprocessed_data/deepseek-r1/',

    # Loadgen settings
    loadgen_fields.min_duration: 600000,
    loadgen_fields.min_query_count: 26328,
    llm_fields.warmup_iterations: 0,
    llm_fields.use_token_latencies: True,

    # Build flags for generation workers
    llm_fields.trtllm_build_flags: {
        'max_beam_width': 1,
        'kv_cache_type': 'paged',
        'remove_input_padding': 'enable',
        'multiple_profiles': 'enable',
        'use_fused_mlp': 'enable',
        'context_fmha': 'enable',
        'max_num_tokens': 6144,
        'max_input_len': 3200,
        'max_seq_len': 23200,  # Generation workers need full output length
        'use_fp8_context_fmha': 'enable',
        'use_paged_context_fmha': 'enable',
        'enable_attention_dp': True,
    },

    # Runtime flags for generation workers - from Dynamo sbatch gen_config.yaml
    llm_fields.trtllm_runtime_flags: {
        'exclude_input_from_output': True,
        'use_inflight_batching': True,
        'max_batch_size': 128,  # GB300 uses mbs=128
        'max_num_tokens': 6144,
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
        'kvcache_free_gpu_mem_frac': 0.9,
        'enable_chunked_context': False,
        # CUDA graphs for decode - extended for mbs=128
        'cuda_graph_batch_sizes': [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128],
        'cuda_graph_padding_enabled': True,
        # MOE config - CUTEDSL backend with DEP16
        'moe_backend': 'CUTEDSL',
        'use_low_precision_moe_combine': True,
        # Attention DP settings
        'enable_lm_head_tp_in_adp': True,
        # Cache transceiver for receiving KV cache from context workers
        'cache_transceiver_max_tokens': 4608,
        'cache_transceiver_backend': 'UCX',
        # Streaming and post-processing
        'stream_interval': 100,
        # Speculative decoding (MTP)
        'speculative_decoding': {
            'decoding_type': 'MTP',
            'num_nextn_predict_layers': 3,
        },
    },
    harness_fields.use_graphs: True,

    # Precision
    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32',

    # Parallelism - TP=16 for generation workers (4 nodes x 4 GPUs)
    llm_fields.tensor_parallelism: 16,
    llm_fields.pipeline_parallelism: 1,
    llm_fields.moe_expert_parallelism: 16,

    harness_fields.vboost_slider: 1,
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): dynamo_generation,
}

# Alias for dynamo_endpoint orchestrator
dynamo_decode = dynamo_generation

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": dynamo_generation,
        "dynamo_generation": dynamo_generation,
        "dynamo_decode": dynamo_decode,  # Alias for dynamo_endpoint orchestrator
    },
}
