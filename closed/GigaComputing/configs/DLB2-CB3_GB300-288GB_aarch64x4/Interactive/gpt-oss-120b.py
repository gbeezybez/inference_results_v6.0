import copy

import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields

import os
os.environ['TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER'] = '1'
os.environ['MLPINF_HTTP_USE_COMPLETIONS'] = '1'
os.environ['TRTLLM_SERVER_DISABLE_GC'] = '1'
os.environ['TLLM_PROFILE_START_STOP'] = '1000-1100'
os.environ['UCX_CUDA_IPC_ENABLE_MNNVL'] = 'n'
os.environ['UCX_RNDV_SCHEME'] = 'put_zcopy'
os.environ['TRTLLM_WORKER_DISABLE_GC'] = '1'

gen_config = {
    # Data paths. You should not need to change this unless you know what you are doing.
    llm_fields.llm_gen_config_path: 'code/gpt-oss-120b/tensorrt/generation_config_performance.json',
    harness_fields.tensor_path: 'build/data/gpt-oss/v4/perf',

    # Length limits and beam width are set by MLCommons rules and should not be changed.
    loadgen_fields.min_duration: 600000,
    loadgen_fields.min_query_count: 6396 * 1,
    llm_fields.warmup_iterations: 5,
    llm_fields.use_token_latencies: True,
    llm_fields.trtllm_build_flags: {
        'max_beam_width': 1,
        'remove_input_padding': 'enable',
        'max_num_tokens': 4638 * 3,
        'max_input_len': 15536,
        'max_seq_len': 15536 + 10240,
        'enable_attention_dp': True,
        'torch_compile_config': {
            'enable_fullgraph': True,
            'enable_piecewise_cuda_graph': True,
            'enable_userbuffers': False,
        },
    },
    llm_fields.trtllm_runtime_flags: {
        'exclude_input_from_output': True,
        'use_inflight_batching': True,
        'max_num_tokens': 4638 * 3,
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
        'kvcache_free_gpu_mem_frac': 0.95,  # Progressively lower by 0.1/0.05 if you hit OOM errors.
        'enable_chunked_context': True,
        'max_concurrency': 256 + 1024,
        'cuda_graph_batch_sizes': [1, 2, 4, 8, 16, 32, 64, 128, 144, 160, 176, 192],
        'cuda_graph_padding_enabled': True,
        'moe_backend': 'TRTLLM',
        "adp_balancing_enable": False,
        "adp_balancing_batching_wait_iters": 10,
        "adp_balancing_timeout_iters": 500,
        'num_postprocess_workers': 4,
        "stream_interval": 100,
        "cache_transceiver_max_tokens": 15536 + 10240 + 16,
        "sampler_type": "TRTLLMSampler",
    },
    harness_fields.use_graphs: True,

    # Precision fields. You should not need to change these.
    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32',

    # Tune me! If you hit an OOM, decrease the batch size.
    model_fields.gpu_batch_size: {
        'gpt-oss-120b': 192,
    },
    loadgen_fields.server_target_qps: 40,

    # You can try increasing these if you have multiple GPUs.
    llm_fields.tensor_parallelism: 4,
    llm_fields.pipeline_parallelism: 1,
    llm_fields.moe_expert_parallelism: 4,

    # Only supported on Hopper and Blackwell GPUs. On other GPUs, this will not do anything.
    harness_fields.vboost_slider: 1,
    harness_fields.enable_metrics: False,
}

# If 99.9% accuracy target needs different parameters than the default 99% target, you should create a separate
# dictionary or use copy.deepcopy and modify the requisite parameters.
EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): gen_config,
}

# Accuracy-specific overrides (deep merged when test_mode=AccuracyOnly)
ACCURACY_OVERRIDES = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        llm_fields.llm_gen_config_path: 'code/gpt-oss-120b/tensorrt/generation_config_accuracy.json',
        harness_fields.tensor_path: 'build/data/gpt-oss/v4/acc',
        loadgen_fields.min_query_count: 4395,
        llm_fields.trtllm_build_flags: {
            'max_input_len': 3072,
            'max_seq_len': 3072 + 32768,  # 35840
        },
    },
}

# Compliance test overrides (keyed by AuditTest)
COMPLIANCE_OVERRIDES = {
    C.AuditTest.TEST07: {
        C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
            llm_fields.llm_gen_config_path: 'code/gpt-oss-120b/tensorrt/generation_config_performance.json',
            harness_fields.tensor_path: 'build/data/gpt-oss/v4/compliance/test07',
            loadgen_fields.min_query_count: 990,
        },
    },
    C.AuditTest.TEST09: {
        C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
            loadgen_fields.min_query_count: 6396,
            loadgen_fields.min_duration: 0,
        },
    },
}

# Atomic configs for leader mode with multiple variants
# Usage: --mpi_mode=leader --config_id=<variant_name>
ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": gen_config,
    },
}
