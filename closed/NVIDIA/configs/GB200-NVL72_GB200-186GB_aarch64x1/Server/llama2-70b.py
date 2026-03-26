import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields
import os

os.environ['TRTLLM_SERVER_DISABLE_GC'] = '1'
os.environ['TRTLLM_WORKER_DISABLE_GC'] = '1'
os.environ['TRTLLM_ENABLE_PDL'] = '1'
os.environ['UCX_CUDA_IPC_ENABLE_MNNVL'] = 'n'
os.environ['UCX_RNDV_SCHEME'] = 'get_zcopy'
os.environ['TLLM_NUMA_AWARE_WORKER_AFFINITY'] = '1'
os.environ['OMPI_MCA_hwloc_base_binding_policy'] = 'none'
os.environ['OMPI_MCA_rmaps_base_inherit'] = '1'

ifb_config = {
    model_fields.gpu_batch_size: {
        'llama2-70b': 2048,
    },
    model_fields.input_dtype: 'int32',
    llm_fields.llm_gen_config_path: 'code/llama2-70b/tensorrt/generation_config.json',
    loadgen_fields.min_duration: 2400000,
    model_fields.precision: 'fp4',
    loadgen_fields.server_target_qps: 49,
    harness_fields.tensor_path: 'build/preprocessed_data/llama2-70b/',
    llm_fields.tensor_parallelism: 1,
    llm_fields.pipeline_parallelism: 1,
    llm_fields.trtllm_build_flags: {
        'max_beam_width': 1,
        'kv_cache_type': 'paged',
        'remove_input_padding': 'enable',
        'multiple_profiles': 'enable',
        'use_fused_mlp': 'enable',
        'context_fmha': 'enable',
        'max_num_tokens': 3584,
        'max_input_len': 1024,
        'max_seq_len': 2048,
        'use_fp8_context_fmha': 'enable',
        'use_paged_context_fmha': 'enable',
        'tokens_per_block': 32,
        'norm_quant_fusion': 'enable',
        'torch_compile_config': {
            'enable_fullgraph': True,
            'enable_inductor': False,
            'enable_piecewise_cuda_graph': True,
            'enable_userbuffers': True,
            'capture_num_tokens': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584],
        },
    },
    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    llm_fields.trtllm_runtime_flags: {
        'exclude_input_from_output': True,
        'use_inflight_batching': True,
        'max_num_tokens': 3584,
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
        'kvcache_free_gpu_mem_frac': 0.95,
        'enable_chunked_context': True,
        'max_concurrency': 2048,
        'num_postprocess_workers': 4,
        'stream_interval': 50,
        'sampler_type': 'TRTLLMSampler',
    },
    llm_fields.traffic_distribution_policy: 'isl_load_balancing',
    harness_fields.enable_metrics: False,
    harness_fields.use_graphs: False,
    llm_fields.use_token_latencies: True,
    llm_fields.harness_use_hf_tokenizer: True,
    harness_fields.vboost_slider: 1,
}

ctx_config_x72 = {
    llm_fields.llm_gen_config_path: 'code/llama2-70b/tensorrt/generation_config.json',
    harness_fields.tensor_path: 'build/preprocessed_data/llama2-70b/',
    loadgen_fields.min_duration: 2400000,
    loadgen_fields.server_target_qps: 3000,
    llm_fields.use_token_latencies: True,
    llm_fields.trtllm_build_flags: {
        'max_beam_width': 1,
        'kv_cache_type': 'paged',
        'remove_input_padding': 'enable',
        'multiple_profiles': 'enable',
        'use_fused_mlp': 'enable',
        'context_fmha': 'enable',
        'max_num_tokens': 4352,
        'max_input_len': 1024,
        'max_seq_len': 2048,
        'use_fp8_context_fmha': 'enable',
        'use_paged_context_fmha': 'enable',
        'tokens_per_block': 32,
        'norm_quant_fusion': 'enable',
        'enable_attention_dp': False,
    },
    llm_fields.trtllm_runtime_flags: {
        'exclude_input_from_output': True,
        'use_inflight_batching': True,
        'max_num_tokens': 4352,
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
        'kvcache_free_gpu_mem_frac': 0.85,
        'enable_chunked_context': True,
        'stream_interval': 30,
        'cache_transceiver_max_tokens': 2048,
        'disable_overlap_scheduler': True,
        'sampler_type': 'TRTLLMSampler',
    },
    llm_fields.traffic_distribution_policy: 'isl_load_balancing',
    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32',
    model_fields.gpu_batch_size: {
        'llama2-70b': 4096,
    },
    llm_fields.tensor_parallelism: 1,
    llm_fields.pipeline_parallelism: 1,
    harness_fields.enable_metrics: False,
    llm_fields.harness_use_hf_tokenizer: True,
    harness_fields.vboost_slider: 1,
}

gen_config_x72 = {
    llm_fields.llm_gen_config_path: 'code/llama2-70b/tensorrt/generation_config.json',
    harness_fields.tensor_path: 'build/preprocessed_data/llama2-70b/',
    loadgen_fields.min_duration: 2400000,
    loadgen_fields.server_target_qps: 3000,
    llm_fields.use_token_latencies: True,
    llm_fields.trtllm_build_flags: {
        'max_beam_width': 1,
        'kv_cache_type': 'paged',
        'remove_input_padding': 'enable',
        'multiple_profiles': 'enable',
        'use_fused_mlp': 'enable',
        'context_fmha': 'enable',
        'max_num_tokens': 1024,
        'max_input_len': 1024,
        'max_seq_len': 2048,
        'use_fp8_context_fmha': 'enable',
        'use_paged_context_fmha': 'enable',
        'tokens_per_block': 32,
        'norm_quant_fusion': 'enable',
        'enable_attention_dp': False,
    },
    llm_fields.trtllm_runtime_flags: {
        'exclude_input_from_output': True,
        'use_inflight_batching': True,
        'max_num_tokens': 1024,
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
        'kvcache_free_gpu_mem_frac': 0.95,
        'enable_chunked_context': True,
        'cuda_graph_batch_sizes': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 640, 768, 1024],
        'cuda_graph_padding_enabled': True,
        'stream_interval': 150,
        'cache_transceiver_max_tokens': 2048,
        'num_postprocess_workers': 4,
        'sampler_type': 'TRTLLMSampler',
    },
    llm_fields.traffic_distribution_policy: 'isl_load_balancing',
    harness_fields.use_graphs: True,
    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32',
    model_fields.gpu_batch_size: {
        'llama2-70b': 1024,
    },
    llm_fields.tensor_parallelism: 1,
    llm_fields.pipeline_parallelism: 1,
    harness_fields.enable_metrics: False,
    llm_fields.harness_use_hf_tokenizer: True,
    harness_fields.vboost_slider: 1,
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): ifb_config,
}

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": ifb_config,
        "ctx_config_x72": ctx_config_x72,
        "gen_config_x72": gen_config_x72,
    },
}
