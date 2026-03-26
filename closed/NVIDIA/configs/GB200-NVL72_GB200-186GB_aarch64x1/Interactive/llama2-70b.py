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
os.environ['CUDA_SCALE_LAUNCH_QUEUES'] = '4x'

ctx_config = {
    model_fields.gpu_batch_size: {
        'llama2-70b': 4096,
    },
    model_fields.input_dtype: 'int32',
    llm_fields.llm_gen_config_path: 'code/llama2-70b/tensorrt/generation_config.json',
    loadgen_fields.min_duration: 2400000,
    model_fields.precision: 'fp4',
    loadgen_fields.server_target_qps: 310,
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
        'max_num_tokens': 4096,
        'max_input_len': 1024,
        'max_seq_len': 2048,
        'use_fp8_context_fmha': 'enable',
        'use_paged_context_fmha': 'enable',
        'tokens_per_block': 32,
        'norm_quant_fusion': 'enable',
    },
    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    llm_fields.trtllm_runtime_flags: {
        'exclude_input_from_output': True,
        'use_inflight_batching': True,
        'max_num_tokens': 4096,
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
        'kvcache_free_gpu_mem_frac': 0.85,
        'enable_chunked_context': True,
        'stream_interval': 30,
        'cache_transceiver_max_tokens': 1024,
        'cache_transceiver_backend': 'UCX',
        'disable_overlap_scheduler': True,
        'sampler_type': 'TRTLLMSampler',
    },
    llm_fields.traffic_distribution_policy: 'isl_load_balancing',
    harness_fields.use_graphs: False,
    llm_fields.use_token_latencies: True,
    harness_fields.enable_metrics: False,
    llm_fields.harness_use_hf_tokenizer: True,
    harness_fields.vboost_slider: 1,
}

gen_config = {
    model_fields.gpu_batch_size: {
        'llama2-70b': 768,
    },
    model_fields.input_dtype: 'int32',
    llm_fields.llm_gen_config_path: 'code/llama2-70b/tensorrt/generation_config.json',
    loadgen_fields.min_duration: 2400000,
    model_fields.precision: 'fp4',
    loadgen_fields.server_target_qps: 310,
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
        'max_num_tokens': 768,
        'max_input_len': 1024,
        'max_seq_len': 2048,
        'use_fp8_context_fmha': 'enable',
        'use_paged_context_fmha': 'enable',
        'tokens_per_block': 32,
        'norm_quant_fusion': 'enable',
    },
    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    llm_fields.trtllm_runtime_flags: {
        'exclude_input_from_output': True,
        'use_inflight_batching': True,
        'max_num_tokens': 768,
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
        'kvcache_free_gpu_mem_frac': 0.95,
        'cuda_graph_batch_sizes': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 640, 768],
        'cuda_graph_padding_enabled': True,
        'stream_interval': 100,
        'cache_transceiver_max_tokens': 1024,
        'cache_transceiver_backend': 'UCX',
        'num_postprocess_workers': 4,
        'sampler_type': 'TRTLLMSampler',
    },
    llm_fields.traffic_distribution_policy: 'isl_load_balancing',
    llm_fields.use_token_latencies: True,
    harness_fields.use_graphs: True,
    harness_fields.enable_metrics: False,
    llm_fields.harness_use_hf_tokenizer: True,
    harness_fields.vboost_slider: 1,
}

ctx_config_x72 = {
    model_fields.gpu_batch_size: {
        'llama2-70b': 4096,
    },
    model_fields.input_dtype: 'int32',
    llm_fields.llm_gen_config_path: 'code/llama2-70b/tensorrt/generation_config.json',
    loadgen_fields.min_duration: 2400000,
    model_fields.precision: 'fp4',
    loadgen_fields.server_target_qps: 2790,
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
        'max_batch_size': 4096,
        'max_num_tokens': 4352,
        'max_input_len': 1024,
        'max_seq_len': 2048,
        'use_fp8_context_fmha': 'enable',
        'use_paged_context_fmha': 'enable',
        'tokens_per_block': 32,
        'norm_quant_fusion': 'enable',
        'enable_attention_dp': False,
    },
    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
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
        'cache_transceiver_backend': 'UCX',
        'disable_overlap_scheduler': True,
        'sampler_type': 'TRTLLMSampler',
    },
    llm_fields.traffic_distribution_policy: 'isl_load_balancing',
    harness_fields.use_graphs: False,
    llm_fields.use_token_latencies: True,
    harness_fields.enable_metrics: False,
    llm_fields.harness_use_hf_tokenizer: True,
    harness_fields.vboost_slider: 1,
}

gen_config_x72 = {
    model_fields.gpu_batch_size: {
        'llama2-70b': 512,
    },
    model_fields.input_dtype: 'int32',
    llm_fields.llm_gen_config_path: 'code/llama2-70b/tensorrt/generation_config.json',
    loadgen_fields.min_duration: 2400000,
    model_fields.precision: 'fp4',
    loadgen_fields.server_target_qps: 2790,
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
        'max_batch_size': 512,
        'max_num_tokens': 512,
        'max_input_len': 1024,
        'max_seq_len': 2048,
        'use_fp8_context_fmha': 'enable',
        'use_paged_context_fmha': 'enable',
        'tokens_per_block': 32,
        'norm_quant_fusion': 'enable',
        'enable_attention_dp': False,
    },
    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    llm_fields.trtllm_runtime_flags: {
        'exclude_input_from_output': True,
        'use_inflight_batching': True,
        'max_num_tokens': 512,
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
        'kvcache_free_gpu_mem_frac': 0.95,
        'cuda_graph_batch_sizes': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        'cuda_graph_padding_enabled': True,
        'stream_interval': 150,
        'cache_transceiver_max_tokens': 2048,
        'cache_transceiver_backend': 'UCX',
        'num_postprocess_workers': 4,
        'sampler_type': 'TRTLLMSampler',
    },
    llm_fields.traffic_distribution_policy: 'isl_load_balancing',
    llm_fields.use_token_latencies: True,
    harness_fields.use_graphs: True,
    harness_fields.enable_metrics: False,
    llm_fields.harness_use_hf_tokenizer: True,
    harness_fields.vboost_slider: 1,
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): ctx_config,
}

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": ctx_config,
        "ctx_config": ctx_config,
        "gen_config": gen_config,
        "ctx_config_x72": ctx_config_x72,
        "gen_config_x72": gen_config_x72,
    },
}
