import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields


import os
os.environ['TRTLLM_SERVER_DISABLE_GC'] = '1'
os.environ['OMPI_MCA_coll_ucc_enable'] = '0' #Hitting "Deserialization failed: invalid load key" 
os.environ['TLLM_NUMA_AWARE_WORKER_AFFINITY'] = '1' #perf improved from 180->187tps/gpu
os.environ['CUDA_SCALE_LAUNCH_QUEUES'] = '4x'  #perf improved from 152->177 along with cuda graph and mnt1024 changes
os.environ['TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED']= '1'
os.environ['TRTLLM_LLAMA_EAGER_FUSION_DISABLED']= '1'
os.environ['TRTLLM_DISABLE_NVFP4_LAYERNORM_FUSION']= '0'

ifb_config = {
    model_fields.gpu_batch_size: {
        'llama3.1-405b': 256,
    },
    model_fields.input_dtype: 'int32',
    llm_fields.llm_gen_config_path: 'code/llama3_1-405b/tensorrt/generation_config.json',
    loadgen_fields.min_duration: 600000,
    model_fields.precision: 'fp4',
    loadgen_fields.server_target_qps: 1.1,
    harness_fields.tensor_path: 'build/preprocessed_data/llama3.1-405b/',
    llm_fields.tensor_parallelism: 2,
    llm_fields.pipeline_parallelism: 2,
    llm_fields.trtllm_build_flags: {
        'max_beam_width': 1,
        'kv_cache_type': 'paged',
        'remove_input_padding': 'enable',
        'multiple_profiles': 'enable',
        'use_fused_mlp': 'enable',
        'context_fmha': 'enable',
        'max_num_tokens': 1536,
        'max_input_len': 20000,
        'max_seq_len': 22000,
        'use_paged_context_fmha': 'enable',
        'tokens_per_block': 32,
        'use_fp8_context_fmha': 'enable',
        'norm_quant_fusion': 'enable',
        'gemm_allreduce_plugin': 'float16',
        'torch_compile_config': {
            'enable_fullgraph': True,
            'enable_piecewise_cuda_graph': True,
            'enable_userbuffers': False,
        },
    },
    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    llm_fields.trtllm_runtime_flags: {
        'exclude_input_from_output': True,
        'use_inflight_batching': True,
        'max_num_tokens': 1536,
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
        'kvcache_free_gpu_mem_frac': 0.95,
        'enable_chunked_context': True,
        'cuda_graph_batch_sizes': [1, 2, 4, 8, 16,32, 64, 96, 128],
        'cuda_graph_padding_enabled': True,
        'num_postprocess_workers' : 2,
        'stream_interval': 20,
        'sampler_type': 'TRTLLMSampler',
    },
    harness_fields.use_graphs: True,
    llm_fields.use_token_latencies: True,
    llm_fields.readiness_timeout: 600,
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): ifb_config,
}
ctx_config = {
    model_fields.gpu_batch_size: {
        'llama3.1-405b': 128,
    },
    model_fields.input_dtype: 'int32',
    llm_fields.llm_gen_config_path: 'code/llama3_1-405b/tensorrt/generation_config.json',
    loadgen_fields.min_duration: 600000,
    loadgen_fields.min_query_count: 49878,
    loadgen_fields.server_target_qps: 17.7,
    llm_fields.warmup_iterations: 0,
    model_fields.precision: 'fp4',
    harness_fields.tensor_path: 'build/preprocessed_data/llama3.1-405b/',
    llm_fields.tensor_parallelism: 4,
    llm_fields.pipeline_parallelism: 1,
    llm_fields.trtllm_build_flags: {
        'max_beam_width': 1,
        'kv_cache_type': 'paged',
        'remove_input_padding': 'enable',
        'multiple_profiles': 'enable',
        'use_fused_mlp': 'enable',
        'context_fmha': 'enable',
        'max_num_tokens': 4096,
        'max_input_len': 20000,
        'max_seq_len': 20192,
        'use_paged_context_fmha': 'enable',
        'tokens_per_block': 32,
        'use_fp8_context_fmha': 'enable',
        'norm_quant_fusion': 'enable',
        'gemm_allreduce_plugin': 'float16',
        'enable_attention_dp': False,
        'torch_compile_config': {
            'enable_fullgraph': True,
            'enable_piecewise_cuda_graph': True,
            'enable_userbuffers': False,
        },
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
        'kvcache_free_gpu_mem_frac': 0.95,
        'enable_chunked_context': True,
        'disable_overlap_scheduler': True,
        'stream_interval': 20,
        'cache_transceiver_max_tokens': 20192,
        'cache_transceiver_backend': 'UCX',
        'sampler_type': 'TRTLLMSampler',
        'max_concurrency': 2048,
    },
    harness_fields.use_graphs: False,
    llm_fields.use_token_latencies: True,
    llm_fields.readiness_timeout: 600,
}
# Atomic configs for leader mode with multiple variants
# Usage: --mpi_mode=leader --config_id=<variant_name>
ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": ifb_config,
        "ctx": ctx_config,
    },
}

