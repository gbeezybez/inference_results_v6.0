import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields
import code.fields.triton as triton_fields

base = {
    llm_fields.llm_gen_config_path: 'code/llama2-70b/tensorrt/generation_config.json',
    harness_fields.tensor_path: 'build/preprocessed_data/llama2-70b/',
    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32',
    model_fields.gpu_batch_size: {
        'llama2-70b': 2048,
    },
    loadgen_fields.min_duration: 1200000,
    loadgen_fields.server_target_qps: 23,
    
    llm_fields.trtllm_build_flags: {
        'max_beam_width': 1,
        'kv_cache_type': 'paged',
        'remove_input_padding': 'enable',
        'multiple_profiles': 'enable',
        'use_fused_mlp': 'enable',
        'context_fmha': 'enable',
        'max_num_tokens': 2048,
        'max_input_len': 1024,
        'max_seq_len': 2048,
        'use_fp8_context_fmha': 'enable',
        'use_paged_context_fmha': 'enable',
        'tokens_per_block': 32,
        'gemm_plugin': 'fp4',
    },
    llm_fields.tensor_parallelism: 1,
    llm_fields.pipeline_parallelism: 1,
    llm_fields.trtllm_runtime_flags: {
        'exclude_input_from_output': True,
        'use_inflight_batching': True,
        'max_num_tokens': 2048,
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
        'kvcache_free_gpu_mem_frac': 0.95,
        'enable_chunked_context': True,
        'max_concurrency': 1024,
        'workers_per_core': 12,
        'cuda_graph_batch_sizes': [1, 2, 4, 8, 16, 32, 64, *[128 * (x + 1) for x in range(8)]],
        'cuda_graph_padding_enabled': True,
        'use_cuda_graphs': True,
    },

    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    
    harness_fields.use_graphs: True,
    llm_fields.use_token_latencies: True,
    
    harness_fields.vboost_slider: 1,
    harness_fields.numa_config: "0,1,2,3:0-85,172-257&4,5,6,7:86-171,258-343",
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): base,
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.999), C.PowerSetting.MaxP): base,
}

