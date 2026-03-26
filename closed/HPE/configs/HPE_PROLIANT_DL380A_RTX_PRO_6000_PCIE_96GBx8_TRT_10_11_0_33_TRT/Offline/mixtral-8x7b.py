import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields
import code.fields.triton as triton_fields


base = {
    llm_fields.llm_gen_config_path: 'code/mixtral-8x7b/tensorrt/generation_config.json',
    harness_fields.tensor_path: 'build/preprocessed_data/moe/',

    loadgen_fields.min_duration: 1200000,
    llm_fields.use_token_latencies: True,
    llm_fields.trtllm_build_flags: {
        'max_beam_width': 1,
        'kv_cache_type': 'paged',
        'remove_input_padding': 'enable',
        'multiple_profiles': 'enable',
        'use_fused_mlp': 'enable',
        'context_fmha': 'enable',
        'use_paged_context_fmha': 'disable',
        'use_fp8_context_fmha': 'disable',
        'max_num_tokens': 5888,
        'max_input_len': 2048,
        'max_seq_len': 3072,
        'tokens_per_block': 32,
        'gpus_per_node': 8,
    },
    llm_fields.trtllm_runtime_flags: {
        'exclude_input_from_output': True,
        'use_inflight_batching': True,
        'max_num_tokens': 5888,
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
        'kvcache_free_gpu_mem_frac': 0.98,
        'enable_chunked_context': True,
    },

    llm_fields.trtllm_checkpoint_flags: {
        'effective_bits': 8.75,
        'num_score_steps': 4,
        'kv_cache_dtype': 'fp8',
        'num_calib_steps': 16,
    },
    model_fields.precision: 'fp8',
    model_fields.input_dtype: 'int32',

    model_fields.gpu_batch_size: {
        'mixtral-8x7b': 1280,
    },
    loadgen_fields.offline_expected_qps: 300,

    llm_fields.tensor_parallelism: 1,
    llm_fields.pipeline_parallelism: 1,

    harness_fields.use_graphs: False,

    harness_fields.vboost_slider: 1,
}


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): base
}
