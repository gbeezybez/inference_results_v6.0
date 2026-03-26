# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GB200-NVL72_GB200-186GB_aarch64x4 configuration for Qwen3-VL 235B Offline scenario."""

import code.common.constants as C
import code.fields.loadgen as loadgen_fields
from importlib import import_module

q3vl_fields = import_module("code.qwen3-vl-235b-a22b.harness.fields")

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        # Test settings (used by SettingsBuilder via autoconfigure)
        q3vl_fields.test_scenario: "offline",
        q3vl_fields.test_mode: "performance_only",
        loadgen_fields.offline_expected_qps: 80.4816666667,
        q3vl_fields.enable_trace: False,
        q3vl_fields.model_repo_id: "nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.0",
        q3vl_fields.model_revision: "main",
        loadgen_fields.qsl_rng_seed: 2465351861681999779,
        loadgen_fields.sample_index_rng_seed: 14276810075590677512,
        loadgen_fields.schedule_rng_seed: 3936089224930324775,
        q3vl_fields.vllm_cli: [
            "--tensor-parallel-size=1",
            "--pipeline-parallel-size=1",
            "--data-parallel-size=1",
            "--async-scheduling",
            "--max-model-len=32768",
            "--max-num-seqs=1024",
            "--mm-encoder-attn-backend=FLASHINFER",
            "--max-num-batched-tokens=4864",
            "--scheduling-policy=sjf",
            "--compilation-config={\"max_cudagraph_capture_size\":4864,\"cudagraph_capture_sizes\":[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248,256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480,496,512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184, 1216, 1248, 1280, 1312, 1344, 1376, 1408, 1440, 1472, 1504, 1536, 1568, 1600, 1632, 1664, 1696, 1728, 1760, 1792, 1824, 1856, 1888, 1920, 1952, 1984, 2016, 2048, 2080, 2112, 2144, 2176, 2208, 2240, 2272, 2304, 2336, 2368, 2400, 2432, 2464, 2496, 2528, 2560, 2592, 2624, 2656, 2688, 2720, 2752, 2784, 2816, 2848, 2880, 2912, 2944, 2976, 3008, 3040, 3072, 3104, 3136, 3168, 3200, 3232, 3264, 3296, 3328, 3360, 3392, 3424, 3456, 3488, 3520, 3552, 3584, 3616, 3648, 3680, 3712, 3744, 3776, 3808, 3840, 3872, 3904, 3936, 3968, 4000, 4032, 4064, 4096, 4128, 4160, 4192, 4224, 4256, 4288, 4320, 4352, 4384, 4416, 4448, 4480, 4512, 4544, 4576, 4608, 4640, 4672, 4704, 4736, 4768, 4800, 4832,4864]}",
            "--override-generation-config={\"max_new_tokens\":150}",
            "--limit-mm-per-prompt.video=0",
            "--no-enable-prefix-caching",
            "--enable-multimodal",
            "--connector=none",
            "--kv-events-config={\"publisher\":\"null\"}",
            "--mm-processor-cache-gb=0",
        ],
        q3vl_fields.vllm_enable_numa_binding: True,
        q3vl_fields.vllm_dyn_log: "debug",
        q3vl_fields.vllm_logging_level: "DEBUG",
        q3vl_fields.vllm_use_flashinfer_sampler: 1,
        q3vl_fields.vllm_use_flashinfer_moe_fp4: 1,
        q3vl_fields.vllm_use_triton_pos_embed: 1,
        q3vl_fields.vllm_mm_encoder_fp8_attn: 1,
        q3vl_fields.use_http_client: True,
        q3vl_fields.max_concurrency: 2048,
        q3vl_fields.vllm_flashinfer_moe_backend: "latency",
        q3vl_fields.vllm_flashinfer_workspace_buffer_size: 6 * 256 * 1024 * 1024,
        q3vl_fields.tokio_worker_threads: 32,
        q3vl_fields.omp_num_threads: 64,
        q3vl_fields.frontend_enable_numa_binding: True,
        q3vl_fields.num_warmup_requests_per_vllm_instance: 400,
    },
}
