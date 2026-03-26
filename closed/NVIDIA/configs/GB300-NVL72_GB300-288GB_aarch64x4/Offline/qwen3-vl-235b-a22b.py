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
            "--max-num-batched-tokens=13824",
            "--scheduling-policy=sjf",
            "--compilation-config={\"max_cudagraph_capture_size\":13824,\"cudagraph_capture_sizes\":[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048, 2112, 2176, 2240, 2304, 2368, 2432, 2496, 2560, 2624, 2688, 2752, 2816, 2880, 2944, 3008, 3072, 3136, 3200, 3264, 3328, 3392, 3456, 3520, 3584, 3648, 3712, 3776, 3840, 3904, 3968, 4032, 4096, 4160, 4224, 4288, 4352, 4416, 4480, 4544, 4608, 4672, 4736, 4800, 4864, 4928, 4992, 5056, 5120, 5184, 5248, 5312, 5376, 5440, 5504, 5568, 5632, 5696, 5760, 5824, 5888, 5952, 6016, 6080, 6144, 6208, 6272, 6336, 6400, 6464, 6528, 6592, 6656, 6720, 6784, 6848, 6912, 6976, 7040, 7104, 7168, 7232, 7296, 7360, 7424, 7488, 7552, 7616, 7680, 7744, 7808, 7872, 7936, 8000, 8064, 8128, 8192, 8256, 8320, 8384, 8448, 8512, 8576, 8640, 8704, 8768, 8832, 8896, 8960, 9024, 9088, 9152, 9216, 9280, 9344, 9408, 9472, 9536, 9600, 9664, 9728, 9792, 9856, 9920, 9984, 10048, 10112, 10176, 10240, 10304, 10368, 10432, 10496, 10560, 10624, 10688, 10752, 10816, 10880, 10944, 11008, 11072, 11136, 11200, 11264, 11328, 11392, 11456, 11520, 11584, 11648, 11712, 11776, 11840, 11904, 11968, 12032, 12096, 12160, 12224, 12288, 12352, 12416, 12480, 12544, 12608, 12672, 12736, 12800, 12864, 12928, 12992, 13056, 13120, 13184, 13248, 13312, 13376, 13440, 13504, 13568, 13632, 13696, 13760, 13824]}",
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
