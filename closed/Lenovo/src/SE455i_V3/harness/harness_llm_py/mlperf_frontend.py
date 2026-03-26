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


import inspect
import os
import pickle as pkl
from typing import List

import numpy as np

from code.common import args_to_string, arguments as common_args, logging, run_command
from code.common.harness import BaseBenchmarkHarness
from code.harness.harness_llm_py.llm_server import LLMDataset, LLMServer

try:
    import mlperf_loadgen as lg
except:
    logging.warning("Loadgen Python bindings are not installed. Installing Loadgen Python bindings!")
    run_command("make build_loadgen")
    import mlperf_loadgen as lg


def create_qsl_cls(dataset_cls: type) -> type:
    """
    Generate a Mlperf-Inference QSL Wrapper for given LLMDataset.
    This is consumed by LLMSUT (which is a LLMServer wrapper).
    """
    assert dataset_cls and issubclass(dataset_cls, LLMDataset), "dataset_cls my be a subclass of LLMDataset."

    class LLMQSL(dataset_cls):
        """ Mlperf-Inference QSL. LLMDataset subclass. """

        FILES = dataset_cls.FILES

        def __init__(self, sample_count: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sample_count = sample_count
            self.mlperf_qsl = lg.ConstructQSL(
                len(self),
                self.sample_count,

                # we load everything to host memory on init
                lambda _: None,
                lambda _: None,
            )

            self.logger.info("Initialized QSL.")
            self.logger.info(f'Total Sample Count: {len(self)}')
            self.logger.info(f'Performance Sample Count: {self.sample_count}')

        def __del__(self):
            lg.DestroyQSL(self.mlperf_qsl)
            self.logger.info("Destroyed QSL.")

    return LLMQSL


class LLMSUT(LLMServer):
    """ Mlperf-Inference SUT. LLMServer subclass. """

    def __init__(self, dataset: LLMDataset, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset = dataset
        self.mlperf_sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.logger.info("Initialized SUT.")

    def __del__(self):
        lg.DestroySUT(self.mlperf_sut)
        self.logger.info("Destroyed SUT.")

    ##### Mlperf-Inference SUT Runner #####

    def run_test(self, test_settings, log_settings):
        lg.StartTestWithLogSettings(
            self.mlperf_sut,
            self.dataset.mlperf_qsl,
            test_settings,
            log_settings
        )
        self.stop_work()

    ##### LLMServer overrides #####

    def issue_queries(self, query_samples: List[lg.QuerySample]):
        qsl_ids = [sample.id for sample in query_samples]
        qsl_indices = [sample.index for sample in query_samples]
        input_tokens = self.dataset.get_input_tokens(qsl_indices)
        stop_tokens = self.dataset.get_stop_tokens(qsl_indices)
        queries = list(zip(qsl_ids, input_tokens, stop_tokens))
        super().issue_queries(queries)

    @staticmethod
    def complete_request(request_id: int, output_tokens: List[int], is_first_token: bool):
        complete_fn = lg.FirstTokenComplete if is_first_token else lg.QuerySamplesComplete
        output_tokens = np.ascontiguousarray(output_tokens, dtype=np.uint32)
        output_seq_len = len(output_tokens)
        output_toks_ptr = output_tokens.ctypes.data
        output_toks_size = output_seq_len * output_tokens.itemsize
        complete_fn([lg.QuerySampleResponse(request_id, output_toks_ptr, output_toks_size, output_seq_len)])


class LLMHarness(BaseBenchmarkHarness):
    """Mlperf-Inference TRTLLM LLMHarness"""

    DATASET_CLS: type = LLMDataset
    CUSTOM_ARGS: List[str] = None

    def __init__(self, args, benchmark):
        super().__init__(args, benchmark)

        harness_args = [
            "devices",
            "use_token_latencies",
            "enable_sort",
            "llm_gen_config_path",
            "trtllm_checkpoint_flags",
            "trtllm_build_flags",
            "trtllm_runtime_flags",
        ]

        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.SHARED_ARGS + harness_args
        if self.CUSTOM_ARGS is not None:
            self.flag_builder_custom_args += self.CUSTOM_ARGS

    def _get_harness_executable(self):
        """Return python command to LLMHarness runner python file"""
        return 'code/harness/harness_llm_py/runner.py'

    def _construct_terminal_command(self, argstr):
        cmd = f'python3 -m {self.executable.replace("/", ".").replace(".py", "")} {argstr}'
        return cmd

    def _get_engine_fpath(self, device_type, _, batch_size):
        # Override this function to pick up the right engine file
        if not self.default_engine_dir:
            return f"{self.engine_dir}/rank0.engine"

        tp_size = self.args['trtllm_build_flags']['tensor_parallelism']
        pp_size = self.args['trtllm_build_flags']['pipeline_parallelism']
        return f"{self.engine_dir}/{self.name}-{self.scenario.valstr()}-{device_type}-b{batch_size}-{self.precision}-tp{tp_size}pp{pp_size}-{self.config_ver}/rank0.engine"

    def _build_custom_flags(self, flag_dict):
        dataset_cls_fpath = inspect.getfile(self.DATASET_CLS)
        dataset_cls_path = dataset_cls_fpath.replace("/work/", "").replace(".py", "").replace("/", ".")
        dataset_cls = dataset_cls_path + f'.{self.DATASET_CLS.__name__}'

        def to_cli(value):
            match value:
                case bool() as b: return str(b).lower()
                case _: return str(value)

        flag_dict |= {
            key: ','.join(f"{k}:{to_cli(v)}" for k, v in value.items())
            for key, value in flag_dict.items()
            if key in ['trtllm_checkpoint_flags', 'trtllm_build_flags', 'trtllm_runtime_flags']
        }

        s = ' '.join([args_to_string(flag_dict),
                      f"--scenario {self.scenario.valstr()}",
                      f"--model {self.name}",
                      f"--dataset_cls {dataset_cls}"])
        return s
