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

from __future__ import annotations
from collections import defaultdict
import os
from pprint import pformat
import signal
from typing import Any, Callable, Dict, List, Tuple

import psutil

from code.common.utils import add_nvtx_scope_wrap, parse_cli_flags

from .config import EngineConfig, HarnessConfig
from .core import LLMCore
from .utils import LLMServerProgressDisplay, add_prefix_logger


@add_prefix_logger()
@add_nvtx_scope_wrap()
class LLMServer():
    def __init__(self,
                 scenario: str,
                 devices: List[int],
                 enable_sort: bool,
                 trtllm_checkpoint_flags: Dict[str, Any],
                 trtllm_build_flags: Dict[str, Any],
                 trtllm_runtime_flags: Dict[str, Any],
                 gpu_engine_dir: os.PathLike,
                 gpu_batch_size: int,
                 verbose: bool,
                 verbose_nvtx: bool,
                 log_dir: str,
                 use_graphs: bool,
                 llm_gen_config_path: os.PathLike):
        self.setup_interrupt_handler()
        self.devices = devices
        self.verbose = verbose
        self.verbose_nvtx = verbose_nvtx

        assert not use_graphs, "LLMHarness does not support use_graphs."
        assert not enable_sort, "LLMHarness does not support enable_sort."

        # NOTE(vir):
        # we ignore trtllm_build_flags['max_batch_size'] if given,
        # use legacy field gpu_batch_size instead
        if trtllm_build_flags.get('max_batch_size', -1) != gpu_batch_size:
            trtllm_build_flags['max_batch_size'] = gpu_batch_size
            self.logger.info(f'Overriding trtllm_build_flags[max_batch_size] with legacy field gpu_batch_size={gpu_batch_size}')

        if 'max_batch_size' not in trtllm_runtime_flags:
            trtllm_runtime_flags['max_batch_size'] = gpu_batch_size
            self.logger.info(f'Using engine max-batch-size as runtime max-batch-size since its not specified in trtllm_runtime_flags')

        if 'max_num_tokens' not in trtllm_runtime_flags:
            trtllm_runtime_flags['max_num_tokens'] = trtllm_build_flags['max_num_tokens']
            self.logger.info(f'Using engine max-num-tokens as runtime max-num-tokens since its not specified in trtllm_runtime_flags')

        # set remaining defaults
        trtllm_runtime_flags |= {
            'batch_scheduler_policy': trtllm_runtime_flags.get('batch_scheduler_policy', 'max_util'),
            'context_chunking_policy': trtllm_runtime_flags.get('context_chunking_policy', 'first_come_first_served'),
            'use_inflight_batching': trtllm_runtime_flags.get('use_inflight_batching', True),
            'enable_batch_size_tuning': trtllm_runtime_flags.get('enable_batch_size_tuning', False),
            'enable_max_num_tokens_tuning': trtllm_runtime_flags.get('enable_max_num_tokens_tuning', False),
            'dynamic_batch_moving_average_window': trtllm_runtime_flags.get('dynamic_batch_moving_average_window', 128),
            'kvcache_free_gpu_mem_frac': trtllm_runtime_flags.get('kvcache_free_gpu_mem_frac', 0.80),
            'enable_chunked_context': trtllm_runtime_flags.get('enable_chunked_context', False),
            'exclude_input_from_output': trtllm_runtime_flags.get('exclude_input_from_output', True)
        }

        # override runtime flags
        if runtime_overrides := os.environ.get('TRTLLM_RUNTIME_FLAGS', None):
            self.logger.info(f"Detected TRTLLM_RUNTIME_FLAGS: {runtime_overrides}")
            runtime_overrides = parse_cli_flags(runtime_overrides)
            for key, override in runtime_overrides.items():
                self.logger.info(f"Overriding {key}: {override}")
                trtllm_runtime_flags[key] = override

        self.engine_config = EngineConfig.from_engine_dir(gpu_engine_dir)
        self.harness_config = HarnessConfig(
            traffic_distribution_policy="load_balancing" if scenario != "Offline" else "round_robin",
            gen_config=HarnessConfig.load_generation_config(llm_gen_config_path),
            trtllm_checkpoint_flags=trtllm_checkpoint_flags,
            trtllm_build_flags=trtllm_build_flags,
            trtllm_runtime_flags=trtllm_runtime_flags,
            log_dir=log_dir,
        )
        self.harness_config.gen_config.streaming &= scenario != 'Offline'
        self.harness_config.validate_compatible_engine(self.engine_config)

        self.logger.info(f'HarnessConfig: \n{pformat(self.harness_config, compact=True)}')
        self.logger.info(f'EngineConfig: \n{pformat(self.engine_config, compact=True)}')

        self.start()

    def setup_interrupt_handler(self):
        current_process = psutil.Process()

        def exit_fn(signum, frame):
            self.logger.info("Received SIGINT. Stop LLMServer and cleanup.")

            children = current_process.children(recursive=True)
            for child in children:
                self.logger.verbose(f"Sending SIGKILL to child process: {child.pid}")
                os.kill(child.pid, signal.SIGKILL)

        signal.signal(signal.SIGINT, exit_fn)

    def start(self):
        self.logger.verbose("start() invoked.")

        # reset state
        self.sample_count = 0

        # initialize progress display
        additional_progress_units = {'tokens/s': 'mean'}
        if self.harness_config.gen_config.streaming:
            additional_progress_units |= {'TTFT(s)': '99%', 'TPOT(ms)': '99%'}
        additional_progress_units |= {'%kvcache_util': 'value'}

        self.progress_display = LLMServerProgressDisplay(
            total=self.sample_count,
            enable=(not self.verbose),
            additional_units=additional_progress_units,
            log_dir=self.harness_config.log_dir,
        )

        # start LLMCores + independent issue thread per core
        self.cores: List[LLMCore] = []
        self.get_next_core: Callable[[], LLMCore] = None
        self.initialize_cores()

        self.logger.verbose("start() completed.")

    def initialize_cores(self):
        """
        Initialize LLMCore instances.
        """
        world_size = self.engine_config.trtllm_config.pretrained_config.mapping.tp_size * \
            self.engine_config.trtllm_config.pretrained_config.mapping.pp_size
        self.num_cores = len(self.devices) // world_size
        self.logger.info(f"Initializing {self.num_cores}xDP instances of LLMCore(world_size={world_size}), on devices: {self.devices}")

        def get_devices_for_core(index: int) -> List[int]:
            return self.devices[index * (world_size): (index + 1) * (world_size)]

        # init cores
        self.cores = [
            LLMCore(
                name=f'core#{index}',
                device_ids=get_devices_for_core(index),
                complete_callback=self.complete_request,
                engine_config=self.engine_config,
                harness_config=self.harness_config,
                progress_display=self.progress_display,
                verbose=self.verbose,
                verbose_nvtx=self.verbose_nvtx,
            )
            for index in range(self.num_cores)
        ]

        # init scheduler
        self.reset_scheduler()

    def reset_scheduler(self) -> Callable[[], LLMCore]:
        """
        Resets the query-issue scheduler state.
        Scheduler determines which core will enqueue the next sample issued to server.

        LLMHarness Current supports two scheduler policies:
        - Round Robin: Cycles through the cores in a circular manner.
        - Load Balancing: Selects the core with the least number of tasks in its queue.

        Returns:
            Callable[[], LLMCore]: A function that, when called, returns the next LLMCore to handle a task.
        """
        scheduler_state = {
            'round_robin_index': -1
        }

        def round_robin() -> int:
            nonlocal scheduler_state
            current = scheduler_state['round_robin_index']
            scheduler_state['round_robin_index'] = next_index = (current + 1) % self.num_cores
            return self.cores[next_index]

        def load_balancing() -> int:
            # NOTE(vir):
            # load-balance by approximate num pending requests per core
            # approximate since its get_num_pending_samples() is not thread-safe
            queue_sizes = {index: core.get_num_pending_samples() for index, core in enumerate(self.cores)}
            least_busy_core = self.cores[min(queue_sizes, key=queue_sizes.get)]
            return least_busy_core

        self.get_next_core = {
            'round_robin': round_robin,
            'load_balancing': load_balancing,
        }[self.harness_config.traffic_distribution_policy]

    def stop_work(self):
        """
        Stop accepting new requests and signal LLMCores to complete all pending work.
        Cleanup corresponding Issue and Response resources.
        """
        self.logger.verbose(f"stop_work() invoked.")
        with self.nvtx_scope("stop_work"):
            # signal LLMCores to exit
            for core in self.cores:
                core.notify_stop()

            # wait to pending queries to complete
            self.flush_queries()

            # close progress display
            self.progress_display.finish()

            # cleanup gpu resources
            self.cores.clear()

        self.logger.info(f"Total Samples Completed={self.sample_count}")
        self.logger.verbose(f"stop_work() completed.")

    def warm_up(self):
        """
        Run Warm up iterations on all LLMCores.
        """
        with self.nvtx_scope("warm_up"):
            for core in self.cores:
                core.warm_up(warm_up_iters=100)

    def issue_queries(self, query_samples: List[Tuple[int, List[int], List[int]]]):
        """
        Issue a list of query samples to the LLMServer ie. queue it for execution.
        Query Sample : Tuple(request_id: int, input_tokens: List[int], stop_tokens: List[int])
        This request_id is passed along with result self.complete_request to "Complete" a query.

        Args:
            query_samples (List[Tuple[int, List[int], List[int]]]): A list of tuples where each tuple contains:
                - An integer representing the request ID.
                - A list of integers representing the input tokens to be processed.
                - A list of integers representing the stop tokens to be used.
        """
        # collect batches to issue to each core
        samples_per_core = defaultdict(list)
        for query in query_samples:
            samples_per_core[self.get_next_core()].append(query)

        # enqueue batches
        for core, samples in samples_per_core.items():
            self.sample_count += core.enqueue(samples)

        num_samples = len(query_samples)
        self.progress_display.update_total(self.sample_count)
        if self.verbose and num_samples > 1:
            queue_sizes = {core.name: core.get_num_pending_samples() for core in self.cores}
            self.logger.verbose(f"Issued +{num_samples} samples. Core Load Status: {queue_sizes}")

    @staticmethod
    def complete_request(request_id: int, output_tokens: List[int], is_first_token: bool):
        """
        "Complete" a given request. Example: Notify source (loadgen) of completed requests.
        """
        raise NotImplementedError()

    def flush_queries(self):
        """
        Flush (blocking await completion) pending queries from the LLMServer.
        """
        self.logger.verbose("flush_queries() invoked.")
        with self.nvtx_scope("flush_queries"):
            # await completion of pending requests
            for core in self.cores:
                core.flush()
        self.logger.verbose("flush_queries() completed.")
