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
from dataclasses import dataclass
import datetime
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Set, Tuple

from code.common.utils import add_nvtx_scope_wrap
import tensorrt_llm
import tensorrt_llm.bindings.executor as trtllm

from .config import EngineConfig, HarnessConfig
from .utils import LLMServerProgressDisplay, add_prefix_logger, track_latencies


@dataclass
class LLMResponse():
    request_id: int
    output_tokens: List[int]
    sender: LLMCore
    enqueue_time: int
    is_first_token: bool = False


@add_prefix_logger(prefix_attr="name")
@add_nvtx_scope_wrap(prefix_attr="name")
class LLMCore():
    """
    LLMCore is a wrapper for TRTLLM Executor, can own multiple gpus (orchestrator of its own domain).
    Sets up trtllm executor by loading model.
    """

    EXECUTOR_WORKER_PATH = Path(tensorrt_llm.__file__).parent / 'bin' / 'executorWorker'

    def __init__(
        self,
        name: str,
        device_ids: List[int],
        complete_callback: Callable,
        engine_config: EngineConfig,
        harness_config: HarnessConfig,
        progress_display: LLMServerProgressDisplay,
        verbose: bool = False,
        verbose_nvtx: bool = False,
    ):
        self.name = name
        self.device_ids = device_ids
        self.engine_config = engine_config
        self.harness_config = harness_config
        self.progress_display = progress_display
        self.verbose = verbose
        self.verbose_nvtx = verbose_nvtx
        self.complete_callback = complete_callback

        scheduler_policy = {
            'max_util': trtllm.CapacitySchedulerPolicy.MAX_UTILIZATION,
            'no_evict': trtllm.CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
            'static': trtllm.CapacitySchedulerPolicy.STATIC_BATCH,
        }[self.harness_config.trtllm_runtime_flags['batch_scheduler_policy']]

        context_chunking_policy = {
            'equal_progress': trtllm.ContextChunkingPolicy.EQUAL_PROGRESS,
            'first_come_first_served': trtllm.ContextChunkingPolicy.FIRST_COME_FIRST_SERVED,
        }[self.harness_config.trtllm_runtime_flags['context_chunking_policy']]

        batching_type = {
            True: trtllm.BatchingType.INFLIGHT,
            False: trtllm.BatchingType.STATIC,
        }[self.harness_config.trtllm_runtime_flags['use_inflight_batching']]

        executor_config = trtllm.ExecutorConfig(
            max_beam_width=self.harness_config.gen_config.runtime_beam_width,
            max_batch_size=int(self.harness_config.trtllm_runtime_flags['max_batch_size']),
            max_num_tokens=int(self.harness_config.trtllm_runtime_flags['max_num_tokens']),
            max_queue_size=-1,
            scheduler_config=trtllm.SchedulerConfig(
                capacity_scheduler_policy=scheduler_policy,
                context_chunking_policy=context_chunking_policy,
                dynamic_batch_config=trtllm.DynamicBatchConfig(
                    enable_batch_size_tuning=self.harness_config.trtllm_runtime_flags['enable_batch_size_tuning'],
                    enable_max_num_tokens_tuning=self.harness_config.trtllm_runtime_flags['enable_max_num_tokens_tuning'],
                    dynamic_batch_moving_average_window=self.harness_config.trtllm_runtime_flags['dynamic_batch_moving_average_window']
                )
            ),
            kv_cache_config=trtllm.KvCacheConfig(
                enable_block_reuse=False,
                free_gpu_memory_fraction=self.harness_config.trtllm_runtime_flags['kvcache_free_gpu_mem_frac'],
            ),
            enable_chunked_context=self.harness_config.trtllm_runtime_flags['enable_chunked_context'],
            batching_type=batching_type,
            parallel_config=trtllm.ParallelConfig(
                communication_type=trtllm.CommunicationType.MPI,
                communication_mode=trtllm.CommunicationMode.ORCHESTRATOR,
                device_ids=device_ids,
                orchestrator_config=trtllm.OrchestratorConfig(
                    is_orchestrator=True,
                    worker_executable_path=str(LLMCore.EXECUTOR_WORKER_PATH),
                )
            )
        )

        self.logger.info(f"Loading TensorRT-LLM engine: {self.engine_config.engine_dir}.")
        with self.nvtx_scope(f"trtllm_executor_init"):
            self.executor = trtllm.Executor(
                model_path=self.engine_config.engine_dir,
                model_type=trtllm.ModelType.DECODER_ONLY,
                executor_config=executor_config
            )
        self.logger.info(f"Executor Using Devices: #{self.device_ids}.")
        assert self.executor.can_enqueue_requests(), "Executor failed to initialize"

        self.output_config = None
        self.sampling_config = None
        self.processed_count = 0
        self.initialize_inference_args()

        # pending samples queue tracks mapping (loadgen-id: trtllm request-id)
        # issue-thread (main-thread) and response-thread access this under mutex
        self.lock = threading.Lock()
        self.pending_samples: Dict[int, Tuple[int, int]] = {}

        self.stop_work = threading.Event()
        self.flush_signal = threading.Condition()
        self.response_thread = None
        self.response_thread_exit = threading.Event()
        self.initialize_response_thread()

    def initialize_inference_args(self):
        """
        Initialize inference configuration arguments for TRTLLM Executor.
        """
        self.output_config = trtllm.OutputConfig(
            exclude_input_from_output=self.harness_config.trtllm_runtime_flags['exclude_input_from_output'],
        )

        self.sampling_config = trtllm.SamplingConfig(
            beam_width=self.harness_config.gen_config.runtime_beam_width,
            temperature=self.harness_config.gen_config.temperature,
            min_tokens=self.harness_config.gen_config.min_output_len,
            top_k=self.harness_config.gen_config.top_k,
            top_p=self.harness_config.gen_config.top_p,
            seed=self.harness_config.random_seed,
        )

    def initialize_response_thread(self):
        """
        Initialize the response thread.
        """
        self.response_thread = threading.Thread(target=self.poll_responses, args=())
        self.response_thread.daemon = True
        self.response_thread.start()

    def __del__(self):
        if not self.response_thread_exit.is_set():
            self.response_thread_exit.wait()

        self.logger.info(f"Completed {self.processed_count} samples.")

    def enqueue(self, queries: List[Tuple[int, List[int], List[int]]]) -> List[int]:
        """
        Enqueue input samples to TRTLLM Executor.

        Args:
            queries (List[Tuple[int, List[int], List[int]]]): A list of tuples where each tuple contains:
                - An integer representing the request ID.
                - A list of integers representing the input tokens to be processed.
                - A list of integers representing the stop tokens to be used.

        Returns:
            int: The number of samples enqueued.
        """
        assert not self.stop_work.is_set(), "Cannot issue queries after stop_work has been signalled to core"

        request_ids, input_tokens, stop_tokens = zip(*queries)
        enqueue_batch = [
            trtllm.Request(input_token_ids=tok_input,
                           max_tokens=self.harness_config.gen_config.max_output_len,
                           streaming=self.harness_config.gen_config.streaming,
                           sampling_config=self.sampling_config,
                           output_config=self.output_config,
                           end_id=self.harness_config.gen_config.eos_token_id,
                           stop_words=tok_stop)
            for tok_input, tok_stop in zip(input_tokens, stop_tokens)
        ]

        with self.lock:
            enqueue_time = time.time()
            trtllm_request_ids = self.executor.enqueue_requests(enqueue_batch)
            self.pending_samples |= {
                trtllm_request_id: (request_id, enqueue_time)
                for request_id, trtllm_request_id in zip(request_ids, trtllm_request_ids)
            }

        return len(enqueue_batch)

    def get_num_pending_samples(self) -> int:
        """
        Get (approx) number of in-flight samples for this core.
        """
        # NOTE(vir):
        # not making this thread safe since its used for load balancing and approx queue size is sufficient
        return len(self.pending_samples)

    @track_latencies
    def poll_responses(self):
        """
        Collected outputs from the executor, complete with self.complete_callback.
        If core has been requested to stop work, response thread will still wait to complete all pending samples.
        """
        self.logger.verbose(f"Core response thread started.")

        # buffers for streaming mode
        output_tokens: Dict[int, List[int]] = defaultdict(list)
        first_token_latencies: Dict[int, int] = {}

        while True:
            with self.lock:
                num_pending = len(self.pending_samples)

            if num_pending == 0:
                with self.flush_signal:
                    self.flush_signal.notify()

                if self.stop_work.is_set():
                    break

            timeout = datetime.timedelta(milliseconds=1)  # datetime.timedelta(milliseconds=1)
            responses = self.executor.await_responses(timeout)

            # batched updates for progress display
            num_completed, num_toks, ttfts, tpots = 0, 0, [], []

            for response in responses:
                if response.has_error():
                    raise RuntimeError(f"{self.name} :: Processing response: {response.request_id} encountered error: {response.error_msg}")

                if self.harness_config.gen_config.streaming:
                    # each response is 1 token in streaming mode
                    is_first_token = response.request_id not in output_tokens
                    is_final_token = response.result.is_final

                    for beam, output_toks_ in enumerate(response.result.output_token_ids):
                        output_tokens[response.request_id].extend(output_toks_)
                        num_output_toks = len(output_tokens[response.request_id])

                        if not (is_first_token or is_final_token):
                            continue

                        assert (is_first_token and num_output_toks >= 1) or \
                               (is_final_token and num_output_toks >= self.harness_config.gen_config.min_output_len)

                        with self.lock:
                            request_id, enqueue_time = self.pending_samples[response.request_id]
                            flight_time = time.time() - enqueue_time

                            if is_final_token:  # stop keeping track of id-mapping
                                del self.pending_samples[response.request_id]

                        output_toks = output_tokens[response.request_id]
                        if num_output_toks <= 1:
                            output_toks += [self.harness_config.gen_config.eos_token_id]
                            num_output_toks += 1

                        self.complete_callback(request_id=request_id,
                                               output_tokens=output_toks,
                                               is_first_token=is_first_token and (not is_final_token))

                        if is_first_token:
                            first_token_latencies[response.request_id] = flight_time
                            ttfts.append(flight_time)

                        if is_final_token:
                            ttft = first_token_latencies[response.request_id]
                            tpot = ttft if num_output_toks <= 1 else ((flight_time - ttft) / (num_output_toks - 1))
                            tpots.append(tpot * 1000)

                            num_completed += 1
                            num_pending -= 1
                            del output_tokens[response.request_id]  # cleanup reference to output list

                        num_toks += num_output_toks
                        self.logger.verbose(f"Completed request #{request_id} "
                                            f"(len={num_output_toks}, is_final={is_final_token}) "
                                            f"[pending={num_pending}]")

                        # we only consier beam=0 since ordering is in descending order of cumLogProbs
                        # NOTE(vir): no model uses both streaming mode and >1 runtime_beams as of yet
                        break

                else:
                    # each response is final in non-streaming mode
                    for beam, output_toks in enumerate(response.result.output_token_ids):
                        num_output_toks = len(output_toks)
                        assert num_output_toks >= self.harness_config.gen_config.min_output_len

                        with self.lock:
                            request_id, enqueue_time = self.pending_samples.pop(response.request_id)

                        if num_output_toks <= 1:
                            output_toks += [self.harness_config.gen_config.eos_token_id]
                            num_output_toks += 1

                        self.complete_callback(request_id=request_id,
                                               output_tokens=output_toks,
                                               is_first_token=False)

                        num_completed += 1
                        num_pending -= 1
                        num_toks += num_output_toks
                        self.logger.verbose(f"Completed request #{request_id} "
                                            f"(len={num_output_toks}, is_final=True)"
                                            f"[pending={num_pending}]")

                        # we only consier beam=0 since ordering is in descending order of cumLogProbs
                        break

            self.processed_count += num_completed
            if self.progress_display.enabled:
                # update progress-display once per batch of responses
                additional_unit_updates = {'tokens/s': num_toks}
                if self.harness_config.gen_config.streaming:
                    additional_unit_updates |= {'TTFT(s)': ttfts, 'TPOT(ms)': tpots}

                # log iteration stats once per batch of responses
                if self.executor.can_enqueue_requests() and (stats := self.executor.get_latest_iteration_stats()):
                    stats = [json.loads(stat.to_json_str()) for stat in stats]
                    self.progress_display.record_iteration_stats(stats)

                    # use latest iteration for progress_display update
                    latest = stats[-1]
                    kvcache_util = 100 * (float(latest["kvCacheStats"]["usedNumBlocks"]) / float(latest["kvCacheStats"]["maxNumBlocks"]))
                    additional_unit_updates |= {'%kvcache_util': kvcache_util}

                self.progress_display.update(completed=num_completed, additional_unit_updates=additional_unit_updates)

        self.executor.shutdown()
        with self.lock:
            assert len(self.pending_samples) == 0, f"Core stopped with self.pending_samples non-empty: {len(self.pending_samples)}"

        self.response_thread_exit.set()  # disable flushing
        with self.flush_signal:  # wake any pending flush
            self.flush_signal.notify()

        self.logger.verbose(f"Core response thread complete.")

    def warm_up(self, warm_up_iters: int = 100):
        """
        Run Warm up iterations on TRTLLM Executor.

        Args:
            warm_up_iters (int): The number of warm-up iterations.
        """
        # raise NotImplementedError()
        pass

    def notify_stop(self):
        """Notify core stop working once all pending requests are completed."""
        self.stop_work.set()
        with self.lock:
            self.logger.verbose(f"notified to stop work, pending: {len(self.pending_samples)}")

    def flush(self):
        """
        Block until all pending requests complete.
        """
        self.logger.verbose(f"flush() invoked.")
        with self.nvtx_scope("flush"):
            if not self.response_thread_exit.is_set():
                with self.flush_signal:
                    self.flush_signal.wait()
        self.logger.verbose(f"flush() completed.")
