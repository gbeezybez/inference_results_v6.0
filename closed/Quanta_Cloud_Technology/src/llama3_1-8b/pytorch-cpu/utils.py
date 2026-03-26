import subprocess
import argparse
import dataclasses
from typing import Tuple, List
import torch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

@dataclasses.dataclass
class RunnerArgs:
    """Arguments for the SGLang runner."""
    batch_size: int = 1
    dataset_path: str = ""
    total_sample_count: int = 1000
    scenario: str = "Offline"
    url: Tuple[str] = ("http://127.0.0.1:3000",)
    workload_name: str = "llama3_1-8b"
    #device: str = "cpu"
    run_name: str = "llama3_1-8b-run"
    accuracy: bool = False
    audit_conf: str = "audit.conf"
    user_conf: str = "user.conf"
    output_log_dir: str = "output-logs"
    enable_log_trace: bool = False
    tensor_parallel: int = 1
    num_workers: int = 1
    quantized: bool = False
    warmup: bool = False
    delegate_batching: bool = False
    streaming: bool = False  # Enable streaming mode
    use_async_server: bool = False  # Use async server for processing requests
    mode: str = "performance"  # 'performance' or 'accuracy'

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=RunnerArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, default=RunnerArgs.batch_size
        )
        parser.add_argument(
            "--dataset-path", type=str, default=RunnerArgs.dataset_path, help="Path to the dataset file."
        )
        parser.add_argument(
            "--total-sample-count", type=int, default=RunnerArgs.total_sample_count, help="Total number of samples to load."
        )
        parser.add_argument("--scenario", type=str, choices=["Offline", "Server", "offline", "server"], default="Offline", help="Scenario")
        parser.add_argument("--url", type=str, nargs='+', required=True, help="URL of the SGLang server(s)")
        parser.add_argument("--workload-name", type=str, default="llama3_1-8b")
        parser.add_argument("--accuracy", action="store_true", help="Run accuracy mode")
        parser.add_argument("--audit-conf", type=str, default="audit.conf", help="audit config for LoadGen settings during compliance runs")
        parser.add_argument("--user-conf", type=str, default="user.conf", help="user config for user LoadGen settings such as target QPS")
        parser.add_argument("--output-log-dir", type=str, default="output-logs", help="Where logs are saved")
        parser.add_argument("--enable-log-trace", action="store_true", help="Enable log tracing. This file can become quite large")
        parser.add_argument("--tensor-parallel", '-tp-size', type=int, default=1, help="Tensor parallel size")
        parser.add_argument("--num-workers", type=int, default=1, help="Number of workers to process queries")
        parser.add_argument("--quantized", action='store_true', help="If using a AWQ quantized model")
        parser.add_argument("--warmup", action='store_true', help="Do warmup")
        parser.add_argument("--delegate-batching", action='store_true', help="Delegate batching to the Instances")
        parser.add_argument("--streaming", action='store_true',
                            help="Enable streaming mode for the runner."
                            " In this mode, the runner will not wait for the entire batch to be processed before returning the results."
                            " Instead, it will return the results as soon as they are available."
                            )
        parser.add_argument("--use-async-server", action='store_true', help="Use async server for processing requests")
        parser.add_argument("--mode", type=str, choices=["performance", "accuracy", "Performance", "Accuracy"], default="performance", help="Mode of the test: performance or accuracy")
        
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )
    
def get_start_cores():
    start_cores = subprocess.check_output('lscpu | grep "NUMA node.* CPU.*" | awk "{print \$4}" | cut -d "-" -f 1', shell=True)
    start_cores = start_cores.decode('ascii').rstrip().split('\n')
    start_cores = [int(_) for _ in start_cores]
    return start_cores

@torch.no_grad
def extend(reqs, model_runner):
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=None,
        model_config=model_runner.model_config,
        enable_overlap=False,
        enable_custom_logit_processor=False,
    )
    batch.prepare_for_extend()
    #_maybe_prepare_dp_attn_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits, batch


@torch.no_grad
def decode(input_token_ids, batch, model_runner):
    batch.output_ids = input_token_ids
    batch.prepare_for_decode()
    #_maybe_prepare_dp_attn_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits

@torch.no_grad
def create_request(
    input_token_ids: List[int] = [],
    req_id: int = 0,
    sampling_params: dict = {},
) -> Req:
    """Create a request for the model runner."""

    req = Req(
            rid=req_id,
            origin_input_text="",
            origin_input_ids=list(input_token_ids),
            sampling_params=sampling_params,
        )
    req.prefix_indices = []
    req.fill_ids = req.origin_input_ids
    req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
    req.logprob_start_len = len(req.origin_input_ids) - 1
    return req

def _maybe_prepare_dp_attn_batch(batch: ScheduleBatch, model_runner):
    if model_runner.server_args.enable_dp_attention:
        Scheduler.prepare_dp_attn_batch_raw(
            batch,
            dp_size=model_runner.server_args.dp_size,
            attn_tp_size=1,
            tp_cpu_group=model_runner.tp_group.cpu_group,
            get_idle_batch=None,
            disable_cuda_graph=model_runner.server_args.disable_cuda_graph,
            speculative_num_draft_tokens=None,
        )
