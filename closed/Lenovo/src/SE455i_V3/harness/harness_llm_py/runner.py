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

import argparse
import gc
from importlib import import_module
from pathlib import Path

from code.common import logging, run_command
from code.common.utils import parse_cli_flags
from code.harness.harness_llm_py import LLMSUT, create_qsl_cls
from code.harness.harness_llm_py.llm_server import LLMChatServer, LLMDataset
from nvmitten.nvidia.cupy import CUDARTWrapper as cudart

try:
    import mlperf_loadgen as lg
except:
    logging.warning("Loadgen Python bindings are not installed. Installing Loadgen Python bindings!")
    run_command("make build_loadgen")
    import mlperf_loadgen as lg


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('true', '1'):
            return True
        elif v.lower() in ('false', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def load_dataset_cls(fpath):
        module_path, class_name = fpath.rsplit('.', 1)
        dataset_cls = getattr(import_module(module_path), class_name)
        return dataset_cls

    parser = argparse.ArgumentParser()

    # Test args
    parser.add_argument("--scenario", choices=["Offline", "Server", "SingleStream"], default="Offline")
    parser.add_argument("--test_mode", choices=["PerformanceOnly", "AccuracyOnly", "SubmissionRun"], default="PerformanceOnly")
    parser.add_argument("--model", choices=['gptj', 'llama2-70b', 'llama3_1-405b', 'llama2-70b-interactive', 'mixtral-8x7b'])  # NOTE(vir): enable models here
    parser.add_argument("--server_num_issue_query_threads", type=int, default=0, help="Number of IssueQuery threads used in Server scenario")
    parser.add_argument("--use_token_latencies", type=str2bool, default=-False, help="Report token throughput and latency information.")

    # QSL args
    parser.add_argument("--tensor_path", type=str)
    parser.add_argument("--performance_sample_count", type=int, default=5000)

    # SUT args
    parser.add_argument("--devices", type=str, default="all", help="Comma-separated numbered devices, use 'all' by default.")
    parser.add_argument("--enable_sort", type=str2bool, default=False, help="Sort the queries before sending to the batch manager.")
    parser.add_argument("--trtllm_checkpoint_flags", type=parse_cli_flags, default={}, help="Dictionary of checkpoint flags for TRTLLM engine(s) being run.")
    parser.add_argument("--trtllm_build_flags", type=parse_cli_flags, default={}, help="Dictionary of build flags for TRTLLM engine(s) being run.")
    parser.add_argument("--trtllm_runtime_flags", type=parse_cli_flags, default={}, help="Dictionary of runtime flags for TRTLLM.")
    parser.add_argument("--gpu_engines", type=str, default="all", help="Path to TRTLLM engine(s) dir.")
    parser.add_argument("--gpu_batch_size", type=int, default=1, help="Max Batch size to use for all devices and engines.")
    parser.add_argument("--verbose", type=str2bool, default=False, help="SUT verbose logging.")
    parser.add_argument("--verbose_nvtx", type=str2bool, default=False, help="SUT enable NVTX scopes ProfilingVerbosity.")
    parser.add_argument("--use_graphs", type=str2bool, default=False, help="Use cuda graph for inference.")
    parser.add_argument("--llm_gen_config_path", type=str, help="Path to json file storing the necessary configs for generation.")

    # Config args
    parser.add_argument("--mlperf_conf_path", help="Path to mlperf.conf")
    parser.add_argument("--user_conf_path", help="Path to user.conf")

    # Log args
    parser.add_argument("--log_mode", type=str, default="AsyncPoll", help="Logging mode for Loadgen")
    parser.add_argument("--log_mode_async_poll_interval_ms", type=int, default=1000, help="Specify the poll interval for asynchrounous logging")
    parser.add_argument("--logfile_outdir", type=str, default='', help="Specify the existing output directory for the LoadGen logs")
    parser.add_argument("--logfile_prefix", type=str, default='', help="Specify the filename prefix for the LoadGen log files")
    parser.add_argument("--logfile_suffix", type=str, default='', help="Specify the filename suffix for the LoadGen log files")

    # LLMHarness args
    parser.add_argument("--dataset_cls", type=str,
                        default='code.harness.harness_llm_py.dataset.LLMDataset',
                        help="Specify the class path for LLMDataset class to use for test.")

    args = parser.parse_args()

    # cleanup device list inplace
    if args.devices == "all":
        device_count = cudart.cudaGetDeviceCount()
        args.devices = list(range(device_count))
    else:
        args.devices = [int(x) for x in args.devices.split(',')]

    # load dataset_cls inplace
    args.dataset_cls = load_dataset_cls(args.dataset_cls)

    return args


def create_test_settings(args):
    scenario_map = {
        "Offline": lg.TestScenario.Offline,
        "SingleStream": lg.TestScenario.SingleStream,
        "Server": lg.TestScenario.Server,
    }
    test_mode_map = {
        "PerformanceOnly": lg.TestMode.PerformanceOnly,
        "AccuracyOnly": lg.TestMode.AccuracyOnly,
        "SubmissionRun": lg.TestMode.SubmissionRun,
    }

    test_settings = lg.TestSettings()
    test_settings.scenario = scenario_map[args.scenario]
    test_settings.mode = test_mode_map[args.test_mode]

    test_settings.FromConfig(args.mlperf_conf_path, args.model, args.scenario, 2)
    test_settings.FromConfig(args.user_conf_path, args.model, args.scenario, 1)
    test_settings.server_coalesce_queries = True
    test_settings.use_token_latencies = args.use_token_latencies
    test_settings.server_num_issue_query_threads = args.server_num_issue_query_threads

    return test_settings


def create_log_settings(args):
    log_mode_map = {
        "AsyncPoll": lg.LoggingMode.AsyncPoll,
        "EndOfTestOnly": lg.LoggingMode.EndOfTestOnly,
        "Synchronous": lg.LoggingMode.Synchronous,
    }

    Path(args.logfile_outdir).mkdir(parents=True, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.logfile_outdir
    log_output_settings.prefix = args.logfile_prefix
    log_output_settings.suffix = args.logfile_suffix
    log_output_settings.copy_summary_to_stdout = True

    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.log_mode = log_mode_map[args.log_mode]
    log_settings.log_mode_async_poll_interval_ms = args.log_mode_async_poll_interval_ms

    return log_settings


def create_qsl(args):
    qsl_cls = create_qsl_cls(args.dataset_cls)
    qsl = qsl_cls(sample_count=args.performance_sample_count,
                  tensor_path=args.tensor_path,
                  verbose=args.verbose)

    return qsl


def create_sut(args, dataset):
    sut = LLMSUT(scenario=args.scenario,
                 dataset=dataset,
                 devices=args.devices,
                 enable_sort=args.enable_sort,
                 trtllm_checkpoint_flags=args.trtllm_checkpoint_flags,
                 trtllm_build_flags=args.trtllm_build_flags,
                 trtllm_runtime_flags=args.trtllm_runtime_flags,
                 gpu_engine_dir=Path(args.gpu_engines).parent,
                 gpu_batch_size=args.gpu_batch_size,
                 verbose=args.verbose,
                 verbose_nvtx=args.verbose_nvtx,
                 log_dir=args.logfile_outdir,
                 use_graphs=args.use_graphs,
                 llm_gen_config_path=args.llm_gen_config_path)

    return sut


def run_chat(args):
    qsl = create_qsl(args)
    server = LLMChatServer(scenario=args.scenario,
                           dataset=qsl,
                           tokenizer_model_path='/work/build/models/Llama2/Llama-2-70b-chat-hf',
                           devices=args.devices,
                           enable_sort=args.enable_sort,
                           trtllm_checkpoint_flags=args.trtllm_checkpoint_flags,
                           trtllm_build_flags=args.trtllm_build_flags,
                           trtllm_runtime_flags=args.trtllm_runtime_flags,
                           gpu_engine_dir=Path(args.gpu_engines).parent,
                           gpu_batch_size=args.gpu_batch_size,
                           verbose=args.verbose,
                           verbose_nvtx=args.verbose_nvtx,
                           log_dir=args.logfile_outdir,
                           use_graphs=args.use_graphs,
                           llm_gen_config_path=args.llm_gen_config_path)

    test_prompt = "Hello world?"
    # can pass text queries and also qsl indices to LLMChatServer.infer([...])
    for request_id, chat_entry in server.infer([test_prompt, 0, 1]).items():
        print('=' * 10)
        print(chat_entry.request_text)
        print('-' * 1)
        print(chat_entry.response_tokens)
        print(len(chat_entry.response_tokens))
        print('=' * 10)

    # server.run_infer_loop()
    server.stop_work()


def run_interactive():
    return run_chat(parse_args())


def run_test(args):
    test_settings = create_test_settings(args)
    log_settings = create_log_settings(args)
    qsl = create_qsl(args)
    sut = create_sut(args, qsl)

    logging.info("Start Warm Up.")
    sut.warm_up()
    logging.info("Warm Up Done.")

    logging.info("Start Test.")
    gc.collect()
    gc.disable()
    sut.run_test(test_settings, log_settings)
    gc.enable()
    logging.info("Test Done.")


def run():
    return run_test(parse_args())


if __name__ == '__main__':
    run()
    # run_interactive()
