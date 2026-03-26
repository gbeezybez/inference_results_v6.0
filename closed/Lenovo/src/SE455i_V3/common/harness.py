# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
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

import re
import os
import sys
from importlib import import_module
from pathlib import Path
import numpy as np
import tensorrt as trt

from nvmitten.nvidia.accelerator import GPU

import code.common.arguments as common_args
from code.common import logging, dict_get, run_command, args_to_string
from code.common.constants import Benchmark, Scenario, G_HIGH_ACC_ENABLED_BENCHMARKS, G_MLCOMMONS_INF_REPO_PATH, VERSION
from code.common.log_parser import from_loadgen_by_keys, get_loadgen_log_keys
from code.common.submission import generate_measurements_entry
from code.common.fix_sys_path import ScopedRestrictedImport
from code.common.systems.system_list import DETECTED_SYSTEM
from code.common.utils import get_e2e_batch_size, ScopeWrap
from code.plugin import get_trt_plugin_paths_by_network
from code import get_cls

_new_path = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, "tools", "submission")] + sys.path
with ScopedRestrictedImport(_new_path):
    submission_checker = import_module("submission_checker")
    # Use v3.1 as the fallback version if not found
    _version_str = "v6.0"
    benchmark_qsl_size_map = {
      "resnet": 1024,
      "retinanet": 64,
      "bert-99": 10833,
      "bert-99.9": 10833,
      "dlrm-v2-99": 204800,
      "dlrm-v2-99.9": 204800,
      "3d-unet-99": 43,
      "3d-unet-99.9": 43,
      "llama3.1-8b": 13368,
      "llama3.1-8b-edge": 5000,
      "llama2-70b-99": 24576,
      "llama2-70b-99.9": 24576,
      "stable-diffusion-xl": 5000,
      "mixtral-8x7b": 15000,
      "llama3.1-405b": 8313,
      "rgat": 788379,
      "pointpainting": 1024,
      "deepseek-r1": 4388,
      "whisper": 1633,
      "gpt-oss-120b": 6396,
      "qwen3-vl-235b-a22b": 48289,
      "wan-2.2-t2v-a14b": 247,
      "dlrm-v3": 34996,
      "yolo-95": 64,
      "yolo-99": 64
    }
    # Set to 2048 since the value in MLCommons repo is 1024, which would cause BS2048 to not be contiguous, which is a
    # commonly used batch size in our A100/H100 configs.
    benchmark_qsl_size_map["resnet"] = 2048
    benchmark_qsl_size_map["resnet50"] = benchmark_qsl_size_map["resnet"]  # submission-checker uses 'resnet' instead of 'resnet50'
    # TODO: Remove these when DLRM is purged from MLPerf
    benchmark_qsl_size_map["dlrm-99"] = benchmark_qsl_size_map["dlrm-99.9"] = 204800
    # TODO: Remove when we add it in loadgen
    benchmark_qsl_size_map["mixtral-8x7b"] = benchmark_qsl_size_map["mixtral-8x7b-99"] = 15000

    # NOTE(vir): primary name/key is 3_1 not 3.1
    benchmark_qsl_size_map['llama3_1-405b'] = benchmark_qsl_size_map['llama3.1-405b']

    # Check for query constraints documented in https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#scenarios
    _min_queries = {
    "resnet": {
        "SingleStream": 1024,
        "MultiStream": 270336,
        "Server": 270336,
        "Offline": 1,
    },
    "retinanet": {
        "SingleStream": 1024,
        "MultiStream": 270336,
        "Server": 270336,
        "Offline": 1,
    },
    "bert-99": {"SingleStream": 1024, "Offline": 1},
    "bert-99.9": {"SingleStream": 1024, "Offline": 1},
    "dlrm-v2-99": {"Server": 270336, "Offline": 1},
    "dlrm-v2-99.9": {"Server": 270336, "Offline": 1},
    "3d-unet-99": {"SingleStream": 1024, "Offline": 1},
    "3d-unet-99.9": {"SingleStream": 1024, "Offline": 1},
    "llama3.1-8b": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
    "llama3.1-8b-edge": {"SingleStream": 1024, "Offline": 1},
    "llama2-70b-99": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
    "llama2-70b-99.9": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
    "stable-diffusion-xl": {
        "SingleStream": 1024,
        "Server": 270336,
        "Offline": 1,
    },
    "mixtral-8x7b": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
    "llama3.1-405b": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
    "rgat": {"SingleStream": 1024, "Offline": 1},
    "pointpainting": {"SingleStream": 1024},
    "deepseek-r1": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
    "whisper": {"SingleStream": 1024, "Offline": 1},
    "gpt-oss-120b": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
    "qwen3-vl-235b-a22b": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
    "dlrm-v3": {"Server": 270336, "Offline": 1},
    "yolo-95": {"SingleStream": 1024, "MultiStream": 270336, "Offline": 1},
    "yolo-99": {"SingleStream": 1024, "MultiStream": 270336, "Offline": 1},
    "wan-2.2-t2v-a14b": {"SingleStream": 247, "Offline": 1}
}
    # Offline uses min. samples/query since min query count is always 1. For other scenarios, these values are the same
    # across benchmarks.
    QUERY_METRIC_CONSTRAINTS = {
        "Offline": ("effective_samples_per_query", 24576),
        "Server": ("effective_min_query_count", _min_queries["resnet"]["Server"]),
        "MultiStream": ("effective_min_query_count", _min_queries["resnet"]["MultiStream"]),
        "SingleStream": ("effective_min_query_count", _min_queries["resnet"]["SingleStream"]),
    }


class BaseBenchmarkHarness:
    """Base class for benchmark harnesses."""

    def __init__(self, args, benchmark):
        self.args = args
        self.benchmark = benchmark
        self.name = self.benchmark.valstr()
        self.workload_setting = dict_get(args, "workload_setting", default=None)
        self.verbose = dict_get(args, "verbose", default=None)
        self.verbose_nvtx = dict_get(args, "verbose_nvtx", default=None)
        self.verbose_glog = dict_get(args, "verbose_glog", default=None)
        if self.verbose:
            logging.info(f"===== Harness arguments for {self.name} =====")
            for key in args:
                logging.info("{:}={:}".format(key, args[key]))

        self.system_id = args["system_id"]
        self.scenario = args["scenario"]
        self.config_ver = args["config_ver"]
        self.engine_dir = "./build/engines/{:}/{:}/{:}".format(self.system_id, self.name, self.scenario.valstr())
        self.default_engine_dir = True
        if "engine_dir" in args:
            self.engine_dir = args["engine_dir"]
            self.default_engine_dir = False
        self.precision = args["precision"]
        self.has_gpu = dict_get(args, "gpu_batch_size", default=None) is not None
        self.has_dla = dict_get(args, "dla_batch_size", default=None) is not None
        self.nms_type = dict_get(args, "nms_type", default=None)

        # Enumerate engine files
        # Engine not needed if we are only generating measurements/ entries
        self.skip_file_checks = dict_get(self.args, "skip_file_checks", False)
        self.gpu_engine = None
        self.dla_engine = None
        self.use_triton = dict_get(self.args, "use_triton", False)
        if self.use_triton:
            self.verify_triton_repo()

        self.enumerate_engines()

        # Enumerate harness executable
        self.executable = self._get_harness_executable()
        self.check_file_exists(self.executable)

        self.mitten_workload = self._get_harness_mitten_workload()
        self.use_jemalloc = False

        self.env_vars = os.environ.copy()
        self.flag_builder_custom_args = []

        # run harness under modified vboost if set (for SM >= 90)
        if int(DETECTED_SYSTEM.extras["primary_compute_sm"]) >= 90 and 'vboost_slider' in args:
            def set_vboost(value=0):
                logging.info(f"setting vboost to {value if value != 0 else 'gpu default'}")
                run_command(f"sudo nvidia-smi boost-slider --vboost {value}")

            self.set_vboost_fn = lambda: set_vboost(self.args['vboost_slider'])
            self.reset_vboost_fn = set_vboost
        else:
            self.set_vboost_fn = self.reset_vboost_fn = None

    def _get_harness_executable(self):
        raise NotImplementedError("BaseBenchmarkHarness cannot be called directly")

    def _get_harness_mitten_workload(self):
        return None

    def _construct_terminal_command(self, argstr):
        return f"{self.executable} {argstr}"

    def _build_custom_flags(self, flag_dict):
        """
        Handles any custom flags to insert into flag_dict. Can return either a flag_dict, or a converted arg string.
        """
        return flag_dict

    def _get_engine_fpath(self, device_type, component, component_batch_size):
        return f"{self.engine_dir}/{self.name}-{self.scenario.valstr()}-{device_type}-{component.valstr()}-b{component_batch_size}-{self.precision}.{self.config_ver}.plan"

    def _append_config_ver_name(self, system_name):
        if "maxq" in self.config_ver.lower():
            system_name += "_MaxQ"
        if "hetero" in self.config_ver.lower():
            system_name += "_HeteroMultiUse"
        return system_name

    def get_system_name(self, add_trt=True):
        override_system_name = dict_get(self.args, "system_name", default=None)
        if override_system_name not in {None, ""}:
            return override_system_name

        system_name = self.system_id

        if add_trt:
            system_name = f"{system_name}_TRT"

        return self._append_config_ver_name(system_name)

    def _get_submission_benchmark_name(self):
        full_benchmark_name = self.name
        if dict_get(self.args, "accuracy_level", "99%") == "99.9%":
            full_benchmark_name += "-99.9"
        elif self.name in G_HIGH_ACC_ENABLED_BENCHMARKS:
            full_benchmark_name += "-99"
        return full_benchmark_name

    def get_full_log_dir(self):
        return os.path.join(self.args["log_dir"], self.get_system_name(), self._get_submission_benchmark_name(),
                            self.scenario.valstr())

    def verify_triton_repo(self):
        """
            Check if the triton model repository has the reqd config.pbtxt
        """
        num_servers = dict_get(self.args, "triton_num_servers", 1)
        models = [Path(f"/work/build/triton_model_repo_{i}/") for i in range(num_servers)]
        for model in models:
            assert (model / "config.pbtxt").exists, f"Model {model} is missing its config.pbtxt. Please run make generate_triton_config"

    def enumerate_engines(self):
        # e2e batch size calculate and check
        # set e2e batch size and engine batch size
        if self.has_gpu:
            gpu_e2e_batch_size = get_e2e_batch_size(self.args["gpu_batch_size"])
            self.args["gpu_engine_batch_size"] = self.args["gpu_batch_size"]
            self.args["gpu_batch_size"] = gpu_e2e_batch_size

            gpu_engine_list = []
            gpu_engine_batch_size_list = []
            for component, component_batch_size in self.args["gpu_engine_batch_size"].items():
                engine_path = self._get_engine_fpath("gpu", component, component_batch_size)
                if not self.use_triton:
                    self.check_file_exists(engine_path)
                gpu_engine_list.append(engine_path)
                gpu_engine_batch_size_list.append(str(component_batch_size))
            self.gpu_engine = ','.join(gpu_engine_list)
            self.args["gpu_engine_batch_size"] = ','.join(gpu_engine_batch_size_list)

        if self.has_dla:
            dla_e2e_batch_size = get_e2e_batch_size(self.args["dla_batch_size"])
            self.args["dla_engine_batch_size"] = self.args["dla_batch_size"]
            self.args["dla_batch_size"] = dla_e2e_batch_size

            dla_engine_list = []
            dla_engine_batch_size_list = []
            for component, component_batch_size in self.args["dla_engine_batch_size"].items():
                engine_path = self._get_engine_fpath("dla", component, component_batch_size)
                self.check_file_exists(engine_path)
                dla_engine_list.append(engine_path)
                dla_engine_batch_size_list.append(str(component_batch_size))
            self.dla_engine = ','.join(dla_engine_list)
            self.args["dla_engine_batch_size"] = ','.join(gpu_engine_batch_size_list)

    def check_file_exists(self, f):
        """Check if file exists. Complain if configured to do so."""

        if not os.path.isfile(f):
            if self.skip_file_checks:
                print(f"Note: File {f} does not exist. Attempting to continue regardless, as hard file checks are disabled.")
                return False
            else:
                raise RuntimeError("File {:} does not exist.".format(f))
        return True

    def build_default_flags(self):
        flag_dict = {}
        flag_dict["verbose"] = self.verbose
        flag_dict["verbose_nvtx"] = self.verbose_nvtx
        flag_dict["v"] = self.verbose_glog

        # Handle plugins
        plugins = get_trt_plugin_paths_by_network(self.name, self.args)
        if len(plugins) > 0:
            logging.info(f"The harness will load {len(plugins)} plugins: {plugins}")
            flag_dict["plugins"] = ",".join(plugins)

        # Generate flags for logfile names.
        log_dir = self.get_full_log_dir()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        flag_dict["logfile_outdir"] = log_dir
        flag_dict["logfile_prefix"] = "mlperf_log_"

        # Handle performance sample count
        perf_sample_count = dict_get(self.args, "performance_sample_count", None)
        perf_sample_count_override = dict_get(self.args, "performance_sample_count_override", None)
        if perf_sample_count_override is not None:
            flag_dict["performance_sample_count"] = perf_sample_count_override
        elif perf_sample_count is not None:
            flag_dict["performance_sample_count"] = perf_sample_count
        elif benchmark_qsl_size_map[self._get_submission_benchmark_name()] > 0:
            flag_dict["performance_sample_count"] = benchmark_qsl_size_map[self._get_submission_benchmark_name()]
        else:
            flag_dict["performance_sample_count"] = self.args["gpu_batch_size"]

        # Handle custom arguments
        for arg in self.flag_builder_custom_args:
            val = dict_get(self.args, arg, None)
            if val is not None:
                flag_dict[arg] = val

        return flag_dict

    def build_scenario_specific_flags(self):
        """Return flags specific to current scenario."""

        flag_dict = {}

        scenario_keys = common_args.getScenarioMetricArgs(self.scenario)

        for arg in scenario_keys:
            val = dict_get(self.args, arg, None)
            if val is None:
                raise ValueError("Missing required key {:}".format(arg))
            flag_dict[arg] = val

        # Handle RUN_ARGS
        for arg in scenario_keys:
            val = dict_get(self.args, arg, None)
            if val is not None:
                flag_dict[arg] = val

        return flag_dict

    def build_non_custom_flags(self):
        """Returns the flag_dict for all flags excluding custom ones.
        """
        flag_dict = self.build_default_flags()
        flag_dict.update(self.build_scenario_specific_flags())

        # Handle engines
        if self.has_gpu:
            flag_dict["gpu_engines"] = self.gpu_engine

        # MLPINF-853: Special handing of --test_run. Use min_duration=60000 and min_query_count=1.
        if flag_dict.get("test_run", False):
            if "min_duration" not in flag_dict:
                flag_dict["min_duration"] = 60000
            if "min_query_count" not in flag_dict:
                # let 3D-UNet to run at least 3x of number of samples
                flag_dict["min_query_count"] = 129 if self.benchmark == Benchmark.UNET3D else 1
            flag_dict["test_run"] = None
        if "min_duration" in flag_dict:
            logging.info(f"min_duration is overwritten to {flag_dict['min_duration']}.")
        if "min_query_count" in flag_dict:
            logging.info(f"min_query_cnt is overwritten to {flag_dict['min_query_count']}.")

        return flag_dict

    def prepend_ld_preload(self, so_path):
        if "LD_PRELOAD" in self.env_vars:
            self.env_vars["LD_PRELOAD"] = ":".join([so_path, self.env_vars["LD_PRELOAD"]])
        else:
            self.env_vars["LD_PRELOAD"] = so_path

        logging.info("Updated LD_PRELOAD: " + self.env_vars["LD_PRELOAD"])

    def run_harness(self, flag_dict=None, skip_generate_measurements=False, use_py_harness=False):
        if flag_dict is None:
            flag_dict = self.build_non_custom_flags()

        if not skip_generate_measurements:
            # Generates the entries in the `measurements/` directory, and updates flag_dict accordingly
            generate_measurements_entry(
                self.get_system_name(),
                self.name,
                self._get_submission_benchmark_name(),
                self.scenario,
                self.args["input_dtype"],
                self.args["precision"],
                flag_dict)

        argstr = self._build_custom_flags(flag_dict)
        if type(argstr) is dict:
            argstr = args_to_string(flag_dict)

        # Handle environment variables
        if self.use_jemalloc:
            import platform
            self.prepend_ld_preload(f"/usr/lib/{platform.processor()}-linux-gnu/libjemalloc.so.2")

        with ScopeWrap(self.set_vboost_fn, self.reset_vboost_fn):
            if use_py_harness and self.benchmark in [Benchmark.ResNet50, Benchmark.Retinanet, Benchmark.UNET3D]:
                logging.info(f"Using pybind11 for this workload: {self.benchmark} {self.scenario} {self.system_id} ...")
                workload = get_cls(self.mitten_workload)(flag_dict)
                workload.run()
            else:
                if use_py_harness:
                    logging.warning("This workload does not support the new Python harness.")
                logging.info("Using harness launch command...")
                output = run_command(self._construct_terminal_command(argstr),
                                     get_output=True,
                                     custom_env=self.env_vars)

        # Return harness result.
        scenario_key = get_loadgen_log_keys(self.scenario, self.benchmark)
        query_metric_key = QUERY_METRIC_CONSTRAINTS[self.scenario.valstr()][0]
        loadgen_query_keys = ["result_validity", scenario_key, "early_stopping_met", query_metric_key, "effective_min_duration_ms"]

        # Add result sample per second (QPS) when running Llama2 offline
        if self.benchmark in [
            Benchmark.LLAMA2,
            Benchmark.LLAMA2_Interactive,
            Benchmark.Mixtral8x7B,
        ] and self.scenario in [Scenario.Offline]:
            loadgen_query_keys.append("result_samples_per_second")

        results = from_loadgen_by_keys(os.path.join(self.args["log_dir"],
                                                    self.get_system_name(),
                                                    self._get_submission_benchmark_name(),
                                                    self.scenario.valstr()),
                                       loadgen_query_keys)
        test_mode = flag_dict.get("test_mode", "PerformanceOnly")
        satisfies_query_constraint = float(results.get(query_metric_key, "0.0")) \
            >= QUERY_METRIC_CONSTRAINTS[self.scenario.valstr()][1]
        results.update({"system_name": self.get_system_name(),
                        "benchmark_short": self.benchmark.valstr(),
                        "benchmark_full": self._get_submission_benchmark_name(),
                        "scenario": self.scenario.valstr(),
                        "test_mode": test_mode,
                        "tensorrt_version": trt.__version__,
                        "detected_system": DETECTED_SYSTEM.summary_description(),
                        "scenario_key": scenario_key,
                        "satisfies_query_constraint": satisfies_query_constraint
                        })

        # Special DLRM thing
        if self.benchmark in [Benchmark.DLRMv2] \
                and test_mode == "PerformanceOnly":
            partitions = np.load(os.path.expandvars(self.args["sample_partition_path"]))
            partition_mean_size = np.mean(partitions[1:] - partitions[:-1])
            results["dlrm_partition_mean_size"] = partition_mean_size
            results["dlrm_pairs_per_second"] = results[scenario_key] * partition_mean_size

        return results
