
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

import os
from mpi4py import MPI
from nvmitten.configurator import autoconfigure, bind
from code.ops.harness import BenchmarkHarnessOp, Vboost
from mlperf_inf_mm_q3vl.schema import (
    Verbosity
)
from mlperf_inf_mm_q3vl.benchmark import run_benchmark
from mlperf_inf_mm_q3vl_nv.deploy import MpiDynamoVllmEndpointDeployer
from mlperf_inf_mm_q3vl_nv.log import setup_loguru_for_benchmark_nv
from . import fields
from .utils import build_settings, build_dynamo_endpoint, build_dataset, build_env_vars



@autoconfigure
@bind(fields.random_seed, "random_seed")
class Qwen3VL235BHarnessOp(BenchmarkHarnessOp):
    def __init__(self, *args, random_seed: bool = 12345, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = build_dataset()
        self.settings = build_settings()
        self.dynamo = build_dynamo_endpoint()
        self.vllm_env_vars = build_env_vars()
        self.random_seed = random_seed

    def run(self, scratch_space, dependency_outputs):
        """Run the benchmark using Python code.

        Args:
            scratch_space: The scratch space for temporary files.
            dependency_outputs: Outputs from dependency operations.

        Returns:
            dict: Dictionary containing the result metadata and log directory.
        """
        for key, value in self.vllm_env_vars.items():
            if value is None:
                continue
            os.environ[key] = str(value)
        comm = MPI.COMM_WORLD
        # Ensure log output directory matches Workload log_dir.
        log_outdir = self.wl.log_dir
        if self.settings.logging.log_output is None or not hasattr(self.settings.logging.log_output, "outdir"):
            self.settings.logging.log_output = {
                "outdir": log_outdir,
                "prefix": "mlperf_log_",
                "suffix": "",
            }
        else:
            self.settings.logging.log_output.outdir = log_outdir
        setup_loguru_for_benchmark_nv(settings=self.settings, verbosity=Verbosity.INFO)
        with Vboost(), \
                self.power_monitor(), \
                MpiDynamoVllmEndpointDeployer(endpoint=self.dynamo, settings=self.settings):
            if comm.Get_rank() == 0:
                run_benchmark(
                    settings=self.settings,
                    dataset=self.dataset,
                    endpoint=self.dynamo,
                    random_seed=self.random_seed,
                )
            comm.Barrier()
        return self.load_run_results()
