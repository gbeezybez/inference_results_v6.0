# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import ctypes
import os
import yaml

from code.common import logging
from code.common.constants import TRT_LOGGER
from code.common.systems.system_list import DETECTED_SYSTEM
from code.plugin import load_trt_plugin_by_network

import tensorrt as trt

from code.internal.dlsim import DeviceConfig


class ProfilerHarness:
    """Wraps around harness to profile it."""

    def __init__(self, harness, profiler):
        self.harness = harness

        self.apply_profiler(profiler)

    def run_harness(self, *args, **kwargs):
        return self.harness.run_harness(*args, **kwargs)

    def apply_profiler(self, profile):
        if profile is None:
            return

        accuracy_level = self.harness.args["accuracy_level"][:-1].replace(".", "p")
        network_name = "_".join([self.harness.name, self.harness.scenario.valstr, self.harness.system_id, accuracy_level])
        self._set_profiler_clk(network_name)
        self.harness.executable = self._get_profiler_cmd(profile, self.harness.executable, network_name)
        try:
            self._dump_engine_info(self.harness.gpu_engine, network_name)
        except Exception as e:
            print("Cannot dump engine info! Error: {}".format(e))

    def get_full_log_dir(self):
        return self.harness.get_full_log_dir()

    def get_system_name(self):
        return self.harness.get_system_name()

    def _set_profiler_clk(self, network_name):
        print("Creating {}.yml".format(network_name))
        gpu_clk = os.getenv('GPUCLK', 1000)
        DeviceConfig(
            self.harness.precision,
            self.harness.system_id,
            int(gpu_clk),
            self.harness.args["gpu_batch_size"],
            network_name,
            self.harness.name,
            self.harness.scenario.valstr
        ).create_dlsim_config()

    def _get_profiler_cmd(self, profile, executable, network_name):
        """Return profiler cmdline with args."""

        if "gpt" in network_name and profile != "nsys":
            logging.error(f"{profile} for gpt has not been supported! Aborting...")
            return

        if profile == "nvprof":
            executable = "/usr/local/cuda/bin/nvprof --profile-api-trace all  --demangling on --profile-from-start on --force-overwrite --print-gpu-trace --csv --log-file {}.log --export-profile {}.nvvp {} ".format(
                network_name, network_name, executable)
        elif profile == "nsys":
            nsys_args = "-c cudaProfilerApi --capture-range-end=none --output={}_pid%p.sqlite --trace=cuda,nvtx --export=sqlite --force-overwrite true".format(
                network_name)
            if "is_orin" in DETECTED_SYSTEM.extras["tags"]:
                nsys_args += "--accelerator-trace=tegra-accelerators"
            executable = "/usr/local/bin/nsys profile {} {} ".format(nsys_args, executable)
        elif profile == "pic-c":
            pic_c_args = "--metrics-set triage --summarize --show-output --start-stop-method cuda-profiler-api --outdir={} --clobber".format(network_name)
            executable = "pic-c profile {} {}".format(pic_c_args, executable)
        elif profile == "ncu":
            ncu_args = dict()
            ncu_yml_path = "internal/correlation/ncu_args.yml"
            if os.path.isfile(ncu_yml_path):
                with open(ncu_yml_path, "r") as stream:
                    ncu_args.update(yaml.safe_load(stream))
            if ncu_args:
                ncu_metrics = ",".join(ncu_args["metrics"])
                ncu_sections = " ".join("--section \"{}\"".format(section) for section in ncu_args["sections"])
                ncu_ext_args = " ".join(ncu_args["ext_args"])
                base_args_dict = {"clock-control": "none", "profile-from-start": "off", "nvtx": "", "force-overwrite": ""}
                ncu_binary = "/usr/local/NsightCompute/ncu"
                ncu_section_folder = "/usr/local/NsightCompute/sections/"

                ncu_base_args = ""
                for key, value in base_args_dict.items():
                    if value:
                        ncu_base_args += "--{} {} ".format(key, value)
                    else:
                        ncu_base_args += "--{} ".format(key)
                executable = "{} -o {} {} {} --section-folder {} {} --metrics {} {}".format(ncu_binary, network_name,
                                                                                            ncu_base_args, ncu_ext_args, ncu_section_folder, ncu_sections, ncu_metrics, executable)

        return executable

    def _dump_engine_info(self, engine_path, network_name):
        if engine_path is None or not os.path.exists(engine_path):
            print("Engine does not exist. Skipped engine info dumping.")
            return

        # Load plugins.
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        load_trt_plugin_by_network(self.harness.benchmark)

        # Deserialize the engine.
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)

        if engine is None:
            print("Engine deserialization failure. Skipped engine info dumping.")
            return

        # Dump engine layer info.
        inspector = engine.create_engine_inspector()
        engine_info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        with open("{}.engine.json".format(network_name), "w") as f:
            print(engine_info, file=f)

        print("Finished dumping engine information to {}.engine.json!".format(network_name))
