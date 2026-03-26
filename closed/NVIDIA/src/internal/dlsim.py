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

import getpass
import socket
import nrsu
import yaml


class DeviceConfig:
    def __init__(self, op, gpus, gpu_clk, batches, name, benchmark, scenario):
        self.ofile = name
        self.batches = batches
        self.op = self.parse_precision(op)
        self.gpus = gpus
        self.gpu_clk = gpu_clk  # in MHz
        self.batches = batches
        self.scenario = scenario
        self.benchmark = benchmark

    def parse_precision(self, op):
        precision_map = {
            "int8": "imma",
            "fp16": "hmma",
        }
        return precision_map.get(op.lower(), "")

    def create_dlsim_config(self):
        dlsim_dict = {}
        dlsim_dict['op'] = self.op
        dlsim_dict['gpus'] = self.gpus
        dlsim_dict['batches'] = self.batches
        dlsim_dict['gpu_clk'] = self.gpu_clk

        # Writing config as yaml
        with open("{}.yml".format(self.ofile), 'w') as f:
            yaml.dump(dlsim_dict, f)

        # Write NRSU info as yml
        with open("{}_system_info.nrsu.yml".format(self.ofile), "w") as fp:
            yaml.dump(self.get_nrsu_system_info(), fp, default_flow_style=False)

    def get_nrsu_system_info(self):

        return {
            "username": getpass.getuser(),
            "hostname": socket.gethostname(),
            "ip": nrsu.os.get_ip_address(),
            "nvidia_driver_version": self.get_nrsu_driver_version(),
            "nvidia_device_count": self.get_nrsu_device_count(),
            "os_properties": self.get_nrsu_os_properties(),
            "cpu_properties": self.get_nrsu_cpu_properties(),
            "gpu_properties": self.get_nrsu_gpu_properties(),
            "system_properties": self.get_nrsu_system_properties()
        }

    def get_nrsu_driver_version(self):
        try:
            driver_ver = nrsu.gpu.get_driver_version()
        except nrsu.exceptions.SysUtilsError:
            driver_ver = 418.0
        return driver_ver

    def get_nrsu_device_count(self):
        try:
            device_count = nrsu.gpu.get_device_count()
        except nrsu.exceptions.SysUtilsError:
            device_count = 1
        return device_count

    def get_nrsu_os_properties(self):
        try:
            os_properties = nrsu.os.get_properties()
        except nrsu.exceptions.SysUtilsError:
            os_properties = None
        return os_properties

    def get_nrsu_cpu_properties(self):
        try:
            cpu_properties = nrsu.cpu.get_properties()
        except nrsu.exceptions.SysUtilsError:
            cpu_properties = None
        return cpu_properties

    def get_nrsu_gpu_properties(self):
        try:
            gpu_properties = [gpu.query.get_properties() for gpu in nrsu.gpu.get_gpus()]
        except:
            gpu = nrsu.gpu.GPUProperties()
            gpu.device_brand_name = "NVIDIA Jetson"
            gpu.device_product_name = "NVIDIA Jetson"  # DLSim user should change this manually.
            gpu.nvidia_driver_version = self.get_nrsu_driver_version()
            gpu.nvidia_device_count = self.get_nrsu_device_count()
            gpu.cuda_attributes = {}
            gpu_properties = [gpu]
        return gpu_properties

    def get_nrsu_system_properties(self):
        try:
            system_properties = nrsu.system.get_properties()
        except:
            system_properties = None
        return system_properties
