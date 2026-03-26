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

import os
import sys
sys.path.insert(0, os.getcwd())

from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *
from configs.resnet50 import GPUBaseConfig


class SingleStreamGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.SingleStream
    gpu_batch_size = {'resnet50': 1}
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class GH200_96GB_aarch64x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.GH200_96GB_ARMx1
    single_stream_expected_latency_ns = 660000
    disable_beta1_smallk = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L4x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.L4x1
    single_stream_expected_latency_ns = 660000
    disable_beta1_smallk = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class L40x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.L40x1
    single_stream_expected_latency_ns = 660000
    disable_beta1_smallk = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx1(SingleStreamGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GBx1
    single_stream_expected_latency_ns = 660000
    disable_beta1_smallk = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GBx1_Triton(H100_PCIe_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_PCIe_80GB_aarch64x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.H100_PCIe_80GB_ARMx1
    single_stream_expected_latency_ns = 660000
    disable_beta1_smallk = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1(SingleStreamGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx1
    single_stream_expected_latency_ns = 660000
    disable_beta1_smallk = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100_SXM_80GBx1_Triton(H100_SXM_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H200_SXM_141GBx1(SingleStreamGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GBx1
    single_stream_expected_latency_ns = 660000
    disable_beta1_smallk = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx1
    single_stream_expected_latency_ns = 660000
    disable_beta1_smallk = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1_Triton(A100_PCIe_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1_TritonUnified(A100_PCIe_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_ARMx1
    single_stream_expected_latency_ns = 660000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1_Triton(A100_PCIe_80GB_aarch64x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1_TritonUnified(A100_PCIe_80GB_aarch64x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_PCIe_40GB_ARMx1
    single_stream_expected_latency_ns = 660000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1_Triton(A100_PCIe_aarch64x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1_TritonUnified(A100_PCIe_aarch64x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx1
    start_from_device = True
    single_stream_expected_latency_ns = 660000
    disable_beta1_smallk = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1_Triton(A100_SXM_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1_TritonUnified(A100_SXM_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARMx1
    single_stream_expected_latency_ns = 490000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x1_Triton(A100_SXM_80GB_aarch64x1):
    use_triton = True
    single_stream_expected_latency_ns = 450000


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x1_TritonUnified(A100_SXM_80GB_aarch64x1):
    use_triton = True
    single_stream_expected_latency_ns = 450000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM4_40GBx1
    start_from_device = True
    single_stream_expected_latency_ns = 660000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1_Triton(A100_SXM4_40GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_40GBx1_TritonUnified(A100_SXM4_40GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.T4x1
    single_stream_expected_latency_ns = 996648


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1_Triton(T4x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1_TritonUnified(T4x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class Orin(SingleStreamGPUBaseConfig):
    system = KnownSystem.Orin
    gpu_copy_streams = 2
    use_direct_host_access = True
    single_stream_expected_latency_ns = 722240
    disable_beta1_smallk = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Orin_Triton(Orin):
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class Orin_TritonUnified(Orin):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class Orin_MaxQ(Orin):
    soc_cpu_freq = 652800
    soc_gpu_freq = 612000000
    soc_dla_freq = 0
    soc_emc_freq = 2133000000
    soc_pva_freq = 115000000
    orin_num_cores = 4
    gpu_copy_streams = 1
    single_stream_expected_latency_ns = 898255


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class Orin_NX(SingleStreamGPUBaseConfig):
    system = KnownSystem.Orin_NX
    gpu_copy_streams = 2
    use_direct_host_access = True
    single_stream_expected_latency_ns = 1200000
    disable_beta1_smallk = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class Orin_NX_MaxQ(Orin_NX):
    soc_cpu_freq = 806400
    soc_gpu_freq = 816000000
    soc_dla_freq = 0
    soc_emc_freq = 2133000000
    soc_pva_freq = 0
    orin_num_cores = 4
    gpu_copy_streams = 1
    single_stream_expected_latency_ns = 1200000
