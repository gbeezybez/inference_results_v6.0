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

from nvmitten.constants import ByteSuffix, CPUArchitecture
from nvmitten.interval import NumericRange
from nvmitten.json_utils import load
from nvmitten.nvidia.accelerator import GPU, DLA
from nvmitten.system.component import Description
from nvmitten.system.system import System
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Set

from code.common.systems.known_hardware import *


# Dynamically build Enum for known systems
_system_confs = dict()


def add_systems(name_format_string: str,
                id_format_string: str,
                cpu: KnownCPU,
                accelerator: KnownGPU,
                accelerator_counts: List[int],
                mem_requirement: Memory,
                target_dict: Dict[str, Description] = _system_confs,
                tags: List[str] = None,
                n_dlas: int = 0):
    """Adds a Description to a dictionary.

    Args:
        name_format_string (str): A Python format to generate the name for the Enum member. Can have a single format
                                  item to represent the count.
        id_format_string (str): A Python format to generate the system ID to use. The system ID is used for the systems/
                                json file. Can contain a single format item to represent the count.
        cpu (KnownCPU): The CPU that the system uses
        accelerator (KnownGPU): The Accelerator that the system uses
        accelerator_counts (List[int]): The list of various counts to use for accelerators.
        mem_requirement (Memory): The minimum memory requirement to have been tested for the hardware configuration.
        target_dict (Dict[str, Description]): The dictionary to add the Description to.
                                              (Default: _system_confs)
        tags (List[str]): A list of strings denoting certain tags used for classifying the system. (Default: None)
        n_dlas (int): The number of DLAs present on the system (Default: 0)
    """
    def _mem_cmp(m):
        thresh = NumericRange(mem_requirement._num_bytes * 0.95)
        return thresh.contains_numeric(m.capacity._num_bytes)

    for count in accelerator_counts:
        def _accelerator_cmp(count=0):
            def _f(d):
                # Check GPUs
                if len(d[GPU]) != count:
                    return False

                for i in range(count):
                    if not accelerator.matches(d[GPU][i]):
                        return False

                # Check DLAs
                if len(d[DLA]) != n_dlas:
                    return False
                return True
            return _f

        k = name_format_string.format(count)
        v = Description(System,
                        _match_ignore_fields=["extras"],
                        cpu=cpu,
                        host_memory=_mem_cmp,
                        accelerators=_accelerator_cmp(count=count),
                        extras={"id": id_format_string.format(count),
                                "tags": set(tags) if tags else set()})

        target_dict[k] = v


# Blackwell Systems
add_systems("B200_SXM_180GBx{}",
            "B200-SXM-180GBx{}",
            KnownCPU.x86_64_Intel_Generic,
            KnownGPU.B200_SXM_180GB,
            [1, 2, 4, 8],
            Memory(100, ByteSuffix.GiB),
            tags=["start_from_device_enabled"])

# Grace-Blackwell NVL
add_systems("GB200_NVL_186GB_ARMx{}",
            "GB200-NVL_GB200-NVL-186GB_aarch64x{}",
            KnownCPU.ARMGeneric,
            KnownGPU.GB200_GraceBlackwell_186GB,
            [1, 2, 4],
            Memory(500, ByteSuffix.GiB),
            tags=["start_from_device_enabled", "end_on_device_enabled"])

# Grace-Hopper Superchip Systems
add_systems("GH200_96GB_ARMx{}",
            "GH200-GraceHopper-Superchip_GH200-96GB_aarch64x{}",
            KnownCPU.ARMGeneric,
            KnownGPU.GH200_GraceHopper_96GB,
            [1],
            Memory(500, ByteSuffix.GiB),
            tags=["start_from_device_enabled", "end_on_device_enabled"])
add_systems("GH200_144GB_ARMx{}",
            "GH200-GraceHopper-Superchip_GH200-144GB_aarch64x{}",
            KnownCPU.ARMGeneric,
            KnownGPU.GH200_GraceHopper_144GB,
            [1, 2],
            Memory(600, ByteSuffix.GiB),
            tags=["start_from_device_enabled", "end_on_device_enabled"])

# ADA Systems
add_systems("L40x{}",
            "L40x{}",
            KnownCPU.x86_64_AMD_Generic,
            KnownGPU.L40,
            [1, 2, 4, 8],
            Memory(100, ByteSuffix.GiB))
add_systems("L4x{}",
            "L4x{}",
            KnownCPU.x86_64_AMD_Generic,
            KnownGPU.L4,
            [1, 2, 4, 8],
            Memory(100, ByteSuffix.GiB))
add_systems("L40Sx{}",
            "L40Sx{}",
            KnownCPU.x86_64_AMD_Generic,
            KnownGPU.L40S,
            [1, 2, 4, 8],
            Memory(100, ByteSuffix.GiB))

# Hopper systems
add_systems("H200_SXM_141GBx{}",
            "H200-SXM-141GBx{}",
            KnownCPU.x86_64_Intel_Generic,
            KnownGPU.H200_SXM_141GB,
            [1, 2, 4, 8],
            Memory(100, ByteSuffix.GiB),
            tags=["start_from_device_enabled"])
add_systems("H200_SXM_141GB_CTSx{}",
            "H200-SXM-141GB-CTSx{}",
            KnownCPU.x86_64_Intel_Generic,
            KnownGPU.H200_SXM_141GB_CTS,
            [1, 2, 4, 8],
            Memory(2, ByteSuffix.TB),
            tags=["start_from_device_enabled"])
add_systems("H100_NVL_94GBx{}",
            "H100-NVL-94GBx{}",
            KnownCPU.x86_64_Intel_Generic,
            KnownGPU.H100_NVL,
            [1, 2, 4, 8],
            Memory(100, ByteSuffix.GiB))
add_systems("H100_SXM_80GBx{}",
            "DGX-H100_H100-SXM-80GBx{}",
            KnownCPU.x86_64_Intel_Generic,
            KnownGPU.H100_SXM_80GB,
            [1, 2, 4, 8],
            Memory(100, ByteSuffix.GiB),
            tags=["start_from_device_enabled"])
add_systems("H100_PCIe_80GBx{}",
            "H100-PCIe-80GBx{}",
            KnownCPU.x86_64_AMD_Generic,
            KnownGPU.H100_PCIe_80GB,
            [1, 2, 4, 8],
            Memory(100, ByteSuffix.GiB))
add_systems("H100_PCIe_80GB_ARMx{}",
            "H100-PCIe-80GB_aarch64x{}",
            KnownCPU.ARMGeneric,
            KnownGPU.H100_PCIe_80GB,
            [1, 2, 4, 8],
            Memory(100, ByteSuffix.GiB))

# A100_PCIe_40GB and 80GB based systems:
add_systems("A100_PCIe_40GB_ARMx{}",
            "A100-PCIe_aarch64x{}",
            KnownCPU.ARMGeneric,
            KnownGPU.A100_PCIe_40GB,
            [1, 2, 4],
            Memory(100, ByteSuffix.GiB),
            tags=["deprecated"])
add_systems("A100_PCIe_80GBx{}",
            "A100-PCIe-80GBx{}",
            KnownCPU.x86_64_Generic,
            KnownGPU.A100_PCIe_80GB,
            [1, 8],
            Memory(100, ByteSuffix.GiB),
            tags=["deprecated"])
add_systems("A100_PCIe_80GB_ARMx{}",
            "A100-PCIe-80GB_aarch64x{}",
            KnownCPU.ARMGeneric,
            KnownGPU.A100_PCIe_80GB,
            [1, 2, 4],
            Memory(100, ByteSuffix.GiB),
            tags=["deprecated"])

# A100_SXM4_40GB and SXM_80GB based systems:
add_systems("A100_SXM4_40GBx{}",
            "DGX-A100_A100-SXM4-40GBx{}",
            KnownCPU.x86_64_Generic,
            KnownGPU.A100_SXM4_40GB,
            [1, 8],
            Memory(100, ByteSuffix.GiB),
            tags=["deprecated", "start_from_device_enabled", "end_on_device_enabled"])
add_systems("A100_SXM_80GBx{}",
            "DGX-A100_A100-SXM-80GBx{}",
            KnownCPU.x86_64_Generic,
            KnownGPU.A100_SXM_80GB,
            [1, 8],
            Memory(100, ByteSuffix.GiB),
            tags=["deprecated", "start_from_device_enabled", "end_on_device_enabled"])
add_systems("A100_SXM_80GB_ARMx{}",
            "A100-SXM-80GB_aarch64x{}",
            KnownCPU.ARMGeneric,
            KnownGPU.A100_SXM_80GB,
            [1, 8],
            Memory(1, ByteSuffix.TB),
            tags=["deprecated", "start_from_device_enabled", "end_on_device_enabled"])


# Other Ampere based systems
add_systems("GeForceRTX_3080x{}",
            "GeForceRTX3080x{}",
            Any,
            KnownGPU.GeForceRTX_3080,
            [1],
            Memory(30, ByteSuffix.GiB),
            tags=["deprecated"])
add_systems("GeForceRTX_3090x{}",
            "GeForceRTX3090x{}",
            Any,
            KnownGPU.GeForceRTX_3090,
            [1],
            Memory(30, ByteSuffix.GiB),
            tags=["deprecated"])
add_systems("A10x{}",
            "A10x{}",
            KnownCPU.x86_64_AMD_Generic,
            KnownGPU.A10,
            [1, 8],
            Memory(0.9, ByteSuffix.TiB),
            tags=["deprecated"])
add_systems("A30x{}",
            "A30x{}",
            KnownCPU.x86_64_AMD_Generic,
            KnownGPU.A30,
            [1, 8],
            Memory(0.5, ByteSuffix.TiB),
            tags=["deprecated"])
add_systems("A2x{}",
            "A2x{}",
            Any,
            KnownGPU.A2,
            [1, 2],
            Memory(100, ByteSuffix.GiB),
            tags=["deprecated"])

# Turing based systems
add_systems("T4x{}",
            "T4x{}",
            KnownCPU.x86_64_Generic,
            KnownGPU.T4_32GB,
            [1, 8, 20],
            Memory(32, ByteSuffix.GiB),
            tags=["deprecated"])

# Embedded systems
add_systems("Orin",
            "Orin",
            KnownCPU.ARMGeneric,
            KnownGPU.Orin,
            [1],
            Memory(7, ByteSuffix.GiB),
            n_dlas=2)
add_systems("Orin_NX",
            "Orin_NX",
            KnownCPU.ARMGeneric,
            KnownGPU.OrinNX,
            [1],
            Memory(7, ByteSuffix.GiB),
            n_dlas=2)

# Handle custom systems to better support partner drops.
custom_system_file = Path("code/common/systems/custom_list.json")
if custom_system_file.exists():
    with custom_system_file.open() as f:
        custom_systems = load(f)

    for k, v in custom_systems.items():
        if k in _system_confs:
            raise KeyError(f"SystemEnum member {k} already exists")

        # Set up 'extras'
        if "extras" not in v.mapping:
            v.mapping["extras"] = dict()
        v._match_ignore_fields.add("extras")

        if "tags" not in v.mapping["extras"]:
            v.mapping["extras"]["tags"] = set()

        v.mapping["extras"]["id"] = k
        v.mapping["extras"]["tags"].add("custom")

        _system_confs[k] = v

KnownSystem = SimpleNamespace(**_system_confs)


def classification_tags(system: System) -> Set[str]:
    tags = set()

    # This may break for non-homogeneous systems.
    gpus = system.accelerators[GPU]
    if len(gpus) > 0:
        tags.add("gpu_based")

        primary_sm = int(gpus[0].compute_sm)
        if primary_sm == 90:
            tags.add("is_hopper")
        if primary_sm == 89:
            tags.add("is_ada")
        if primary_sm in (80, 86, 87, 89):
            tags.add("is_ampere")
        if primary_sm == 75:
            tags.add("is_turing")

        if gpus[0].name.startswith("Orin") and primary_sm == 87:
            tags.add("is_orin")
            tags.add("is_soc")

    if len(gpus) > 1:
        tags.add("multi_gpu")

    if system.cpu.architecture == CPUArchitecture.aarch64:
        tags.add("is_aarch64")

    return tags


DETECTED_SYSTEM = System.detect()[0]
for name, sys_desc in _system_confs.items():
    if sys_desc.matches(DETECTED_SYSTEM):
        DETECTED_SYSTEM.extras["id"] = sys_desc.mapping["extras"]["id"]
        DETECTED_SYSTEM.extras["tags"] = sys_desc.mapping["extras"]["tags"].union(classification_tags(DETECTED_SYSTEM))
        DETECTED_SYSTEM.extras["name"] = name

        # Convenience field
        if len(DETECTED_SYSTEM.accelerators[GPU]) > 0:
            DETECTED_SYSTEM.extras["primary_compute_sm"] = DETECTED_SYSTEM.accelerators[GPU][0].compute_sm
        else:
            DETECTED_SYSTEM.extras["primary_compute_sm"] = None
        break
