# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SE455I_V3(OfflineGPUBaseConfig):
    system = KnownSystem.SE455I_V3

    gpu_batch_size = {'3d-unet': 1}
    offline_expected_qps = 1.3*2
    slice_overlap_patch_kernel_cg_impl = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SE455IV3_HighAccuracy(SE455I_V3):
    pass

