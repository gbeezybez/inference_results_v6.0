# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SE455I_V3(OfflineGPUBaseConfig):
    system = KnownSystem.SE455I_V3

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):

    scenario = Scenario.Offline

    run_infer_on_copy_streams = False
    use_graphs = False
    gpu_inference_streams = 1
    gpu_copy_streams = 2
    gpu_batch_size = {'resnet50': 32}
    offline_expected_qps = 13000 * 2
    use_graphs = True



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SE455I_V3_HighAccuracy(SE455I_V3):
    pass

