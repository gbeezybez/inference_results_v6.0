# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SE455I_V3(MultiStreamGPUBaseConfig):
    system = KnownSystem.SE455I_V3
    gpu_batch_size = {'resnet50': 8}
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    multi_stream_samples_per_query = 8
    multi_stream_target_latency_percentile = 99
    use_graphs = True

    multi_stream_expected_latency_ns = 830000
    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SE455I_V3_Highaccuracy(SE455I_V3):
    pass

