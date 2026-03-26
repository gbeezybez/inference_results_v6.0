# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SE455I_V3(SingleStreamGPUBaseConfig):
    system = KnownSystem.SE455I_V3
    scenario = Scenario.SingleStream

    gpu_batch_size = {'bert': 1}
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    use_graphs = True
    bert_opt_seqlen = 270
    use_small_tile_gemm_plugin = False
    single_stream_expected_latency_ns = 1700000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SE455I_V3_HighAccuracy(SE455I_V3):
    precision = "fp16"
    single_stream_expected_latency_ns = SE455I_V3.single_stream_expected_latency_ns * 2


#@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
#class SE455I_V3(SE455IV3):
#    system = KnownSystem.SE455I_V3
    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
#    single_stream_expected_latency_ns = 1700000

#@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
#class SE455IV3_HighAccuracy(SE455IV3_HighAccuracy):
#    precision = "fp16"
#    single_stream_expected_latency_ns = 1700000
