# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SE455I_V3(SingleStreamGPUBaseConfig):
    system = KnownSystem.SE455I_V3
    scenario = Scenario.SingleStream

    gpu_batch_size = {'resnet50': 1}
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    use_graphs = True
    single_stream_expected_latency_ns = 660000
    disable_beta1_smallk = True
    use_cuda_thread_per_device = True
    
    workspace_size = 2147483648


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SE455I_V3_HighAccuracy(SE455I_V3):
    pass

