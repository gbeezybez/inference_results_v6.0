# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SE455IV3(OfflineGPUBaseConfig):
    system = KnownSystem.SE455iV3
    
    gpu_batch_size = {'bert': 128}

    offline_expected_qps = 2300.0

    bert_opt_seqlen = 384

    use_graphs = True
    graphs_max_seqlen = 384

    workspace_size = 8589934592  

    use_fp8 = True 
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SE455IV3_HighAccuracy(SE455IV3):
    precision = "fp16"
    offline_expected_qps = SE455IV3.offline_expected_qps / 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SE455I_V3(OfflineGPUBaseConfig):
    system = KnownSystem.SE455I_V3
    gpu_batch_size = {'bert': 128}
    offline_expected_qps = 2350.0

    use_graphs = True
    graphs_max_seqlen = 384 
    
    num_concurrent_batchers = 4
    num_concurrent_issuers = 4
    
    workspace_size = 10737418240 
    engine_dir = "build/engines/SE455I_V3/bert/Offline"
    
    batch_triton_requests = True
    output_pinned_memory = True 

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SE455I_V3_HighAccuracy(SE455I_V3):
    pass

