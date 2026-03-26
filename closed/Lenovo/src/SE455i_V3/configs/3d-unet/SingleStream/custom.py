# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SE455IV3(SingleStreamGPUBaseConfig):
    system = KnownSystem.SE455iV3

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size = {'3d-unet': 1}
   # map_path: str = ''
   # tensor_path: str = ''

    # Optional fields:
   # active_sms: int = 0
   # cache_file: str = ''
   # complete_threads: int = 0
   # engine_dir: str = ''
    single_stream_expected_latency_ns = 650000000
    #single_stream_target_latency_percentile: float = 0.0
    slice_overlap_patch_kernel_cg_impl = True
    #unet3d_sw_gaussian_patch_path: str = ''
    #use_batcher_thread_per_device: bool = False
    use_cuda_thread_per_device = True
    use_deque_limit = True
    #use_same_context: bool = False
    vboost_slider = 1
    warmup_duration = 30.0
    workspace_size = 7000000000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SE455IV3_HighAccuracy(SE455IV3):
    pass



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SE455I_V3(SingleStreamGPUBaseConfig):
    system = KnownSystem.SE455I_V3
    single_stream_expected_latency_ns = 650000000
    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
#    gpu_batch_size: Dict = {}
#    map_path: str = ''
#    tensor_path: str = ''

    # Optional fields:
#    active_sms: int = 0
#    cache_file: str = ''
#    complete_threads: int = 0
#    engine_dir: str = ''
#    single_stream_expected_latency_ns: int = 0
#    single_stream_target_latency_percentile: float = 0.0
#    slice_overlap_patch_kernel_cg_impl: bool = False
#    unet3d_sw_gaussian_patch_path: str = ''
#    use_batcher_thread_per_device: bool = False
#    use_cuda_thread_per_device: bool = False
#    use_deque_limit: bool = False
#    use_same_context: bool = False
#    vboost_slider: int = 0
#    warmup_duration: float = 0.0
#    workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SE455I_V3_HighAccuracy(SE455I_V3):
    pass

