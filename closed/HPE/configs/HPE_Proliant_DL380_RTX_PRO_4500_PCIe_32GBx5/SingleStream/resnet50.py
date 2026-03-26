import code.common.constants as C
import code.fields.models as model_fields
import code.fields.harness as harness_fields
import code.fields.loadgen as loadgen_fields
import code.fields.gen_engines as gen_engines_fields

EXPORTS = {
        C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {

        # Tune me!
        model_fields.gpu_batch_size: {
            'resnet50': 16,
        },
        #
        model_fields.input_dtype: 'int8',
        model_fields.precision: 'int8',

        # Do not change input format from 'linear' unless you know what you are doing.
        model_fields.input_format: 'linear',


        loadgen_fields.single_stream_expected_latency_ns: 281736,

        # Data paths. You should not need to change this unless you know what you are doing.
        harness_fields.map_path: 'data_maps/imagenet/val_map.txt',
        harness_fields.tensor_path: 'build/preprocessed_data/imagenet/ResNet50/int8_linear',

        harness_fields.gpu_copy_streams: 2,
        harness_fields.gpu_inference_streams: 1,



        # These flags are tune-able but are probably un-tested for your system and are not guaranteed to work.
        harness_fields.run_infer_on_copy_streams: False,
        harness_fields.use_graphs: False,

    },
}

"""
EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        # Data paths. You should not need to change this unless you know what you are doing.
        harness_fields.map_path: 'data_maps/imagenet/val_map.txt',
        harness_fields.tensor_path: 'build/preprocessed_data/imagenet/ResNet50/int8_linear',

        #
        model_fields.input_dtype: 'int8',
        model_fields.precision: 'int8',

        # Do not change input format from 'linear' unless you know what you are doing.
        model_fields.input_format: 'linear',

        # Tune me!
        model_fields.gpu_batch_size: {
            'resnet50': 32,
        },
        loadgen_fields.offline_expected_qps: 20000,
        harness_fields.gpu_copy_streams: 2,
        harness_fields.gpu_inference_streams: 1,

        # These flags are tune-able but are probably un-tested for your system and are not guaranteed to work.
        harness_fields.run_infer_on_copy_streams: False,
        harness_fields.use_graphs: False,

    },
}
"""
