# Copyright 2024, MangoBoost, Inc. All rights reserved.

import logging
import os
import threading
import numpy as np
import array
from queue import Empty
from tqdm import tqdm

from llmboost import LLMBoost
from dataset import get_dataset_info

import mlperf_loadgen as lg

from absl import app, flags
from llmboost.mlperf.mlperf_sut import SUT, set_model_map

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("MB-MLPERF-INFERENCE-MAIN")

model_map = {
    "llama2-70b": "/models/amd2025_model/model/llama2-70b-chat-hf/quantized",
    "llama3_1-405b": "/models/models/Llama-3.1-405B-Instruct-quark-fp8",
}

FLAGS = flags.FLAGS

flags.DEFINE_enum("model_name", "llama2-70b", model_map.keys(), "Model name")
flags.DEFINE_string("mlperf_conf", "conf/mlperf.conf", "Path to mlperf.conf")
flags.DEFINE_string("user_conf", "None", "Path to user.conf")
flags.DEFINE_enum(
    "test_mode", "Offline", ["Offline", "Server"], "type of test to perform"
)
flags.DEFINE_boolean("accuracy_test", False, "run accuracy test")
flags.DEFINE_integer("max_tokens", 1024, "max tokens to generate")
flags.DEFINE_integer("tp", 1, "tensor parallelism")
flags.DEFINE_integer("dp", 8, "number of workers")
flags.DEFINE_integer("beam_width", 0, "beam width if used")
flags.DEFINE_enum(
    "load_format",
    "auto",
    ["bitsandbytes", "bitsandbytesint8", "auto"],
    "Quantized model to use",
)
flags.DEFINE_enum(
    "quantization",
    "None",
    ["bitsandbytes", "fp8", "None"],
    "Quantization scheme to use",
)
flags.DEFINE_enum("kv_cache_dtype", "auto", ["auto", "fp8"], "KV cache dtype")
flags.DEFINE_string("quantization_param_path", None, "Path to quantization parameters")
flags.DEFINE_string("quantization_weight_path", None, "Path to quantization weights")
flags.DEFINE_integer(
    "max_num_seqs",
    0,
    "max number of sequences to process in parallel - 0 means default",
)
flags.DEFINE_string("load_balancing_mode", "auto", "Load Balancing method")
flags.DEFINE_integer(
    "gpu_batch_size",
    48,
    "The max number of queries to be sent to each worker as a batch.",
)
flags.DEFINE_float(
    "batcher_threshold",
    0.4,
    "The max duration (in sec) that we wait to send the pending queries.",
)

# Ugly
flags.DEFINE_bool("drain_per_worker", False, "Drain per worker")
flags.DEFINE_bool(
    "output_semi_batching",
    False,
    "Per query only send the outputs twice (first token and rest tokerns)",
)

flags.register_validator(
    "max_num_seqs",
    lambda x: x >= 0,
    message="max_num_seqs must be greater than or equal 0",
)
flags.register_validator(
    "tp", lambda x: x > 0 and x < 9, message="tp must be between 1 and 8"
)
flags.register_validator(
    "dp", lambda x: x > 0 and x < 9, message="dp must be between 1 and 8"
)

# Logging
flags.DEFINE_string(
    "result_dir", "mlperf-logs", "The directory to put the result loggings"
)

flags.DEFINE_integer(
    "max_model_len",
    None,
    "The max model len of input",
)


def main(argv):
    del argv

    set_model_map(model_map)

    settings = lg.TestSettings()
    # to avoid double configuration files error, we only include user.conf
    # settings.FromConfig(FLAGS.mlperf_conf, FLAGS.model_name, FLAGS.test_mode)
    if FLAGS.user_conf != "None":
        log.info(f"Loading user config from {FLAGS.user_conf}")
        settings.FromConfig(FLAGS.user_conf, FLAGS.model_name, FLAGS.test_mode)

    if FLAGS.test_mode == "Server":
        settings.scenario = lg.TestScenario.Server
    else:
        settings.scenario = lg.TestScenario.Offline

    if FLAGS.accuracy_test:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    settings.use_token_latencies = (
        True  # "defaulty record metrics of Tokens per second"
    )

    if os.path.exists(FLAGS.result_dir):
        log.info(f"[Overwrite] result directory already exists: {FLAGS.result_dir}")

    os.makedirs(FLAGS.result_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = FLAGS.result_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    sut = SUT(flags=FLAGS, model_name=FLAGS.model_name)

    sut.start()
    lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
    log.info("Starting Benchmark run")
    lg.StartTestWithLogSettings(lgSUT, sut.qsl, settings, log_settings, "audit.config")

    sut.stop()
    log.info("Run Completed!")
    log.info("Destroying SUT...")
    lg.DestroySUT(lgSUT)
    log.info("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
    app.run(main)
