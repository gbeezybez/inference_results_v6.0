# Copyright 2024, MangoBoost, Inc. All rights reserved.

import threading
import requests
import array
import logging
import os
import json
from tqdm import tqdm

from dataset import get_dataset_info

import mlperf_loadgen as lg
import numpy as np

from absl import app, flags
from llmboost.mlperf.mlperf_client_func import QSL, QDL

model_name_list = [
    "gptj",
    "llama2-70b",
    "mixtral-8x7b",
    "llama3_1-405b",
]

FLAGS = flags.FLAGS

flags.DEFINE_enum("model_name", "llama2-70b", model_name_list, "Model name")
flags.DEFINE_enum(
    "test_mode", "Server", ["Offline", "Server"], "type of test to perform"
)
flags.DEFINE_boolean("accuracy_test", False, "Perform accuracy test")
flags.DEFINE_integer("parallel_requests", 10, "Number of parallel requests")
flags.DEFINE_integer("qps", 128, "Number of parallel requests")
flags.DEFINE_integer("batched_queries", 128, "Number of batched requests")
flags.DEFINE_string("mlperf_conf", "mlperf.conf", "Path to mlperf.conf")
flags.DEFINE_string("user_conf", "None", "Path to user.conf")
flags.DEFINE_list("sut_server_addr", ["http://localhost"], "List of server addresses")
flags.DEFINE_integer(
    "port", 8000, "Servers port, where all the ports should use the same port number"
)
flags.DEFINE_enum(
    "scheduler",
    "round_robin",
    ["round_robin", "weighted_random"],
    "The scheduler algorithm for scheduling the requests among nodes. Options- `round_robin`: split the workload evenly across the nodes. `weighted_random`: split the workload to all the nodes according to the specific weights in --scheduler_weights.",
)
flags.DEFINE_list(
    "scheduler_weights",
    [],
    "The scheduling weights of each node. Only works when the scheduler is weighted_random.",
)
flags.DEFINE_bool(
    "client_send_batch_requests",
    False,
    "Client send the requests to each node in a batching way during the Server Scenario",
)
flags.DEFINE_bool(
    "offline_delay_receive_response",
    False,
    "If set True, the offline servers will only send the responses back until all the requests are got",
)
# Testing
flags.DEFINE_bool(
    "ws_stress_test",
    False,
    "Enable the socket stress testing, please only enable it when developing or testing",
)
# Logging
flags.DEFINE_string("result_dir", "mlperf-logs", "Results recording")


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("MB-MLPERF-INFERENCE-CLIENT")


def main(argv):
    del argv
    settings = lg.TestSettings()
    # To avoid double configuration files error, we comment this out
    # settings.FromConfig(FLAGS.mlperf_conf, FLAGS.model_name, FLAGS.test_mode)
    if FLAGS.user_conf != "None":
        settings.FromConfig(FLAGS.user_conf, FLAGS.model_name, FLAGS.test_mode)

    if FLAGS.test_mode == "Offline":
        settings.scenario = lg.TestScenario.Offline
    else:
        settings.scenario = lg.TestScenario.Server

    settings.use_token_latencies = True

    if FLAGS.accuracy_test:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    os.makedirs(FLAGS.result_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = FLAGS.result_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    qsl = QSL(model_name=FLAGS.model_name)
    qdl = QDL(qsl=qsl, sut_server_addr=FLAGS.sut_server_addr, flags=FLAGS)

    log.info("Starting Benchmark run")
    lg.StartTestWithLogSettings(
        qdl.qdl, qsl.qsl, settings, log_settings, "audit.config"
    )

    log.info("Run Completed!")
    log.info("Destroying QSL...")
    lg.DestroyQSL(qsl.qsl)


if __name__ == "__main__":
    app.run(main)
