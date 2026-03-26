"""
openvino backend (https://docs.openvino.ai/)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np
from openvino import Core, get_version, CompiledModel, Output
import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
import backend

import queue
import threading

class BackendOpenvino(backend.Backend):
    def __init__(self) -> None:
        super().__init__()
        self.core: Core | None = None
        self.compiled_model: CompiledModel | None = None
        self.ml_scenario: str = "SingleStream"
        self.ov_device: str = "CPU"
        self.max_batchsize: int = 1
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.output_ports: Dict[str, Output] = {}

    def version(self) -> str:
        return get_version()

    def name(self) -> str:
        """Name of the runtime."""
        return "openvino"

    def image_format(self) -> str:
        """image_format. Please use --data-format=NHWC for alternative layout."""
        return "NCHW"

    def load(
        self,
        model_path: str,
        inputs: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
        ) -> "BackendOpenvino":

        num_req = 1
        num_str = 1
        batch_size = 1

        """Load model and find input/outputs from the model file."""
        self.core = self.core or Core()
        self.counter = 0

        print(f"[Info] Target MLPerf OpenVINO Device: {self.ov_device}")

        if self.ml_scenario == "SingleStream":
            perf_mode = hints.PerformanceMode.LATENCY
        elif self.ml_scenario == "MultiStream":
            perf_mode = hints.PerformanceMode.LATENCY
            if "GPU" in self.ov_device:
                num_str = 2
                batch_size = 8
            elif self.ov_device == "NPU":
                num_req = 8
                batch_size = 8
            elif self.ov_device == "CPU":
                num_req = 8
                num_str = 8
        elif self.ml_scenario == "Offline":
            perf_mode = hints.PerformanceMode.THROUGHPUT
            if "GPU" in self.ov_device:
                num_str = 2
                batch_size = self.max_batchsize
            elif self.ov_device == "NPU":
                num_req = 4
            elif self.ov_device == "CPU":
                num_req = self.max_batchsize
                num_str = num_req

        self.core.set_property(self.ov_device, {hints.performance_mode: perf_mode})
        model = self.core.read_model(model_path)

        #shape_map = {}
        #for port in model.inputs:
        #    orig_shape = list(port.partial_shape)
        #    if len(orig_shape) <4:
        #        continue
        #    orig_shape[0] = batch_size
        #    shape_map[port] = orig_shape
        #if shape_map:
        #    model.reshape(shape_map)
        model.reshape([1,3,224,224])
        ov_config = {
            hints.num_requests: str(num_req),
            props.enable_profiling: False,
            hints.performance_mode: perf_mode,
        }

        if "GPU" in self.ov_device:
            ov_config[props.streams.num] = str(num_str)
            ov_config[hints.inference_precision] = ov.Type.f16
        elif self.ov_device == "CPU":
            ov_config[props.streams.num] = str(num_str)

        self.compiled_model = self.core.compile_model(
            model,
            self.ov_device,
            ov_config
        )

        self.req = self.compiled_model.create_infer_request()

        if self.ml_scenario == "MultiStream" and self.ov_device in ["CPU"]:
            self.infer_queue = ov.AsyncInferQueue(self.compiled_model, 8)
        else:
            self.infer_queue = ov.AsyncInferQueue(self.compiled_model, self.max_batchsize)

        self.inputs  = list(inputs) if inputs else [p.get_any_name() for p in self.compiled_model.inputs]
        self.outputs = list(outputs) if outputs else [p.get_any_name() for p in self.compiled_model.outputs]

        self._output_ports = {p.get_any_name(): p for p in self.compiled_model.outputs}
        return self

    def predict(self, feed: Dict[str, np.ndarray]):
        """Run the prediction."""

        if self.compiled_model is None:
            raise RuntimeError("Model has not been loaded - call .load() first.")

        if self.ml_scenario == "SingleStream":
            for name, arr in feed.items():
                self.req.get_tensor(name).data[...] = arr
            self.req.infer()
            if len(self._output_ports) == 1:
                output = self.req.get_output_tensor().data
                return [output]
            else:
                return [self.req.get_output_tensor(i).data for i in range(len(self._output_ports))]

        if self.ml_scenario == "MultiStream":
            if self.ov_device == "CPU":
                array_len = 1
                for name, arr in feed.items():
                    array_len = len(arr)
                    [self.infer_queue.start_async({0:ov.Tensor(split_tensor,shared_memory=True)}) for split_tensor in np.array_split(arr, array_len)]
                self.infer_queue.wait_all()

                if len(self._output_ports) == 1:
                    output = [self.infer_queue[i].get_output_tensor().data for i in range(len(self._output_ports))]
                    return [output]
                else:
                    # Needs more work - Paul
                    output = []
                    for i in range(array_len):
                        output.append([self.infer_queue[i].get_output_tensor(j).data for j in range(len(self._output_ports))])
                    return output[0]
            else:
                for name, arr in feed.items():
                    self.req.get_tensor(name).data[...] = arr
                self.req.infer()
                if len(self._output_ports) == 1:
                    output = self.req.get_output_tensor().data
                    return [output]
                else:
                    return [self.req.get_output_tensor(i).data for i in range(len(self._output_ports))]

        # Needs overall optimizations for infer queue or not creating infer requests for each sample - Paul
        elif self.ml_scenario == "Offline":
            if "GPU" in self.ov_device:
                ireq = self.compiled_model.create_infer_request()
                for name, arr in feed.items():
                    self.req.get_tensor(name).data[...] = arr
                ireq.infer()
                if len(self._output_ports) == 1:
                    output = self.req.get_output_tensor().data
                    return [output]
            elif self.ov_device in ["CPU", "NPU"]:
                infer_queue = ov.AsyncInferQueue(self.compiled_model, self.max_batchsize)
                array_len = 1
                for name, arr in feed.items():
                    array_len = len(arr)
                    for i in range(array_len):
                        infer_queue.start_async({0: [arr[i]]})
                infer_queue.wait_all()
                if len(self._output_ports) == 1:
                    output = [infer_queue[i].get_output_tensor().data[0] for i in range(len(infer_queue))]
                    return [output]
                else:
                    return [self.infer_queue[i].get_output_tensor(j).data for i in range(array_len) for j in range(len(self._output_ports))]
