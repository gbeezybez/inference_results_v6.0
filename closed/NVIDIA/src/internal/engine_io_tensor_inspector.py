# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging

import tensorrt as trt

from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-path",
                        help="Specify where the engine file is")

    args = parser.parse_args()
    engine_path = args.engine_path
    logging.info(f"Loading TensorRT engine: {engine_path}")
    trt_engine = engine_from_bytes(bytes_from_path(engine_path))
    for idx in range(trt_engine.num_io_tensors):
        tensor_name = trt_engine[idx]
        tensor_shape = trt_engine.get_tensor_shape(tensor_name)
        tensor_dtype = trt_engine.get_tensor_dtype(tensor_name)
        tensor_mode = trt_engine.get_tensor_mode(tensor_name)
        tensor_format = trt_engine.get_tensor_format(tensor_name)
        print(f"{tensor_mode} | {tensor_name}: {tensor_shape} in {tensor_dtype}, {tensor_format}")


if __name__ == '__main__':
    main()
