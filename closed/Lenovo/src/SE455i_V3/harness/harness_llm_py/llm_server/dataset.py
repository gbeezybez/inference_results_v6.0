# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import os
from pathlib import Path
from typing import List, Union

from .utils import add_prefix_logger

import numpy as np
import torch


@add_prefix_logger()
class LLMDataset:
    """
    Loads and serves tensors for LLMHarness.
    """

    FILES = []

    def __init__(self, tensor_path: os.PathLike, verbose: bool):
        """
        Initialize the Dataset with the given tensor path.

        Args:
            tensor_path (str): The path to the directory containing the tensor files.
            verbose (bool): Enable verbose logging for data pipeline debugging.

        Raises:
            AssertionError: If the required tensor files are not found in the specified path.
        """
        self.tensor_path = tensor_path
        self.verbose = verbose

        def remove_extension(f_name: os.PathLike): return f_name.split('.')[0]

        def load_tensor(f_name: os.PathLike, pin_memory: bool = True):
            full_path = Path(tensor_path, f_name)
            assert full_path.exists(), f"Missing {f_name} at: {tensor_path}"

            ret_val = np.load(full_path)
            ret_val = ret_val if not pin_memory else torch.tensor(ret_val).pin_memory()
            self.logger.info(f"Loaded tensor to {'pinned' if pin_memory else 'non-pinned'} memory: "
                             f"{f_name} (dtype={ret_val.dtype}, shape={ret_val.shape})")

            return ret_val

        self.tensors = {
            remove_extension(f_name): load_tensor(f_name)
            for f_name in self.FILES
        }

        # also add tensors as top level attrs
        for name, tensor in self.tensors.items():
            setattr(self, name, tensor)

        # any pre-processing on dataset load to be done in post_init
        self.post_init()

    def get_input_tokens(self, sample_indices: List[int]) -> List[List[int]]:
        """
        Get the input tokens for the given sample indices.

        Args:
            sample_indices (List[int]): The list of sample indices to retrieve tokens for.

        Returns:
            List[List[int]]: The input tokens as list of lists.
        """
        raise NotImplementedError()

    def get_stop_tokens(self, sample_indices: List[int]) -> List[List[int]]:
        """
        Get the stop tokens for the given sample indices.

        Args:
            sample_indices (List[int]): The list of sample indices to retrieve tokens for.

        Returns:
            List[List[int]]: The stop tokens as list of lists.
        """
        return [None for _ in sample_indices]

    def __len__(self) -> int:
        """Get size of Dataset"""
        raise NotImplementedError()

    def post_init(self):
        """
        Hook for any pre-processing on dataset load.
        """
        pass
