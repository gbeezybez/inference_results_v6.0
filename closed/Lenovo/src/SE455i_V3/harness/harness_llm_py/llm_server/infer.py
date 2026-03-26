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
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from transformers import AutoTokenizer

from .dataset import LLMDataset
from .server import LLMServer


class LLMChatServer(LLMServer):
    """
    LLMChatServer is a subclass of LLMServer designed to handle chat-based inference.
    Uses pre-trained language model tokenizer to render request and response texts.
    Can be provided optional LLMDataset which enables infering directly from dataset sample indices.
    """

    class LLMChatEntry:
        request_text: str
        response_text: str
        response_tokens: List[int]

    def __init__(self, tokenizer_model_path: str, dataset: Optional[LLMDataset] = None, *args, **kwargs):
        """
        Args:
            tokenizer_model_path (str): The path to the pre-trained model to use with transformers.AutoTokenizer.
            dataset (Optional[LLMDataset]): The dataset to be used when `infer` is given int sample indices .
            *args: Additional arguments for the LLMServer.
            **kwargs: Additional keyword arguments for LLMServer.
        """
        super().__init__(*args, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
        self.dataset = dataset

        self.next_request_id = 1
        self.text_log: Dict[int, LLMChatServer.LLMChatEntry] = {}

    def get_next_id(self, reset: bool = False):
        """
        Generates the next request ID.

        Args:
            reset (bool): If True, resets the request ID counter to 1. Defaults to False.

        Returns:
            int: The next request ID.
        """
        self.next_request_id = 1 if reset else self.next_request_id
        ret_val = self.next_request_id
        self.next_request_id += 1
        return ret_val

    def infer(self, input: List[Union[str, int]] = None) -> Dict[int, Tuple[str, str]]:
        """
        Performs (blocking) inference on the given input and returns the request and response texts.

        Args:
            input (List[Union[str, int]], optional): A list of input samples, either strings or integers.

        Returns:
            Dict[int, LLMChatEntry]: A dictionary mapping request IDs to LLMChatEntry(request_text, response_text) texts.
        """
        self.text_log = defaultdict(LLMChatServer.LLMChatEntry)

        self.issue_queries(input)
        self.flush_queries()

        return self.text_log

    def run_infer_loop(self):
        """
        Runs an interactive inference loop, accepting user input and providing model responses.
        """
        while True:
            prompt = input("[llm-server] <-- ")
            if prompt == 'exit':
                break

            response = list(self.infer([prompt]).values())[0].response_text
            self.logger.info(f"[llm-server] --> {str(response)}")

    ##### LLMServer overrides #####

    def issue_queries(self, query_samples: List[Union[str, int]]):
        def prepare_query_sample(sample: Union[str, int]):
            if type(sample) is int:
                input_tokens = self.dataset.get_input_tokens([sample])[0]
                stop_tokens = self.dataset.get_stop_tokens([sample])[0]
                sample = self.tokenizer.decode(input_tokens, skip_special_tokens=True)

            else:
                input_tokens = self.tokenizer.batch_encode_plus([sample])['input_ids'][0]
                stop_tokens = None

            request_id = self.get_next_id()
            self.text_log[request_id].request_text = sample
            return (request_id, input_tokens, stop_tokens)

        super().issue_queries([
            prepare_query_sample(sample)
            for sample in query_samples
        ])

    def complete_request(self, request_id, output_tokens, is_first_token):
        output_tokens = np.asarray(output_tokens).astype(np.uint32).reshape(-1)
        output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)

        self.text_log[request_id].response_text = output_text
        self.text_log[request_id].response_tokens = output_tokens
