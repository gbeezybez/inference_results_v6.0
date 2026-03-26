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

from __future__ import annotations
import builtins
import dataclasses
from functools import wraps
import json
import os
from pathlib import Path
from typing import Any, Optional, Dict

from code.common import logging


def dataclass_slice(cls: type) -> type:
    """
    makes a dataclass where extra kwargs passed to constructor will be ignored.
    usecase example:
        - this can represent subset of a json file, which can be marshalled via dataclass semantics.
        - ignores values not declared as field
    """

    def wrap(cls: type) -> type:
        @wraps(cls.__init__)
        def __init__(self, **kwargs):
            def create(type_str: str, value: Any) -> Any:
                py_type = getattr(builtins, type_str, eval(type_str))
                return py_type(value) if not isinstance(value, dict) else py_type(**value)

            fields = {field.name: field.type for field in dataclasses.fields(cls)}
            for name, value in kwargs.items():
                if name in fields:
                    setattr(self, name, create(fields[name], value))

        cls.__init__ = __init__
        return cls

    return wrap(dataclasses.dataclass(cls))


@dataclasses.dataclass
class EngineConfig:
    @dataclass_slice
    class TRTLLMConfig:
        """
        TRTLLM Engine specific config.json.
        @dataclass_slice lets other values present in file be ignored.
        """

        @dataclass_slice
        class BuildConfig:
            @dataclass_slice
            class PluginConfig:
                use_paged_context_fmha: bool = True

            max_input_len: int = 1024
            max_seq_len: int = 2048
            max_batch_size: int = 1
            max_beam_width: int = 1
            max_num_tokens: int = 8192
            plugin_config: EngineConfig.TRTLLMConfig.BuildConfig.PluginConfig = dataclasses.field(default_factory=PluginConfig)

        @dataclass_slice
        class PretrainedConfig:
            @dataclass_slice
            class Mapping:
                tp_size: int = 1
                pp_size: int = 1

            mapping: EngineConfig.TRTLLMConfig.PretrainedConfig.Mapping

        build_config: EngineConfig.TRTLLMConfig.BuildConfig
        pretrained_config: EngineConfig.TRTLLMConfig.PretrainedConfig

    engine_dir: os.PathLike
    trtllm_config: EngineConfig.TRTLLMConfig

    @staticmethod
    def from_engine_dir(engine_dir: os.PathLike) -> EngineConfig:
        """
        Load EngineConfig from a directory containing a config.json file.

        Args:
            engine_dir (os.PathLike): The directory containing the engine configuration.

        Returns:
            EngineConfig: The loaded EngineConfig object.
        """
        config_file = Path(engine_dir) / 'config.json'
        assert config_file.exists(), f"TRTLLM Engine config file not found at: {config_file}"

        with config_file.open() as config_str:
            trtllm_config = EngineConfig.TRTLLMConfig(**json.load(config_str))

        return EngineConfig(engine_dir=engine_dir, trtllm_config=trtllm_config)


@dataclasses.dataclass
class HarnessConfig:
    @dataclass_slice
    class GenerationConfig:
        """
        Model specific generation_config.json relevant to LLMHarness.
        @dataclass_slice lets other values present in file be ignored.
        """
        eos_token_id: int = 2
        max_output_len: int = 1024
        min_output_len: int = 1
        name: str = "llama"
        runtime_beam_width: int = 1
        streaming: bool = True
        temperature: float = 1.0
        top_k: int = 1
        top_p: float = 0.001
        use_stop_tokens: bool = False

    random_seed: Optional[int] = 0
    traffic_distribution_policy: str = "load_balancing"

    gen_config: HarnessConfig.GenerationConfig = dataclasses.field(default_factory=GenerationConfig)
    trtllm_checkpoint_flags: Dict[str, Any] = dataclasses.field(default_factory=dict)
    trtllm_build_flags: Dict[str, Any] = dataclasses.field(default_factory=dict)
    trtllm_runtime_flags: Dict[str, Any] = dataclasses.field(default_factory=dict)

    log_dir: Optional[str] = None

    @staticmethod
    def load_generation_config(path: os.PathLike) -> HarnessConfig.GenerationConfig:
        """
        Load GenerationConfig from a JSON file.

        Args:
            path (os.PathLike): The path to the JSON file containing the generation configuration.

        Returns:
            HarnessConfig.GenerationConfig: The loaded GenerationConfig object.
        """
        with Path(path).open() as config_str:
            gen_config = HarnessConfig.GenerationConfig(**json.load(config_str)['generation_config'])
        return gen_config

    def validate_compatible_engine(self, engine_config: EngineConfig):
        """
        Validate that the HarnessConfig is compatible with the given EngineConfig.

        Args:
            engine_config (EngineConfig): The engine configuration to validate against.

        Raises:
            AssertionError: If the configurations are not compatible.
        """
        assert self.gen_config.runtime_beam_width <= engine_config.trtllm_config.build_config.max_beam_width
        assert self.gen_config.max_output_len <= (engine_config.trtllm_config.build_config.max_seq_len - engine_config.trtllm_config.build_config.max_input_len)

        assert self.trtllm_build_flags['tensor_parallelism'] == engine_config.trtllm_config.pretrained_config.mapping.tp_size
        assert self.trtllm_build_flags['pipeline_parallelism'] == engine_config.trtllm_config.pretrained_config.mapping.pp_size
        assert self.trtllm_runtime_flags['max_batch_size'] <= engine_config.trtllm_config.build_config.max_batch_size
        assert self.trtllm_runtime_flags['max_num_tokens'] <= engine_config.trtllm_config.build_config.max_num_tokens

        if (batch_size := self.trtllm_runtime_flags['max_batch_size']) != engine_config.trtllm_config.build_config.max_batch_size:
            logging.warning(f"runtime max_batch_size ({batch_size}) is different from engine's maximum ({engine_config.trtllm_config.build_config.max_batch_size})")
            logging.warning(f"using runtime max_batch_size={batch_size}")

        if (runtime_mnt := self.trtllm_runtime_flags['max_num_tokens']) != engine_config.trtllm_config.build_config.max_num_tokens:
            logging.warning(f"runtime max_num_tokens ({runtime_mnt}) is different from engine's maximum ({engine_config.trtllm_config.build_config.max_num_tokens})")
            logging.warning(f"using runtime max_num_tokens={runtime_mnt}")
