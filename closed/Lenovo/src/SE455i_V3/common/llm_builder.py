# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
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

from abc import ABC
import importlib
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Optional

from code.common.utils import parse_cli_flags
from nvmitten.utils import logging


class TRTLLMBuilder(ABC):
    """
    TRTLLMBuilder, Used as a Mixin for an Operation.
    Uses TRTLLM provided build.py to create engine directly from weights or TRTLLM-checkpoint.
    Quantized TRTLLM checkpoint is created if needed, using model-opt framework.
    """

    TRTLLM_BUILD_SCRIPT = Path("tensorrt_llm/commands/build.py")
    TRTLLM_CHECKPOINT_SCRIPT = Path("examples/quantization/quantize.py")

    def __init__(
        self,
        model_name: str,
        model_path: os.PathLike,
        checkpoint_path: os.PathLike,

        # model
        precision: str = "fp16",
        max_batch_size: int = 16,

        # checkpoint (quantization)
        calib_dataset_path: os.PathLike = None,
        calib_batch_size: int = 1024,
        calib_max_batches: int = 1,

        trtllm_build_flags: Dict[str, Any] = {},
        trtllm_checkpoint_flags: Dict[str, Any] = {},

        # modules
        trtllm_path: os.PathLike = "/work/build/TRTLLM",
        quant_module_path: Optional[os.PathLike] = None,  # overrides default TRTLLM quantize module

        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name

        self.trtllm_path = Path(trtllm_path)
        self.quant_module_path = Path(quant_module_path or trtllm_path)
        self.model_path = Path(model_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.calib_dataset_path = calib_dataset_path

        if self.calib_dataset_path is None:
            logging.info(f"Calibration dataset not specified. See model README.md for instructions regarding quantization and checkpoint generation.")
            assert self.checkpoint_path.exists(), f"TRTLLM Checkpoint not found at: {self.checkpoint_path}"

        self.precision = precision
        self.max_batch_size = max_batch_size

        # NOTE(vir): these are not flags for trtllm build command
        self.tp_size = trtllm_build_flags.pop('tensor_parallelism')
        self.pp_size = trtllm_build_flags.pop('pipeline_parallelism')
        self.world_size = self.tp_size * self.pp_size
        assert self.world_size > 0

        # TODO(vir): fix: process overrides within __init__ to enable sub-classes
        self.build_args = {
            'workers': self.world_size if self.world_size > 1 else None,
            'max_batch_size': self.max_batch_size,
        }

        self.checkpoint_args = {
            'tp_size': self.tp_size,
            'pp_size': self.pp_size,
            'dtype': 'float16',  # dtype of weights, and activations of non-quantized part (embedding, lm-head)
            'qformat': str('nvfp4' if self.precision == 'fp4' else self.precision),

            'model_dir': str(self.model_path.absolute()),
            'output_dir': str(self.checkpoint_path.absolute()),
            'calib_dataset': self.calib_dataset_path,
            'calib_size': calib_batch_size * calib_max_batches,
        }

        self.build_args |= trtllm_build_flags
        self.checkpoint_args |= trtllm_checkpoint_flags
        self.custom_env = os.environ.copy()

    def process_checkpoint_overrides(self):
        if checkpoint_overrides := os.environ.get('TRTLLM_CHECKPOINT_FLAGS', None):
            logging.info(f"Detected TRTLLM_CHECKPOINT_FLAGS: {checkpoint_overrides}")
            checkpoint_overrides = parse_cli_flags(checkpoint_overrides)
            for key, override in checkpoint_overrides.items():
                logging.info(f"Overriding {key}: {override}")
                self.checkpoint_args[key] = override

    def process_build_overrides(self):
        if build_overrides := os.environ.get('TRTLLM_BUILD_FLAGS', None):
            logging.info(f"Detected TRTLLM_BUILD_FLAGS: {build_overrides}")
            build_overrides = parse_cli_flags(build_overrides)

            self.tp_size = build_overrides.pop('tensor_parallelism', self.tp_size)
            self.pp_size = build_overrides.pop('pipeline_parallelism', self.pp_size)
            self.world_size = self.tp_size * self.pp_size

            if build_overrides.pop('max_batch_size', None):
                logging.info(f"Ignoring max_batch_size override in favor of legacy field --gpu_batch_size={self.max_batch_size}")

            self.build_args |= {
                'workers': self.world_size if self.world_size > 1 else None,
                'max_batch_size': self.max_batch_size,
            }

            for key, override in build_overrides.items():
                logging.info(f"Overriding {key}: {override}")
                self.build_args[key] = override

        # NOTE(vir): assumes all trtllm build.py flags use same convention
        # update inplace: 'fp4' -> 'nvfp4'
        self.build_args |= {
            key: 'nvfp4'
            for key, value in self.build_args.items()
            if value == 'fp4'
        }

    def generate_checkpoint(self):
        self.process_checkpoint_overrides()

        if self.calib_dataset_path is None:
            raise FileNotFoundError("Calibration Dataset not specified.")

        if not importlib.util.find_spec("tensorrt_llm"):
            raise ModuleNotFoundError("Cannot import tensorrt_llm module. Please run `make build_trt_llm`.")

        if not (script_location := self.quant_module_path / self.TRTLLM_CHECKPOINT_SCRIPT).exists():
            raise FileNotFoundError(f"Could not locate quant script at: {script_location}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Could not find model weights at: {self.model_path}")

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_path.exists():
            logging.warning(f"{self.checkpoint_path} already exists. This will be overwritten.")
        else:
            self.checkpoint_path.mkdir(parents=True)

        quant_flags = [f"--{key}={value}" for key, value in self.checkpoint_args.items() if value is not None]
        quant_cmd = [sys.executable, '-m', '.'.join(self.TRTLLM_CHECKPOINT_SCRIPT.with_suffix('').parts)] + quant_flags

        if 'mixtral' in self.model_name and self.precision == 'fp4':
            raise NotImplementedError("Mixtral checkpoint generation for fp4 is not supported - please use published checkpoint at " + \
                                      "/work/build/models/Mixtral/fp4-quantized-modelopt/mixtral-8x7b-instruct-v0.1-tp1pp1-fp4-e7.25-passing_acuracies." +\
                                      "If not found, go to /opt/fp4-quantized-modelopt/ and untar the tarball.")
        logging.info(f"Generating {self.model_name} {self.precision} TRTLLM checkpoint in: {self.checkpoint_path}.")
        logging.info(f"Command executing in {self.quant_module_path}: {' '.join(quant_cmd)}")

        tik = time.time()
        ret = subprocess.run(quant_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(self.quant_module_path), env=self.custom_env)
        tok = time.time()

        # Save stdout and stderr logs
        with (self.checkpoint_path / 'stdout.txt').open(mode='w') as f:
            f.write(ret.stdout)
        with (self.checkpoint_path / 'stderr.txt').open(mode='w') as f:
            f.write(ret.stderr)

        if ret.returncode != 0:
            logging.error(ret.stderr)
            raise RuntimeError(f"Checkpoint generation failed. Logs dumped to: {self.checkpoint_path}")

        logging.info(f"Quantization complete in {tok - tik}s. Saved to: {self.checkpoint_path}")

    def build_engine(self, engine_dir: os.PathLike):
        self.process_build_overrides()

        if not importlib.util.find_spec("tensorrt_llm"):
            raise ModuleNotFoundError("Cannot import tensorrt_llm module. Please run `make build_trt_llm`.")

        if not (build_script := self.trtllm_path / TRTLLMBuilder.TRTLLM_BUILD_SCRIPT).exists():
            raise FileNotFoundError(f"Could not locate TRTLLM build script ({build_script}), please run `make clone_trt_llm`")

        if not (self.checkpoint_path / 'rank0.safetensors').exists():
            logging.info(f"Could not find valid TRTLLM checkpoint at: {self.checkpoint_path}")
            logging.info(f"Generating {self.precision} TRTLLM checkpoint at: {self.checkpoint_path}")
            self.generate_checkpoint()

        # For TRT-LLM, the engine name is fixed; engines are differentiated by dir.
        engine_dir.mkdir(parents=True, exist_ok=True)
        engine_file = Path(engine_dir) / 'rank0.plan'
        if engine_file.exists():
            logging.warning(f"{engine_file} already exists. This will be overwritten.")

        build_flags = [f"--{key}={value}" for key, value in self.build_args.items() if value is not None]
        build_flags += [
            f"--checkpoint_dir={str(self.checkpoint_path.absolute())}",
            f"--output_dir={str(engine_dir.absolute())}"
        ]

        build_cmd = [sys.executable, '-m', '.'.join(TRTLLMBuilder.TRTLLM_BUILD_SCRIPT.with_suffix('').parts)] + build_flags
        custom_env = os.environ.copy()

        logging.info(f"Building engine in: {engine_dir}")
        logging.info(f"Command executing in {self.trtllm_path}: {' '.join(build_cmd)}")

        tik = time.time()
        ret = subprocess.run(build_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(self.trtllm_path), env=custom_env)
        tok = time.time()

        # Save stdout and stderr logs
        with engine_file.with_suffix(".stdout").open(mode='w') as f:
            f.write(ret.stdout)
        with engine_file.with_suffix(".stderr").open(mode='w') as f:
            f.write(ret.stderr)

        if ret.returncode != 0:
            logging.error(ret.stderr)
            raise RuntimeError(f"Engine build failed. Logs dumped to: {engine_dir}.")

        logging.info(f"Engine build complete in {tok - tik}s. Saved to: {engine_dir}")

    def __call__(self, engine_dir: os.PathLike):
        self.build_engine(engine_dir)
