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

from code.actionhandler.base import ActionHandler
from code.common import logging
from code.common.triton.base_config import G_TRITON_BASE_CONFIG
from nvmitten.nvidia.accelerator import GPU
from pathlib import Path
import shutil


class GenerateTritonConfigHandler(ActionHandler):
    def __init__(self, benchmark_conf, main_args, overwrite=False):
        """
        Args:
            benchmark_conf (Dict[str, Any]): The benchmark configuration in dictionary form (Legacy behavior)
            main_args (Dict[str, Any]): The arguments
            overwrite (bool): Skip generation if repo exists, else overwrite
        """
        self.overwrite = overwrite
        self.system_id = benchmark_conf['system_id']
        self.workload_name = benchmark_conf['benchmark'].valstr().lower()

        assert benchmark_conf.get('use_triton', False), "Please run with --harness_type=triton in RUN_ARGS and set use_triton=True in the config class"

        self.model_version = '1'

        scenarios = main_args['scenarios'].split(',')
        assert len(scenarios) == 1, "Please specify only 1 scenario"
        self.scenario = scenarios[0].lower()
        self.decoupled = self.scenario == 'server'  # TODO: get from gen config

        gpus = benchmark_conf['system'].accelerators[GPU]
        logging.info("Accelerators: ")
        for gpu in gpus:
            logging.info(gpu.pretty_string())

        self.num_gpus = len(gpus)
        self.num_servers = benchmark_conf['triton_num_servers']

        trtllm_runtime_flags = benchmark_conf['trtllm_runtime_flags']
        trtllm_build_flags = benchmark_conf['trtllm_build_flags']

        self.enable_chunked_context = trtllm_runtime_flags['enable_chunked_context']
        self.max_num_tokens = trtllm_runtime_flags['max_num_tokens']
        self.num_gpus_per_model = trtllm_build_flags['tensor_parallelism'] * trtllm_build_flags['pipeline_parallelism']

        self.num_triton_models = self.num_gpus // self.num_gpus_per_model

        def get_default_engine_dir(config: dict, scenario: str):
            system_id = config['system_id']
            workload_name = config['benchmark'].valstr().replace('_', '.')
            batch_size = config['gpu_batch_size'][workload_name]
            precision = config['precision']
            build_flags = config['trtllm_build_flags']
            tp_size, pp_size = build_flags['tensor_parallelism'], build_flags['pipeline_parallelism']
            config_ver = config['config_ver']
            default_engine_dir = "./build/engines/{:}/{:}/{:}".format(system_id, workload_name, scenario.capitalize())
            default_engine_dir += f"/{workload_name}-{scenario.capitalize()}-gpu-b{batch_size}-{precision}-tp{tp_size}pp{pp_size}-{config_ver}"
            return default_engine_dir

        default_engine_dir = get_default_engine_dir(benchmark_conf, self.scenario.capitalize())
        self.engine_dir = benchmark_conf.get('engine_dir', default_engine_dir)
        self.engine_dir = Path(self.engine_dir).absolute()

        self.beam_width = 1  # TODO get from generation_config.json

        self.model_store_path_prefix = f"/work/build/triton_model_repos/{self.system_id}/{self.workload_name}/{self.scenario}/repo"

    def cleanup(self, success: bool):
        pass

    def setup(self):
        pass

    def handle(self) -> bool:
        num_devices_per_server = self.num_gpus // self.num_servers
        num_models_per_server = self.num_triton_models // self.num_servers
        for repo_idx in range(self.num_servers):
            model_path_str = f"{self.model_store_path_prefix}_{repo_idx}"
            model_repo_path = Path(model_path_str)
            if model_repo_path.exists():
                if not self.overwrite:
                    logging.info(f"Directory {model_path_str} exists, skipping regeneration")
                    continue
                logging.info(f"Directory {model_path_str} already exists, this will be overwritten")
                shutil.rmtree(model_path_str)
            else:
                logging.info(f"Creating {model_path_str}")

            triton_model_name_prefix = "model"
            for m_idx in range(num_models_per_server):
                model_idx = m_idx + (repo_idx * num_models_per_server)
                triton_model_name = f"{triton_model_name_prefix}-{str(model_idx)}"

                gpu_start_idx = model_idx * self.num_gpus_per_model
                gpu_start_idx %= num_devices_per_server
                gpu_idcs = list(range(gpu_start_idx, gpu_start_idx + self.num_gpus_per_model))
                gpu_idcs = list(map(str, gpu_idcs))
                gpu_idcs = ','.join(gpu_idcs)
                model_dir = model_repo_path.joinpath(triton_model_name, self.model_version)
                model_dir.mkdir(parents=True, exist_ok=False)
                config_file_path = model_repo_path.joinpath(triton_model_name, "config.pbtxt")

                assert (self.engine_dir / "rank0.engine").exists(), "Please specify valid --engine_dir in RUN_ARGS, no engine found at {self.engine_dir}"
                logging.info(f"Using TRTLLM engine at {self.engine_dir}")

                engine_file_name = str(self.engine_dir)

                with config_file_path.open(mode='w', encoding='utf-8') as f:
                    f.write(G_TRITON_BASE_CONFIG.format(
                        model_name=triton_model_name,
                        is_decoupled=self.decoupled,
                        beam_width=self.beam_width,
                        engine_path=engine_file_name,
                        gpu_device_idx=gpu_idcs,
                        enable_chunked_context=self.enable_chunked_context,
                        max_num_tokens=self.max_num_tokens))
            logging.info(f"Generated triton repository at {model_repo_path}")

    def handle_failure(self):
        pass
