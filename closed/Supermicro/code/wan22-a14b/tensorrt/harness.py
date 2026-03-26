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

"""
WAN22 T2V Harness for MLPerf Inference.

This module implements the harness for running WAN2.2 Text-to-Video benchmark
with MLPerf LoadGen integration. Supports both single-GPU and multi-GPU modes.
"""

import contextlib
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import mlperf_loadgen as lg
from diffusers.utils import export_to_video
from diffusers.utils import logging as diffusers_logging

# Disable diffusers progress bar for DiT inference steps
diffusers_logging.disable_progress_bar()
from nvmitten.configurator import autoconfigure, bind
from tqdm import tqdm

from code.common import logging
from code.fields import general as general_fields
from code.ops.harness import PyHarnessOp
from code.ops.loadgen import LoadgenConfFilesOp

from . import fields as wan22_fields
from .dataset import Wan22Dataset
from .utils import (
    find_free_port,
    setup_distributed_env,
    video_to_numpy,
    video_to_loadgen_response,
    create_placeholder_video,
)

from visual_gen import setup_configs
from visual_gen.models.transformers.wan_transformer import ditWanTransformer3DModel as WanTransformer3DModel
from visual_gen.models.vaes.wan_vae import ditWanAutoencoderKL
from visual_gen.pipelines.wan_pipeline import ditWanPipeline as WanPipeline
from visual_gen.utils import create_default_dit_configs

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Wan22Response:
    """Response object for WAN22 video generation."""
    sample_ids: List[int]
    sample_indices: List[int]
    generated_videos: List[np.ndarray]


# =============================================================================
# Pipeline Handler
# =============================================================================

class VflyPipelineHandler:
    """Handles VisionFly pipeline initialization and inference."""

    def __init__(
        self,
        model_path: str,
        num_frames: int,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        guidance_scale_2: float,
        attn_type: str,
        linear_type: str,
        ulysses_size: int,
        cfg_size: int,
        enable_cpu_offload: bool,
        cpu_offload_stride: int,
        # Context parallelism
        cp_size: int = 1,
        ring_size: int = 1,
        # Tensor parallelism
        tp_size: int = 1,
        # Data parallelism
        dp_size: int = 1,
        # FSDP
        fsdp_size: int = 1,
        t5_fsdp_size: int = 1,
        # VAE parallelism
        disable_parallel_vae: bool = False,
        parallel_vae_split_dim: str = "width",
    ):
        self.model_path = model_path
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.guidance_scale_2 = guidance_scale_2
        self.attn_type = attn_type
        self.linear_type = linear_type
        # Context parallelism
        self.ulysses_size = ulysses_size
        self.cfg_size = cfg_size
        self.cp_size = cp_size
        self.ring_size = ring_size
        # Tensor parallelism
        self.tp_size = tp_size
        # Data parallelism
        self.dp_size = dp_size
        # FSDP
        self.fsdp_size = fsdp_size
        self.t5_fsdp_size = t5_fsdp_size
        # VAE parallelism
        self.disable_parallel_vae = disable_parallel_vae
        self.parallel_vae_split_dim = parallel_vae_split_dim
        # CPU offload
        self.enable_cpu_offload = enable_cpu_offload
        self.cpu_offload_stride = cpu_offload_stride
        self.pipe = None
        self.local_rank = 0

    def initialize(self):
        """Initialize the VisionFly pipeline (handles distributed setup if needed)."""
        torch.autograd.set_grad_enabled(False)

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # Total parallel size includes all parallelism types
        total_parallel_size = self.tp_size * self.ulysses_size * self.ring_size * self.cp_size * self.dp_size * self.cfg_size

        if not dist.is_initialized():
            # Set default environment variables for single-process mode if not set
            if "RANK" not in os.environ:
                os.environ["RANK"] = "0"
            if "WORLD_SIZE" not in os.environ:
                os.environ["WORLD_SIZE"] = "1"
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = "localhost"
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"
            dist.init_process_group(backend="nccl")
            self.local_rank = dist.get_rank()
        torch.cuda.set_device(self.local_rank)
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize the VisionFly pipeline components."""
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        logging.info(f"Initializing WAN2.2 pipeline (rank {self.local_rank}/{world_size})...")
        logging.info(f"Multi-device config: ulysses_size={self.ulysses_size}, cfg_size={self.cfg_size}, "
                     f"cp_size={self.cp_size}, ring_size={self.ring_size}, tp_size={self.tp_size}")

        # Configure visual_gen
        vfly_configs = create_default_dit_configs()
        vfly_configs["pipeline"]["enable_torch_compile"] = True
        vfly_configs["pipeline"]["torch_compile_models"] = ["transformer"]
        vfly_configs["pipeline"]["torch_compile_mode"] = "max-autotune-no-cudagraphs"
        vfly_configs["pipeline"]["fuse_qkv"] = True
        vfly_configs["attn"]["type"] = self.attn_type
        vfly_configs["linear"]["type"] = self.linear_type

        # Context parallelism settings
        vfly_configs["parallel"]["dit_ulysses_size"] = self.ulysses_size
        vfly_configs["parallel"]["dit_cfg_size"] = self.cfg_size
        vfly_configs["parallel"]["dit_cp_size"] = self.cp_size
        vfly_configs["parallel"]["dit_ring_size"] = self.ring_size

        # Tensor parallelism settings
        vfly_configs["parallel"]["dit_tp_size"] = self.tp_size

        # Data parallelism settings
        vfly_configs["parallel"]["dit_dp_size"] = self.dp_size

        # FSDP settings
        vfly_configs["parallel"]["dit_fsdp_size"] = self.fsdp_size
        vfly_configs["parallel"]["t5_fsdp_size"] = self.t5_fsdp_size

        # VAE parallelism settings
        vfly_configs["parallel"]["disable_parallel_vae"] = self.disable_parallel_vae
        vfly_configs["parallel"]["parallel_vae_split_dim"] = self.parallel_vae_split_dim

        # Refiner settings (mirror main DiT settings)
        vfly_configs["parallel"]["refiner_dit_ulysses_size"] = self.ulysses_size
        vfly_configs["parallel"]["refiner_dit_cfg_size"] = self.cfg_size
        vfly_configs["parallel"]["refiner_dit_cp_size"] = self.cp_size
        vfly_configs["parallel"]["refiner_dit_ring_size"] = self.ring_size
        vfly_configs["parallel"]["refiner_dit_tp_size"] = self.tp_size
        vfly_configs["parallel"]["refiner_dit_dp_size"] = self.dp_size
        vfly_configs["parallel"]["refiner_dit_fsdp_size"] = self.fsdp_size

        setup_configs(**vfly_configs)

        # Load components
        vae = ditWanAutoencoderKL.from_pretrained(
            self.model_path, subfolder="vae", torch_dtype=torch.float32
        )
        transformer = WanTransformer3DModel.from_pretrained(
            self.model_path, subfolder="transformer", torch_dtype=torch.bfloat16
        )

        # Create pipeline
        self.pipe = WanPipeline.from_pretrained(
            self.model_path,
            vae=vae,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            **vfly_configs,
        )

        self._configure_device_placement()
        logging.info(f"[Rank {self.local_rank}] WAN2.2 pipeline initialized successfully")

    def _configure_device_placement(self):
        """Configure pipeline device placement."""
        if self.enable_cpu_offload:
            model_wise = ["text_encoder"]
            block_wise = ["transformer"]
            if "Wan2.2" in self.model_path:
                block_wise.append("transformer_2")
            self.pipe.enable_async_cpu_offload(
                model_wise=model_wise,
                block_wise=block_wise,
                offloading_stride=self.cpu_offload_stride,
            )
            logging.info(f"[Rank {self.local_rank}] Enabled async CPU offload")
        else:
            self.pipe.to(f"cuda:{self.local_rank}")
            logging.info(f"[Rank {self.local_rank}] Pipeline moved to cuda:{self.local_rank}")

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        latents: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
    ):
        """Generate video from prompt."""
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            guidance_scale=self.guidance_scale,
            guidance_scale_2=self.guidance_scale_2,
            num_inference_steps=num_inference_steps or self.num_inference_steps,
            latents=latents,
        )
        return output.frames[0]

    @property
    def is_ready(self) -> bool:
        return self.pipe is not None


# =============================================================================
# Error Handler
# =============================================================================

class GenerationErrorHandler:
    """Handles generation errors."""

    def __init__(self, num_frames: int, height: int, width: int):
        self.num_frames = num_frames
        self.height = height
        self.width = width

    def handle_error(self, error: Exception, context: str = "") -> np.ndarray:
        """Handle generation error and return placeholder."""
        logging.error(f"Error generating video{' - ' + context if context else ''}: {error}")
        return create_placeholder_video(self.num_frames, self.height, self.width)


# =============================================================================
# Inference Handler
# =============================================================================

class InferenceHandler:
    """Handles inference execution with error handling."""

    def __init__(self, pipeline_handler: VflyPipelineHandler, error_handler: GenerationErrorHandler):
        self.pipeline = pipeline_handler
        self.error_handler = error_handler

    def generate_batch(
        self,
        prompts: List[str],
        negative_prompt: Optional[str] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> List[np.ndarray]:
        """Generate videos for a batch of prompts."""
        results = []
        for idx, prompt in enumerate(prompts):
            logging.info(f"Generating video {idx + 1}/{len(prompts)}: \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"")
            frames = self.pipeline.generate(prompt, negative_prompt, latents)
            if frames is not None:
                result = video_to_numpy(frames)
                results.append(result)
                logging.debug(f"Generated video with shape: {result.shape}")
            else:
                placeholder = self.error_handler.handle_error(
                    Exception("Generation returned None"), prompt[:50]
                )
                results.append(placeholder)
        return results

    def warmup(self, prompt: str, negative_prompt: Optional[str], iterations: int = 2):
        """Run warmup inference."""
        logging.info(f"Running warm up with {iterations} iterations")
        for _ in range(iterations):
            self.pipeline.generate(prompt, negative_prompt, num_inference_steps=2)


class LoadGenResponseHandler:
    """Handles LoadGen response processing."""

    def __init__(self, accuracy_mode: bool = False):
        self.total_samples = 0
        self.accuracy_mode = accuracy_mode
        # Store videos and their prompts for accuracy mode
        self.accuracy_videos: Dict[int, np.ndarray] = {}

    def process_response(self, response: Wan22Response):
        """Process response and report to LoadGen."""
        qsr = []
        for idx, sample_id in enumerate(response.sample_ids):
            video_data = response.generated_videos[idx]
            if isinstance(video_data, torch.Tensor):
                video_data = video_data.cpu().numpy()
            qsr.append(video_to_loadgen_response(sample_id, video_data))

            # Store video for accuracy mode
            if self.accuracy_mode:
                sample_idx = response.sample_indices[idx]
                self.accuracy_videos[sample_idx] = video_data

        lg.QuerySamplesComplete(qsr)
        self.total_samples += len(response.sample_ids)
        logging.debug(f"Reported {len(response.sample_ids)} samples")


# =============================================================================
# Multi-Process Worker
# =============================================================================

def _worker_fn(
    rank: int,
    world_size: int,
    master_port: int,
    init_barrier: mp.Barrier,
    work_queue: mp.Queue,
    result_queue: mp.Queue,
    shutdown_event: mp.Event,
    pipeline_kwargs: dict,
    gpu_ids: Optional[str] = None,
):
    """Worker function for multi-GPU inference processes."""
    # Set CUDA_VISIBLE_DEVICES first if specified
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    setup_distributed_env(rank, world_size, master_port)

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    torch.autograd.set_grad_enabled(False)

    handler = VflyPipelineHandler(**pipeline_kwargs)
    handler.local_rank = rank
    handler._initialize_pipeline()

    logging.info(f"[Worker {rank}] Initialized and ready")
    init_barrier.wait()

    while not shutdown_event.is_set():
        try:
            work_item = work_queue.get(timeout=0.1)
            if work_item is None:
                break

            prompt, negative_prompt, latents = work_item
            frames = handler.generate(prompt, negative_prompt, latents)

            if rank == 0:
                result_queue.put(video_to_numpy(frames))

        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"[Worker {rank}] Error: {e}")
            if rank == 0:
                result_queue.put(None)

    dist.destroy_process_group()
    logging.info(f"[Worker {rank}] Shutdown complete")


# =============================================================================
# Main Server Class
# =============================================================================

@autoconfigure
@bind(general_fields.verbose)
@bind(wan22_fields.model_path, "model_path")
@bind(wan22_fields.num_frames, "num_frames")
@bind(wan22_fields.height, "height")
@bind(wan22_fields.width, "width")
@bind(wan22_fields.num_inference_steps, "num_inference_steps")
@bind(wan22_fields.guidance_scale, "guidance_scale")
@bind(wan22_fields.guidance_scale_2, "guidance_scale_2")
@bind(wan22_fields.attn_type, "attn_type")
@bind(wan22_fields.linear_type, "linear_type")
# Context parallelism fields
@bind(wan22_fields.ulysses_size, "ulysses_size")
@bind(wan22_fields.cfg_size, "cfg_size")
@bind(wan22_fields.cp_size, "cp_size")
@bind(wan22_fields.ring_size, "ring_size")
# Tensor parallelism fields
@bind(wan22_fields.tp_size, "tp_size")
# Data parallelism fields
@bind(wan22_fields.dp_size, "dp_size")
# FSDP fields
@bind(wan22_fields.fsdp_size, "fsdp_size")
@bind(wan22_fields.t5_fsdp_size, "t5_fsdp_size")
# VAE parallelism fields
@bind(wan22_fields.disable_parallel_vae, "disable_parallel_vae")
@bind(wan22_fields.parallel_vae_split_dim, "parallel_vae_split_dim")
# CPU offload fields
@bind(wan22_fields.enable_visual_gen_cpu_offload, "enable_visual_gen_cpu_offload")
@bind(wan22_fields.cpu_offload_stride, "cpu_offload_stride")
# GPU selection
@bind(wan22_fields.gpu_ids, "gpu_ids")
class Wan22Server:
    """
    Server for WAN2.2 T2V inference.
    Supports both single-GPU and multi-GPU modes.
    For multi-GPU, spawns worker processes for distributed inference.
    """

    def __init__(
        self,
        dataset: Wan22Dataset,
        model_path: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        num_frames: int = 81,
        height: int = 720,
        width: int = 1280,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.0,
        guidance_scale_2: float = 3.0,
        attn_type: str = "sage-attn",
        linear_type: str = "default",
        # Context parallelism
        ulysses_size: int = 1,
        cfg_size: int = 1,
        cp_size: int = 1,
        ring_size: int = 1,
        # Tensor parallelism
        tp_size: int = 1,
        # Data parallelism
        dp_size: int = 1,
        # FSDP
        fsdp_size: int = 1,
        t5_fsdp_size: int = 1,
        # VAE parallelism
        disable_parallel_vae: bool = False,
        parallel_vae_split_dim: str = "width",
        # CPU offload
        enable_visual_gen_cpu_offload: bool = False,
        cpu_offload_stride: int = 1,
        # GPU selection
        gpu_ids: Optional[str] = None,
        verbose: bool = False,
        # Accuracy mode for video dumping
        accuracy_mode: bool = False,
    ):
        # Set CUDA_VISIBLE_DEVICES first, before any CUDA operations
        self.gpu_ids = gpu_ids
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            logging.info(f"Setting CUDA_VISIBLE_DEVICES={gpu_ids}")

        self.dataset = dataset
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.verbose = verbose
        self.accuracy_mode = accuracy_mode

        # Parallelism configuration
        # Total parallel size = tp * ulysses * ring * cp * dp * cfg
        self.total_parallel_size = tp_size * ulysses_size * ring_size * cp_size * dp_size * cfg_size
        self.is_multiprocess = self.total_parallel_size > 1

        # Pipeline configuration
        self.pipeline_kwargs = {
            "model_path": model_path,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "guidance_scale_2": guidance_scale_2,
            "attn_type": attn_type,
            "linear_type": linear_type,
            # Context parallelism
            "ulysses_size": ulysses_size,
            "cfg_size": cfg_size,
            "cp_size": cp_size,
            "ring_size": ring_size,
            # Tensor parallelism
            "tp_size": tp_size,
            # Data parallelism
            "dp_size": dp_size,
            # FSDP
            "fsdp_size": fsdp_size,
            "t5_fsdp_size": t5_fsdp_size,
            # VAE parallelism
            "disable_parallel_vae": disable_parallel_vae,
            "parallel_vae_split_dim": parallel_vae_split_dim,
            # CPU offload
            "enable_cpu_offload": enable_visual_gen_cpu_offload,
            "cpu_offload_stride": cpu_offload_stride,
        }

        # Handlers
        self.error_handler = GenerationErrorHandler(num_frames, height, width)
        self.response_handler = LoadGenResponseHandler(accuracy_mode=accuracy_mode)

        # Multi-process state
        self.workers = []
        self.work_queues = []
        self.result_queue = None
        self.shutdown_event = None

        # Initialize pipeline or workers
        if self.is_multiprocess:
            self._init_multiprocess()
        else:
            self._init_single_process()

        # Server threading
        self.sample_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.sample_count = 0
        self.processed_count = 0

        # Progress bar for video generation
        self.progress_bar = tqdm(
            total=0,
            desc="Generating videos",
            unit="videos",
            bar_format='{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            dynamic_ncols=True,
        )
        self.progress_lock = threading.Lock()

        self.worker_thread = threading.Thread(target=self._process_samples, daemon=True)
        self.response_thread = threading.Thread(target=self._process_responses, daemon=True)
        self.worker_thread.start()
        self.response_thread.start()

    def _init_single_process(self):
        """Initialize in single-process mode."""
        self.pipeline_handler = VflyPipelineHandler(**self.pipeline_kwargs)
        self.pipeline_handler.initialize()
        self.inference_handler = InferenceHandler(self.pipeline_handler, self.error_handler)

    def _init_multiprocess(self):
        """Initialize in multi-process mode by spawning workers."""
        logging.info(f"Initializing multi-GPU mode with {self.total_parallel_size} processes")

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        master_port = find_free_port()
        self.shutdown_event = mp.Event()
        self.result_queue = mp.Queue()
        init_barrier = mp.Barrier(self.total_parallel_size + 1)
        self.work_queues = [mp.Queue() for _ in range(self.total_parallel_size)]

        for rank in range(self.total_parallel_size):
            p = mp.Process(
                target=_worker_fn,
                args=(
                    rank, self.total_parallel_size, master_port, init_barrier,
                    self.work_queues[rank], self.result_queue, self.shutdown_event,
                    self.pipeline_kwargs, self.gpu_ids,
                ),
            )
            p.start()
            self.workers.append(p)

        logging.info("Waiting for workers to initialize...")
        init_barrier.wait()
        logging.info("All workers initialized")

        self.pipeline_handler = None
        self.inference_handler = None

    def _process_samples(self):
        """Worker thread to process samples."""
        while True:
            samples = self.sample_queue.get()
            if samples is None:
                self.sample_queue.task_done()
                break
            self._generate_videos(samples)
            self.sample_queue.task_done()

    def _process_responses(self):
        """Response thread to report results to LoadGen."""
        while True:
            response = self.response_queue.get()
            if response is None:
                self.response_queue.task_done()
                break
            self.response_handler.process_response(response)
            # Update progress bar
            num_completed = len(response.sample_ids)
            with self.progress_lock:
                self.processed_count += num_completed
                self.progress_bar.update(num_completed)
            self.response_queue.task_done()

    def _generate_videos(self, samples):
        """Generate videos for the given samples."""
        sample_indices = [q.index for q in samples]
        sample_ids = [q.id for q in samples]

        logging.debug(f"Running inference on samples {sample_indices}")

        prompts = [self.dataset.get_prompt(idx) for idx in sample_indices]

        # Log which prompt and id is being processed (only in verbose mode)
        if self.verbose:
            for idx, (sample_id, sample_idx, prompt) in enumerate(zip(sample_ids, sample_indices, prompts)):
                logging.info(f"Processing sample {idx + 1}/{len(samples)}: id={sample_id}, index={sample_idx}, prompt=\"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"")

        negative_prompt = self.dataset.get_negative_prompt()
        fixed_latent = self.dataset.get_fixed_latents()

        if self.is_multiprocess:
            generated_videos = self._generate_multiprocess(prompts, negative_prompt, fixed_latent)
        else:
            generated_videos = self.inference_handler.generate_batch(
                prompts=prompts,
                negative_prompt=negative_prompt,
                latents=fixed_latent,
            )

        response = Wan22Response(sample_ids=sample_ids, sample_indices=sample_indices, generated_videos=generated_videos)
        self.response_queue.put(response)

    def _generate_multiprocess(self, prompts, negative_prompt, latents):
        """Generate videos using multiple worker processes."""
        generated_videos = []
        for idx, prompt in enumerate(prompts):
            logging.info(f"Generating video {idx + 1}/{len(prompts)} (multiprocess): \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"")
            work_item = (prompt, negative_prompt, latents)
            for wq in self.work_queues:
                wq.put(work_item)

            try:
                result = self.result_queue.get(timeout=600)
                if result is not None:
                    generated_videos.append(result)
                    logging.debug(f"Generated video with shape: {result.shape}")
                else:
                    generated_videos.append(
                        self.error_handler.handle_error(Exception("Worker returned None"), prompt[:50])
                    )
            except Exception as e:
                generated_videos.append(self.error_handler.handle_error(e, prompt[:50]))

        return generated_videos

    def warm_up(self, warm_up_iters: int = 2):
        """Perform warm-up inference."""
        test_prompt = "A cat walking in a garden."
        negative_prompt = self.dataset.get_negative_prompt()

        if self.is_multiprocess:
            logging.info(f"Running warm up with {warm_up_iters} iterations (multi-process)")
            for _ in range(warm_up_iters):
                for wq in self.work_queues:
                    wq.put((test_prompt, negative_prompt, None))
                try:
                    self.result_queue.get(timeout=300)
                except Exception:
                    pass
        else:
            self.inference_handler.warmup(test_prompt, negative_prompt, warm_up_iters)

    def issue_queries(self, query_samples):
        """Issue queries to the server."""
        num_samples = len(query_samples)
        logging.info(f"[Server] Received {num_samples} samples")
        self.sample_count += num_samples

        # Update progress bar total
        with self.progress_lock:
            self.progress_bar.total = self.sample_count
            self.progress_bar.refresh()

        for sample in query_samples:
            self.sample_queue.put([sample])

    def flush_queries(self):
        """Flush queries (no-op for WAN22)."""
        pass

    def finish_test(self):
        """Finish the test and cleanup."""
        logging.debug("SUT finished!")
        logging.info(f"[Server] Received {self.sample_count} total samples")

        self.sample_queue.put(None)
        self.sample_queue.join()

        self.response_queue.put(None)
        self.response_queue.join()

        # Close progress bar
        self.progress_bar.close()

        logging.info(f"Reported {self.response_handler.total_samples} samples")
        self.worker_thread.join()
        self.response_thread.join()

        if self.is_multiprocess:
            logging.info("Shutting down worker processes...")
            self.shutdown_event.set()
            for wq in self.work_queues:
                wq.put(None)
            for worker in self.workers:
                worker.join(timeout=10)
                if worker.is_alive():
                    worker.terminate()
            logging.info("All worker processes terminated")

    def save_accuracy_videos(self, log_dir: Path):
        """Save generated videos to disk for accuracy mode.

        Videos are saved to <log_dir>/video with name <prompt>-0.mp4

        Args:
            log_dir: The log directory path
        """
        if not self.accuracy_mode:
            logging.warning("save_accuracy_videos called but not in accuracy mode")
            return

        video_dir = log_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Saving {len(self.response_handler.accuracy_videos)} videos to {video_dir}")

        for sample_idx, video_data in self.response_handler.accuracy_videos.items():
            prompt = self.dataset.get_prompt(sample_idx)
            video_path = video_dir / f"{prompt}-0.mp4"

            try:
                # Convert numpy array to list of PIL images for export_to_video
                # video_data shape: (num_frames, height, width, 3)
                from PIL import Image

                # Convert to uint8 if needed (handle float data in 0-1 or 0-255 range)
                if video_data.dtype != np.uint8:
                    if video_data.max() <= 1.0:
                        # Data is in 0-1 range, scale to 0-255
                        video_data = (video_data * 255).clip(0, 255).astype(np.uint8)
                    else:
                        # Data is likely in 0-255 range but wrong dtype
                        video_data = video_data.clip(0, 255).astype(np.uint8)

                frames = [Image.fromarray(frame) for frame in video_data]
                export_to_video(frames, str(video_path), fps=16)
                logging.debug(f"Saved video for sample {sample_idx}: {video_path}")
            except Exception as e:
                logging.error(f"Failed to save video for sample {sample_idx}: {e}")

        logging.info(f"Saved {len(self.response_handler.accuracy_videos)} videos to {video_dir}")


# =============================================================================
# Harness Operation
# =============================================================================

@autoconfigure
class Wan22HarnessOp(PyHarnessOp):
    """WAN22 T2V harness operation for MLPerf LoadGen integration."""

    @classmethod
    def immediate_dependencies(cls):
        return {LoadgenConfFilesOp}

    @classmethod
    def output_keys(cls):
        return ["log_dir", "result_metadata"]

    def __init__(self, *args, **kwargs):
        super().__init__(Wan22Dataset, *args, **kwargs)
        self._server_inst = None

    def issue_queries(self, query_samples: List[lg.QuerySample]):
        """Issue queries to the WAN22 server."""
        self._server_inst.issue_queries(query_samples)

    def flush_queries(self):
        """Flush queries."""
        self._server_inst.flush_queries()

    @contextlib.contextmanager
    def wrap_lg_test(self, scratch_space, dependency_outputs):
        """Context manager for LoadGen test."""
        accuracy_mode = (self.test_mode == "AccuracyOnly")
        self._server_inst = Wan22Server(dataset=self._qsl_inst, accuracy_mode=accuracy_mode)
        logging.info("Starting Warm Up!")
        self._server_inst.warm_up()
        logging.info("Warm Up Done!")
        yield None
        self._server_inst.finish_test()

        # Save videos at teardown for AccuracyOnly mode
        if accuracy_mode:
            logging.info("AccuracyOnly mode: Saving generated videos...")
            self._server_inst.save_accuracy_videos(self.wl.log_dir)
