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
import datetime
import os
import shutil
import tempfile

# Disable tqdm progress bars globally (for visual_gen pipeline's denoising loop)
# Must be set before tqdm is imported by any library
os.environ["TQDM_DISABLE"] = "1"

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image
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
from code.common.constants import AuditTest
from code.fields import general as general_fields
from code.fields import harness as harness_fields
from code.ops.harness import PyHarnessOp
from code.ops.loadgen import LoadgenConfFilesOp

from . import fields as wan22_fields
from .dataset import Wan22Dataset
from .utils import (
    video_to_numpy,
    video_to_loadgen_response,
    create_placeholder_video
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
        # Warmup
        warmup_iters: int = 1,
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
        # Warmup
        self.warmup_iters = warmup_iters
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
                     f"cp_size={self.cp_size}, ring_size={self.ring_size}, tp_size={self.tp_size}, dp_size={self.dp_size}")

        # Configure visual_gen
        vfly_configs = create_default_dit_configs()
        vfly_configs["pipeline"]["enable_torch_compile"] = True
        vfly_configs["pipeline"]["torch_compile_models"] = ["transformer", "transformer_2"]
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
        
        # Run warmup inference during initialization
        # This happens before any work queue processing, avoiding the queue leak issue
        if self.warmup_iters > 0:
            logging.info(f"[Rank {self.local_rank}] Running {self.warmup_iters} warmup iteration(s) during initialization...")
            warmup_batch_size = self.dp_size if self.dp_size > 1 else 1
            warmup_prompts = ["A fox walking in a garden"] * warmup_batch_size
            warmup_negative = "low quality, blurry"
            
            for i in range(self.warmup_iters):
                try:
                    _ = self.pipe(
                        prompt=warmup_prompts,
                        negative_prompt=[warmup_negative] * warmup_batch_size,
                        num_frames=self.num_frames,
                        height=self.height,
                        width=self.width,
                        guidance_scale=self.guidance_scale,
                        guidance_scale_2=self.guidance_scale_2,
                        num_inference_steps=self.num_inference_steps,
                    )
                    logging.info(f"[Rank {self.local_rank}] Warmup iteration {i+1}/{self.warmup_iters} completed")
                except Exception as e:
                    logging.warning(f"[Rank {self.local_rank}] Warmup iteration {i+1} failed: {e}")
            
            # Clear CUDA cache after warmup
            torch.cuda.empty_cache()
            logging.info(f"[Rank {self.local_rank}] Warmup completed successfully")
        else:
            logging.info(f"[Rank {self.local_rank}] Skipping warmup (warmup_iters=0)")
        
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
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        latents: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
    ):
        """Generate video from prompt(s).

        Args:
            prompt: Single prompt string or list of prompts for batched generation.
            negative_prompt: Single negative prompt or list matching prompts length.
            latents: Optional latent tensor.
            num_inference_steps: Override default inference steps.

        Returns:
            Single video frames if prompt is a string, or list of video frames if prompt is a list.
        """
        is_batched = isinstance(prompt, list)

        # Handle negative prompt for batched input
        if is_batched and negative_prompt is not None and isinstance(negative_prompt, str):
            # Replicate negative prompt for each prompt in batch
            negative_prompt = [negative_prompt] * len(prompt)

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

        if is_batched:
            # Return list of video frames for batched input
            return output.frames
        else:
            # Return single video frames for single prompt
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
            self.pipeline.generate(prompt, negative_prompt, num_inference_steps=20)


class LoadGenResponseHandler:
    """Handles LoadGen response processing."""

    def __init__(self, accuracy_mode: bool = False):
        self.total_samples = 0
        self.accuracy_mode = accuracy_mode
        # Store videos by sample_idx for accuracy mode
        self.accuracy_videos: Dict[int, np.ndarray] = {}

    def process_response(self, response: Wan22Response):
        """Process response and report to LoadGen."""
        qsr = []
        for idx, sample_id in enumerate(response.sample_ids):
            video_data = response.generated_videos[idx]
            if isinstance(video_data, torch.Tensor):
                video_data = video_data.cpu().numpy()
            qsr.append(video_to_loadgen_response(sample_id, video_data))

            # Store video for accuracy mode using sample_idx
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
    init_file: str,
    init_barrier: mp.Barrier,
    shutdown_barrier: mp.Barrier,
    work_queue: mp.Queue,
    result_queue: mp.Queue,
    shutdown_event: mp.Event,
    pipeline_kwargs: dict,
    gpu_ids: Optional[str] = None,
):
    """Worker function for multi-GPU inference processes.

    Always expects prompts as a list (even for batch_size=1) and returns a list of videos.
    This unified approach treats single prompts as batch_size=1 case.
    """
    # Set CUDA_VISIBLE_DEVICES first if specified
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    # Set up distributed environment variables
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Use FileStore instead of TCPStore to avoid "Broken pipe" errors during shutdown
    # FileStore doesn't have the heartbeat monitor issues that TCPStore has
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=300)
    )
    torch.cuda.set_device(rank)
    torch.autograd.set_grad_enabled(False)

    handler = VflyPipelineHandler(**pipeline_kwargs)
    handler.local_rank = rank
    handler._initialize_pipeline()

    logging.info(f"[Worker {rank}] Initialized and ready")
    init_barrier.wait()

    # Main work loop - only exit via None sentinel to ensure all workers exit together
    while True:
        try:
            work_item = work_queue.get(timeout=1.0)
            if work_item is None:
                logging.info(f"[Worker {rank}] Received shutdown signal")
                break

            prompts, negative_prompt, latents = work_item

            # Debug: Log the prompts being processed by this worker
            if rank == 0:
                logging.info(f"[Worker {rank}] Processing {len(prompts)} prompts:")
                for i, p in enumerate(prompts):
                    logging.info(f"[Worker {rank}]   prompt[{i}]: \"{p[:100]}{'...' if len(p) > 100 else ''}\"")

            # Always expect prompts as a list (unified batched approach)
            frames = handler.generate(prompts, negative_prompt, latents)

            if rank == 0:
                # Always return as a list (frames is already a list from batched generate)
                logging.info(f"[Worker {rank}] Got {len(frames)} frames for {len(prompts)} prompts")
                results = [video_to_numpy(f) for f in frames]
                result_queue.put(results)

        except queue.Empty:
            # Check shutdown event during idle periods
            if shutdown_event.is_set():
                logging.info(f"[Worker {rank}] Shutdown event detected")
                break
            continue
        except Exception as e:
            logging.error(f"[Worker {rank}] Error: {e}")
            if rank == 0:
                result_queue.put(None)

    # Synchronize all workers at shutdown barrier BEFORE any cleanup
    # This ensures all workers exit the work loop together before any process group operations
    # Use a short timeout - if barrier fails, proceed with cleanup anyway
    logging.info(f"[Worker {rank}] Waiting at shutdown barrier...")
    try:
        shutdown_barrier.wait(timeout=30)
        logging.info(f"[Worker {rank}] Shutdown barrier passed")
    except Exception as e:
        logging.warning(f"[Worker {rank}] Shutdown barrier timeout/error (proceeding with cleanup): {e}")

    # Cleanup: Destroy process group
    logging.info(f"[Worker {rank}] Starting cleanup...")
    if dist.is_initialized():
        try:
            # Use NCCL barrier with short timeout to synchronize before destruction
            # If barrier fails, proceed with cleanup anyway
            dist.barrier()
            logging.info(f"[Worker {rank}] NCCL barrier passed")
        except Exception as e:
            logging.warning(f"[Worker {rank}] NCCL barrier failed (proceeding with cleanup): {e}")

        try:
            dist.destroy_process_group()
            logging.info(f"[Worker {rank}] Process group destroyed")
        except Exception as e:
            logging.warning(f"[Worker {rank}] destroy_process_group failed: {e}")

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
# Warmup fields
@bind(wan22_fields.warmup_iters, "warmup_iters")
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
        # Warmup
        warmup_iters: int = 4,
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

        # Store dp_size for batching - when dp_size > 1, we need to batch prompts
        self.dp_size = dp_size

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
            # Warmup
            "warmup_iters": warmup_iters,
        }

        # Handlers
        self.error_handler = GenerationErrorHandler(num_frames, height, width)
        self.response_handler = LoadGenResponseHandler(accuracy_mode=accuracy_mode)

        # Multi-process state
        self.workers = []
        self.work_queues = []
        self.result_queue = None
        self.shutdown_event = None
        self.shutdown_barrier = None
        self.init_file = None  # FileStore init file path

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

        # Use FileStore instead of TCPStore to avoid "Broken pipe" errors during shutdown
        # Create a temporary file for the FileStore - must be deleted before workers start
        self.init_file = tempfile.NamedTemporaryFile(delete=False, prefix="wan22_init_").name
        # Delete the file so FileStore can create it fresh (required by FileStore)
        os.unlink(self.init_file)
        logging.info(f"Using FileStore init file: {self.init_file}")

        self.shutdown_event = mp.Event()
        self.result_queue = mp.Queue()
        init_barrier = mp.Barrier(self.total_parallel_size + 1)
        # Shutdown barrier ensures all workers synchronize before cleanup
        # Only workers participate (no +1), main process sends signals and waits
        self.shutdown_barrier = mp.Barrier(self.total_parallel_size)
        self.work_queues = [mp.Queue() for _ in range(self.total_parallel_size)]

        for rank in range(self.total_parallel_size):
            p = mp.Process(
                target=_worker_fn,
                args=(
                    rank, self.total_parallel_size, self.init_file, init_barrier,
                    self.shutdown_barrier, self.work_queues[rank], self.result_queue,
                    self.shutdown_event, self.pipeline_kwargs, self.gpu_ids,
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

        logging.info(f"Running inference on {len(samples)} samples: indices={sample_indices}")

        # Debug: Log dataset info
        logging.info(f"Dataset has {len(self.dataset.prompts)} prompts, first prompt: \"{self.dataset.prompts[0][:60]}...\"")

        prompts = [self.dataset.get_prompt(idx) for idx in sample_indices]

        # Always log sample_idx to prompt mapping in accuracy mode for verification
        if self.accuracy_mode:
            logging.info(f"Prompt mapping for batch (sample_idx -> prompt):")
            for sample_idx, prompt in zip(sample_indices, prompts):
                prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
                logging.info(f"  sample_idx={sample_idx:3d} -> \"{prompt_preview}\"")

        # Log which prompt and id is being processed (only in verbose mode)
        if self.verbose:
            for idx, (sample_id, sample_idx, prompt) in enumerate(zip(sample_ids, sample_indices, prompts)):
                logging.info(f"Processing sample {idx + 1}/{len(samples)}: id={sample_id}, sample_idx={sample_idx}, prompt=\"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"")

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
        """Generate videos using multiple worker processes.

        Always processes prompts in batches. When dp_size > 1, batches are sized
        to meet the minimum batch size requirement for data parallelism.
        When dp_size = 1, processes one prompt at a time (batch_size=1).
        """
        generated_videos = []
        batch_size = 1
        if self.dp_size > 1 and self.dp_size > batch_size:
            batch_size = self.dp_size
            logging.info(f"Set batch size = {self.dp_size} for self.dp_size > 1")
        else:
            batch_size = 1

        # Process prompts in batches
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            actual_batch_size = len(batch_prompts)

            # Pad batch to required size if needed (for dp_size > 1 with incomplete last batch)
            if actual_batch_size < batch_size:
                padding_count = batch_size - actual_batch_size
                batch_prompts = batch_prompts + [batch_prompts[-1]] * padding_count
                logging.debug(f"Padded batch from {actual_batch_size} to {batch_size} prompts")

            logging.info(f"Generating videos {batch_end}/{len(prompts)} (multiprocess, batch_size={len(batch_prompts)}, actual={actual_batch_size})")
            logging.info(f"Prompts being processed (actual={actual_batch_size}, padded to {len(batch_prompts)}):")
            for i, p in enumerate(batch_prompts):
                is_padding = i >= actual_batch_size
                padding_marker = " [PADDING]" if is_padding else ""
                logging.info(f"  [{i}]{padding_marker} \"{p[:100]}{'...' if len(p) > 100 else ''}\"")

            # Always send as a list (even for batch_size=1)
            work_item = (batch_prompts, negative_prompt, latents)
            for wq in self.work_queues:
                wq.put(work_item)

            try:
                results = self.result_queue.get(timeout=600)
                if results is not None:
                    # Verify we got the expected number of results
                    logging.info(f"Received {len(results)} results for {len(batch_prompts)} prompts (actual={actual_batch_size})")
                    if len(results) != len(batch_prompts):
                        logging.error(f"MISMATCH: Expected {len(batch_prompts)} results but got {len(results)}!")
                        logging.error("This suggests visual_gen pipeline is not gathering all dp_size results")
                    
                    # Only take the actual results (exclude padding)
                    for i in range(actual_batch_size):
                        if i < len(results):
                            generated_videos.append(results[i])
                            logging.info(f"Video {batch_start + i + 1}: shape={results[i].shape}, prompt=\"{batch_prompts[i][:60]}...\"")
                        else:
                            logging.error(f"Missing video for index {i}, prompt=\"{batch_prompts[i][:60]}...\"")
                            generated_videos.append(
                                self.error_handler.handle_error(
                                    Exception(f"Missing result at index {i}"),
                                    batch_prompts[i][:50]
                                )
                            )
                else:
                    # Handle error for all prompts in batch
                    for i in range(actual_batch_size):
                        generated_videos.append(
                            self.error_handler.handle_error(
                                Exception("Worker returned None"),
                                batch_prompts[i][:50]
                            )
                        )
            except Exception as e:
                for i in range(actual_batch_size):
                    generated_videos.append(
                        self.error_handler.handle_error(e, batch_prompts[i][:50])
                    )

        return generated_videos


    def issue_queries(self, query_samples):
        """Issue queries to the server."""
        num_samples = len(query_samples)
        logging.info(f"[Server] Received {num_samples} samples")
        self.sample_count += num_samples

        # Update progress bar total
        with self.progress_lock:
            self.progress_bar.total = self.sample_count
            self.progress_bar.refresh()

        # Batch samples based on dp_size for efficient data parallelism
        # When dp_size > 1, group samples to utilize all data parallel ranks
        batch_size = self.dp_size if self.dp_size > 1 else 1

        if batch_size > 1:
            logging.info(f"[Server] Batching samples with batch_size={batch_size} for dp_size={self.dp_size}")
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                batch = list(query_samples[batch_start:batch_end])
                self.sample_queue.put(batch)
            logging.info(f"[Server] Created {(num_samples + batch_size - 1) // batch_size} batches")
        else:
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
            # Send None to all work queues FIRST to signal workers to exit
            # This ensures all workers receive the shutdown signal before any exit
            for wq in self.work_queues:
                wq.put(None)

            # Set shutdown event as backup (workers check this during idle)
            self.shutdown_event.set()

            # Wait for workers to finish - they need time to:
            # 1. Receive None and break from loop
            # 2. Pass shutdown barrier
            # 3. Pass NCCL barrier
            # 4. Destroy process group
            for i, worker in enumerate(self.workers):
                worker.join(timeout=120)
                if worker.is_alive():
                    logging.warning(f"Worker {i} (pid={worker.pid}) did not shutdown gracefully, terminating...")
                    worker.terminate()
                    worker.join(timeout=5)
            logging.info("All worker processes terminated")

            # Clean up FileStore init file
            if self.init_file and os.path.exists(self.init_file):
                try:
                    os.unlink(self.init_file)
                    logging.info(f"Cleaned up init file: {self.init_file}")
                except Exception as e:
                    logging.warning(f"Failed to clean up init file {self.init_file}: {e}")

    def save_accuracy_videos(self, log_dir: Path):
        """Save generated videos to disk for accuracy mode.

        Videos are saved to <log_dir>/video with name <prompt>-0.mp4
        Uses sample_idx to get the correct prompt via dataset.get_prompt(sample_idx).

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

            # Log verification: sample_idx -> prompt -> video file
            prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
            logging.info(f"  sample_idx={sample_idx:3d} -> \"{prompt_preview}\" -> {video_path.name}")

            try:
                # Convert numpy array to list of PIL images for export_to_video
                # video_data shape: (num_frames, height, width, 3)

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
                logging.debug(f"Saved video for sample_idx {sample_idx}: {video_path}")
            except Exception as e:
                logging.error(f"Failed to save video for sample_idx {sample_idx}: {e}")

        logging.info("=" * 60)
        logging.info(f"Saved {len(self.response_handler.accuracy_videos)} videos to {video_dir}")


def copy_accuracy_sample_videos_to_videos_folder(log_dir: Path) -> None:
    """Create a 'videos' folder in log_dir and copy videos listed in accuracy/samples.txt into it.

    Reads accuracy/samples.txt (or accuracy/sample.txt) where each line is 'sample_id, prompt'.
    For each line, copies log_dir/video/{prompt}-0.mp4 to log_dir/videos/ if the source exists.
    Only runs when in AccuracyOnly mode; call after save_accuracy_videos().

    Args:
        log_dir: The log directory path (videos are read from log_dir/video).
    """
    accuracy_dir = Path(__file__).resolve().parent / "accuracy"
    sample_list_path = accuracy_dir / "samples.txt"
    if not sample_list_path.exists():
        sample_list_path = accuracy_dir / "sample.txt"
    if not sample_list_path.exists():
        logging.warning(
            f"Accuracy sample list not found at {sample_list_path} or {accuracy_dir / 'sample.txt'}; "
            "skipping copy_accuracy_sample_videos_to_videos_folder"
        )
        return

    video_src_dir = log_dir / "video"
    videos_dst_dir = log_dir / "videos"
    videos_dst_dir.mkdir(parents=True, exist_ok=True)

    if not video_src_dir.exists():
        logging.warning(f"Source video directory {video_src_dir} does not exist; skipping copy")
        return

    copied = 0
    with open(sample_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: "sample_id, prompt" (prompt may contain commas)
            parts = line.split(",", 1)
            if len(parts) < 2:
                continue
            prompt = parts[1].strip()
            src_path = video_src_dir / f"{prompt}-0.mp4"
            if src_path.exists():
                dst_path = videos_dst_dir / src_path.name
                try:
                    shutil.copy2(src_path, dst_path)
                    copied += 1
                    logging.debug(f"Copied {src_path.name} -> {videos_dst_dir}")
                except Exception as e:
                    logging.error(f"Failed to copy {src_path} to {dst_path}: {e}")
            else:
                logging.debug(f"Source video not found (skipping): {src_path}")

    logging.info(f"Copied {copied} videos from accuracy sample list to {videos_dst_dir}")


# =============================================================================
# Harness Operation
# =============================================================================

@autoconfigure
@bind(harness_fields.audit_test)
@bind(wan22_fields.total_sample_count, "total_sample_count")
class Wan22HarnessOp(PyHarnessOp):
    """WAN22 T2V harness operation for MLPerf LoadGen integration."""

    @classmethod
    def immediate_dependencies(cls):
        return {LoadgenConfFilesOp}

    @classmethod
    def output_keys(cls):
        return ["log_dir", "result_metadata"]

    def __init__(self, *args, audit_test: Optional[AuditTest] = None, total_sample_count: Optional[int] = None, **kwargs):
        super().__init__(Wan22Dataset, *args, total_sample_count=total_sample_count, **kwargs)
        self._server_inst = None
        self.audit_test = audit_test

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
        # Also enable video dumping for audit TEST01 (needs accuracy log for verification)
        dump_videos = accuracy_mode or (self.audit_test == AuditTest.TEST01)
        self._server_inst = Wan22Server(dataset=self._qsl_inst, accuracy_mode=dump_videos)

        # Reset progress bar timer after warmup to exclude initialization and warmup time
        # from the elapsed time calculation for accurate performance metrics
        if hasattr(self._server_inst, 'progress_bar') and self._server_inst.progress_bar is not None:
            import time
            self._server_inst.progress_bar.start_t = time.time()
            self._server_inst.progress_bar.last_print_t = time.time()
            logging.info("Progress bar timer reset after warmup")

        yield None
        self._server_inst.finish_test()

        # Save videos at teardown for AccuracyOnly mode or Audit TEST01
        if dump_videos:
            mode_str = "AccuracyOnly" if accuracy_mode else f"Audit {self.audit_test.valstr}"
            logging.info(f"{mode_str} mode: Saving generated videos...")
            self._server_inst.save_accuracy_videos(self.wl.log_dir)
            if accuracy_mode:
                copy_accuracy_sample_videos_to_videos_folder(self.wl.log_dir)
