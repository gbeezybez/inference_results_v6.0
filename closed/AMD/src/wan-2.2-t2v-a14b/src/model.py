import array
import functools
import math
import os
import logging
import torch
import torch.distributed as dist
import mlperf_loadgen as lg
import torchao
from concurrent.futures import ThreadPoolExecutor
from torchao.quantization.quantize_.common import KernelPreference
from dataclasses import dataclass
from diffusers import WanPipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from pathlib import Path
from torch.profiler import profile, ProfilerActivity, record_function
from typing import Any, Generator, Optional, Union, Dict
from video_utils import export_to_video_bytes
from xfuser.model_executor.models.transformers.transformer_wan import xFuserWanAttnProcessor
from xfuser import xFuserArgs
from xfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
    initialize_runtime_state,
    is_dp_last_group,
)
from xfuser.core.distributed import (
    get_world_group,
    get_sp_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_runtime_state,
    shard_dit,
    shard_t5_encoder,
)

@dataclass(frozen=True)
class ModelConfig:
    negative_prompt: str
    guidance_scale: float
    guidance_scale_2: float
    num_frames: int
    num_inference_steps: int
    height: int
    width: int
    base_seed: int
    fps: int
    sage_fraction: float

def get_pipe_kwargs(prompts: list[str], config: ModelConfig, latents: Optional[torch.Tensor] = None) -> dict[str, Any]:
    batch_size = len(prompts)
    kwargs = {
        "prompt": prompts,
        "negative_prompt": [config.negative_prompt.strip()] * batch_size,
        "num_inference_steps": config.num_inference_steps,
        "num_frames": config.num_frames,
        "guidance_scale": config.guidance_scale,
        "guidance_scale_2": config.guidance_scale_2,
        "height": config.height,
        "width": config.width,
        "generator": torch.Generator(device="cuda").manual_seed(config.base_seed),
        "latents": latents if latents is None else latents.repeat(batch_size, 1, 1, 1, 1),
    }
    return kwargs

def maybe_transformer_2(transformer_2):
    if transformer_2 is not None:
        return functools.wraps(transformer_2.__class__.forward)
    else:
        return (lambda f:f)

def parallelize_transformer(pipe):
    transformer = pipe.transformer
    transformer_2 = pipe.transformer_2
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    @maybe_transformer_2(transformer_2)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        
        get_runtime_state().increment_step_counter()

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # 1. RoPE
        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            # We only reach this for Wan2.1, when doing cross attention with image embeddings
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # Part of sequence parallel: given the resolution, we may need to pad the sequence length to match this prior to chunking
        max_chunked_sequence_length = int(math.ceil(hidden_states.shape[1] / get_sequence_parallel_world_size())) * get_sequence_parallel_world_size()
        sequence_pad_amount = max_chunked_sequence_length - hidden_states.shape[1]
        hidden_states = torch.cat([
            hidden_states,
            torch.zeros(batch_size, sequence_pad_amount, hidden_states.shape[2], device=hidden_states.device, dtype=hidden_states.dtype)
        ], dim=1)
        hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]

        if ts_seq_len is not None: # (wan2.2 ti2v)
            temb = torch.cat([
                temb,
                torch.zeros(batch_size, sequence_pad_amount, temb.shape[2], device=temb.device, dtype=temb.dtype)
            ], dim=1)
            timestep_proj = torch.cat([
                timestep_proj,
                torch.zeros(batch_size, sequence_pad_amount, timestep_proj.shape[2], timestep_proj.shape[3], device=timestep_proj.device, dtype=timestep_proj.dtype)
            ], dim=1)
            temb = torch.chunk(temb, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
            timestep_proj = torch.chunk(timestep_proj, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

        freqs_cos, freqs_sin = rotary_emb

        def get_rotary_emb_chunk(freqs, sequence_pad_amount):
            freqs = torch.cat([
                freqs,
                torch.zeros(1, sequence_pad_amount, freqs.shape[2], freqs.shape[3], device=freqs.device, dtype=freqs.dtype)
            ], dim=1)
            freqs = torch.chunk(freqs, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
            return freqs

        freqs_cos = get_rotary_emb_chunk(freqs_cos, sequence_pad_amount)
        freqs_sin = get_rotary_emb_chunk(freqs_sin, sequence_pad_amount)
        rotary_emb = (freqs_cos, freqs_sin)


        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = get_sp_group().all_gather(hidden_states, dim=-2)

        # Removing excess padding to get back to original sequence length
        hidden_states = hidden_states[:, :math.prod([post_patch_num_frames, post_patch_height, post_patch_width]), :]

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    new_forward_1 = new_forward.__get__(transformer)
    transformer.forward = new_forward_1

    for block in transformer.blocks:
        block.attn1.processor = xFuserWanAttnProcessor()
        block.attn2.processor = xFuserWanAttnProcessor(use_parallel_attention=False)

    if transformer_2 is not None:
        new_forward_2 = new_forward.__get__(transformer_2)
        transformer_2.forward = new_forward_2

        for block in transformer_2.blocks:
            block.attn1.processor = xFuserWanAttnProcessor()
            block.attn2.processor = xFuserWanAttnProcessor(use_parallel_attention=False)

def quantize_linear_layers_to_fp8(modules_to_quantize: list[torch.nn.Module]):
    """ Quantize all linear layers in the given modules """
    for module in modules_to_quantize:
        _quantize_module_linear_layers_to_fp8(module)

def _quantize_module_linear_layers_to_fp8(module: torch.nn.Module|list[torch.nn.Module], parent=None, name=None, link=None):
    """ Quantize all linear layers in the given module to FP8  """
    for child_name, child in module.named_children():
        _quantize_module_linear_layers_to_fp8(child, module, child_name)

    if isinstance(module, torch.nn.Linear):
        setattr(parent, name, module.to(torch.bfloat16))
        torchao.quantization.quantize_(
              module,
              config=torchao.quantization.Float8DynamicActivationFloat8WeightConfig(
                  granularity=torchao.quantization.PerTensor(),
                  set_inductor_config=False,
                  kernel_preference=KernelPreference.AUTO
            )
        )

def set_timestep_embedding_dtype(pipe, dtype: torch.dtype):
    logging.info(f"Setting timestep embedding dtype to {dtype}")
    pipe.transformer.condition_embedder.time_embedder = pipe.transformer.condition_embedder.time_embedder.to(dtype)
    if pipe.transformer_2 is not None:
        pipe.transformer_2.condition_embedder.time_embedder = pipe.transformer_2.condition_embedder.time_embedder.to(dtype)


def setup_sage_attention():
    """
    monkey patch to use Triton implemented fav3 sage attn
    """
    import xfuser.model_executor.layers.usp as usp_mod
    from xfuser.model_executor.layers.usp import _get_attention_function, concat_joint_tensors_decorator
    from xfuser.core.distributed.attention_backend import AttentionBackendType, _aiter_attn_call
    from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func
    from functools import partial

    def fav3_sage_custom_attention(query, key, value, dropout_p, is_causal):
        return fav3_sage_wrapper_func(query, key, value, layout="bhsd", USE_MXFP4_SAGE=True), None

    def _custom_get_attention():
        attn_type = get_runtime_state().fp8_decision_vector[get_runtime_state().step_counter]
        if attn_type == AttentionBackendType.AITER_FP8:
            func = fav3_sage_custom_attention
        else:
            func = _aiter_attn_call
        return concat_joint_tensors_decorator(func)

    # Monkey-patch
    usp_mod._get_attention_function = _custom_get_attention


def chunk_list(lst: list[Any], chunk_size: int) -> Generator[list[Any], None, None]:
    """Split a list into chunks of specified size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]

class Model:
    def __init__(self, model_path: str, prompts: list[str], config: ModelConfig, fixed_latent: Optional[torch.Tensor] = None, batch_size: int = 1, video_output_path: str = None, performance_sample_count: int = 248):
        self.prompts = prompts
        self.config = config
        self.fixed_latent = fixed_latent
        self.batch_size = batch_size
        self.video_output_path = video_output_path
        self.performance_sample_count = performance_sample_count
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.pipe = WanPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
        )
        # Store videos for later saving to filesystem
        self.saved_videos = []  # List of tuples: (prompt_index, video_bytes)
        
        # Thread pool for parallel video encoding (only on rank 0)
        if self.rank == 0:
            # Use up to 32 workers - balance parallelism with CPU overhead
            # Video encoding is CPU-bound (ffmpeg), so more threads = better throughput
            num_workers = min(32, max(1, self.batch_size))
            self.video_encoder_pool = ThreadPoolExecutor(
                max_workers=num_workers,
                thread_name_prefix="video_encoder"
            )
            logging.info(f"Initialized video encoder thread pool with {num_workers} workers")
        else:
            self.video_encoder_pool = None
        
        self._prepare_model()

    def _prepare_model(self) -> None:
        device = torch.device(f"cuda:{self.local_rank}")

        engine_args = xFuserArgs(ulysses_degree=8)
        engine_config, _ = engine_args.create_config()

        initialize_runtime_state(self.pipe, engine_config)
        logging.info(f"Setup sage attention")
        setup_sage_attention()
        parallelize_transformer(self.pipe)
        self.pipe.to(device)
        logging.info(f"Model moved to {device=}")
        
        torch.set_float32_matmul_precision('high')

        set_timestep_embedding_dtype(self.pipe, torch.bfloat16)

        logging.info("Quantizing GEMMs to FP8")
        quantize_linear_layers_to_fp8(self.pipe.transformer.blocks)
        if self.pipe.transformer_2 is not None:
            quantize_linear_layers_to_fp8(self.pipe.transformer_2.blocks)

        guidance_scale = self.config.guidance_scale
        multiplier = 2 if guidance_scale > 1.0 else 1 # CFG is switched on in this case and double the transformers are called
        total_steps = self.config.num_inference_steps * multiplier # Total number of transformer calls during the denoising process
        sage_steps_threshold = round(total_steps * self.config.sage_fraction)
        sage_decision_vector = torch.tensor(
            [i < sage_steps_threshold for i in range(total_steps)], dtype=torch.bool)
        get_runtime_state().use_hybrid_fp8_attn = True
        get_runtime_state().set_hybrid_attn_parameters(sage_decision_vector)

        logging.info("Compiling model")
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="max-autotune-no-cudagraphs")
        if self.pipe.transformer_2 is not None:
            self.pipe.transformer_2 = torch.compile(self.pipe.transformer_2, mode="max-autotune-no-cudagraphs")

        # Warmup the torch compiler for all batch sizes that will be used
        # With torch.compile, each unique batch size triggers a separate compilation
        # We need to warmup both the full batch size and the remainder batch
        batch_sizes_to_warmup = self._get_batch_sizes_to_warmup()
        
        logging.info(f"Warming up model for batch sizes: {batch_sizes_to_warmup}")
        for warmup_bs in batch_sizes_to_warmup:
            warmup_pipe_kwargs = get_pipe_kwargs(
                prompts=["A cat playing with a ball"] * warmup_bs,
                config=self.config,
                latents=self.fixed_latent,
            )
            logging.info(f"Warming up model with batch_size={warmup_bs}...")
            self.pipe(**warmup_pipe_kwargs)
            logging.info(f"Warmup complete for batch_size={warmup_bs}")
        
        logging.info(f"All warmups complete")
    
    def _get_batch_sizes_to_warmup(self) -> list[int]:
        """
        Calculate which batch sizes will be used during inference.
        Returns list of unique batch sizes to warmup.
        """
        batch_sizes = set()
        
        # Full batch size
        batch_sizes.add(self.batch_size)
        
        # Remainder batch size (if performance_sample_count is not evenly divisible)
        remainder = self.performance_sample_count % self.batch_size
        if remainder > 0:
            batch_sizes.add(remainder)
        
        # Return sorted list for consistent warmup order
        return sorted(list(batch_sizes))

    def benchmark_model(self, batch_size: int):
        """Benchmark model, without loadgen instrumentation."""
        pipe_kwargs = get_pipe_kwargs(prompts=self.prompts[:batch_size],
                                        config=self.config,
                                        latents=self.fixed_latent,
                                    )
        events = {
            "start": torch.cuda.Event(enable_timing=True),
            "end": torch.cuda.Event(enable_timing=True),
        }
        for step_index in range(pipe_kwargs["num_inference_steps"]):
            events[f"inference_step_end_{step_index}"] = torch.cuda.Event(enable_timing=True)

        # Set up a callback function that records event timing after each inference/denoising step.
        # This may be used to infer VAE decode time as time between end of last inference step and end of pipe.
        def record_inference_step_end_time_cb(pipeline, step_index: int, timestep: int, callback_kwargs: dict):
            events[f"inference_step_end_{step_index}"].record()
            return callback_kwargs

        pipe_kwargs["callback_on_step_end"] = record_inference_step_end_time_cb

        events["start"].record()
        output = self.pipe(**pipe_kwargs)
        events["end"].record()
        torch.cuda.synchronize()

        return output, events

    def profile_model(self, batch_size: int):
        """Profile model, without loadgen instrumentation."""
        pipe_kwargs = get_pipe_kwargs(prompts=self.prompts[:batch_size],
                                                config=self.config,
                                                latents=self.fixed_latent,
                                            )
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        ) as prof:
            with record_function("model_inference"):
                output = self.pipe(**pipe_kwargs)
            torch.cuda.synchronize()
        return output, prof

    def _process_one_batch(self, i_chunk: list[int], q_chunk: list[int]) -> list[tuple]:
        """
        Process a single batch of queries. Called by both rank 0 (via issue_queries)
        and non-zero ranks (via worker_loop).
        
        Returns list of tuples (i, q, future) where future completes to video_bytes (only populated on rank 0).
        Video encoding happens in background threads - this method returns immediately after inference.
        """
        
        # All ranks prepare the same prompts and run inference
        pipe_kwargs = get_pipe_kwargs(prompts=[self.prompts[i] for i in i_chunk],
                                      config=self.config,
                                      latents=self.fixed_latent,
                                    )
        # All ranks participate in inference (required for xFuser parallelism)
        output = self.pipe(**pipe_kwargs)
        
        encoding_futures = []
        # Only rank 0 encodes videos and prepares responses
        if self.rank == 0:
            # Submit all video encoding tasks to thread pool - return immediately without waiting
            for i, q, video_array in zip(i_chunk, q_chunk, output.frames):
                future = self.video_encoder_pool.submit(
                    export_to_video_bytes,
                    video_array,
                    fps=self.config.fps
                )
                encoding_futures.append((i, q, future))
        
        return encoding_futures
    
    def issue_queries(self, query_samples: list[lg.QuerySample]) -> None:
        """
        Process queries from LoadGen. Only called on rank 0.
        Broadcasts work to all ranks for distributed inference.
        Video encoding happens in background - batches can overlap GPU inference with CPU encoding.
        """
        if self.rank != 0:
            logging.error("issue_queries should only be called on rank 0")
            return
        
        idx_chunks = [lst for lst in chunk_list([q.index for q in query_samples], chunk_size=self.batch_size)]
        query_ids_chunks = [lst for lst in chunk_list([q.id for q in query_samples], chunk_size=self.batch_size)]
        num_batches = len(idx_chunks)
        
        logging.info(f"Received query with {len(query_samples)} samples, splitting into {num_batches} batches")
        
        all_encoding_futures = []
        for ii, (i_chunk, q_chunk) in enumerate(zip(idx_chunks, query_ids_chunks)):
            logging.info(f"Broadcasting batch {ii+1}/{num_batches}")
            
            # Broadcast batch information to all ranks
            if dist.is_initialized():
                batch_size = len(i_chunk)
                max_batch_size = self.batch_size
                
                # Pad to fixed size for broadcasting
                i_chunk_padded = i_chunk + [0] * (max_batch_size - len(i_chunk))
                q_chunk_padded = q_chunk + [0] * (max_batch_size - len(q_chunk))
                
                # Combine all data into a single tensor for one broadcast operation
                # Layout: [batch_size, i_chunk[0..max_batch_size-1], q_chunk[0..max_batch_size-1]]
                combined_data = [batch_size] + i_chunk_padded + q_chunk_padded
                combined_tensor = torch.tensor(combined_data, dtype=torch.long, device=f"cuda:{self.local_rank}")
                
                # Single broadcast instead of three separate ones
                dist.broadcast(combined_tensor, src=0)
            
            # Process batch - encoding happens in background, we get futures immediately
            # This allows the next batch to start GPU inference while this batch encodes on CPU
            encoding_futures = self._process_one_batch(i_chunk, q_chunk)
            all_encoding_futures.extend(encoding_futures)
        
        # Now wait for all encodings to complete and build responses
        all_responses = []
        response_array_refs = []
        for i, q, future in all_encoding_futures:
            video_bytes = future.result()  # Wait for encoding to complete
            
            # Store video bytes and prompt index for later filesystem saving if needed
            if self.video_output_path is not None:
                self.saved_videos.append((i, video_bytes))
            
            # Create response array from encoded MP4 bytes
            response_array = array.array("B", video_bytes)
            response_array_refs.append(response_array)
            bi = response_array.buffer_info()
            all_responses.append(lg.QuerySampleResponse(q, bi[0], bi[1]))
        
        # Only rank 0 sends responses to LoadGen
        lg.QuerySamplesComplete(all_responses)


    def flush_queries(self):
        pass
    
    def worker_loop(self):
        """
        Worker loop for non-zero ranks. Continuously listens for batches
        broadcasted by rank 0 and participates in distributed inference.
        """
        if self.rank == 0:
            logging.warning("worker_loop() called on rank 0, which should be running LoadGen")
            return
        
        logging.info(f"Rank {self.rank} entering worker loop...")
        
        if not dist.is_initialized():
            logging.error("torch.distributed not initialized, cannot participate in inference")
            return
        
        # Keep listening for work until rank 0 signals completion
        while True:
            # Wait for batch information from rank 0
            max_batch_size = self.batch_size
            
            # Receive combined tensor: [batch_size, i_chunk[0..max_batch_size-1], q_chunk[0..max_batch_size-1]]
            combined_tensor = torch.zeros(1 + 2 * max_batch_size, dtype=torch.long, device=f"cuda:{self.local_rank}")
            dist.broadcast(combined_tensor, src=0)
            
            # Unpack the combined tensor
            batch_size = combined_tensor[0].item()
            i_chunk_tensor = combined_tensor[1:1 + max_batch_size]
            q_chunk_tensor = combined_tensor[1 + max_batch_size:]
            
            # Negative batch size signals end of work
            if batch_size < 0:
                logging.info(f"Rank {self.rank} received termination signal")
                break
            
            # If zero batch size, something is wrong
            if batch_size == 0:
                logging.warning(f"Rank {self.rank} received batch_size=0, skipping")
                continue
            
            # Extract the actual batch data (remove padding)
            i_chunk = i_chunk_tensor[:batch_size].cpu().tolist()
            q_chunk = q_chunk_tensor[:batch_size].cpu().tolist()
            
            # Process batch (participate in distributed inference)
            self._process_one_batch(i_chunk, q_chunk)
    
    def signal_workers_done(self):
        """
        Signal all worker ranks that processing is complete.
        Only rank 0 should call this.
        """
        if self.rank != 0:
            return
        
        if not dist.is_initialized():
            return
        
        logging.info("Signaling workers that processing is complete...")
        # Send combined tensor with negative batch size to signal completion
        # Layout: [batch_size=-1, padding...]
        max_batch_size = self.batch_size
        combined_data = [-1] + [0] * (2 * max_batch_size)
        combined_tensor = torch.tensor(combined_data, dtype=torch.long, device=f"cuda:{self.local_rank}")
        dist.broadcast(combined_tensor, src=0)
    
    def cleanup(self):
        """
        Cleanup resources. Should be called when done with the model.
        """
        if self.rank == 0 and self.video_encoder_pool is not None:
            logging.info("Shutting down video encoder thread pool...")
            self.video_encoder_pool.shutdown(wait=True)
            logging.info("Video encoder thread pool shut down successfully")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup
    
    def save_videos_to_filesystem(self):
        """
        Save all generated videos to filesystem.
        Only rank 0 should call this, and only if video_output_path was specified.
        """
        if self.rank != 0:
            logging.warning("save_videos_to_filesystem called on non-zero rank, skipping")
            return
        
        if self.video_output_path is None:
            logging.info("No video_output_path specified, skipping filesystem save")
            return
        
        if not self.saved_videos:
            logging.info("No videos to save")
            return
        
        logging.info(f"Saving {len(self.saved_videos)} videos to {self.video_output_path}")
        for prompt_index, video_bytes in self.saved_videos:
            output_path = Path(self.video_output_path, f"{self.prompts[prompt_index]}-0.mp4")
            logging.info(f"Writing video to {output_path}")
            with open(output_path, "wb") as f:
                f.write(video_bytes)
        
        logging.info(f"Successfully saved {len(self.saved_videos)} videos to filesystem")