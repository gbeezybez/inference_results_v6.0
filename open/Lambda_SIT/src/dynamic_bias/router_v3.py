import ctypes
import os
from typing import Optional

import torch


def _default_lib_path() -> str:
    here = os.path.dirname(__file__)
    return os.path.join(here, "libdynamic_bias_router_v3.so")


class DynamicBiasRouterV3:
    """
    Python wrapper around the v3 CUDA kernel via ctypes.

    This is the production-path router for TRT-LLM plugin integration.
    For validation/prototyping without CUDA compilation, use triton_router.py.
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        topk: int,
        alpha: float,
        guardrail_threshold: float,
        ema_beta: float,
        alpha2: float = 0.0,
        lib_path: Optional[str] = None,
    ):
        lib_path = lib_path or os.environ.get("DYNAMIC_BIAS_LIB_PATH",
                                              _default_lib_path())
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"Dynamic bias CUDA library not found: {lib_path}")

        self._lib = ctypes.CDLL(lib_path)
        self._init_symbols()

        handle = self._lib.create_router_v3(
            ctypes.c_int(num_layers),
            ctypes.c_int(num_experts),
            ctypes.c_int(topk),
            ctypes.c_float(alpha),
            ctypes.c_float(guardrail_threshold),
            ctypes.c_float(ema_beta),
            ctypes.c_float(alpha2),
        )
        if not handle:
            raise RuntimeError("Failed to create DynamicBiasRouterV3 handle")
        self._handle = ctypes.c_void_p(handle)
        self._num_layers = num_layers
        self._num_experts = num_experts
        self._topk = topk

    def _init_symbols(self) -> None:
        # create / destroy
        self._lib.create_router_v3.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ]
        self._lib.create_router_v3.restype = ctypes.c_void_p

        self._lib.destroy_router_v3.argtypes = [ctypes.c_void_p]
        self._lib.destroy_router_v3.restype = None

        # forward (int32 indices)
        self._lib.router_v3_forward.argtypes = [
            ctypes.c_void_p,  # router
            ctypes.c_int,     # layer_idx
            ctypes.c_void_p,  # logits
            ctypes.c_void_p,  # bias_base
            ctypes.c_void_p,  # out_indices (int32)
            ctypes.c_void_p,  # out_weights
            ctypes.c_void_p,  # out_flags
            ctypes.c_int,     # num_tokens
            ctypes.c_void_p,  # stream
        ]
        self._lib.router_v3_forward.restype = None

        # step (with num_tokens for normalization + optional global_counts)
        self._lib.router_v3_step.argtypes = [
            ctypes.c_void_p,  # router
            ctypes.c_int,     # num_tokens_this_step
            ctypes.c_void_p,  # global_counts (nullable)
            ctypes.c_void_p,  # stream
        ]
        self._lib.router_v3_step.restype = None

        # reset
        self._lib.router_v3_reset.argtypes = [ctypes.c_void_p]
        self._lib.router_v3_reset.restype = None

        # param setters (CUDA Graph safe)
        self._lib.router_v3_set_alpha.argtypes = [
            ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p,
        ]
        self._lib.router_v3_set_alpha.restype = None

        self._lib.router_v3_set_guardrail.argtypes = [
            ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p,
        ]
        self._lib.router_v3_set_guardrail.restype = None

        self._lib.router_v3_set_alpha2.argtypes = [
            ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p,
        ]
        self._lib.router_v3_set_alpha2.restype = None

        self._lib.router_v3_set_params.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_void_p,
        ]
        self._lib.router_v3_set_params.restype = None

        # accuracy stats
        self._lib.router_v3_get_accuracy_stats.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
        ]
        self._lib.router_v3_get_accuracy_stats.restype = ctypes.POINTER(ctypes.c_int32)

        self._lib.router_v3_reset_accuracy_stats.argtypes = [ctypes.c_void_p]
        self._lib.router_v3_reset_accuracy_stats.restype = None

        self._lib.router_v3_set_accuracy_stats.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
        ]
        self._lib.router_v3_set_accuracy_stats.restype = None

        # monitoring
        self._lib.router_v3_get_load_ema.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
        ]
        self._lib.router_v3_get_load_ema.restype = ctypes.POINTER(ctypes.c_float)

        self._lib.router_v3_get_counts.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
        ]
        self._lib.router_v3_get_counts.restype = ctypes.POINTER(ctypes.c_int32)

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def topk(self) -> int:
        return self._topk

    def forward(
        self,
        layer_idx: int,
        logits: torch.Tensor,       # float32, CUDA, contiguous
        bias_base: torch.Tensor,     # float32, CUDA, contiguous
        out_indices: torch.Tensor,   # int32 (not int64!)
        out_weights: torch.Tensor,
        out_flags: torch.Tensor,
    ) -> None:
        if not logits.is_cuda or logits.dtype != torch.float32:
            raise ValueError("logits must be float32 CUDA tensor")
        if not logits.is_contiguous():
            raise ValueError("logits must be contiguous")
        if not bias_base.is_cuda or bias_base.dtype != torch.float32:
            raise ValueError("bias_base must be float32 CUDA tensor")
        if out_indices.dtype != torch.int32:
            raise ValueError("out_indices must be int32 (v3 change from v2)")

        num_tokens = logits.shape[0]
        stream = torch.cuda.current_stream(device=logits.device).cuda_stream
        self._lib.router_v3_forward(
            self._handle,
            ctypes.c_int(layer_idx),
            ctypes.c_void_p(logits.data_ptr()),
            ctypes.c_void_p(bias_base.data_ptr()),
            ctypes.c_void_p(out_indices.data_ptr()),
            ctypes.c_void_p(out_weights.data_ptr()),
            ctypes.c_void_p(out_flags.data_ptr()),
            ctypes.c_int(num_tokens),
            ctypes.c_void_p(stream),
        )

    def step(
        self,
        num_tokens: int = 0,
        global_counts: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        stream = torch.cuda.current_stream(
            device=device).cuda_stream if device else torch.cuda.current_stream().cuda_stream
        gc_ptr = ctypes.c_void_p(global_counts.data_ptr()) if global_counts is not None else None
        self._lib.router_v3_step(
            self._handle,
            ctypes.c_int(num_tokens),
            gc_ptr,
            ctypes.c_void_p(stream),
        )

    def reset(self) -> None:
        self._lib.router_v3_reset(self._handle)

    def set_alpha(self, alpha: float) -> None:
        stream = torch.cuda.current_stream().cuda_stream
        self._lib.router_v3_set_alpha(
            self._handle, ctypes.c_float(alpha), ctypes.c_void_p(stream))

    def set_guardrail(self, threshold: float) -> None:
        stream = torch.cuda.current_stream().cuda_stream
        self._lib.router_v3_set_guardrail(
            self._handle, ctypes.c_float(threshold), ctypes.c_void_p(stream))

    def set_alpha2(self, alpha2: float) -> None:
        stream = torch.cuda.current_stream().cuda_stream
        self._lib.router_v3_set_alpha2(
            self._handle, ctypes.c_float(alpha2), ctypes.c_void_p(stream))

    def set_params(self, alpha: float, threshold: float, alpha2: float) -> None:
        stream = torch.cuda.current_stream().cuda_stream
        self._lib.router_v3_set_params(
            self._handle,
            ctypes.c_float(alpha), ctypes.c_float(threshold), ctypes.c_float(alpha2),
            ctypes.c_void_p(stream),
        )

    def set_accuracy_stats(self, enabled: bool) -> None:
        self._lib.router_v3_set_accuracy_stats(
            self._handle, ctypes.c_int(1 if enabled else 0))

    def get_accuracy_stats(self, layer_idx: int) -> dict:
        """Returns {flip_cnt, lock_cnt, drop_cnt} for one layer."""
        ptr = self._lib.router_v3_get_accuracy_stats(
            self._handle, ctypes.c_int(layer_idx))
        if not ptr:
            return {"flip_cnt": 0, "lock_cnt": 0, "drop_cnt": 0}
        result = {
            "flip_cnt": ptr[0],
            "lock_cnt": ptr[1],
            "drop_cnt": ptr[2],
        }
        # Free the C-allocated buffer (platform-independent)
        import ctypes.util
        _libc_name = ctypes.util.find_library("c")
        if _libc_name:
            _libc = ctypes.CDLL(_libc_name)
            _libc.free.argtypes = [ctypes.c_void_p]
            _libc.free(ptr)
        return result

    def reset_accuracy_stats(self) -> None:
        self._lib.router_v3_reset_accuracy_stats(self._handle)

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle:
            try:
                self._lib.destroy_router_v3(handle)
            except (AttributeError, OSError):
                pass
            self._handle = None
