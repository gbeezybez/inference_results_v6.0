import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val is not None else default


@dataclass(frozen=True)
class DynamicBiasParams:
    num_layers: int
    num_experts: int
    topk: int
    alpha: float
    alpha2: float
    guardrail_threshold: float
    ema_beta: float
    use_cuda: bool


class DynamicBiasRouterManager:
    """
    Wraps either the CUDA (router_v3.py) or Triton (triton_router.py) backend.

    Usage:
        manager = get_dynamic_bias_manager(device, num_layers=36, ...)
        indices, weights = manager.route(layer_idx, logits)
        ...after all 36 layers...
        manager.step(num_tokens)
    """

    def __init__(self, params: DynamicBiasParams, device: torch.device):
        self.params = params
        self.device = device
        self._last_num_tokens = 0

        if params.use_cuda:
            if params.num_experts != 128 or params.topk != 4:
                raise ValueError(
                    "DynamicBiasRouterV3 CUDA kernel is fixed to 128 experts and topk=4"
                )

            from tensorrt_llm.dynamic_bias import router_v3

            self._router = router_v3.DynamicBiasRouterV3(
                num_layers=params.num_layers,
                num_experts=params.num_experts,
                topk=params.topk,
                alpha=params.alpha,
                guardrail_threshold=params.guardrail_threshold,
                ema_beta=params.ema_beta,
                alpha2=params.alpha2,
            )
            self._bias_base = torch.zeros(params.num_experts,
                                          device=device,
                                          dtype=torch.float32)
        else:
            from tensorrt_llm.dynamic_bias import triton_router

            self._router = triton_router.DynamicBiasRouterV3(
                num_layers=params.num_layers,
                num_experts=params.num_experts,
                topk=params.topk,
                alpha=params.alpha,
                alpha2=params.alpha2,
                guardrail_threshold=params.guardrail_threshold,
                ema_beta=params.ema_beta,
                device=str(device),
            )

    def route(self, layer_idx: int,
              logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_tokens = logits.shape[0]
        self._last_num_tokens = num_tokens

        if self.params.use_cuda:
            logits_f32 = logits.float().contiguous()
            # v3: int32 indices
            indices = torch.empty((num_tokens, self.params.topk),
                                  device=logits_f32.device,
                                  dtype=torch.int32)
            weights = torch.empty((num_tokens, self.params.topk),
                                  device=logits_f32.device,
                                  dtype=torch.float32)
            flags = torch.empty((num_tokens,),
                                device=logits_f32.device,
                                dtype=torch.int32)
            self._router.forward(layer_idx, logits_f32, self._bias_base,
                                 indices, weights, flags)
            return indices, weights

        logits_f32 = logits.float().contiguous()
        indices, weights, _flags = self._router.route(layer_idx, logits_f32,
                                                      None)
        return indices, weights

    def step(self, num_tokens: int = 0) -> None:
        nt = num_tokens if num_tokens > 0 else self._last_num_tokens
        if self.params.use_cuda:
            self._router.step(num_tokens=nt, device=self.device)
        else:
            self._router.step(num_tokens=nt)


_MANAGERS: Dict[int, DynamicBiasRouterManager] = {}


def get_dynamic_bias_manager(
    device: torch.device,
    num_layers: int,
    num_experts: int,
    topk: int,
    alpha: Optional[float] = None,
    alpha2: Optional[float] = None,
    guardrail_threshold: Optional[float] = None,
    ema_beta: Optional[float] = None,
    use_cuda: Optional[bool] = None,
) -> DynamicBiasRouterManager:
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Dynamic bias router requires CUDA device")

    use_cuda = _env_flag("DYNAMIC_BIAS_USE_CUDA", "1") if use_cuda is None else use_cuda

    params = DynamicBiasParams(
        num_layers=num_layers,
        num_experts=num_experts,
        topk=topk,
        alpha=_env_float("DYNAMIC_BIAS_ALPHA", 0.15)
        if alpha is None else alpha,
        alpha2=_env_float("DYNAMIC_BIAS_ALPHA2", 0.0)
        if alpha2 is None else alpha2,
        guardrail_threshold=_env_float("DYNAMIC_BIAS_GUARDRAIL", 2.0)
        if guardrail_threshold is None else guardrail_threshold,
        ema_beta=_env_float("DYNAMIC_BIAS_EMA_BETA", 0.85)
        if ema_beta is None else ema_beta,
        use_cuda=use_cuda,
    )

    key = device.index if device.index is not None else torch.cuda.current_device()
    manager = _MANAGERS.get(key)
    if manager is None:
        manager = DynamicBiasRouterManager(params=params, device=device)
        _MANAGERS[key] = manager
        return manager

    if manager.params != params:
        raise RuntimeError(
            f"Dynamic bias router already initialized with different params on device {key}: "
            f"{manager.params} vs {params}")
    return manager


def dynamic_bias_step(device: torch.device, num_tokens: int = 0) -> None:
    device = torch.device(device)
    key = device.index if device.index is not None else torch.cuda.current_device()
    manager = _MANAGERS.get(key)
    if manager is not None:
        manager.step(num_tokens=num_tokens)
