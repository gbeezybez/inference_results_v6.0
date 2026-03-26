import importlib.util
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    import tensorrt_llm
    HAS_TRTLLM = True
except ImportError:
    HAS_TRTLLM = False

try:
    import torch.distributed as dist
    HAS_DIST = True
except ImportError:
    HAS_DIST = False

try:
    from tensorrt_llm.dynamic_bias import triton_router
except Exception:
    _module_path = Path(__file__).resolve().parent / "triton_router.py"
    _spec = importlib.util.spec_from_file_location("triton_router",
                                                   _module_path)
    if _spec is None or _spec.loader is None:
        raise
    triton_router = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(triton_router)

# ---------------------------------------------------------------------------
# gpt-oss-120b constants
# ---------------------------------------------------------------------------
GPT_OSS_NUM_EXPERTS = 128
GPT_OSS_TOPK = 4
GPT_OSS_NUM_LAYERS = 36
GPT_OSS_HIDDEN_SIZE = 2880


@dataclass
class DynamicBiasConfig:
    """Configuration for gpt-oss-120b dynamic bias routing."""

    # Architecture
    num_experts: int = GPT_OSS_NUM_EXPERTS
    topk: int = GPT_OSS_TOPK
    num_layers: int = GPT_OSS_NUM_LAYERS
    hidden_size: int = GPT_OSS_HIDDEN_SIZE

    # Dynamic bias
    alpha: float = 0.15
    alpha2: float = 0.0          # intra-step feedback (0=off, 0.01-0.05 typical)
    guardrail_threshold: float = 1.5
    ema_beta: float = 0.85

    # EP global aggregation
    ep_global_sync: bool = False  # True = AllReduce counts across EP ranks
    ep_group: Optional[object] = field(default=None, repr=False)

    # Alpha auto-tuning
    alpha_auto_tune: bool = True
    auto_tune_interval: int = 50
    target_cv: float = 0.35
    alpha_lr: float = 0.01
    alpha_min: float = 0.01
    alpha_max: float = 0.5

    # Layer 3: α safety upper bound (gap-based clamp)
    alpha_safety_enabled: bool = True
    alpha_safety_margin: float = 0.8
    gap_sample_interval: int = 50
    gap_sample_size: int = 256
    gap_sample_layers: Tuple = (0, 11, 23, 35)

    # Layer 4: flip/drop rate monitoring + auto-reduce
    flip_monitor_enabled: bool = True
    max_flip_rate: float = 0.05
    max_drop_rate: float = 0.001
    flip_reduce_factor: float = 0.7
    drop_reduce_factor: float = 0.5
    max_lock_rate: float = 0.90
    min_lock_rate: float = 0.50
    threshold_adjust_step: float = 0.1

    # Sampling window
    sample_on_steps: int = 2
    sample_off_steps: int = 48

    # Monitoring
    log_interval: int = 100


class DynamicBiasRouterManager:
    """
    Manages v3 dynamic bias routing across all 36 MoE layers.

    6-layer accuracy protection:
      Layer 1: Guardrail        → sem gap > threshold locks top-1
      Layer 2: Weights = sem    → mixture weights from semantic baseline
      Layer 3: α safety clamp   → α * max(load) + α2 * max(count) < p10(gap)
      Layer 4: Drop/Flip monitor→ drop>0.1% → α*0.5; flip>5% → α*0.7
      Layer 5: FP4 quantization → quantization noise >> bias perturbation
      Layer 6: MLPerf accuracy  → final validation via AccuracyTarget(0.99)

    Usage:
        manager = DynamicBiasRouterManager(config)
        for layer_idx in range(36):
            indices, weights = manager.route(layer_idx, logits, bias)
        manager.step(num_tokens)  # ONCE per step
    """

    def __init__(self, config: DynamicBiasConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.step_count = 0

        self.router = triton_router.DynamicBiasRouterV3(
            num_layers=config.num_layers,
            num_experts=config.num_experts,
            topk=config.topk,
            alpha=config.alpha,
            alpha2=config.alpha2,
            guardrail_threshold=config.guardrail_threshold,
            ema_beta=config.ema_beta,
            device=device,
        )

        self._layer_locked_sum = torch.zeros(config.num_layers, device=device)
        self._layer_token_sum = torch.zeros(config.num_layers, device=device)

        # Pre-allocate global counts buffer for EP sync
        if config.ep_global_sync:
            total = config.num_layers * config.num_experts
            self._global_counts = torch.zeros(total, dtype=torch.int32, device=device)
        else:
            self._global_counts = None

        # Cache last logits/bias for gap estimation
        self._last_logits = {}
        self._last_bias = {}

    def route(
        self,
        layer_idx: int,
        logits: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route tokens for one layer. Returns (indices_i32, weights)."""
        indices, weights, flags = self.router.route(layer_idx, logits, bias)

        self._layer_locked_sum[layer_idx] += flags.sum().float()
        self._layer_token_sum[layer_idx] += logits.shape[0]

        # Cache for gap sampling
        if layer_idx in self.config.gap_sample_layers:
            self._last_logits[layer_idx] = logits.detach()
            self._last_bias[layer_idx] = bias.detach() if bias is not None else None

        return indices, weights

    def step(self, num_tokens: int = 0):
        """
        Call ONCE per decode step after all 36 layers.

        If ep_global_sync=True, does AllReduce(SUM) on counts across EP ranks
        before EMA update. Cost: 18KB AllReduce per step.
        """
        global_counts = None

        if self.config.ep_global_sync and HAS_DIST and dist.is_initialized():
            self._global_counts.copy_(self.router.counts)
            group = self.config.ep_group
            dist.all_reduce(self._global_counts, op=dist.ReduceOp.SUM, group=group)
            global_counts = self._global_counts

        self.router.step(num_tokens=num_tokens, global_counts=global_counts)
        self.step_count += 1

        # ---- Sampling window: toggle accuracy stats collection ----
        cfg = self.config
        cycle_len = cfg.sample_on_steps + cfg.sample_off_steps
        cycle_pos = self.step_count % cycle_len
        should_collect = cycle_pos < cfg.sample_on_steps
        self.router.accuracy_stats_enabled = should_collect

        # ---- At end of sampling window: run Layer 4 checks ----
        at_sample_boundary = (cycle_pos == cfg.sample_on_steps)

        if at_sample_boundary and cfg.flip_monitor_enabled:
            self._check_drop_rate()    # hard safety first
            self._check_flip_rate()    # then soft metric
            self._adjust_threshold_by_lock_rate()
            self.router.reset_accuracy_stats()

        # ---- Periodic α tuning + Layer 3 clamp ----
        interval_hit = (self.step_count % cfg.auto_tune_interval == 0)
        if interval_hit:
            if cfg.alpha_safety_enabled:
                self._sample_gap_stats()
            if cfg.alpha_auto_tune:
                self._auto_tune_alpha()
            if cfg.alpha_safety_enabled:
                self._clamp_alpha_to_safe_bound()

        if (cfg.log_interval > 0 and
                self.step_count % cfg.log_interval == 0):
            self._log_stats()

    def _auto_tune_alpha(self):
        """CV-based α tuning."""
        cvs = []
        for layer_idx in self.config.gap_sample_layers:
            stats = self.router.get_layer_stats(layer_idx)
            cvs.append(stats["coeff_var"])
        avg_cv = sum(cvs) / len(cvs)

        cfg = self.config
        if avg_cv > cfg.target_cv:
            new_alpha = min(self.router.alpha + cfg.alpha_lr, cfg.alpha_max)
        else:
            new_alpha = max(self.router.alpha - cfg.alpha_lr * 0.5, cfg.alpha_min)

        self.router.alpha = new_alpha

    # ------------------------------------------------------------------
    # Layer 3: α safety upper bound
    # ------------------------------------------------------------------

    def _sample_gap_stats(self):
        for layer_idx in self.config.gap_sample_layers:
            logits = self._last_logits.get(layer_idx)
            if logits is None:
                continue
            bias = self._last_bias.get(layer_idx)
            self.router.sample_gap_stats(
                layer_idx, logits, bias,
                max_samples=self.config.gap_sample_size,
            )

    def _clamp_alpha_to_safe_bound(self):
        """Layer 3: clamp α to gap-derived safe upper bound."""
        safe_alpha = self.router.compute_safe_alpha(
            safety_margin=self.config.alpha_safety_margin,
        )

        if self.router.alpha > safe_alpha:
            self.router.alpha = max(safe_alpha, self.config.alpha_min)

    # ------------------------------------------------------------------
    # Layer 4: flip/drop rate monitoring + auto-reduce
    # ------------------------------------------------------------------

    def _check_drop_rate(self):
        """Layer 4 HARD SAFETY: top-1 out of top-4 → α *= 0.5."""
        cfg = self.config
        drop_rates = []
        for layer_idx in cfg.gap_sample_layers:
            stats = self.router.get_accuracy_stats(layer_idx)
            drop_rates.append(stats["drop_rate"])

        if not drop_rates:
            return

        max_drop = max(drop_rates)
        if max_drop > cfg.max_drop_rate:
            self.router.alpha = max(
                self.router.alpha * cfg.drop_reduce_factor,
                cfg.alpha_min,
            )

    def _check_flip_rate(self):
        """Layer 4: top-1 flipped → α *= 0.7."""
        cfg = self.config
        flip_rates = []
        for layer_idx in cfg.gap_sample_layers:
            stats = self.router.get_accuracy_stats(layer_idx)
            flip_rates.append(stats["flip_rate"])

        if not flip_rates:
            return

        max_flip = max(flip_rates)
        if max_flip > cfg.max_flip_rate:
            self.router.alpha = max(
                self.router.alpha * cfg.flip_reduce_factor,
                cfg.alpha_min,
            )

    def _adjust_threshold_by_lock_rate(self):
        """Layer 4: keep lock_rate in 50-90% band."""
        cfg = self.config
        lock_rates = []
        for layer_idx in cfg.gap_sample_layers:
            stats = self.router.get_accuracy_stats(layer_idx)
            lock_rates.append(stats["lock_rate"])

        if not lock_rates:
            return

        avg_lock = sum(lock_rates) / len(lock_rates)

        # Lock condition: gap > threshold
        # Too many locks  → raise threshold (harder to lock) → fewer locks
        # Too few locks   → lower threshold (easier to lock) → more locks
        if avg_lock > cfg.max_lock_rate:
            self.router.guardrail_threshold = min(
                self.router.guardrail_threshold + cfg.threshold_adjust_step,
                10.0,
            )
        elif avg_lock < cfg.min_lock_rate:
            self.router.guardrail_threshold = max(
                self.router.guardrail_threshold - cfg.threshold_adjust_step,
                0.1,
            )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_stats(self):
        print(f"\n=== Dynamic Bias Router v3 (step {self.step_count}) ===")
        for i in self.config.gap_sample_layers:
            stats = self.router.get_layer_stats(i)
            acc = self.router.get_accuracy_stats(i)
            token_sum = self._layer_token_sum[i].item()
            lock_rate = self._layer_locked_sum[i].item() / max(token_sum, 1)

            safe_alpha = self.router.compute_safe_alpha(self.config.alpha_safety_margin)
            print(
                f"  L{i:02d}: CV={stats['coeff_var']:.3f} "
                f"α={self.router.alpha:.3f} "
                f"α_safe={safe_alpha:.3f} "
                f"thr={self.router.guardrail_threshold:.2f} "
                f"lock={lock_rate:.3f} "
                f"flip={acc['flip_rate']:.4f} "
                f"drop={acc['drop_rate']:.5f} "
                f"gap_p10={acc['gap_p10']:.2f}"
            )
        print()

    def reset(self):
        self.router.reset()
        self._layer_locked_sum.zero_()
        self._layer_token_sum.zero_()
        self.step_count = 0


class TRTLLMRouterHook:
    """
    Hooks into TRT-LLM's MoE to replace routing decisions.
    Python-level hook for validation. Production: use TRT plugin.
    """

    def __init__(self, config: DynamicBiasConfig, device: str = "cuda"):
        self.manager = DynamicBiasRouterManager(config, device)
        self.config = config
        self._step_tokens = 0

    def intercept_moe_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        router_weights: torch.Tensor,
        expert_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = torch.matmul(hidden_states.float(), router_weights.float())
        self._step_tokens = logits.shape[0]
        return self.manager.route(layer_idx, logits, expert_bias)

    def on_step_complete(self):
        self.manager.step(num_tokens=self._step_tokens)
        self._step_tokens = 0
