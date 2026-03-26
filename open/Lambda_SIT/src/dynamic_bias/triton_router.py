import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# gpt-oss-120b constants
# ---------------------------------------------------------------------------
GPT_OSS_NUM_EXPERTS = 128
GPT_OSS_TOPK = 4
GPT_OSS_NUM_LAYERS = 36
GPT_OSS_HIDDEN_SIZE = 2880


# ---------------------------------------------------------------------------
# Triton Kernels
# ---------------------------------------------------------------------------

@triton.jit
def _dynamic_bias_route_v3_kernel(
    # Pointers
    logits_ptr,          # [num_tokens, num_experts]
    bias_ptr,            # [num_experts]
    load_ema_ptr,        # [num_experts] normalized (1.0 = avg)
    counts_ptr,          # [num_experts] current step counts (for intra-step feedback)
    output_indices_ptr,  # [num_tokens, topk] int32
    output_weights_ptr,  # [num_tokens, topk]
    output_flags_ptr,    # [num_tokens]
    out_counts_ptr,      # [num_experts] atomicAdd target
    # Scalars
    num_tokens,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    alpha,
    alpha2,              # intra-step feedback (0 = disabled)
    guardrail_threshold,
):
    """
    Fused kernel: semantic baseline + bias adjustment + guardrail + top-k + weights.
    Each program instance handles one token.

    Semantic baseline: sem = raw + bias_base
      - Guardrail gap computed on sem
      - Routing: sem - α*load_ema (- α2*counts)
      - Weights: softmax(sem[selected]) — same for locked/unlocked
    """
    pid = tl.program_id(0)
    token_idx = pid

    if token_idx >= num_tokens:
        return

    expert_offsets = tl.arange(0, num_experts)
    mask = expert_offsets < num_experts
    logits_offset = token_idx * num_experts + expert_offsets

    raw_logits = tl.load(logits_ptr + logits_offset, mask=mask, other=-1e9)
    bias = tl.load(bias_ptr + expert_offsets, mask=mask, other=0.0)
    load = tl.load(load_ema_ptr + expert_offsets, mask=mask, other=0.0)

    # Semantic baseline: what model would compute without dynamic bias
    sem = raw_logits + bias

    # Routing scores
    adjusted = sem - alpha * load
    if alpha2 > 0.0:
        cur_counts = tl.load(counts_ptr + expert_offsets, mask=mask, other=0).to(tl.float32)
        adjusted = adjusted - alpha2 * cur_counts

    # ---- Guardrail: gap on SEMANTIC baseline ----
    top1_val = tl.max(sem, axis=0)
    masked_for_top2 = tl.where(sem == top1_val, -1e9, sem)
    top2_val = tl.max(masked_for_top2, axis=0)

    affinity_gap = top1_val - top2_val
    is_locked = affinity_gap > guardrail_threshold

    # ---- Top-K selection ----
    # Locked: top-1 forced from sem, remaining from adjusted
    # Unlocked: all from adjusted
    scores = tl.where(is_locked, sem, adjusted)

    # Pass 0: select top-1
    idx0 = tl.argmax(scores, axis=0)
    is_sel_0 = expert_offsets == idx0
    remaining = tl.where(is_sel_0, -1e9, adjusted)

    # Pass 1
    idx1 = tl.argmax(remaining, axis=0)
    remaining = tl.where(expert_offsets == idx1, -1e9, remaining)

    # Pass 2
    idx2 = tl.argmax(remaining, axis=0)
    remaining = tl.where(expert_offsets == idx2, -1e9, remaining)

    # Pass 3
    idx3 = tl.argmax(remaining, axis=0)

    # ---- Store indices (int32) ----
    tl.store(output_indices_ptr + token_idx * topk + 0, idx0.to(tl.int32))
    tl.store(output_indices_ptr + token_idx * topk + 1, idx1.to(tl.int32))
    tl.store(output_indices_ptr + token_idx * topk + 2, idx2.to(tl.int32))
    tl.store(output_indices_ptr + token_idx * topk + 3, idx3.to(tl.int32))

    # ---- Weights: softmax(sem[selected]) — SAME for locked and unlocked ----
    s0 = tl.load(logits_ptr + token_idx * num_experts + idx0) + tl.load(bias_ptr + idx0)
    s1 = tl.load(logits_ptr + token_idx * num_experts + idx1) + tl.load(bias_ptr + idx1)
    s2 = tl.load(logits_ptr + token_idx * num_experts + idx2) + tl.load(bias_ptr + idx2)
    s3 = tl.load(logits_ptr + token_idx * num_experts + idx3) + tl.load(bias_ptr + idx3)

    max_val = tl.maximum(tl.maximum(s0, s1), tl.maximum(s2, s3))
    e0 = tl.exp(s0 - max_val)
    e1 = tl.exp(s1 - max_val)
    e2 = tl.exp(s2 - max_val)
    e3 = tl.exp(s3 - max_val)
    inv_sum = 1.0 / (e0 + e1 + e2 + e3)

    tl.store(output_weights_ptr + token_idx * topk + 0, e0 * inv_sum)
    tl.store(output_weights_ptr + token_idx * topk + 1, e1 * inv_sum)
    tl.store(output_weights_ptr + token_idx * topk + 2, e2 * inv_sum)
    tl.store(output_weights_ptr + token_idx * topk + 3, e3 * inv_sum)

    # ---- Guardrail flag ----
    tl.store(output_flags_ptr + token_idx, is_locked.to(tl.int32))

    # ---- Atomic count update ----
    tl.atomic_add(out_counts_ptr + idx0, 1)
    tl.atomic_add(out_counts_ptr + idx1, 1)
    tl.atomic_add(out_counts_ptr + idx2, 1)
    tl.atomic_add(out_counts_ptr + idx3, 1)


@triton.jit
def _ema_update_v3_kernel(
    load_ema_ptr,    # [total_elements]
    counts_ptr,      # [total_elements]
    beta,
    inv_mean,        # NUM_EXPERTS / (num_tokens * TOPK)
    total_elements,
    BLOCK: tl.constexpr,
):
    """Normalized EMA update + counts reset. All layers in one launch."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements

    old_ema = tl.load(load_ema_ptr + offsets, mask=mask, other=0.0)
    raw_count = tl.load(counts_ptr + offsets, mask=mask, other=0).to(tl.float32)

    # Normalize: 1.0 = average load
    norm_count = raw_count * inv_mean

    updated = beta * old_ema + (1.0 - beta) * norm_count

    tl.store(load_ema_ptr + offsets, updated, mask=mask)
    tl.store(counts_ptr + offsets, tl.zeros_like(raw_count).to(tl.int32), mask=mask)


# ---------------------------------------------------------------------------
# Python Interface
# ---------------------------------------------------------------------------

class DynamicBiasRouterV3:
    """
    Dynamic Bias Router v3 for gpt-oss-120b.

    Key changes from v2:
      - Normalized load_ema (1.0 = average expert)
      - Unified sem = raw + bias as baseline for guardrail/weights
      - Optional intra-step feedback (alpha2)
      - step() supports global_counts for EP AllReduce
      - int32 indices
      - Accuracy monitoring: flip/drop/lock rate with sampling window
      - Safe alpha bound accounts for alpha2 contribution

    Usage:
        router = DynamicBiasRouterV3()
        for layer_idx in range(36):
            indices, weights, flags = router.route(layer_idx, logits, bias)
        router.step(num_tokens)  # pass token count for normalization
    """

    def __init__(
        self,
        num_layers: int = GPT_OSS_NUM_LAYERS,
        num_experts: int = GPT_OSS_NUM_EXPERTS,
        topk: int = GPT_OSS_TOPK,
        alpha: float = 0.15,
        alpha2: float = 0.0,               # intra-step (0 = disabled)
        guardrail_threshold: float = 1.5,
        ema_beta: float = 0.85,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.topk = topk
        self.alpha = alpha
        self.alpha2 = alpha2
        self.guardrail_threshold = guardrail_threshold
        self.ema_beta = ema_beta
        self.device = device

        total = num_layers * num_experts
        self.counts = torch.zeros(total, dtype=torch.int32, device=device)
        self.load_ema = torch.zeros(total, dtype=torch.float32, device=device)

        self.step_count = 0
        self._last_num_tokens = 0

        # ---- Accuracy monitoring (Layer 3 + 4) ----
        self._flip_counts = [0] * num_layers    # top-1 flipped (unlocked only)
        self._drop_counts = [0] * num_layers    # top-1 NOT in final top-4 (hard safety)
        self._lock_counts = [0] * num_layers    # locked token count
        self._token_counts = [0] * num_layers   # total token count
        # Per-layer gap stats (updated periodically via sample_gap_stats)
        self._gap_p10 = [float('inf')] * num_layers
        self._gap_p20 = [float('inf')] * num_layers
        self._max_load_ema = [1.0] * num_layers
        # Sampling control
        self.accuracy_stats_enabled = False

    def route(
        self,
        layer_idx: int,
        logits: torch.Tensor,                  # [num_tokens, num_experts]
        bias: Optional[torch.Tensor] = None,   # [num_experts]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens for one layer. Returns (indices_i32, weights, flags)."""
        num_tokens = logits.shape[0]
        assert logits.shape[1] == self.num_experts

        logits_f = logits.float().contiguous()
        if bias is None:
            bias = torch.zeros(self.num_experts, dtype=torch.float32, device=self.device)
        else:
            bias = bias.float().contiguous()

        offset = layer_idx * self.num_experts
        layer_ema = self.load_ema[offset:offset + self.num_experts]
        layer_counts = self.counts[offset:offset + self.num_experts]

        # int32 indices (not int64)
        indices = torch.empty(num_tokens, self.topk, dtype=torch.int32, device=self.device)
        weights = torch.empty(num_tokens, self.topk, dtype=torch.float32, device=self.device)
        flags = torch.empty(num_tokens, dtype=torch.int32, device=self.device)

        grid = (num_tokens,)
        _dynamic_bias_route_v3_kernel[grid](
            logits_f, bias, layer_ema, layer_counts,
            indices, weights, flags,
            layer_counts,  # atomicAdd target
            num_tokens,
            self.num_experts,
            self.topk,
            self.alpha,
            self.alpha2,
            self.guardrail_threshold,
        )

        self._last_num_tokens = num_tokens

        # ---- Accuracy monitoring: detect flip + drop (only when sampling enabled) ----
        if self.accuracy_stats_enabled:
            sem = logits_f + bias.unsqueeze(0)                 # [N, 128]
            sem_top1_idx = sem.argmax(dim=1)                   # [N]
            actual_top1 = indices[:, 0].long()                 # [N]

            unlocked = (flags == 0)

            # Flip = unlocked token whose final top-1 != semantic top-1
            flipped = (sem_top1_idx != actual_top1) & unlocked
            self._flip_counts[layer_idx] += flipped.sum().item()

            # Drop = semantic top-1 NOT in final top-4 at all (hard safety metric)
            sem_top1_expanded = sem_top1_idx.unsqueeze(1)      # [N, 1]
            top1_in_topk = (indices.long() == sem_top1_expanded).any(dim=1)  # [N]
            dropped = (~top1_in_topk)
            self._drop_counts[layer_idx] += dropped.sum().item()

            self._lock_counts[layer_idx] += flags.sum().item()
            self._token_counts[layer_idx] += num_tokens

        return indices, weights, flags

    def step(
        self,
        num_tokens: int = 0,
        global_counts: Optional[torch.Tensor] = None,
    ):
        """
        Normalized EMA update + reset counts.
        Call ONCE per decode step after all 36 layers.

        Args:
            num_tokens: tokens this step (for normalization). 0 = use last route() value.
            global_counts: optional AllReduced counts [36*128] for EP global view.
        """
        total = self.num_layers * self.num_experts
        nt = num_tokens if num_tokens > 0 else self._last_num_tokens
        if nt <= 0:
            nt = 1
        inv_mean = float(self.num_experts) / (float(nt) * float(self.topk))

        counts_to_use = global_counts if global_counts is not None else self.counts

        BLOCK = 256
        grid = ((total + BLOCK - 1) // BLOCK,)
        _ema_update_v3_kernel[grid](
            self.load_ema, counts_to_use,
            self.ema_beta, inv_mean, total, BLOCK,
        )

        # If global_counts was used, still reset local counts
        if global_counts is not None:
            self.counts.zero_()

        self.step_count += 1

    def reset(self):
        self.counts.zero_()
        self.load_ema.zero_()
        self.step_count = 0

    def get_layer_stats(self, layer_idx: int) -> dict:
        offset = layer_idx * self.num_experts
        ema = self.load_ema[offset:offset + self.num_experts].float()
        counts = self.counts[offset:offset + self.num_experts].float()
        mean_ema = ema.mean().item()
        return {
            "layer": layer_idx,
            "ema_mean": mean_ema,
            "ema_std": ema.std().item(),
            "ema_max": ema.max().item(),
            "ema_min": ema.min().item(),
            "coeff_var": (ema.std() / (mean_ema + 1e-8)).item(),
            "step_count_sum": counts.sum().item(),
            "top5_loaded": torch.topk(ema, 5).indices.tolist(),
        }

    # ------------------------------------------------------------------
    # Accuracy monitoring (Layer 3 + Layer 4)
    # ------------------------------------------------------------------

    def sample_gap_stats(
        self,
        layer_idx: int,
        logits: torch.Tensor,                   # [num_tokens, 128]
        bias: Optional[torch.Tensor] = None,     # [128]
        max_samples: int = 256,
    ):
        """
        Sample gap(top1, top5) distribution on sem for α safety bound.
        Call periodically (e.g. every 50 steps) — NOT every route() call.
        """
        if bias is None:
            bias = torch.zeros(self.num_experts, dtype=torch.float32, device=self.device)

        sem = logits.float() + bias.float().unsqueeze(0)

        N = sem.shape[0]
        if N > max_samples:
            idx = torch.randperm(N, device=self.device)[:max_samples]
            sem = sem[idx]

        top5_vals, _ = torch.topk(sem, 5, dim=1)
        gap_top1_top5 = top5_vals[:, 0] - top5_vals[:, 4]

        self._gap_p10[layer_idx] = torch.quantile(gap_top1_top5, 0.10).item()
        self._gap_p20[layer_idx] = torch.quantile(gap_top1_top5, 0.20).item()

        offset = layer_idx * self.num_experts
        ema = self.load_ema[offset:offset + self.num_experts]
        self._max_load_ema[layer_idx] = ema.max().item()

    def compute_safe_alpha(self, safety_margin: float = 0.8) -> float:
        """
        Layer 3: compute α upper bound that guarantees semantic top-1
        stays in the final top-4 for ~90% of tokens.

        Math: α * max(load_ema) + α2 * max(cur_count) < gap(top1, top5)
              → α_safe = (p10(gap) - α2 * max_count_est) / max(load_ema) * margin

        If alpha2 > 0, the intra-step count term eats into the gap budget.
        We estimate max(cur_count) conservatively as 3× the average count.
        """
        safe_alphas = []
        for layer_idx in range(self.num_layers):
            gap = self._gap_p10[layer_idx]
            max_load = self._max_load_ema[layer_idx]
            if max_load > 0.01 and gap < float('inf'):
                # Subtract alpha2's worst-case contribution from gap budget
                effective_gap = gap
                if self.alpha2 > 0.0:
                    avg_count = max(self._last_num_tokens * self.topk / self.num_experts, 1.0)
                    max_count_est = avg_count * 3.0
                    effective_gap = gap - self.alpha2 * max_count_est
                    if effective_gap <= 0:
                        safe_alphas.append(0.01)
                        continue

                safe_alphas.append(effective_gap / max_load)

        if not safe_alphas:
            return self.alpha

        return min(safe_alphas) * safety_margin

    def get_accuracy_stats(self, layer_idx: int) -> dict:
        tc = self._token_counts[layer_idx]
        if tc == 0:
            return {
                "layer": layer_idx,
                "flip_rate": 0.0, "drop_rate": 0.0, "lock_rate": 0.0,
                "reroute_rate": 0.0,
                "gap_p10": self._gap_p10[layer_idx],
                "gap_p20": self._gap_p20[layer_idx],
                "max_load_ema": self._max_load_ema[layer_idx],
                "total_tokens": 0,
            }
        return {
            "layer": layer_idx,
            "flip_rate": self._flip_counts[layer_idx] / tc,
            "drop_rate": self._drop_counts[layer_idx] / tc,
            "lock_rate": self._lock_counts[layer_idx] / tc,
            "reroute_rate": (tc - self._lock_counts[layer_idx]) / tc,
            "gap_p10": self._gap_p10[layer_idx],
            "gap_p20": self._gap_p20[layer_idx],
            "max_load_ema": self._max_load_ema[layer_idx],
            "total_tokens": tc,
        }

    def reset_accuracy_stats(self):
        for i in range(self.num_layers):
            self._flip_counts[i] = 0
            self._drop_counts[i] = 0
            self._lock_counts[i] = 0
            self._token_counts[i] = 0


# Backward-compatible alias
DynamicBiasRouterV2 = DynamicBiasRouterV3
