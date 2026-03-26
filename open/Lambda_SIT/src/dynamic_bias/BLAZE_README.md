# BLAZE: Bias-Driven Load-Aware Zero-Overhead Expert Routing

MLPerf Inference v6.0 Open Track | GPT-OSS-120B Server | TRT-LLM | B200 Blackwell (sm_100)

## Results (MLPerf v6.0 Open Track)

| Metric | Value |
|--------|-------|
| Accuracy | 83.66% (threshold >= 82.30%) |
| Performance | 16,167 completed tokens/s (Server, VALID) |
| TTFT P99 usage | 66.7% of limit |
| TPOT P99 usage | 12.8% of limit |
| CV reduction | 18.1% (expert load imbalance) |

## Improvement vs Baseline (Server, B200-SXM-180GBx8_TRT)

Comparison between the BLAZE run and the benchmark run, both VALID with
`target_qps=12`, `min_query_count=51168`, and identical RNG seeds.

| Metric | BLAZE | Benchmark | Delta |
|--------|-------------------|------------------------|-------|
| Completed tokens/s | 16,167.37 | 15,559.85 | +607.52 (+3.9%) |
| Mean TTFT (s) | 2.13 | 3.52 | -1.39 (-39.5%) |
| Mean TPOT (ms) | 10.28 | 9.58 | +0.70 (+7.3%) |
| TTFT P99 usage | 66.7% | 96.8% | -30.1% |
| TPOT P99 usage | 12.8% | 12.0% | +0.8% |

Log references:
`/work/build/logs/2026.02.13-16.23.16/B200-SXM-180GBx8_TRT/gpt-oss-120b/Server/`
and
`/work/build/logs/2026.02.18-22.50.46/B200-SXM-180GBx8_TRT/gpt-oss-120b/Server/`.

## What We Changed (Code)

We added a dynamic-bias router to the TRT-LLM MoE routing path for GPT-OSS-120B.
The router applies a lightweight load penalty to expert routing scores at runtime,
reducing load imbalance across experts without modifying model weights or retraining.

The primary changes are:

| File | Description |
|------|-------------|
| `tensorrt_llm/dynamic_bias/dynamic_bias_router_v3.cu` | CUDA kernel (sm_100) |
| `tensorrt_llm/dynamic_bias/router.py` | ctypes wrapper for the CUDA .so |
| `tensorrt_llm/dynamic_bias/triton_router.py` | Triton reference implementation (validation) |
| `tensorrt_llm/dynamic_bias/router_manager.py` | Singleton manager for parameters, EMA state, per-step update |
| `tensorrt_llm/dynamic_bias/trtllm_integration.py` | TRT-LLM hook |
| `tensorrt_llm/dynamic_bias/mlperf_adapter.py` | MLPerf LoadGen harness adapter |
| `tensorrt_llm/dynamic_bias/test_router.py` | 15 correctness tests + benchmarks |
| `tensorrt_llm/dynamic_bias/build.sh` | Build script (sm_100) |
| `tensorrt_llm/_torch/modules/fused_moe/routing.py` | Integration point in fused MoE routing |
| `tensorrt_llm/_torch/models/modeling_gpt_oss.py` | Environment-controlled enablement and per-step call |

## Method Summary

### Motivation

In large MoE models under tensor/expert parallelism, expert load imbalance causes
slower ranks to block AllReduce synchronization. With 72+ sync points per forward
pass, even small imbalances compound into significant overhead. BLAZE applies a 
negative-feedback bias to expert routing scores at runtime, reducing load variance 
without modifying model weights or retraining.

### Core Algorithm

For each token and layer, we compute a load-penalized routing score:
```
base[e]  = logits[e] + bias_base[e]              # original routing score
score[e] = base[e] - alpha * load_ema[e]          # load-penalized score
```

- Top-K expert selection uses `score` (load-aware).
- Expert mixture weights use `softmax(base)` -- load penalty never enters weight computation.
- `load_ema` is normalized: 1.0 = average expert load, stable across batch sizes.
- Updated once per decode step via exponential moving average (beta = 0.85).

### Affinity Guardrail

When the gap between the original top-1 and top-2 expert logits exceeds
`threshold`, the top-1 expert is locked in place -- identical to the unmodified
model. The remaining K-1 slots use load-adjusted scores to rank the top-4 experts. This ensures high-confidence routing decisions stay unchanged while only ambiguous slots
participate in load balancing.

### Accuracy Protection

**Protection 1 — Affinity Guardrail**: For each token, compute the gap between
the original top-1 and top-2 expert logits (before any load penalty is applied).
If this gap exceeds `guardrail_threshold` (1.5), the top-1 expert is locked —
it stays selected regardless of load penalty. Only the remaining K-1 slots are
subject to load-based reranking.

**Protection 2 — Weight Preservation**: Expert mixture weights always come from
`softmax(original_logits)`, never from load-adjusted scores. The model's
output blending is unchanged even when expert selection is load-adjusted.

**Protection 3 — Drop/Flip Monitor**: Tracks two runtime metrics:
- **Drop rate**: original top-1 expert not in final top-K. Hard violation. alpha *= 0.5
- **Flip rate**: original top-1 in top-K but demoted from rank-0. Soft signal. alpha *= 0.7

## Architecture: Feedback Loop

BLAZE operates as a closed-loop feedback controller embedded inside the TRT-LLM
decode pipeline. Here is the complete flow for one decode step:

### Phase 1 — Forward Pass (per layer, 36 times)

For each of the 36 MoE layers in GPT-OSS-120B, the router intercepts the
original expert logits and applies the current bias:
```
route(layer_id, logits, bias):
    1. base[e] = logits[e] + bias_base[e]           # original routing score
    2. score[e] = base[e] - alpha * load_ema[e]      # subtract load penalty
    3. Check guardrail:
       - gap = base[top1] - base[top2]
       - If gap > threshold: lock top-1, select remaining 3 from score[]
       - Else: select all top-4 from score[]
    4. weights = softmax(base[selected])              # weights from original scores only
    5. atomicAdd(counts[layer_id][selected_experts])  # record which experts got chosen
    6. Return selected expert indices + weights to TRT-LLM MoE kernel
```

At this point, `counts[36][128]` contains how many tokens expert_id 128 received
in the current step, on layer 36. This is the raw signal for the feedback loop.

### Phase 2 — End-of-Step Update (once per decode step)

After all 36 layers complete their forward pass, `step(num_tokens)` runs the
feedback update. This is ONE CUDA kernel processing all 36×128 entries:
```
step(num_tokens):
    # --- 1. Global Aggregation (EP only) ---
    # If EP > 1, each GPU only has partial counts (its own experts).
    # AllReduce merges counts across all EP ranks so every GPU sees full picture.
    if ep_global_sync:
        nccl_allreduce(counts)           

    # --- 2. Normalize Counts ---
    # Convert raw counts to relative load. 1.0 = average load.
    # A value of 2.0 means this expert got 2x average tokens.
    for each layer L:
        avg = sum(counts[L]) / num_experts   
        normalized[L][e] = counts[L][e] / avg  # 1.0 = average, 2.0 = 2x overloaded

    # --- 3. EMA Update ---
    # Smooth the normalized counts into a running average.
    # This prevents the bias from overreacting to single-step spikes.
    for each layer L, each expert e:
        load_ema[L][e] = beta * load_ema[L][e] + (1 - beta) * normalized[L][e]
        # beta=0.85: 85% old history + 15% new observation

    # --- 4. Reset Counts ---
    # Clear counts for the next step.
    memset(counts, 0)

    # --- 5. Safety Checks (only during sampling window) ---
    # Sampling window: collect stats for 2 steps, skip 48 steps (~4% duty cycle).
    # This avoids atomicAdd contention on flip/drop counters during normal operation.
    if in_sampling_window:
        for each layer L:
            drop_rate = drop_cnt[L] / total_tokens[L]
            flip_rate = flip_cnt[L] / total_tokens[L]

            # Protection 3: emergency alpha reduction
            if drop_rate > 0.001:       # original top-1 lost entirely
                alpha *= 0.5            # emergency: halve the penalty
            elif flip_rate > 0.05:      # original top-1 demoted from rank-0
                alpha *= 0.7            # softer reduction

        # Write updated alpha to device memory (CUDA Graph safe)
        params[0] = alpha
```

### Timing

The entire `step()` runs in ~890μs — a single CUDA kernel covering all 36 layers
× 128 experts. The forward pass for one decode step takes hundreds of milliseconds,
so the feedback overhead is <0.1% of total step time.

### State Layout

```
counts[36][128]    int32    per-step atomic counters (reset each step)
load_ema[36][128]  float32  normalized smoothed load (persists across steps)
params[3]          float32  device scalars: [alpha, threshold, alpha2] (CUDA Graph safe)
flip_cnt[36]       int32    per-layer flip counter (sampling window)
lock_cnt[36]       int32    per-layer lock counter
drop_cnt[36]       int32    per-layer drop counter
```

## Submission Configuration

| Category | Parameter | Value | Notes |
|----------|-----------|-------|-------|
| **System** | GPUs | 8x NVIDIA B200 SXM 180GB | Single node |
| | Tensor Parallelism | 4 | |
| | Expert Parallelism | 4 | |
| | Framework | TensorRT-LLM | |
| | Precision | FP4 weights, FP8 KV cache | |
| | Max concurrency | 600 | |
| | Target QPS | 12 | |
| | Max num tokens | 4096 | |
| | Batch scheduler | max_util, chunked context | |
| **BLAZE** | `alpha` | 0.15 | Load penalty on normalized EMA |
| | `guardrail_threshold` | 1.5 | Logit gap cutoff for locking top-1 |
| | `ema_beta` | 0.85 | EMA smoothing factor |
| | `max_drop_rate` | 0.001 | Hard safety trigger, alpha *= 0.5 |
| | `max_flip_rate` | 0.05 | Soft safety trigger, alpha *= 0.7 |
| | `alpha_safety_margin` | 0.8 | Protection 3 headroom multiplier |

## How to Reproduce

### 1. Build the CUDA router library

```bash
cd <TRTLLM_ROOT>/tensorrt_llm/dynamic_bias
GPU_ARCH=sm_100 ./build.sh cuda
```

This produces `libdynamic_bias_router.so` in the same directory.

### 2. Set environment variables

```bash
export ENABLE_DYNAMIC_BIAS_ROUTER=1
export DYNAMIC_BIAS_USE_CUDA=1
export DYNAMIC_BIAS_LIB_PATH=<TRTLLM_ROOT>/tensorrt_llm/dynamic_bias/libdynamic_bias_router.so
export DYNAMIC_BIAS_ALPHA=0.15
export DYNAMIC_BIAS_ALPHA2=0.0
export DYNAMIC_BIAS_GUARDRAIL=1.5
export DYNAMIC_BIAS_EMA_BETA=0.85
```

### 3. Run MLPerf harness (Server scenario)

```bash
cd <MLPERF_ROOT>

# Accuracy run
make run_llm_server RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Server \
  --core_type=trtllm_endpoint --test_mode=AccuracyOnly"

# Performance run
make run_llm_server RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Server \
  --core_type=trtllm_endpoint --test_mode=PerformanceOnly"
```

### 4. Validate and package

```bash
make stage_results
make truncate_results SUBMITTER=Lambda_SIT
make copy_results_artifacts
make check_submission SUBMITTER=Lambda_SIT
make pack_submission
```

For code, please see (this repo)[https://github.com/RyanAIResearch/dynamic_bias]
For technical report, please see (here)[https://www.researchgate.net/publication/401331074_BLAZE_Bias-Driven_Load-Aware_Zero-Overhead_Expert_Routing]